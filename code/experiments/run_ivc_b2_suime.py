"""B2: enhancement outputs and reference quality for all SUIM-E images at working size.

Outputs are stored as 8-bit PNG (the image a system would store or transmit).
Resumable: rows already present in an action's quality file are skipped after
their saved PNG hash is re-verified.
"""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT / 'work/metric_weights/torchhub'))
import lpips  # noqa: E402
import ivc_b_actions as actions_module  # noqa: E402

FOLDS = ROOT / 'outputs/ivc_b3_suime_folds_v1/folds.json'
OUT = ROOT / 'outputs/ivc_b2_suime_v1'
STORE = ROOT / 'work/ivc_b2_suime_outputs'


def now():
    return datetime.now(timezone.utc).isoformat()


def ssim(a, b):
    a = a.astype(np.float64) / 255
    b = b.astype(np.float64) / 255
    return float(np.mean([structural_similarity(a[:, :, c], b[:, :, c], win_size=7, data_range=1,
                                                gaussian_weights=False, use_sample_covariance=True) for c in range(3)]))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return float(-10 * math.log10(max(mse, 1e-12)))


def to_lpips_tensor(x):
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div(127.5).sub(1).unsqueeze(0)


def main():
    global OUT, STORE
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', nargs='+', default=actions_module.BUILTIN)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--store', default=str(STORE))
    parser.add_argument('--format', choices=['png', 'webp'], default='png')
    parser.add_argument('--records', default=str(FOLDS), help='records JSON with id, group_id, fold, input, reference')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    OUT, STORE = Path(args.out), Path(args.store)
    torch.set_num_threads(args.threads)
    records = json.loads(Path(args.records).read_text(encoding='utf-8'))['records']
    if args.limit:
        records = records[:args.limit]
    records = records[args.shard::args.shards]
    acts = actions_module.Actions(args.actions, args.device)
    device = torch.device(args.device)
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().requires_grad_(False).to(device)
    OUT.mkdir(parents=True, exist_ok=True)
    env_path = OUT / 'environment.json'
    if not env_path.exists():
        env_path.write_text(json.dumps(dict(at_utc=now(), torch=torch.__version__, threads=args.threads, device=args.device,
                                            short_side=actions_module.SHORT_SIDE, multiple=actions_module.MULTIPLE,
                                            output_format='PNG 8-bit RGB', reference_resize='bicubic to working size',
                                            ssim='skimage uniform 7x7, sample covariance, channel mean',
                                            lpips='alex v0.1, inputs scaled to [-1,1]',
                                            weights=acts.weights), indent=2), encoding='utf-8')
    done = {}
    for name in args.actions:
        path = OUT / f'quality_{name}.jsonl'
        done[name] = {}
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                done[name][row['id']] = row
    start = time.time()
    with torch.inference_mode():
        for i, r in enumerate(records):
            pending = [a for a in args.actions if r['id'] not in done[a]]
            if not pending:
                continue
            image, native = actions_module.load_working(ROOT / r['input'])
            reference = np.asarray(Image.open(ROOT / r['reference']).convert('RGB').resize(image.size, Image.Resampling.BICUBIC), dtype=np.uint8)
            ref_t = to_lpips_tensor(reference).to(device)
            for name in pending:
                target = STORE / name / (r['id'].replace('/', '__') + '.' + args.format)
                t0 = time.perf_counter()
                out = acts.run(name, image)
                elapsed = time.perf_counter() - t0
                target.parent.mkdir(parents=True, exist_ok=True)
                if args.format == 'png':
                    Image.fromarray(out).save(target, optimize=False, compress_level=6)
                else:
                    Image.fromarray(out).save(target, lossless=True, method=4, exact=True)
                saved = np.asarray(Image.open(target).convert('RGB'), dtype=np.uint8)
                assert np.array_equal(saved, out)
                row = dict(id=r['id'], group_id=r['group_id'], fold=r.get('fold', -1), action=name,
                           working_size_wh=list(image.size), native_size_wh=native,
                           output=target.relative_to(STORE).as_posix(), output_sha256=actions_module.sha(target),
                           ssim=ssim(out, reference), psnr_db=psnr(out, reference),
                           lpips=float(metric(to_lpips_tensor(out).to(device), ref_t).item()), action_seconds=elapsed,
                           device=args.device)
                with (OUT / f'quality_{name}.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
                done[name][r['id']] = row
            if i % 25 == 0:
                (OUT / f'progress_{args.shard}.json').write_text(json.dumps(dict(at_utc=now(), index=i, total=len(records),
                                                                   elapsed_seconds=round(time.time() - start, 1))), encoding='utf-8')
    (OUT / f'progress_{args.shard}.json').write_text(json.dumps(dict(at_utc=now(), index=len(records), total=len(records), complete=True,
                                                       elapsed_seconds=round(time.time() - start, 1), actions=args.actions)), encoding='utf-8')


if __name__ == '__main__':
    main()
