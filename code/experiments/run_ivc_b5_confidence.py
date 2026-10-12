"""One confidence pass over completed segmentation checkpoints.

Writes mean max-softmax, entropy and top-2 margin on the same upsampled grid
used for IoU. Does not retrain and does not rewrite eval.jsonl or complete.json.
"""
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_ivc_b3_seg as seg


def _forward_chunk(model, chunk, device):
    rows = []
    by_shape = {}
    for rec, action, image, native in chunk:
        by_shape.setdefault(image.shape, []).append((rec, action, image, native))
    with torch.inference_mode():
        for group in by_shape.values():
            for start in range(0, len(group), 8):
                part = group[start:start + 8]
                batch = np.stack([item[2] for item in part])
                x = seg.to_tensor(batch, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)['out'].float()
                for n, (rec, action, _, native) in enumerate(part):
                    up = F.interpolate(logits[n:n + 1], size=native, mode='bilinear', align_corners=False)
                    prob = torch.softmax(up[0], dim=0)
                    top2 = torch.topk(prob, k=2, dim=0).values
                    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(0)
                    rows.append(dict(id=rec['id'], group_id=rec['group_id'], action=action,
                                     mean_max_prob=float(prob.max(0).values.mean()),
                                     mean_entropy=float(entropy.mean()),
                                     mean_top2_margin=float((top2[0] - top2[1]).mean())))
    return rows


def confidence_rows(model, data, test, actions, device):
    jobs = [(rec, action) for rec in test for action in actions]

    def load(job):
        rec, action = job
        image = seg.load_rgb(data.output(rec, action))
        return rec, action, image, data.mask_native(rec).shape

    rows = []
    pending = []
    with ThreadPoolExecutor(8) as pool:
        for item in pool.map(load, jobs, chunksize=4):
            pending.append(item)
            if len(pending) >= 32:
                rows.extend(_forward_chunk(model, pending, device))
                pending = []
        if pending:
            rows.extend(_forward_chunk(model, pending, device))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--folds', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--classes', type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    seg.CLASSES = a.classes
    payload = json.loads(Path(a.folds).read_text(encoding='utf-8'))
    holdout = set(payload.get('holdout_experiments') or [])
    if any(r.get('experiment') in holdout for r in payload['records']):
        raise SystemExit('holdout records present')
    data = seg.Data(a.root, a.store, a.folds, a.masks)
    device = torch.device('cuda')
    runs = sorted(p for p in Path(a.runs).iterdir() if p.is_dir() and (p / 'complete.json').exists())
    print('runs', len(runs), flush=True)
    for run in runs:
        dest = run / 'confidence.jsonl'
        info = json.loads((run / 'complete.json').read_text(encoding='utf-8'))
        test = [r for r in data.records if r['fold'] == info['fold'] and r['task_geometry_ok']]
        actions = sorted({json.loads(line)['action'] for line in (run / 'eval.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()})
        expected = len(test) * len(actions)
        if dest.exists() and sum(1 for line in dest.read_text(encoding='utf-8').splitlines() if line.strip()) == expected:
            print('skip', run.name, flush=True)
            continue
        final = run / 'final.pt'
        if not zipfile.is_zipfile(final):
            print('bad_checkpoint', run.name, flush=True)
            continue
        model, _ = seg.build(info['arch'])
        ck = torch.load(final, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'])
        model = model.to(device).eval()
        rows = confidence_rows(model, data, test, actions, device)
        if len(rows) != expected:
            raise SystemExit(f'{run.name} wrote {len(rows)} rows, expected {expected}')
        dest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        print('confidence', run.name, len(rows), flush=True)
        del model
        torch.cuda.empty_cache()
    print('CONFIDENCE_DONE', flush=True)


if __name__ == '__main__':
    main()
