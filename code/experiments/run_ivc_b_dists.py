"""Secondary DISTS scores for stored 8-bit outputs against the same resized references.

This does not rerun enhancement. It only reads records already listed in the
quality split, so TUB holdout scenes stay unread. Lower DISTS is better.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from piq import DISTS


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def output_path(store, rec, action):
    base = Path(store) / action / rec['id'].replace('/', '__')
    png = base.with_suffix('.png')
    return png if png.exists() else base.with_suffix('.webp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--quality-dir', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    payload = json.loads(Path(a.records).read_text(encoding='utf-8'))
    holdout = set(payload.get('holdout_experiments') or [])
    records = payload['records']
    leaked = [r['id'] for r in records if r.get('experiment') in holdout]
    if leaked:
        raise SystemExit('holdout records are in the split: ' + str(len(leaked)))
    by_id = {r['id']: r for r in records}
    quality_dir = Path(a.quality_dir)
    device = torch.device(a.device)
    metric = DISTS(reduction='none').to(device).eval()
    files = sorted(quality_dir.glob('quality_*.jsonl'))
    if not files:
        raise SystemExit('no quality jsonl in ' + str(quality_dir))
    done_n = 0
    total_n = 0
    for path in files:
        action = path.name[len('quality_'):-len('.jsonl')]
        out = quality_dir / f'dists_{action}.jsonl'
        done = set()
        if out.exists():
            done = {json.loads(line)['id'] for line in out.read_text(encoding='utf-8').splitlines() if line.strip()}
        pending = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            total_n += 1
            if row['id'] not in done and row['id'] in by_id:
                pending.append(row['id'])
        done_n += len(done)
        if not pending:
            print('dists complete', action, len(done), flush=True)
            continue
        with out.open('a', encoding='utf-8') as stream, torch.inference_mode():
            for start in range(0, len(pending), a.batch):
                chunk = pending[start:start + a.batch]
                refs, outs, ids = [], [], []
                shape = None
                bucket = []
                def flush(bucket):
                    if not bucket:
                        return
                    ref_b = torch.from_numpy(np.stack([x[0] for x in bucket])).permute(0, 3, 1, 2).float().div_(255).to(device)
                    out_b = torch.from_numpy(np.stack([x[1] for x in bucket])).permute(0, 3, 1, 2).float().div_(255).to(device)
                    values = metric(out_b, ref_b).detach().float().cpu().view(-1).tolist()
                    for rec_id, group_id, value in zip([x[2] for x in bucket], [x[3] for x in bucket], values):
                        stream.write(json.dumps(dict(id=rec_id, group_id=group_id, action=action, dists=float(value))) + '\n')
                    stream.flush()
                for rec_id in chunk:
                    rec = by_id[rec_id]
                    pred = load_rgb(output_path(a.store, rec, action))
                    ref = np.asarray(Image.open(Path(a.root) / rec['reference']).convert('RGB').resize((pred.shape[1], pred.shape[0]), Image.Resampling.BICUBIC), dtype=np.uint8)
                    if shape != pred.shape:
                        flush(bucket)
                        bucket, shape = [], pred.shape
                    bucket.append((np.ascontiguousarray(ref), np.ascontiguousarray(pred), rec_id, rec['group_id']))
                flush(bucket)
                done_n += len(chunk)
                print(f'dists {action} {done_n}/{total_n}', flush=True)
    print('DISTS_DONE', done_n, flush=True)


if __name__ == '__main__':
    main()
