"""One probe rule: pick the action with the higher raw-model confidence.

Uses the saved raw LR-ASPP and DeepLab max-probability, averaged with equal
weight. No threshold and no second rule. Development readout only.
"""
import json
from pathlib import Path

import numpy as np

from run_ivc_b5_cnn import ACTIONS, load_joined


def confidence(b3):
    score = {}
    for arch in ('lraspp', 'deeplabv3'):
        for fold in range(5):
            path = Path(b3) / f'{arch}__raw__fold{fold}' / 'confidence.jsonl'
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                score.setdefault((row['id'], row['action']), []).append(row['mean_max_prob'])
    return {key: float(np.mean(vals)) for key, vals in score.items() if len(vals) == 2}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--quality-dir', required=True)
    ap.add_argument('--b3', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    records, ids, gains, fold_of = load_joined(a.records, a.quality_dir, a.b3)
    score = confidence(a.b3)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    by_fold = {fold: [] for fold in range(5)}
    for i in ids:
        vals = [score.get((i, action)) for action in ACTIONS]
        if any(v is None for v in vals):
            continue
        pick = int(np.argmax(vals))
        true = gains[i]
        row = dict(id=i, group_id=records[i]['group_id'], fold=fold_of[i], chosen=ACTIONS[pick],
                   confidence=[float(v) for v in vals], joint=float(true[pick].sum()),
                   oracle=float(true.sum(axis=1).max()))
        lines.append(json.dumps(row) + '\n')
        by_fold[fold_of[i]].append(row)
    (out / 'predictions.jsonl').write_text(''.join(lines), encoding='utf-8')
    summary = []
    for fold, rows in by_fold.items():
        if not rows:
            continue
        fixed = int(np.mean([gains[i] for i in ids if fold_of[i] != fold], axis=0).sum(axis=1).argmax())
        summary.append(dict(fold=fold, test_images=len(rows), fixed_action=ACTIONS[fixed],
                            mean_joint_probe=float(np.mean([r['joint'] for r in rows])),
                            mean_joint_fixed=float(np.mean([gains[r['id']][fixed].sum() for r in rows])),
                            mean_joint_oracle=float(np.mean([r['oracle'] for r in rows]))))
    (out / 'summary.json').write_text(json.dumps(dict(images=len(lines), rule='mean raw max-probability',
                                                      folds=summary), indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    print('PROBE_DONE', len(lines), flush=True)


if __name__ == '__main__':
    main()
