"""B4: split each action's task effect into a retraining-removable shift and a residual.

For action a and image i (out-of-fold predictions, LR-ASPP):
  total    = IoU_raw(a)        - IoU_raw(identity)    raw-trained model on a vs on the input
  residual = IoU_matched_a(a)  - IoU_raw(identity)    a-trained model on a vs raw model on the input
  shift    = total - residual
  robust   = IoU_mixed(a)      - IoU_mixed(identity)  model trained on all actions
Group-macro means and harmed-group counts (loss > 0.01) are reported per action.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TOL = 0.01


def rows(b3, run, metric):
    out = {}
    for fold in range(5):
        p = Path(b3) / f'{run}__fold{fold}' / 'eval.jsonl'
        if not p.exists():
            return None
        for line in open(p):
            r = json.loads(line)
            out[(r['id'], r['action'])] = (r[metric], r['group_id'])
    return out


def macro(values, groups):
    per = defaultdict(list)
    for v, g in zip(values, groups):
        per[g].append(v)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b3', required=True)
    ap.add_argument('--actions', nargs='+', required=True)
    ap.add_argument('--arch', default='lraspp')
    ap.add_argument('--metric', default='iou')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    raw = rows(a.b3, f'{a.arch}__raw', a.metric)
    mixed = rows(a.b3, f'{a.arch}__mixed', a.metric)
    result = {}
    for act in a.actions:
        if act == 'identity':
            continue
        matched = rows(a.b3, f'{a.arch}__matched-{act}', a.metric)
        ids = sorted(i for (i, x) in raw if x == 'identity' and (i, act) in raw and matched and (i, act) in matched)
        if not ids:
            continue
        g = [raw[(i, 'identity')][1] for i in ids]
        base = np.array([raw[(i, 'identity')][0] for i in ids])
        total = np.array([raw[(i, act)][0] for i in ids]) - base
        resid = np.array([matched[(i, act)][0] for i in ids]) - base
        robust = (np.array([mixed[(i, act)][0] for i in ids]) - np.array([mixed[(i, 'identity')][0] for i in ids])) if mixed else None
        res = {}
        for name, v in [('total', total), ('residual', resid), ('shift', total - resid)] + ([('mixed', robust)] if robust is not None else []):
            per = macro(v, g)
            gm = np.array([np.mean(x) for x in per.values()])
            res[name] = dict(group_macro=float(gm.mean()),
                             harmed_groups=int(sum(bool((np.array(x) < -TOL).any()) for x in per.values())))
        per_t, per_r = macro(total, g), macro(resid, g)
        harmed_total = {k for k, x in per_t.items() if (np.array(x) < -TOL).any()}
        still = {k for k in harmed_total if (np.array(per_r[k]) < -TOL).any()}
        res['groups'] = len(per_t)
        res['harmed_under_raw_model'] = len(harmed_total)
        res['still_harmed_after_matched_retraining'] = len(still)
        res['share_still_harmed'] = len(still) / len(harmed_total) if harmed_total else None
        result[act] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    for act, r in result.items():
        print(f"{act:20s} total={r['total']['group_macro']:+.4f} residual={r['residual']['group_macro']:+.4f} "
              f"shift={r['shift']['group_macro']:+.4f} harmed={r['harmed_under_raw_model']} still={r['still_harmed_after_matched_retraining']}"
              + (f" mixed={r['mixed']['group_macro']:+.4f}" if 'mixed' in r else ''))


if __name__ == '__main__':
    main()
