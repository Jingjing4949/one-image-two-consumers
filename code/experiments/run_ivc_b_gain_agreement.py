"""Agreement of per-image task gains between endpoint seeds and architectures.

Reads saved eval.jsonl files only. Gains are relative to the original image under the
same endpoint, so the anchor's own variation is removed before comparison.
"""
import json
from pathlib import Path

import numpy as np

from run_ivc_b_finish_dev import ACTIONS

OUT = Path('/root/ivcb/out')
CODE = Path('/root/ivcb/code')
SETS = {'suim': ('outputs/ivc_b2_suime_gpu_v1', OUT / 'b3', 'iou'), 'tub': ('outputs/ivc_b2_tub_v1', OUT / 'tub_b3', 'fg_iou')}
TOL = 0.01


def table(b3, name, metric):
    t = {}
    for fold in range(5):
        for line in (b3 / name.format(fold=fold) / 'eval.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                t[(r['id'], r['action'])] = r[metric]
    return t


def gains(t, ids):
    return np.array([[t[(i, a)] - t[(i, 'identity')] for a in ACTIONS[1:]] for i in ids])


def groups_of(qdir):
    g = {}
    for line in (CODE / qdir / 'quality_identity.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            g[r['id']] = r['group_id']
    return g


def compare(a, b, groups):
    d = np.abs(a - b)
    harm_a, harm_b = a < -TOL, b < -TOL
    gain_a, gain_b = a > TOL, b > TOL
    names = np.array(groups)
    keys = sorted(set(groups))
    ga = np.array([bool(harm_a[names == k].any()) for k in keys])
    gb = np.array([bool(harm_b[names == k].any()) for k in keys])
    big = np.abs(a) > TOL
    return dict(pairs=int(d.size), mean_abs=float(d.mean()), median_abs=float(np.median(d)), p95_abs=float(np.quantile(d, .95)),
                frac_abs_above_tol=float((d > TOL).mean()),
                harm_label_agreement=float((harm_a == harm_b).mean()),
                harm_rate=[float(harm_a.mean()), float(harm_b.mean())],
                gain_rate=[float(gain_a.mean()), float(gain_b.mean())],
                sign_agreement_where_first_exceeds_tol=float((np.sign(a[big]) == np.sign(b[big])).mean()),
                pearson=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                action_mean_gain_pearson=float(np.corrcoef(a.mean(0), b.mean(0))[0, 1]),
                group_any_harm_agreement=float((ga == gb).mean()),
                group_any_harm=[int(ga.sum()), int(gb.sum())])


def main():
    result = {}
    for name, (qdir, b3, metric) in SETS.items():
        base = table(b3, 'lraspp__raw__fold{fold}', metric)
        seed2 = table(b3, 'lraspp__raw__fold{fold}__seed2', metric)
        deep = table(b3, 'deeplabv3__raw__fold{fold}', metric)
        gmap = groups_of(qdir)
        ids = sorted(i for i in gmap if all((i, a) in base and (i, a) in seed2 and (i, a) in deep for a in ACTIONS))
        groups = [gmap[i] for i in ids]
        g0, g1, g2 = gains(base, ids), gains(seed2, ids), gains(deep, ids)
        result[name] = dict(images=len(ids), groups=len(set(groups)),
                            lraspp_seed1_vs_seed2=compare(g0, g1, groups),
                            lraspp_vs_deeplab=compare(g0, g2, groups))
        print(name, json.dumps(result[name]), flush=True)
    (OUT / 'b_gain_agreement.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('AGREEMENT_DONE', flush=True)


if __name__ == '__main__':
    main()
