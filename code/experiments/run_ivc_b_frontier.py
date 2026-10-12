"""Coverage frontier and single-factor tolerance grid on existing scores.

The 0.01 result is not replaced. New solves use a fixed grid and a 60-second
limit; an unsolved point stays unsolved.
"""
import json
from pathlib import Path

import numpy as np

import ivc_b_compat as C

ACTIONS = ['identity', 'gray_world_mean', 'funie_s20260913', 'funie_s20261013', 'lite_s20260913',
           'lite_s20261013', 'semi_uir', 'phaseformer_uieb', 'deep_wavenet_euvp', 'uir_polykernel',
           'shallow_uwnet', 'syreanet', 'ushape_lsui', 'tacl', 'uiec2net']
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
VECTORS = {
    'base_0.01': dict(ssim=0.01, lpips=0.01, iou=0.01),
    'ssim_0.005': dict(ssim=0.005, lpips=0.01, iou=0.01),
    'ssim_0.02': dict(ssim=0.02, lpips=0.01, iou=0.01),
    'lpips_0.005': dict(ssim=0.01, lpips=0.005, iou=0.01),
    'lpips_0.02': dict(ssim=0.01, lpips=0.02, iou=0.01),
    'iou_0.005': dict(ssim=0.01, lpips=0.01, iou=0.005),
    'iou_0.02': dict(ssim=0.01, lpips=0.01, iou=0.02),
}


def safe(g, metrics, tol):
    return np.all(np.stack([g[:, :, C.IDX[m]] >= -tol[m] for m in metrics], axis=0), axis=0)


def capacity(g, groups, anchor, metrics, tol):
    ok = safe(g, metrics, tol)
    ok[:, anchor] = False
    names = np.array(groups)
    return int(sum(bool(ok[names == k].any()) for k in set(groups)))


def count_harm(g, groups, choices, metrics, tol):
    sel = g[np.arange(len(groups)), choices]
    names = sorted(set(groups))
    members = {k: np.flatnonzero(np.array(groups) == k) for k in names}
    total = 0
    for ix in members.values():
        block = sel[ix]
        bad = np.zeros(len(ix), dtype=bool)
        for m in metrics:
            bad |= block[:, C.IDX[m]] < -tol[m]
        total += int(bad.any())
    return total


def harm_at(g, groups, anchor, metrics, other, target, tol):
    old = C.TOL
    C.TOL = min(tol.values()) if len(set(tol.values())) == 1 else old
    try:
        if len(set(tol.values())) != 1:
            spec = C.build(g, groups, anchor, metrics)
            spec['allowed'] = safe(g, metrics, tol)
            spec['allowed'][:, anchor] = False
            n, A, G = spec['n'], spec['A'], spec['G']
            bad = np.stack([g[:, :, C.IDX[m]] < -tol[m] for m in other], axis=0).any(axis=0)
            extra = [({n * A + j: 1 for j in range(G)}, target, target)]
            base = spec['count']
            for j, k in enumerate(spec['names']):
                for i in spec['members'][k]:
                    for a in range(A):
                        if bad[i, a] and spec['allowed'][i, a]:
                            extra.append(({i * A + a: 1, base + j: -1}, -np.inf, 0))
            obj = np.r_[np.zeros(base), np.ones(G)]
            res = C.run_milp(spec, obj, extra_vars=G, extra_rows=extra, time_limit=60)
            if res.x is None:
                return None
            return count_harm(g, groups, C.choices_from(spec, res.x, anchor), other, tol)
        value, _ = C.min_cross_harm(g, groups, anchor, metrics, other, target)
        return value
    finally:
        C.TOL = old


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--quality', required=True)
    ap.add_argument('--b3', required=True)
    ap.add_argument('--metric', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, groups, scores = C.load(a.quality, a.b3, 'lraspp', 'raw', ACTIONS, a.metric)
    anchor_id = ACTIONS.index('identity')
    macro = scores.mean(axis=0)
    fixed = int((macro[:, 0] - macro[:, 1]).argmax())
    g_id = C.gains(scores, anchor_id)
    g_fix = C.gains(scores, fixed)
    base = VECTORS['base_0.01']
    frontier = []
    for label, g, anchor in (('identity', g_id, anchor_id), ('best_fixed_quality', g_fix, fixed)):
        qmax = capacity(g, groups, anchor, C.FAMILIES['quality'], base)
        tmax = capacity(g, groups, anchor, C.FAMILIES['task'], base)
        for fraction in FRACTIONS:
            for kind, total, metrics, other in (
                    ('quality_coverage', qmax, C.FAMILIES['quality'], C.FAMILIES['task']),
                    ('task_coverage', tmax, C.FAMILIES['task'], C.FAMILIES['quality'])):
                target = int(round(fraction * total))
                frontier.append(dict(anchor=label, anchor_action=ACTIONS[anchor], kind=kind, fraction=fraction,
                                      target_groups=target, max_groups=total,
                                      min_other_harm=harm_at(g, groups, anchor, metrics, other, target, base)))
                print('frontier', a.name, label, kind, fraction, frontier[-1]['min_other_harm'], flush=True)
    tolerance = []
    for name, tol in VECTORS.items():
        tolerance.append(dict(name=name, tol=tol,
                              quality=capacity(g_id, groups, anchor_id, C.FAMILIES['quality'], tol),
                              task=capacity(g_id, groups, anchor_id, C.FAMILIES['task'], tol),
                              joint=capacity(g_id, groups, anchor_id, C.FAMILIES['joint'], tol)))
        print('tolerance', a.name, name, tolerance[-1], flush=True)
    path.write_text(json.dumps(dict(dataset=a.name, images=len(ids), groups=len(set(groups)),
                                           fixed_quality_action=ACTIONS[fixed], frontier=frontier,
                                           tolerance=tolerance), indent=1), encoding='utf-8')
    print('GRID_DONE', a.name, flush=True)


if __name__ == '__main__':
    main()
