"""B3: quality-task compatibility on SUIM-E for each task endpoint (out-of-fold predictions).

Per endpoint and anchor this computes
  * maximum switched groups for quality-only, task-only and joint objectives (MILP, A13 rules),
  * the minimum other-objective harm at each single-objective maximum (MILP, A15 rule) and the
    counting lower bound max(0, N - B_J),
  * the raw safe-group capacities as the action pool grows (fixed random orders).
Rules: per-image harm tolerance 0.01 on SSIM, LPIPS and IoU; mean effects SSIM >= 0.001, LPIPS
reduction >= 0, IoU >= 0.001, all averaged within groups and then equally across all groups.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

TOL = 0.01
DELTA = {'ssim': 0.001, 'lpips': 0.0, 'iou': 0.001}
FAMILIES = {'quality': ['ssim', 'lpips'], 'task': ['iou'], 'joint': ['ssim', 'lpips', 'iou']}


def load(quality_dir, b3_dir, arch, regime, actions, task_metric='iou'):
    q = {}
    for a in actions:
        for line in open(Path(quality_dir) / f'quality_{a}.jsonl'):
            r = json.loads(line)
            q[(r['id'], a)] = (r['ssim'], r['lpips'], r['group_id'])
    t = {}
    for fold in range(5):
        path = Path(b3_dir) / f'{arch}__{regime}__fold{fold}' / 'eval.jsonl'
        for line in open(path):
            r = json.loads(line)
            t[(r['id'], r['action'])] = r[task_metric]
    ids = sorted({i for (i, a) in t if all((i, b) in t and (i, b) in q for b in actions)})
    scores = np.zeros((len(ids), len(actions), 3))
    groups = []
    for n, i in enumerate(ids):
        groups.append(q[(i, actions[0])][2])
        for k, a in enumerate(actions):
            s, l, _ = q[(i, a)]
            scores[n, k] = (s, l, t[(i, a)])
    return ids, groups, scores


def gains(scores, anchor):
    g = scores - scores[:, anchor:anchor + 1, :]
    g[:, :, 1] *= -1  # LPIPS reduction is the favourable direction
    return g  # [image, action, metric(ssim, lpips, iou)]


IDX = {'ssim': 0, 'lpips': 1, 'iou': 2}


def safe_mask(g, metrics):
    return np.all(g[:, :, [IDX[m] for m in metrics]] >= -TOL, axis=2)


def build(g, groups, anchor, metrics, extra_rows=None):
    n, A, _ = g.shape
    names = sorted(set(groups))
    members = {k: np.flatnonzero(np.array(groups) == k) for k in names}
    G = len(names)
    allowed = safe_mask(g, metrics)
    allowed[:, anchor] = False
    count = n * A + G
    rows = []
    for i in range(n):
        rows.append(({j: 1 for j in range(i * A, i * A + A)}, -np.inf, 1))
    for j, k in enumerate(names):
        ix = members[k]
        d = {i * A + a: 1 for i in ix for a in range(A)}
        rows.append(({**d, n * A + j: -1}, 0, np.inf))
        rows.append(({**d, n * A + j: -len(ix)}, -np.inf, 0))
    w = np.array([1 / len(members[k]) for k in groups])
    for m in metrics:
        coef = (g[:, :, IDX[m]] * w[:, None]).ravel() * 1e5
        rows.append(({j: c for j, c in enumerate(coef) if c != 0}, G * DELTA[m] * 1e5, np.inf))
    return dict(n=n, A=A, G=G, names=names, members=members, allowed=allowed, count=count, rows=rows)


def run_milp(spec, objective, extra_vars=0, extra_rows=(), extra_upper=None, time_limit=120):
    count = spec['count'] + extra_vars
    rows = list(spec['rows']) + list(extra_rows)
    M = lil_matrix((len(rows), count))
    lo = np.empty(len(rows))
    hi = np.empty(len(rows))
    for r, (d, a, b) in enumerate(rows):
        for j, c in d.items():
            M[r, j] = c
        lo[r], hi[r] = a, b
    upper = np.r_[spec['allowed'].ravel().astype(float), np.ones(spec['G']), np.ones(extra_vars) if extra_upper is None else extra_upper]
    res = milp(objective, integrality=np.ones(count), bounds=Bounds(np.zeros(count), upper),
               constraints=LinearConstraint(M.tocsr(), lo, hi), options={'time_limit': time_limit, 'mip_rel_gap': 0})
    return res


def choices_from(spec, x, anchor):
    z = np.round(x[:spec['n'] * spec['A']]).reshape(spec['n'], spec['A'])
    return np.where(z.sum(1) > 0, z.argmax(1), anchor)


def harm_groups(g, spec, choices, metrics):
    sel = g[np.arange(spec['n']), choices]
    return int(sum(bool((sel[ix][:, [IDX[m] for m in metrics]] < -TOL).any()) for ix in spec['members'].values()))


def max_coverage(g, groups, anchor, metrics):
    spec = build(g, groups, anchor, metrics)
    obj = np.r_[np.zeros(spec['n'] * spec['A']), -np.ones(spec['G'])]
    res = run_milp(spec, obj)
    if res.x is None:
        return spec, None, 0
    ch = choices_from(spec, res.x, anchor)
    N = int(sum(bool((ch[ix] != anchor).any()) for ix in spec['members'].values()))
    return spec, ch, N


def min_cross_harm(g, groups, anchor, metrics, other, N):
    """Fix coverage N for `metrics`; minimise groups harmed (> TOL loss) on `other` metrics."""
    spec = build(g, groups, anchor, metrics)
    n, A, G = spec['n'], spec['A'], spec['G']
    bad = np.any(g[:, :, [IDX[m] for m in other]] < -TOL, axis=2)
    extra = []
    base = spec['count']
    extra.append(({n * A + j: 1 for j in range(G)}, N, N))
    for j, k in enumerate(spec['names']):
        for i in spec['members'][k]:
            for a in range(A):
                if bad[i, a] and spec['allowed'][i, a]:
                    extra.append(({i * A + a: 1, base + j: -1}, -np.inf, 0))
    obj = np.r_[np.zeros(base), np.ones(G)]
    res = run_milp(spec, obj, extra_vars=G, extra_rows=extra)
    if res.x is None:
        return None, None
    ch = choices_from(spec, res.x, anchor)
    return harm_groups(g, spec, ch, other), int(round(res.fun))


def joint_capacity(g, groups, anchor):
    safe = safe_mask(g, ['ssim', 'lpips', 'iou'])
    safe[:, anchor] = False
    names = sorted(set(groups))
    return sum(bool(safe[np.array(groups) == k].any()) for k in names)


def raw_capacities(g, groups, anchor, subset):
    out = {}
    names = np.array(groups)
    for fam, metrics in FAMILIES.items():
        s = safe_mask(g, metrics)
        # an alternative must also strictly improve at least one active metric to count as an opportunity
        better = np.any(g[:, :, [IDX[m] for m in metrics]] > 0, axis=2)
        ok = s & better
        ok[:, anchor] = False
        mask = np.zeros(g.shape[1], bool)
        mask[subset] = True
        ok &= mask[None, :]
        out[fam] = int(sum(bool(ok[names == k].any()) for k in set(groups)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quality', required=True)
    ap.add_argument('--b3', required=True)
    ap.add_argument('--arch', required=True)
    ap.add_argument('--regime', required=True)
    ap.add_argument('--actions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clean-only', default='', help='folds.json path: keep images without LSUI/UIEB links')
    ap.add_argument('--task-metric', default='iou', choices=['iou', 'fg_iou'])
    a = ap.parse_args()
    ids, groups, scores = load(a.quality, a.b3, a.arch, a.regime, a.actions, a.task_metric)
    if a.clean_only:
        linked = {r['id'] for r in json.load(open(a.clean_only, encoding='utf-8'))['records'] if r['linked_old_datasets']}
        keep = [n for n, i in enumerate(ids) if i not in linked]
        ids = [ids[n] for n in keep]
        groups = [groups[n] for n in keep]
        scores = scores[keep]
    G = len(set(groups))
    macro = {}
    for k, act in enumerate(a.actions):
        per = defaultdict(list)
        for n, gid in enumerate(groups):
            per[gid].append(scores[n, k])
        macro[act] = np.mean([np.mean(v, axis=0) for v in per.values()], axis=0).tolist()
    best_quality = max(a.actions, key=lambda x: macro[x][0] - macro[x][1])
    anchors = {'identity': a.actions.index('identity'), 'best_fixed_quality': a.actions.index(best_quality)}
    result = dict(arch=a.arch, regime=a.regime, actions=a.actions, images=len(ids), groups=G,
                  group_macro=dict(zip(a.actions, [dict(ssim=m[0], lpips=m[1], iou=m[2]) for m in macro.values()])),
                  best_fixed_quality_action=best_quality, anchors={})
    rng = np.random.default_rng(20261010)
    for name, anchor in anchors.items():
        g = gains(scores, anchor)
        r = {}
        for fam, metrics in FAMILIES.items():
            spec, ch, N = max_coverage(g, groups, anchor, metrics)
            r[f'max_{fam}'] = N
        bj = joint_capacity(g, groups, anchor)
        r['joint_safe_capacity'] = bj
        hq, _ = min_cross_harm(g, groups, anchor, FAMILIES['quality'], ['iou'], r['max_quality'])
        ht, _ = min_cross_harm(g, groups, anchor, FAMILIES['task'], ['ssim', 'lpips'], r['max_task'])
        r['min_task_harm_at_quality_max'] = hq
        r['min_quality_harm_at_task_max'] = ht
        r['counting_bound_task_harm'] = max(0, r['max_quality'] - bj)
        r['counting_bound_quality_harm'] = max(0, r['max_task'] - bj)
        others = [k for k in range(len(a.actions)) if k != anchor]
        curve = []
        for rep in range(20):
            order = rng.permutation(others)
            for size in range(1, len(others) + 1):
                curve.append(dict(rep=rep, pool_size=size, **raw_capacities(g, groups, anchor, list(order[:size]) + [anchor])))
        r['pool_size_curve'] = curve
        result['anchors'][name] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'pool_size_curve'} for k, v in result['anchors'].items()}
    print(json.dumps(dict(endpoint=f'{a.arch}/{a.regime}', groups=G, best_fixed=best_quality, **summary)))


if __name__ == '__main__':
    main()
