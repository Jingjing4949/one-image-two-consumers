"""Paper summaries computed from saved development results.

Reads saved quality, DISTS, segmentation eval and selector prediction files. Trains
nothing and opens no TUB holdout record.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import lsui_router_features as features
from ivc_b_compat import TOL
from run_ivc_b_finish_dev import ACTIONS, decoupled, open_rgb

CODE = Path('/root/ivcb/code')
OUT = Path('/root/ivcb/out')
SETS = {
    'suim': dict(records='outputs/ivc_b3_suime_folds_v1/folds.json', quality='outputs/ivc_b2_suime_gpu_v1',
                 b3=OUT / 'b3', metric='iou', cnn=OUT / 'b5_cnn_suim'),
    'tub': dict(records='outputs/ivc_b_tub_v1/records.json', quality='outputs/ivc_b2_tub_v1',
                b3=OUT / 'tub_b3', metric='fg_iou', cnn=OUT / 'b5_cnn_tub'),
}
ENDPOINTS = [('lraspp', 'raw', ''), ('deeplabv3', 'raw', ''), ('lraspp', 'mixed', ''),
             ('deeplabv3', 'mixed', ''), ('lraspp', 'raw', 'seed2')]


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def run_dir(b3, arch, regime, fold, tag):
    return Path(b3) / (f'{arch}__{regime}__fold{fold}' + (f'__{tag}' if tag else ''))


def task_table(b3, arch, regime, tag, metric):
    table, fold_of = {}, {}
    for fold in range(5):
        path = run_dir(b3, arch, regime, fold, tag) / 'eval.jsonl'
        if not path.exists():
            return None, None
        for row in jsonl(path):
            table[(row['id'], row['action'])] = row[metric]
            fold_of[row['id']] = fold
    return table, fold_of


def quality_table(qdir):
    q, d = {}, {}
    for action in ACTIONS:
        for row in jsonl(CODE / qdir / f'quality_{action}.jsonl'):
            q[(row['id'], action)] = (row['ssim'], row['lpips'], row['group_id'])
        path = CODE / qdir / f'dists_{action}.jsonl'
        if path.exists():
            for row in jsonl(path):
                d[(row['id'], action)] = row['dists']
    return q, d


def assemble(q, task):
    ids = sorted({i for (i, a) in task if all((i, b) in task and (i, b) in q for b in ACTIONS)})
    scores = np.array([[(q[(i, a)][0], q[(i, a)][1], task[(i, a)]) for a in ACTIONS] for i in ids])
    groups = [q[(i, 'identity')][2] for i in ids]
    return ids, groups, scores


def gain_array(scores):
    g = scores - scores[:, :1, :]
    g[:, :, 1] *= -1
    return g


def group_macro(values, groups):
    per = defaultdict(list)
    for v, k in zip(values, groups):
        per[k].append(v)
    return float(np.mean([np.mean(v) for v in per.values()]))


def safe_groups(g, groups, cols):
    sub = g[:, 1:, :][:, :, cols]
    ok = np.all(sub >= -TOL, axis=2) & np.any(sub > 0, axis=2)
    names = np.array(groups)
    return int(sum(bool(ok[names == k].any()) for k in set(groups)))


def endpoint_summary(q, task, metric_name):
    ids, groups, scores = assemble(q, task)
    g = gain_array(scores)
    task_change = {a: float(group_macro(list(g[:, k, 2]), groups)) for k, a in enumerate(ACTIONS) if a != 'identity'}
    dec = decoupled(ids, groups, scores, ACTIONS)
    return dict(images=len(ids), groups=len(set(groups)), metric=metric_name,
                identity_task=float(group_macro(list(scores[:, 0, 2]), groups)),
                task_change_vs_identity=task_change,
                decoupled={k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'action_counts'})
                           for k, v in dec.items()},
                safe_groups=dict(quality=safe_groups(g, groups, [0, 1]), task=safe_groups(g, groups, [2]),
                                 joint=safe_groups(g, groups, [0, 1, 2]))), ids, groups, scores


def seed_noise(t1, t2, ids):
    diffs, base = [], []
    for i in ids:
        for a in ACTIONS:
            if (i, a) in t1 and (i, a) in t2:
                diffs.append(abs(t1[(i, a)] - t2[(i, a)]))
                if a == 'identity':
                    base.append(abs(t1[(i, a)] - t2[(i, a)]))
    diffs, base = np.array(diffs), np.array(base)
    return dict(pairs=int(len(diffs)), mean_abs=float(diffs.mean()), median_abs=float(np.median(diffs)),
                p95_abs=float(np.quantile(diffs, .95)), frac_above_tol=float((diffs > TOL).mean()),
                identity_mean_abs=float(base.mean()), identity_frac_above_tol=float((base > TOL).mean()))


def dists_summary(q, d, ids, groups, scores):
    if not d:
        return None
    dist = np.array([[d[(i, a)] for a in ACTIONS] for i in ids])
    red = dist[:, :1] - dist
    lp = scores[:, :1, 1] - scores[:, :, 1]
    per_action = {}
    for k, a in enumerate(ACTIONS):
        if a == 'identity':
            continue
        per_action[a] = dict(dists_reduction=float(group_macro(list(red[:, k]), groups)),
                             lpips_reduction=float(group_macro(list(lp[:, k]), groups)))
    names = [a for a in ACTIONS if a != 'identity']
    x = np.array([per_action[a]['lpips_reduction'] for a in names])
    y = np.array([per_action[a]['dists_reduction'] for a in names])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    spearman = float(np.corrcoef(rank(x), rank(y))[0, 1])
    g = gain_array(scores)
    qsafe = np.all(g[:, 1:, :2] >= -TOL, axis=2)
    dsafe = red[:, 1:] >= -TOL
    alt = g.copy()
    alt[:, :, 1] = red
    lp_best = (scores[:, :, 0] - scores[:, :, 1]).argmax(1)
    d_best = (scores[:, :, 0] - dist).argmax(1)
    return dict(per_action=per_action, spearman_lpips_vs_dists_across_actions=spearman,
                quality_safe_pairs=int(qsafe.sum()), quality_safe_pairs_also_dists_safe=int((qsafe & dsafe).sum()),
                joint_safe_groups_with_dists_in_place_of_lpips=safe_groups(alt, groups, [0, 1, 2]),
                same_quality_best_action_fraction=float((lp_best == d_best).mean()))


def selector_groups(choice, g_by_id, groups_by_id, ids):
    """Group-level outcome of a per-image choice."""
    per = defaultdict(list)
    for i in ids:
        k = ACTIONS.index(choice[i])
        per[groups_by_id[i]].append((k, g_by_id[i][k]))
    harmed = recovered = 0
    for rows in per.values():
        bad = any(bool((v < -TOL).any()) for _, v in rows)
        good = any(k != 0 and bool((v >= -TOL).all()) and bool((v > 0).any()) for k, v in rows)
        harmed += bad
        recovered += (good and not bad)
    return dict(groups=len(per), harmed_groups=int(harmed), safe_improved_groups=int(recovered))


def selector_outcomes(name, cfg, records, ids, groups, scores):
    g = gain_array(scores)
    g_by_id = {i: g[n] for n, i in enumerate(ids)}
    groups_by_id = dict(zip(ids, groups))
    capacity = safe_groups(g, groups, [0, 1, 2])
    out = dict(joint_safe_capacity=capacity)
    cnn = {}
    fixed = {}
    for fold in range(5):
        for row in jsonl(cfg['cnn'] / f'fold{fold}.jsonl'):
            cnn[row['id']] = row['chosen']
            fixed[row['id']] = row['fixed']
    common = [i for i in ids if i in cnn]
    out['cnn'] = selector_groups(cnn, g_by_id, groups_by_id, common)
    out['fixed_training_fold'] = selector_groups(fixed, g_by_id, groups_by_id, common)
    oracle = {i: ACTIONS[int(g_by_id[i].sum(1).argmax())] for i in common}
    out['joint_oracle'] = selector_groups(oracle, g_by_id, groups_by_id, common)
    probe_path = None
    for path in sorted(OUT.glob('**/predictions.jsonl')):
        rows = jsonl(path)
        if rows and 'confidence' in rows[0] and rows[0]['id'] in groups_by_id and len(rows) > 0.9 * len(ids):
            probe_path = path
            break
    if probe_path is not None:
        probe = {r['id']: r['chosen'] for r in jsonl(probe_path)}
        common_p = [i for i in common if i in probe]
        out['probe'] = selector_groups(probe, g_by_id, groups_by_id, common_p)
        out['probe_source'] = str(probe_path)
        folds = defaultdict(list)
        fold_of = {r['id']: r['fold'] for r in jsonl(probe_path)}
        for i in common_p:
            folds[fold_of[i]].append(i)
        out['probe_folds'] = []
        for fold, members in sorted(folds.items()):
            fx = ACTIONS.index(fixed[members[0]])
            out['probe_folds'].append(dict(fold=fold, images=len(members),
                                           joint_probe=float(np.mean([g_by_id[i][ACTIONS.index(probe[i])].sum() for i in members])),
                                           joint_fixed=float(np.mean([g_by_id[i][fx].sum() for i in members])),
                                           fixed=ACTIONS[fx]))
    stats, fold_of = {}, {}
    for i in common:
        stats[i] = features.extract_features(open_rgb(CODE / records[i]['input']))['stats52']
    for fold in range(5):
        for row in jsonl(cfg['cnn'] / f'fold{fold}.jsonl'):
            fold_of[row['id']] = fold
    x_all = np.stack([stats[i] for i in common])
    y_all = np.stack([g_by_id[i] for i in common])
    ridge = {}
    for fold in range(5):
        train = np.array([fold_of[i] != fold for i in common])
        mu, sd = x_all[train].mean(0), x_all[train].std(0)
        sd = np.where(sd < 1e-8, 1, sd)
        xt = (x_all[train] - mu) / sd
        coef = np.linalg.solve(xt.T @ xt + np.eye(xt.shape[1]), xt.T @ y_all[train].reshape(train.sum(), -1))
        pred = (((x_all[~train] - mu) / sd) @ coef).reshape(-1, len(ACTIONS), 3)
        for i, p in zip([c for c, t in zip(common, train) if not t], pred):
            ridge[i] = ACTIONS[int(p.sum(1).argmax())]
    out['ridge'] = selector_groups(ridge, g_by_id, groups_by_id, common)
    counts = defaultdict(lambda: defaultdict(int))
    for label, table in (('cnn', cnn), ('ridge', ridge), ('fixed', fixed)):
        for i in common:
            counts[label][table[i]] += 1
    out['choice_counts'] = {k: dict(v) for k, v in counts.items()}
    return out


def tub_by_stage(records, ids, groups, scores):
    stage_of = {i: records[i]['stage'] for i in ids}
    turb = defaultdict(list)
    for i in ids:
        turb[stage_of[i]].append(records[i].get('turbidity'))
    g = gain_array(scores)
    q = scores[:, :, 0] - scores[:, :, 1]
    t = scores[:, :, 2]
    rows = []
    for stage in sorted(set(stage_of.values())):
        ix = [n for n, i in enumerate(ids) if stage_of[i] == stage]
        safe = np.all(g[ix][:, 1:, :] >= -TOL, axis=2) & np.any(g[ix][:, 1:, :] > 0, axis=2)
        rows.append(dict(stage=stage, images=len(ix),
                         turbidity=[v for v in turb[stage] if v is not None][:1],
                         identity_ssim=float(scores[ix, 0, 0].mean()), identity_lpips=float(scores[ix, 0, 1].mean()),
                         identity_task=float(scores[ix, 0, 2].mean()),
                         same_best_fraction=float((q[ix].argmax(1) == t[ix].argmax(1)).mean()),
                         images_with_joint_safe_alternative=int(safe.any(1).sum()),
                         best_task_change=float(g[ix][:, 1:, 2].mean(0).max()),
                         best_task_change_action=ACTIONS[1 + int(g[ix][:, 1:, 2].mean(0).argmax())]))
    return rows


def main():
    result = {}
    for name, cfg in SETS.items():
        payload = json.loads((CODE / cfg['records']).read_text(encoding='utf-8'))
        holdout = set(payload.get('holdout_experiments') or [])
        if any(r.get('experiment') in holdout for r in payload['records']):
            raise SystemExit('holdout records present')
        records = {r['id']: r for r in payload['records']}
        q, d = quality_table(cfg['quality'])
        part = dict(endpoints={})
        tables = {}
        base = None
        for arch, regime, tag in ENDPOINTS:
            task, _ = task_table(cfg['b3'], arch, regime, tag, cfg['metric'])
            key = f'{arch}/{regime}' + (f'/{tag}' if tag else '')
            if task is None:
                part['endpoints'][key] = 'missing'
                continue
            tables[key] = task
            summary, ids, groups, scores = endpoint_summary(q, task, cfg['metric'])
            part['endpoints'][key] = summary
            if key == 'lraspp/raw':
                base = (ids, groups, scores)
            print(name, key, summary['images'], summary['safe_groups'], flush=True)
        if 'lraspp/raw/seed2' in tables:
            part['seed_noise_lraspp_raw'] = seed_noise(tables['lraspp/raw'], tables['lraspp/raw/seed2'], base[0])
        part['dists'] = dists_summary(q, d, *base)
        part['selectors'] = selector_outcomes(name, cfg, records, *base)
        if name == 'tub':
            part['by_stage'] = tub_by_stage(records, *base)
        result[name] = part
        print('SET_DONE', name, flush=True)
    dest = OUT / 'b_paper_extras.json'
    dest.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('EXTRAS_DONE', dest, flush=True)


if __name__ == '__main__':
    main()
