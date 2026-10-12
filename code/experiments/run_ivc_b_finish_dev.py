"""Finish the pre-specified development analyses that do not open the TUB holdout.

Decoupled label reference, category-order pool counts, input-output appearance
strata, and one stats-52 ridge. No threshold search and no retraining of enhancers.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab

import ivc_b_actions as actions_module
import lsui_router_features as features
from ivc_b_compat import gains, load, raw_capacities

ACTIONS = ['identity', 'gray_world_mean', 'shallow_uwnet', 'funie_s20260913', 'funie_s20261013',
           'lite_s20260913', 'lite_s20261013', 'deep_wavenet_euvp', 'uiec2net', 'uir_polykernel',
           'syreanet', 'semi_uir', 'phaseformer_uieb', 'ushape_lsui', 'tacl']
CATEGORIES = [
    ['gray_world_mean'],
    ['shallow_uwnet'],
    ['funie_s20260913', 'funie_s20261013', 'lite_s20260913', 'lite_s20261013'],
    ['deep_wavenet_euvp'],
    ['uiec2net'],
    ['uir_polykernel'],
    ['syreanet'],
    ['semi_uir'],
    ['phaseformer_uieb'],
    ['ushape_lsui'],
    ['tacl'],
]


def output_path(store, rec, action):
    base = Path(store) / action / rec['id'].replace('/', '__')
    png = base.with_suffix('.png')
    return png if png.exists() else base.with_suffix('.webp')


def open_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def group_mean(pairs):
    per = defaultdict(list)
    for value, group in pairs:
        per[group].append(value)
    if not per:
        return None
    return float(np.mean([np.mean(v) for v in per.values()]))


def decoupled(ids, groups, scores, actions):
    q = scores[:, :, 0] - scores[:, :, 1]
    t = scores[:, :, 2]
    joint = q + t
    names = {'quality': q.argmax(1), 'task': t.argmax(1), 'forced': joint.argmax(1)}
    out = {}
    for name, choice in names.items():
        chosen_q = q[np.arange(len(ids)), choice]
        chosen_t = t[np.arange(len(ids)), choice]
        out[name] = dict(quality=group_mean(zip(chosen_q, groups)), task=group_mean(zip(chosen_t, groups)),
                         action_counts={actions[i]: int((choice == i).sum()) for i in range(len(actions))})
    same = names['quality'] == names['task']
    out['same_action_image_fraction'] = float(same.mean())
    out['images'] = len(ids)
    out['groups'] = len(set(groups))
    return out


def category_curve(quality, b3, arch, regime, metric):
    ids, groups, scores = load(quality, b3, arch, regime, ACTIONS, metric)
    anchor = ACTIONS.index('identity')
    g = gains(scores, anchor)
    subset = [anchor]
    curve = []
    for block in CATEGORIES:
        subset.extend(ACTIONS.index(a) for a in block)
        curve.append(dict(added=block, pool_size=len(subset) - 1, **raw_capacities(g, groups, anchor, subset)))
    return dict(images=len(ids), groups=len(set(groups)), curve=curve)


def class_shift(b3, arch, classes, metric_name):
    """Per-class IoU of each action minus identity, from saved confusion matrices."""
    raw = {}
    for fold in range(5):
        path = Path(b3) / f'{arch}__raw__fold{fold}' / 'eval.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                raw[(row['id'], row['action'])] = row
    per_action = {}
    for action in ACTIONS:
        if action == 'identity':
            continue
        deltas = []
        for (i, a), row in raw.items():
            if a != action or (i, 'identity') not in raw:
                continue
            def iou(confusion):
                cm = np.asarray(confusion, dtype=np.float64).reshape(classes, classes)
                tp = np.diag(cm)
                union = cm.sum(0) + cm.sum(1) - tp
                return np.divide(tp, union, out=np.full(classes, np.nan), where=union > 0)
            base = raw[(i, 'identity')]
            deltas.append((iou(row['confusion']) - iou(base['confusion']), base['group_id']))
        if not deltas:
            continue
        stacked = np.vstack([d[0] for d in deltas])
        groups = [d[1] for d in deltas]
        class_means = []
        for c in range(classes):
            class_means.append(group_mean((stacked[n, c], groups[n]) for n in range(len(groups)) if np.isfinite(stacked[n, c])))
        per_action[action] = class_means
    return per_action


def appearance(records_path, store, root, device):
    payload = json.loads(Path(records_path).read_text(encoding='utf-8'))
    holdout = set(payload.get('holdout_experiments') or [])
    if any(r.get('experiment') in holdout for r in payload['records']):
        raise SystemExit('holdout records present')
    records = payload['records']
    metric = __import__('lpips').LPIPS(net='alex', version='0.1', verbose=False).eval().to(device)
    rows, stats = [], {}
    with torch.inference_mode():
        for n, rec in enumerate(records):
            ident = open_rgb(output_path(store, rec, 'identity'))
            ident_t = torch.from_numpy(ident.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1)[None].to(device)
            gray = ident.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
            g0 = np.hypot(np.diff(gray, axis=1)[:-1], np.diff(gray, axis=0)[:, :-1])
            stats[rec['id']] = features.extract_features(open_rgb(Path(root) / rec['input']))['stats52']
            for action in ACTIONS:
                if action == 'identity':
                    rows.append(dict(id=rec['id'], group_id=rec['group_id'], action=action, delta_e=0.0, gradient_corr=1.0, lpips=0.0))
                    continue
                pred = open_rgb(output_path(store, rec, action))
                if pred.shape != ident.shape:
                    pred = np.asarray(Image.fromarray(pred).resize((ident.shape[1], ident.shape[0]), Image.Resampling.BICUBIC), dtype=np.uint8)
                lab_a = rgb2lab(ident / 255.0)
                lab_b = rgb2lab(pred / 255.0)
                delta = float(np.mean(np.linalg.norm(lab_a - lab_b, axis=2)))
                gray_b = pred.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
                g1 = np.hypot(np.diff(gray_b, axis=1)[:-1], np.diff(gray_b, axis=0)[:, :-1])
                if g0.shape != g1.shape:
                    corr = None
                else:
                    a, b = g0.ravel(), g1.ravel()
                    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
                pred_t = torch.from_numpy(pred.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1)[None].to(device)
                dist = float(metric(pred_t, ident_t).item())
                rows.append(dict(id=rec['id'], group_id=rec['group_id'], action=action, delta_e=delta,
                                 gradient_corr=corr, lpips=dist))
            if n % 50 == 0:
                print('appearance', n, len(records), flush=True)
    return rows, stats


def tertile_summary(rows, key, task_gain):
    values = np.array([r[key] for r in rows if r['action'] != 'identity' and r[key] is not None], dtype=np.float64)
    edges = np.quantile(values, [0, 1/3, 2/3, 1])
    bins = []
    for b in range(3):
        lo, hi = edges[b], edges[b + 1]
        chosen = [r for r in rows if r['action'] != 'identity' and r[key] is not None and (lo <= r[key] <= hi if b == 2 else lo <= r[key] < hi)]
        gains = [(task_gain[(r['id'], r['action'])], r['group_id']) for r in chosen if (r['id'], r['action']) in task_gain]
        bins.append(dict(bin=b, lo=float(lo), hi=float(hi), pairs=len(gains), task_gain=group_mean(gains)))
    return bins


def task_gains(b3, arch, metric):
    out = {}
    base = {}
    for fold in range(5):
        for line in (Path(b3) / f'{arch}__raw__fold{fold}' / 'eval.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row['fg_iou'] if 'fg_iou' in row else row['iou']
            if row['action'] == 'identity':
                base[row['id']] = value
            out[(row['id'], row['action'])] = value
    return {key: value - base[key[0]] for key, value in out.items() if key[0] in base}


def ridge_readout(ids, stats, gains_by_id, fold_of):
    summary = []
    x_all = np.stack([stats[i] for i in ids])
    y_all = np.stack([gains_by_id[i] for i in ids])
    for fold in range(5):
        train = np.array([fold_of[i] != fold for i in ids])
        test = ~train
        mu, sd = x_all[train].mean(0), x_all[train].std(0)
        sd = np.where(sd < 1e-8, 1, sd)
        def norm(x):
            return (x - mu) / sd
        xt, yt = norm(x_all[train]), y_all[train].reshape(train.sum(), -1)
        coef = np.linalg.solve(xt.T @ xt + np.eye(xt.shape[1]), xt.T @ yt)
        pred = (norm(x_all[test]) @ coef).reshape(test.sum(), len(ACTIONS), 3)
        true = y_all[test]
        pick = pred.sum(2).argmax(1)
        fixed = int(yt.reshape(-1, len(ACTIONS), 3).sum(2).mean(0).argmax())
        summary.append(dict(fold=fold, test_images=int(test.sum()), fixed_action=ACTIONS[fixed],
                            mean_joint_pred=float(true[np.arange(len(pick)), pick].sum(1).mean()),
                            mean_joint_fixed=float(true[:, fixed].sum(1).mean()),
                            mean_joint_oracle=float(true.sum(2).max(1).mean())))
    return summary


def gains_from_scores(ids, scores, actions, id_index):
    anchor = actions.index('identity')
    table = {}
    for n, i in enumerate(ids):
        table[i] = scores[n] - scores[n, anchor]
        table[i][:, 1] *= -1
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--records', required=True)
    ap.add_argument('--quality', required=True)
    ap.add_argument('--b3', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--arch', default='lraspp')
    ap.add_argument('--classes', type=int, required=True)
    ap.add_argument('--metric', choices=['iou', 'fg_iou'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-pixels', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, groups, scores = load(a.quality, a.b3, a.arch, 'raw', ACTIONS, a.metric)
    result = dict(name=a.name, images=len(ids), groups=len(set(groups)),
                  decoupled=decoupled(ids, groups, scores, ACTIONS),
                  category_curve=category_curve(a.quality, a.b3, a.arch, 'raw', a.metric),
                  class_iou_delta=class_shift(a.b3, a.arch, a.classes, a.metric))
    (out / 'fast.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('FAST_DONE', a.name, flush=True)
    if a.skip_pixels:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, stats = appearance(a.records, a.store, a.root, device)
    gain = task_gains(a.b3, a.arch, a.metric)
    strata = {key: tertile_summary(rows, key, gain) for key in ('delta_e', 'gradient_corr', 'lpips')}
    payload = json.loads(Path(a.records).read_text(encoding='utf-8'))
    if all('object_fraction' in r for r in payload['records']):
        fractions = {r['id']: r['object_fraction'] for r in payload['records']}
        for row in rows:
            row['object_fraction'] = fractions.get(row['id'])
        strata['object_fraction'] = tertile_summary(rows, 'object_fraction', gain)
    fold_of = {}
    for fold in range(5):
        for line in (Path(a.b3) / f'{a.arch}__raw__fold{fold}' / 'eval.jsonl').read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                fold_of[row['id']] = fold
    gain_table = gains_from_scores(ids, scores, ACTIONS, None)
    common = [i for i in ids if i in stats and i in fold_of and i in gain_table]
    result['strata'] = strata
    result['ridge52'] = ridge_readout(common, stats, gain_table, fold_of)
    (out / 'full.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('FULL_DONE', a.name, flush=True)


if __name__ == '__main__':
    main()
