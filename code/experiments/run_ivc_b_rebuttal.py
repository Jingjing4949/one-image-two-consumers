"""Two post-hoc checks on saved outputs; nothing is fitted or tuned.

A. Development sets: a label-assisted per-image oracle chosen with one task endpoint
   is scored with another endpoint (second seed, second architecture).
B. Held-out TUB scenes, exploratory: DISTS of the locked selectors' choices. The
   choices are recomputed with the locked models and must reproduce confirmation.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from piq import DISTS

from run_ivc_b_finish_dev import ACTIONS

OUT = Path('/root/ivcb/out')
CODE = Path('/root/ivcb/code')
TOL = 0.01
SETS = {'suim': ('outputs/ivc_b2_suime_gpu_v1', OUT / 'b3', 'iou'), 'tub': ('outputs/ivc_b2_tub_v1', OUT / 'tub_b3', 'fg_iou')}
ENDPOINTS = {'lraspp_seed1': 'lraspp__raw__fold{fold}', 'lraspp_seed2': 'lraspp__raw__fold{fold}__seed2',
             'deeplab': 'deeplabv3__raw__fold{fold}'}
PAIRS = [('lraspp_seed1', 'lraspp_seed2'), ('lraspp_seed2', 'lraspp_seed1'), ('lraspp_seed1', 'deeplab'), ('deeplab', 'lraspp_seed1')]


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def task(b3, pattern, metric):
    t = {}
    for fold in range(5):
        for r in jsonl(b3 / pattern.format(fold=fold) / 'eval.jsonl'):
            t[(r['id'], r['action'])] = r[metric]
    return t


def group_outcome(choice, g, groups):
    per = defaultdict(list)
    for n, k in enumerate(groups):
        per[k].append(g[n, choice[n]])
    harmed = improved = 0
    for rows in per.values():
        bad = any(bool((v < -TOL).any()) for v in rows)
        good = any(bool((v >= -TOL).all()) and bool((v > 0).any()) for v in rows)
        harmed += bad
        improved += good and not bad
    return dict(safe_improved_groups=int(improved), harmed_groups=int(harmed))


def part_a():
    out = {}
    for name, (qdir, b3, metric) in SETS.items():
        q = {}
        for a in ACTIONS:
            for r in jsonl(CODE / qdir / f'quality_{a}.jsonl'):
                q[(r['id'], a)] = (r['ssim'], r['lpips'], r['group_id'])
        tables = {k: task(b3, p, metric) for k, p in ENDPOINTS.items()}
        ids = sorted(i for (i, a) in tables['lraspp_seed1'] if a == 'identity'
                     and all((i, b) in q and all((i, b) in t for t in tables.values()) for b in ACTIONS))
        groups = [q[(i, 'identity')][2] for i in ids]
        qual = np.array([[(q[(i, a)][0] - q[(i, 'identity')][0], q[(i, 'identity')][1] - q[(i, a)][1]) for a in ACTIONS] for i in ids])
        gains = {}
        for k, t in tables.items():
            tg = np.array([[t[(i, a)] - t[(i, 'identity')] for a in ACTIONS] for i in ids])
            gains[k] = np.concatenate([qual, tg[:, :, None]], axis=2)
        res = dict(images=len(ids), groups=len(set(groups)), pairs={})
        for a_name, b_name in PAIRS:
            ga, gb = gains[a_name], gains[b_name]
            ja, jb = ga.sum(2), gb.sum(2)
            oracle_a = ja.argmax(1)
            oracle_b = jb.argmax(1)
            fixed_b = int(jb.mean(0).argmax())
            fixed_a = int(ja.mean(0).argmax())
            value = lambda ch: float(jb[np.arange(len(ids)), ch].mean())
            v_oa, v_ob = value(oracle_a), value(oracle_b)
            v_fb, v_fa = value(np.full(len(ids), fixed_b)), value(np.full(len(ids), fixed_a))
            task_oa = float(gb[np.arange(len(ids)), oracle_a, 2].mean())
            res['pairs'][f'{a_name}->{b_name}'] = dict(
                oracle_chosen_with=a_name, scored_with=b_name,
                joint_oracle_transferred=v_oa, joint_oracle_native=v_ob,
                joint_fixed_native=v_fb, fixed_native_action=ACTIONS[fixed_b],
                joint_fixed_transferred=v_fa, fixed_transferred_action=ACTIONS[fixed_a],
                retained_fraction_of_oracle_margin=(v_oa - v_fb) / (v_ob - v_fb) if v_ob > v_fb else None,
                task_gain_oracle_transferred=task_oa,
                task_gain_oracle_native=float(gb[np.arange(len(ids)), oracle_b, 2].mean()),
                same_choice_fraction=float((oracle_a == oracle_b).mean()),
                groups_oracle_transferred=group_outcome(oracle_a, gb, groups),
                groups_oracle_native=group_outcome(oracle_b, gb, groups),
                groups_fixed_native=group_outcome(np.full(len(ids), fixed_b), gb, groups))
            print(name, a_name, b_name, json.dumps(res['pairs'][f'{a_name}->{b_name}']), flush=True)
        out[name] = res
    return out


def part_b():
    sys.path.insert(0, str(CODE / 'experiments'))
    import run_ivc_b_holdout_score as H
    from run_ivc_b5_cnn import ACTIONS as CNN_ACTIONS
    hold = OUT / 'b6_holdout'
    saved = json.loads((hold / 'confirmation.json').read_text(encoding='utf-8'))
    records = json.loads((hold / 'records.json').read_text(encoding='utf-8'))['records']
    quality = H.quality_table(records)
    masks = H.build_masks(records)
    task_l, conf_l = H.forward_all('lraspp', records, masks)
    _, conf_d = H.forward_all('deeplabv3', records, masks)
    probe = {r['id']: CNN_ACTIONS[int(np.argmax([0.5 * (conf_l[(r['id'], a)] + conf_d[(r['id'], a)]) for a in CNN_ACTIONS]))]
             for r in records}
    choices = {'cnn': H.cnn_choice_averaged(records), 'ridge': H.ridge_choice(records), 'probe': probe}
    check = {}
    for k, ch in choices.items():
        m = H.group_macro(records, ch, quality, task_l)
        check[k] = max(abs(m['ssim_gain'] - saved[k]['ssim_gain']), abs(m['lpips_reduction'] - saved[k]['lpips_reduction']),
                       abs(m['fg_iou_gain'] - saved[k]['fg_iou_gain']))
    device = torch.device('cuda')
    metric = DISTS(reduction='none').to(device).eval()
    dists = {}
    with torch.inference_mode():
        for rec in records:
            ref_img = Image.open(CODE / rec['reference']).convert('RGB')
            for action in CNN_ACTIONS:
                pred = np.asarray(Image.open(H.output_file(rec, action)).convert('RGB'), dtype=np.uint8)
                ref = np.asarray(ref_img.resize((pred.shape[1], pred.shape[0]), Image.Resampling.BICUBIC), dtype=np.uint8)
                x = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1)[None].float().div_(255).to(device)
                y = torch.from_numpy(np.ascontiguousarray(ref)).permute(2, 0, 1)[None].float().div_(255).to(device)
                dists[(rec['id'], action)] = float(metric(x, y).item())
    result = dict(reproduces_confirmation_max_abs_difference=check, scenes=len({r['group_id'] for r in records}), selectors={})
    for k, ch in list(choices.items()) + [('identity', {r['id']: 'identity' for r in records})]:
        per = defaultdict(list)
        for rec in records:
            per[rec['group_id']].append(dists[(rec['id'], 'identity')] - dists[(rec['id'], ch[rec['id']])])
        vals = {g: float(np.mean(v)) for g, v in per.items()}
        result['selectors'][k] = dict(dists_reduction_scene_macro=float(np.mean(list(vals.values()))),
                                      scenes_positive=int(sum(v > 0 for v in vals.values())), per_scene=vals)
    per_action = {}
    for action in CNN_ACTIONS:
        per = defaultdict(list)
        for rec in records:
            per[rec['group_id']].append(dists[(rec['id'], 'identity')] - dists[(rec['id'], action)])
        per_action[action] = float(np.mean([np.mean(v) for v in per.values()]))
    result['fixed_action_dists_reduction_scene_macro'] = per_action
    result['choices'] = {k: {r['id']: ch[r['id']] for r in records} for k, ch in choices.items()}
    return result


def main():
    result = dict(part_a=part_a())
    (OUT / 'b_rebuttal.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('PART_A_DONE', flush=True)
    result['part_b_holdout_exploratory'] = part_b()
    (OUT / 'b_rebuttal.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('REBUTTAL_DONE', json.dumps({k: v['dists_reduction_scene_macro'] for k, v in result['part_b_holdout_exploratory']['selectors'].items()}),
          json.dumps(result['part_b_holdout_exploratory']['reproduces_confirmation_max_abs_difference']), flush=True)


if __name__ == '__main__':
    main()
