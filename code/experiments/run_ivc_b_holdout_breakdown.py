"""Per-scene breakdown of the frozen TUB confirmation.

Repeats the locked scoring in run_ivc_b_holdout_score.py with the same models and
rules, then checks that the scene-macro gains equal the saved confirmation.json.
Nothing is fitted on holdout labels and no rule is changed.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_ivc_b_holdout_score as H
from run_ivc_b5_cnn import ACTIONS

OUT = Path('/root/ivcb/out/b6_holdout')


def per_scene(records, chosen, quality, task):
    per = defaultdict(list)
    for rec in records:
        action = chosen[rec['id']]
        s, l, group = quality[(rec['id'], action)]
        s0, l0, _ = quality[(rec['id'], 'identity')]
        per[group].append((s - s0, l0 - l, task[(rec['id'], action)] - task[(rec['id'], 'identity')]))
    return {k: np.mean(v, axis=0).tolist() for k, v in sorted(per.items())}


def main():
    saved = json.loads((OUT / 'confirmation.json').read_text(encoding='utf-8'))
    records = json.loads((OUT / 'records.json').read_text(encoding='utf-8'))['records']
    quality = H.quality_table(records)
    masks = H.build_masks(records)
    task, conf_l = H.forward_all('lraspp', records, masks)
    _, conf_d = H.forward_all('deeplabv3', records, masks)
    probe = {}
    for rec in records:
        probe[rec['id']] = ACTIONS[int(np.argmax([0.5 * (conf_l[(rec['id'], a)] + conf_d[(rec['id'], a)]) for a in ACTIONS]))]
    choices = {'cnn': H.cnn_choice_averaged(records), 'ridge': H.ridge_choice(records), 'probe': probe}
    result = dict(scenes={}, choice_counts={}, check={})
    for name, chosen in choices.items():
        scenes = per_scene(records, chosen, quality, task)
        macro = np.mean(list(scenes.values()), axis=0)
        ref = saved[name]
        diff = max(abs(macro[0] - ref['ssim_gain']), abs(macro[1] - ref['lpips_reduction']), abs(macro[2] - ref['fg_iou_gain']))
        result['scenes'][name] = scenes
        result['check'][name] = dict(max_abs_difference_to_saved=float(diff))
        counts = defaultdict(int)
        for rec in records:
            counts[chosen[rec['id']]] += 1
        result['choice_counts'][name] = dict(counts)
    stage = defaultdict(list)
    for rec in records:
        stage[rec.get('stage')].append(rec['id'])
    result['images'] = len(records)
    worst = max(v['max_abs_difference_to_saved'] for v in result['check'].values())
    result['matches_saved_confirmation'] = bool(worst < 1e-9)
    (OUT / 'breakdown.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('BREAKDOWN_DONE', result['matches_saved_confirmation'], worst, flush=True)


if __name__ == '__main__':
    main()
