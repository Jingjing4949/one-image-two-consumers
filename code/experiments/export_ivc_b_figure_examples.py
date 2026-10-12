"""Export two TUB development examples for the manuscript figure.

Images are picked by a fixed hash among stage-6 development records from different
scenes, before any score is read. Holdout scenes are refused.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run_ivc_b_finish_dev import ACTIONS, output_path

CODE = Path('/root/ivcb/code')
OUT = Path('/root/ivcb/out')
DEST = OUT / 'b_fig_examples'
SALT = 'IVC-B-figure-20261012'
WIDTH = 360


def small(image):
    w, h = image.size
    return image.resize((WIDTH, round(h * WIDTH / w)), Image.Resampling.BICUBIC)


def mask_for(rec, size):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for line in (CODE / 'data/tub/labels' / (rec['id'] + '.txt')).read_text().splitlines():
        vals = line.split()
        if len(vals) >= 7:
            pts = np.array(vals[1:], dtype=float).reshape(-1, 2) * size
            draw.polygon([tuple(p) for p in pts], fill=255)
    return mask


def main():
    payload = json.loads((CODE / 'outputs/ivc_b_tub_v1/records.json').read_text(encoding='utf-8'))
    holdout = set(payload['holdout_experiments'])
    pool = [r for r in payload['records'] if r['experiment'] not in holdout and r['stage'] == 6]
    pool.sort(key=lambda r: hashlib.sha256((SALT + r['id']).encode()).hexdigest())
    picked, scenes = [], set()
    for rec in pool:
        if rec['experiment'] not in scenes:
            picked.append(rec)
            scenes.add(rec['experiment'])
        if len(picked) == 2:
            break
    DEST.mkdir(parents=True, exist_ok=True)
    quality, task = {}, {}
    for action in ACTIONS:
        for line in (CODE / 'outputs/ivc_b2_tub_v1' / f'quality_{action}.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                quality[(row['id'], action)] = dict(ssim=row['ssim'], lpips=row['lpips'])
    for fold in range(5):
        for line in (OUT / 'tub_b3' / f'lraspp__raw__fold{fold}' / 'eval.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                task[(row['id'], row['action'])] = row['fg_iou']
    meta = []
    for rec in picked:
        stem = rec['id'].replace('/', '__')
        ident = Image.open(output_path(OUT / 'tub_b2', rec, 'identity')).convert('RGB')
        reference = Image.open(CODE / rec['reference']).convert('RGB').resize(ident.size, Image.Resampling.BICUBIC)
        small(reference).save(DEST / f'{stem}__reference.png')
        small(mask_for(rec, Image.open(CODE / rec['input']).size).resize(ident.size, Image.Resampling.NEAREST)).save(DEST / f'{stem}__mask.png')
        scores = {}
        for action in ACTIONS:
            small(Image.open(output_path(OUT / 'tub_b2', rec, action)).convert('RGB')).save(DEST / f'{stem}__{action}.png')
            scores[action] = dict(**quality[(rec['id'], action)], fg_iou=task[(rec['id'], action)])
        meta.append(dict(id=rec['id'], stem=stem, experiment=rec['experiment'], camera=rec.get('camera'),
                         stage=rec['stage'], turbidity=rec.get('turbidity'), scores=scores))
    (DEST / 'meta.json').write_text(json.dumps(dict(salt=SALT, rule='stage 6, development, sha256 order, distinct scenes',
                                                    examples=meta), indent=1), encoding='utf-8')
    print('FIG_EXAMPLES_DONE', [m['id'] for m in meta], flush=True)


if __name__ == '__main__':
    main()
