"""TUB preparation: scene split, development records and binary object masks.

Held-out scenes (fixed by hash of the experiment id) are recorded by id only; their
labels and stage-1 references are not opened here. Development scenes get 5 scene folds.
Reference = stage 1 (lowest turbidity) of the same experiment and camera; inputs = stages 2-10.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TUB = ROOT / 'data/tub'
OUT = ROOT / 'outputs/ivc_b_tub_v1'
SALT = 'IVC-B-TUB-20261010'
HOLDOUT = 11
FOLDS = 5


def h(s):
    return hashlib.sha256((SALT + ':' + s).encode()).hexdigest()


def main():
    assert not (OUT / 'records.json').exists(), 'TUB split is frozen once written'
    meta = json.loads((TUB / 'labels.json').read_text())
    images = meta['images']
    exps = sorted({i['experiment'] for i in images})
    order = sorted(exps, key=lambda e: h(f'exp{e}'))
    holdout = sorted(order[:HOLDOUT])
    dev = sorted(order[HOLDOUT:])
    fold_of = {e: k % FOLDS for k, e in enumerate(sorted(dev, key=lambda e: h(f'fold{e}')))}
    by_key = {(i['experiment'], i['camera'], i['stage']): i for i in images}
    records = []
    (OUT / 'masks').mkdir(parents=True, exist_ok=True)
    for i in sorted(images, key=lambda x: (x['experiment'], x['camera'], x['stage'])):
        e, c, s = i['experiment'], i['camera'], i['stage']
        if e in holdout or s == 1:
            continue
        ref = by_key.get((e, c, 1))
        if ref is None:
            continue
        stem = Path(i['file_name']).stem
        label = TUB / 'labels' / (stem + '.txt')
        mask = Image.new('L', (i['width'], i['height']), 0)
        draw = ImageDraw.Draw(mask)
        for line in label.read_text().splitlines():
            vals = line.split()
            if len(vals) < 7:
                continue
            pts = np.array(vals[1:], dtype=float).reshape(-1, 2) * [i['width'], i['height']]
            draw.polygon([tuple(p) for p in pts], fill=1)
        mpath = OUT / 'masks' / (stem + '.png')
        mask.save(mpath)
        arr = np.asarray(mask)
        records.append(dict(id=stem, group_id=f'exp{e}', experiment=e, camera=c, stage=s, turbidity=i['turbidity'],
                            reference_turbidity=ref['turbidity'], fold=fold_of[e],
                            input=f'data/tub/{i["file_name"]}', reference=f'data/tub/{ref["file_name"]}',
                            mask=str(mpath.relative_to(ROOT)), object_fraction=float(arr.mean()),
                            task_geometry_ok=True))
    payload = dict(at_utc=datetime.now(timezone.utc).isoformat(), salt=SALT, holdout_experiments=holdout,
                   development_experiments=dev, folds=FOLDS, fold_of_experiment={str(k): v for k, v in fold_of.items()},
                   reference_rule='stage 1 of same experiment and camera', inputs='stages 2-10',
                   holdout_labels_or_references_read=False, records=records,
                   source_manifest_sha256=hashlib.sha256((TUB / 'download_manifest.json').read_bytes()).hexdigest())
    (OUT / 'records.json').write_text(json.dumps(payload, indent=1))
    print(json.dumps(dict(holdout=holdout, dev_scenes=len(dev), dev_inputs=len(records),
                          per_fold={k: sum(r['fold'] == k for r in records) for k in range(FOLDS)})))


if __name__ == '__main__':
    main()
