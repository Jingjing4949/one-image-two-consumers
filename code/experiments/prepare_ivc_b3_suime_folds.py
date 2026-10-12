"""B3: deterministic 5-fold grouped split of all SUIM-E images for out-of-fold task models.

Reads only input/mask headers and existing manifests; no reference pixels, no quality.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / 'outputs/data_readiness/suime_official_intake_v1.json'
OVERLAP = ROOT / 'outputs/data_readiness/suime_overlap_v1.json'
OUT = ROOT / 'outputs/ivc_b3_suime_folds_v1'
SALT = 'IVC-B3-20261010'
FOLDS = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT / 'folds.json').exists(), 'Folds are frozen once written'
    intake = json.loads(INTAKE.read_text(encoding='utf-8-sig'))
    overlap = json.loads(OVERLAP.read_text(encoding='utf-8-sig'))
    masks = {}
    for f in intake['files']:
        path = Path(f['local_path'])
        if f['role'] == 'mask' and path.parent.name == 'masks':
            masks[f['id']] = f
    group_of = {r['id']: r['group_id'] for r in overlap['records']}
    group_meta = {}
    for g in overlap['groups']:
        for i in g['ids']:
            group_meta[i] = dict(role=g['role'], old_datasets=sorted({m['dataset'] for m in g.get('old_members', [])}))
    records = []
    for r in intake['records']:
        mask = masks[r['id']]
        with Image.open(ROOT / r['input']) as im:
            in_size = list(im.size)
        with Image.open(ROOT / mask['local_path']) as m:
            mask_size = list(m.size)
        records.append(dict(id=r['id'], group_id=group_of[r['id']], author_split=r['author_split'],
                            input=r['input'], input_sha256=r['input_sha256'],
                            reference=r['reference'], reference_sha256=r['reference_sha256'],
                            mask=mask['local_path'], mask_sha256=mask['sha256'],
                            input_size_wh=in_size, mask_size_wh=mask_size,
                            task_geometry_ok=in_size == mask_size,
                            prior_role=group_meta[r['id']]['role'],
                            linked_old_datasets=group_meta[r['id']]['old_datasets']))
    sizes = Counter(r['group_id'] for r in records)
    order = sorted(sizes, key=lambda g: hashlib.sha256((SALT + ':' + g).encode()).hexdigest())
    load = [0] * FOLDS
    fold_of = {}
    for g in order:
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        fold_of[g] = k
        load[k] += sizes[g]
    for r in records:
        r['fold'] = fold_of[r['group_id']]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(images=len(records), groups=len(sizes), images_per_fold=load,
                   groups_per_fold=[sum(1 for g in fold_of if fold_of[g] == k) for k in range(FOLDS)],
                   geometry_mismatch_images=sum(not r['task_geometry_ok'] for r in records),
                   lsui_linked_images=sum('LSUI' in r['linked_old_datasets'] for r in records),
                   prior_roles=dict(Counter(r['prior_role'] for r in records)))
    payload = dict(at_utc=datetime.now(timezone.utc).isoformat(), salt=SALT, folds=FOLDS,
                   intake_sha256=sha(INTAKE), overlap_sha256=sha(OVERLAP), source_sha256=sha(Path(__file__)),
                   reference_pixels_read=False, summary=summary, records=records)
    with (OUT / 'folds.json').open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
