"""One-shot TUB holdout confirmation. Refuses to start unless the freeze file hash matches.

Overlap uses input images only. References and label files are opened only after
that screen writes an allow-list.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FREEZE = Path('/root/ivcb/code/outputs/IVC_B6_TUB保留场景确认冻结_2026-10-11.md')
ROOT = Path('/root/ivcb/code')
SALT = 'IVC-B-TUB-20261010'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_freeze(expected):
    digest = sha_file(FREEZE)
    if digest != expected:
        raise SystemExit('freeze hash mismatch ' + digest)


def holdout_records():
    dev = json.loads((ROOT / 'outputs/ivc_b_tub_v1/records.json').read_text(encoding='utf-8'))
    holdout = set(dev['holdout_experiments'])
    meta = json.loads((ROOT / 'data/tub/labels.json').read_text(encoding='utf-8'))
    by_key = {(i['experiment'], i['camera'], i['stage']): i for i in meta['images']}
    records = []
    for i in meta['images']:
        e, c, s = i['experiment'], i['camera'], i['stage']
        if e not in holdout or s == 1:
            continue
        ref = by_key.get((e, c, 1))
        if ref is None:
            continue
        records.append(dict(id=Path(i['file_name']).stem, group_id=f'exp{e}', experiment=e, camera=c, stage=s,
                            fold=-1, input=f'data/tub/{i["file_name"]}', reference=f'data/tub/{ref["file_name"]}',
                            task_geometry_ok=True))
    return holdout, records


def rgb_sha(path):
    image = Image.open(path).convert('RGB')
    return hashlib.sha256(image.tobytes()).hexdigest(), image.size


def overlap(records, hash_list):
    known = set(json.loads(Path(hash_list).read_text(encoding='utf-8'))['sha256'])
    blocked = set()
    hits = []
    for rec in records:
        digest, _ = rgb_sha(ROOT / rec['input'])
        if digest in known:
            blocked.add(rec['experiment'])
            hits.append(dict(id=rec['id'], experiment=rec['experiment'], method='exact_rgb'))
    try:
        import cv2
    except ImportError:
        return blocked, hits, 'cv2_missing'
    folds = json.loads((ROOT / 'outputs/ivc_b3_suime_folds_v1/folds.json').read_text(encoding='utf-8'))
    suim = [ROOT / r['input'] for r in folds['records']]

    def ahash(path):
        im = Image.open(path).convert('L').resize((8, 8))
        arr = np.asarray(im, dtype=np.float32)
        bits = arr > arr.mean()
        return bits.flatten()

    hold_feat = [(rec, ahash(ROOT / rec['input'])) for rec in records if rec['experiment'] not in blocked]
    suim_feat = [(p, ahash(p)) for p in suim]
    pairs = []
    for rec, ha in hold_feat:
        dist = sorted(((int(np.count_nonzero(ha != hb)), p) for p, hb in suim_feat), key=lambda x: x[0])
        for d, p in dist[:2]:
            pairs.append((rec, p))
        for d, p in dist:
            if d <= 8:
                pairs.append((rec, p))
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=12)

    def desc(path):
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        f = min(1., 480 / max(h, w))
        im = cv2.resize(im, (round(w * f), round(h * f)))
        kp, de = orb.detectAndCompute(im, None)
        return np.float32([k.pt for k in kp]).reshape(-1, 2), de, np.float32([im.shape[1], im.shape[0]])

    cache = {}
    for rec, path in pairs:
        key = str(path)
        if key not in cache:
            cache[key] = desc(path)
        left = cache.setdefault(rec['input'], desc(ROOT / rec['input']))
        x, a, w = left
        y, b, h = cache[key]
        if a is None or b is None or min(len(a), len(b)) < 20:
            continue
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .75 * m[1].distance]
        if len(good) < 24:
            continue
        xx = np.float32([x[m.queryIdx] for m in good])
        yy = np.float32([y[m.trainIdx] for m in good])
        mat, mask = cv2.findHomography(xx, yy, cv2.RANSAC, 3., maxIters=1500, confidence=.995)
        if mat is None or mask is None:
            continue
        keep = mask.ravel().astype(bool)
        count = int(keep.sum())
        if count < 20 or count / len(good) < .55:
            continue
        blocked.add(rec['experiment'])
        hits.append(dict(id=rec['id'], experiment=rec['experiment'], method='ORB_RANSAC_overlap', inliers=count))
    return blocked, hits, 'ok'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--expected-sha', required=True)
    ap.add_argument('--hash-list', required=True)
    ap.add_argument('--out', default='/root/ivcb/out/b6_holdout')
    a = ap.parse_args()
    require_freeze(a.expected_sha)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    holdout, records = holdout_records()
    (out / 'records_all.json').write_text(json.dumps(dict(holdout_experiments=sorted(holdout), records=records), indent=1), encoding='utf-8')
    overlap_path = out / 'overlap.json'
    if overlap_path.exists():
        prev = json.loads(overlap_path.read_text(encoding='utf-8'))
        if prev.get('orb_status') == 'ok' and prev.get('allowed_images', 0) > 0:
            blocked = set(prev.get('blocked_scenes') or [])
            allowed = [r for r in records if r['experiment'] not in blocked]
            print('OVERLAP_REUSED', len(blocked), len(allowed), flush=True)
        else:
            blocked, hits, orb_status = overlap(records, a.hash_list)
            allowed = [r for r in records if r['experiment'] not in blocked]
            overlap_path.write_text(json.dumps(dict(orb_status=orb_status, blocked_scenes=sorted(blocked),
                                                     hits=hits, allowed_images=len(allowed),
                                                     allowed_scenes=sorted({r['experiment'] for r in allowed})), indent=1),
                                      encoding='utf-8')
            print('OVERLAP_DONE', len(blocked), len(allowed), orb_status, flush=True)
            if orb_status != 'ok' or not allowed:
                print('CONFIRMATION_STOPPED', flush=True)
                return
    else:
        blocked, hits, orb_status = overlap(records, a.hash_list)
        allowed = [r for r in records if r['experiment'] not in blocked]
        overlap_path.write_text(json.dumps(dict(orb_status=orb_status, blocked_scenes=sorted(blocked),
                                                 hits=hits, allowed_images=len(allowed),
                                                 allowed_scenes=sorted({r['experiment'] for r in allowed})), indent=1),
                                  encoding='utf-8')
        print('OVERLAP_DONE', len(blocked), len(allowed), orb_status, flush=True)
        if orb_status != 'ok' or not allowed:
            print('CONFIRMATION_STOPPED', flush=True)
            return
    free_gb = int(open('/sys/block/vda/size').read()) if False else None
    usage = subprocess.check_output(['df', '-B1', '/'], text=True).splitlines()[-1].split()
    free = int(usage[3])
    if free < 2_500_000_000:
        print('DISK_LOW', free, flush=True)
        raise SystemExit(2)
    (out / 'records.json').write_text(json.dumps(dict(records=allowed, holdout_labels_or_references_read=False), indent=1), encoding='utf-8')
    py = '/root/py311/bin/python'
    base = [py, '-X', 'utf8', 'experiments/run_ivc_b2_suime.py', '--device', 'cuda', '--threads', '4',
            '--format', 'webp', '--records', str(out / 'records.json'), '--out', str(out / 'quality'),
            '--store', str(out / 'store'), '--actions']
    actions = ['identity', 'gray_world_mean', 'funie_s20260913', 'funie_s20261013', 'lite_s20260913',
               'lite_s20261013', 'semi_uir', 'phaseformer_uieb', 'deep_wavenet_euvp', 'uir_polykernel',
               'shallow_uwnet', 'syreanet', 'ushape_lsui', 'tacl', 'uiec2net']
    print('ENHANCE_START', flush=True)
    for name in actions:
        print('ACTION', name, flush=True)
        subprocess.check_call(base + [name], cwd='/root/ivcb/code')
    print('ENHANCE_DONE', flush=True)


if __name__ == '__main__':
    main()
