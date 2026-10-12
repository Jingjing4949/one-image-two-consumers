"""B3: out-of-fold semantic segmentation on SUIM-E action outputs.

One run = (architecture, training regime, fold). The model trains on the other
four folds and is evaluated on every action output of the held-out fold.
Regimes: raw (identity outputs), mixed (cycle all pool actions, equal exposure),
matched:<action> (that action's outputs only). Fixed final epoch; no early stop.
"""
import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import (deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large,
                                             DeepLabV3_ResNet50_Weights, FCN_ResNet50_Weights,
                                             LRASPP_MobileNet_V3_Large_Weights)

CLASSES = 8
MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
SCHEDULE = {'lraspp': dict(epochs=60, batch=8, crop=512), 'deeplabv3': dict(epochs=40, batch=8, crop=512),
            'fcn': dict(epochs=40, batch=8, crop=512)}


def now():
    return datetime.now(timezone.utc).isoformat()


def build(arch):
    if arch == 'lraspp':
        m = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
        m.classifier.low_classifier = nn.Conv2d(40, CLASSES, 1)
        m.classifier.high_classifier = nn.Conv2d(128, CLASSES, 1)
        heads = ['classifier.low_classifier', 'classifier.high_classifier']
    elif arch == 'deeplabv3':
        m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, aux_loss=True)
        m.classifier[4] = nn.Conv2d(256, CLASSES, 1)
        m.aux_classifier[4] = nn.Conv2d(256, CLASSES, 1)
        heads = ['classifier.4', 'aux_classifier.4']
    elif arch == 'fcn':
        m = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, aux_loss=True)
        m.classifier[4] = nn.Conv2d(512, CLASSES, 1)
        m.aux_classifier[4] = nn.Conv2d(256, CLASSES, 1)
        heads = ['classifier.4', 'aux_classifier.4']
    else:
        raise ValueError(arch)
    return m, heads


def freeze_bn(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def action_offset(image_id, k):
    return int(hashlib.sha256(('B3-mixed:' + image_id).encode()).hexdigest()[:8], 16) % k


class Data:
    def __init__(self, root, store, folds_file, masks_dir):
        self.root, self.store, self.masks_dir = Path(root), Path(store), Path(masks_dir)
        self.records = json.loads(Path(folds_file).read_text(encoding='utf-8'))['records']

    def output(self, rec, action):
        base = self.store / action / rec['id'].replace('/', '__')
        png = base.with_suffix('.png')
        return png if png.exists() else base.with_suffix('.webp')

    def mask_native(self, rec):
        return np.asarray(Image.open(self.masks_dir / (rec['id'].replace('/', '__') + '.png')), dtype=np.uint8)


def to_tensor(x, device='cuda'):
    """uint8 HWC (or NHWC) array -> normalised NCHW float tensor on the device."""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
    if t.ndim == 3:
        t = t[None]
    t = t.permute(0, 3, 1, 2).float().div_(255)
    return (t - MEAN.to(device)) / STD.to(device)


def confusion(y, p):
    return np.bincount((y.astype(np.int64) * CLASSES + p).ravel(), minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)


def fg_iou(m):
    tp = float(m[1, 1])
    union = float(m[1, :].sum() + m[:, 1].sum() - m[1, 1])
    return tp / union if union > 0 else 1.0


def miou(m):
    m = m.astype(np.float64)
    tp = np.diag(m)
    union = m.sum(0) + m.sum(1) - tp
    return float(np.mean(tp[union > 0] / union[union > 0]))


def evaluate_extra(a, out):
    assert (out / 'complete.json').exists() and (out / 'final.pt').exists()
    torch.set_num_threads(4)
    done = {(json.loads(l)['id'], json.loads(l)['action']) for l in (out / 'eval.jsonl').read_text().splitlines()}
    data = Data(a.root, a.store, a.folds, a.masks)
    test = [r for r in data.records if r['fold'] == a.fold and r['task_geometry_ok']]
    ck = torch.load(out / 'final.pt', map_location='cpu', weights_only=True)
    model, _ = build(a.arch)
    model.load_state_dict(ck['model'])
    device = torch.device('cuda')
    model = model.to(device).eval()
    rows = []
    with torch.inference_mode():
        for r in test:
            native = data.mask_native(r)
            for action in a.eval_actions:
                if (r['id'], action) in done:
                    continue
                img = load_rgb(data.output(r, action))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(to_tensor(img).to(device))['out'].float()
                logits = F.interpolate(logits, size=native.shape, mode='bilinear', align_corners=False)
                cm = confusion(native, logits[0].argmax(0).cpu().numpy().astype(np.uint8))
                rows.append(dict(id=r['id'], group_id=r['group_id'], action=action, iou=miou(cm), confusion=cm.ravel().tolist()))
    with (out / 'eval.jsonl').open('a', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    print(json.dumps(dict(run=out.name, appended=len(rows))))


def finish_from_checkpoint(a, out, run_name):
    """Evaluate a run whose final weights were saved before the disk filled."""
    torch.set_num_threads(4)
    data = Data(a.root, a.store, a.folds, a.masks)
    train = [r for r in data.records if r['fold'] != a.fold and r['task_geometry_ok']]
    test = [r for r in data.records if r['fold'] == a.fold and r['task_geometry_ok']]
    counts = np.zeros(CLASSES, dtype=np.float64)
    for r in train:
        counts += np.bincount(data.mask_native(r).ravel(), minlength=CLASSES)
    ck = torch.load(out / 'final.pt', map_location='cpu', weights_only=True)
    model, _ = build(a.arch)
    model.load_state_dict(ck['model'])
    device = torch.device('cuda')
    model = model.to(device).eval()
    rows = []
    test_jobs = [(r, act) for r in test for act in a.pool]
    with ThreadPoolExecutor(a.workers) as pool:
        test_images = dict(zip([(r['id'], act) for r, act in test_jobs],
                               pool.map(lambda job: load_rgb(data.output(*job)), test_jobs)))
    started = time.time()
    with torch.inference_mode():
        for r in test:
            native = data.mask_native(r)
            for action in a.pool:
                img = test_images[(r['id'], action)]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(to_tensor(img).to(device))['out'].float()
                logits = F.interpolate(logits, size=native.shape, mode='bilinear', align_corners=False)
                pred = logits[0].argmax(0).cpu().numpy().astype(np.uint8)
                cm = confusion(native, pred)
                row = dict(id=r['id'], group_id=r['group_id'], action=action, iou=miou(cm), confusion=cm.ravel().tolist())
                if CLASSES == 2:
                    row['fg_iou'] = fg_iou(cm)
                rows.append(row)
    with (out / 'eval.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    ident = [x for x in rows if x['action'] == 'identity']
    total_cm = sum(np.array(x['confusion']).reshape(CLASSES, CLASSES) for x in ident)
    majority = int(np.argmax(counts))
    by_group = {}
    for x in ident:
        by_group.setdefault(x['group_id'], []).append(x['iou'])
    group_iou = float(np.mean([np.mean(v) for v in by_group.values()]))
    maj_group = {}
    for r in test:
        n = data.mask_native(r)
        maj_group.setdefault(r['group_id'], []).append(miou(confusion(n, np.full(n.shape, majority, np.uint8))))
    maj_iou = float(np.mean([np.mean(v) for v in maj_group.values()]))
    predicted_classes = int((total_cm.sum(0) > 0).sum())
    training_seconds = None
    log_path = out / 'train_log.jsonl'
    if log_path.exists() and log_path.stat().st_size:
        last = json.loads([ln for ln in log_path.read_text().splitlines() if ln.strip()][-1])
        training_seconds = last.get('seconds')
    eval_seconds = round(time.time() - started, 1)
    summary = dict(at_utc=now(), run=run_name, arch=a.arch, regime=a.regime, fold=a.fold, train_images=len(train),
                   test_images=len(test), updates=int(ck['updates']), seconds=round((training_seconds or 0) + eval_seconds, 1),
                   identity_group_iou=group_iou, majority_group_iou=maj_iou, predicted_classes=predicted_classes,
                   capability_gate=dict(iou_at_least_0_25=group_iou >= 0.25, beats_majority_by_0_05=group_iou - maj_iou >= 0.05,
                                        at_least_3_classes=predicted_classes >= 3),
                   final_sha256=hashlib.sha256((out / 'final.pt').read_bytes()).hexdigest(), torch=torch.__version__,
                   bf16_autocast=True, bitwise_determinism_claimed=False,
                   evaluation_resumed_from_final=True, training_seconds_before_interrupt=training_seconds,
                   evaluation_seconds=eval_seconds)
    summary['capability_passed'] = all(summary['capability_gate'].values())
    (out / 'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=list(SCHEDULE))
    ap.add_argument('--regime', required=True)  # raw | mixed | matched:<action>
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--pool', nargs='+', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--store', required=True)
    ap.add_argument('--folds', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--classes', type=int, default=8)
    ap.add_argument('--tag', default='', help='suffix for repeat runs, e.g. a second seed')
    ap.add_argument('--eval-actions', nargs='*', default=None,
                    help='Evaluate an already trained run on these extra actions and append to eval.jsonl')
    a = ap.parse_args()
    global CLASSES
    CLASSES = a.classes
    run_name = f"{a.arch}__{a.regime.replace(':', '-')}__fold{a.fold}" + (f'__{a.tag}' if a.tag else '')
    out = Path(a.out) / run_name
    if a.eval_actions:
        evaluate_extra(a, out)
        return
    if (out / 'complete.json').exists():
        print('already complete', run_name)
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.pt').exists():
        finish_from_checkpoint(a, out, run_name)
        return
    if (out / 'train_log.jsonl').exists():
        (out / 'train_log.jsonl').unlink()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed + a.fold)
    data = Data(a.root, a.store, a.folds, a.masks)
    train = [r for r in data.records if r['fold'] != a.fold and r['task_geometry_ok']]
    test = [r for r in data.records if r['fold'] == a.fold and r['task_geometry_ok']]
    sched = SCHEDULE[a.arch]
    if a.regime == 'raw':
        view = lambda r, e: 'identity'
    elif a.regime == 'mixed':
        k = len(a.pool)
        view = lambda r, e: a.pool[(action_offset(r['id'], k) + e) % k]
    elif a.regime.startswith('matched:'):
        fixed = a.regime.split(':', 1)[1]
        assert fixed in a.pool
        view = lambda r, e: fixed
    else:
        raise ValueError(a.regime)
    # Preload every training view and resized mask into memory.
    needed = sorted({view(r, e) for r in train for e in range(sched['epochs'])})
    jobs = [(r, act) for r in train for act in needed]
    with ThreadPoolExecutor(a.workers) as pool:
        arrays = list(pool.map(lambda job: load_rgb(data.output(*job)), jobs))
    images = {(r['id'], act): arr for (r, act), arr in zip(jobs, arrays)}
    counts = np.zeros(CLASSES, dtype=np.float64)
    masks = {}
    for r in train:
        m = data.mask_native(r)
        counts += np.bincount(m.ravel(), minlength=CLASSES)
        h, w = images[(r['id'], needed[0])].shape[:2]
        masks[r['id']] = np.asarray(Image.fromarray(m).resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    device = torch.device('cuda')
    model, heads = build(a.arch)
    model = model.to(device)
    head_params = [p for n, p in model.named_parameters() if any(n.startswith(h) for h in heads)]
    head_ids = {id(p) for p in head_params}
    base_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{'params': base_params, 'lr': 1e-4}, {'params': head_params, 'lr': 1e-3}], weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train) / sched['batch'])
    total = steps_per_epoch * sched['epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    crop = sched['crop']
    log = (out / 'train_log.jsonl').open('a', encoding='utf-8')
    start = time.time()
    step = 0
    for epoch in range(sched['epochs']):
        freeze_bn(model)
        order = rng.permutation(len(train))
        running = 0.0
        for b0 in range(0, len(order), sched['batch']):
            xs, ys = [], []
            for idx in order[b0:b0 + sched['batch']]:
                r = train[idx]
                img = images[(r['id'], view(r, epoch))]
                h, w = img.shape[:2]
                m = masks[r['id']]
                y0 = int(rng.integers(0, h - crop + 1)) if h > crop else 0
                x0 = int(rng.integers(0, w - crop + 1)) if w > crop else 0
                img, m = img[y0:y0 + crop, x0:x0 + crop], m[y0:y0 + crop, x0:x0 + crop]
                if rng.random() < 0.5:
                    img, m = img[:, ::-1], m[:, ::-1]
                xs.append(img)
                ys.append(m)
            x = to_tensor(np.stack(xs), device)
            y = torch.from_numpy(np.stack(ys)).to(device, non_blocking=True).long()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(x)
                loss = loss_fn(outputs['out'].float(), y)
                if 'aux' in outputs:
                    loss = loss + 0.5 * loss_fn(outputs['aux'].float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            step += 1
            running += float(loss)
        log.write(json.dumps(dict(epoch=epoch + 1, updates=step, mean_loss=running / steps_per_epoch,
                                  seconds=round(time.time() - start, 1))) + '\n')
        log.flush()
    torch.save(dict(model=model.state_dict(), arch=a.arch, regime=a.regime, fold=a.fold, updates=step,
                    pool=a.pool, seed=a.seed), out / 'final.pt')
    # Evaluate on every action output of the held-out fold.
    model.eval()
    del images
    rows = []
    test_jobs = [(r, act) for r in test for act in a.pool]
    with ThreadPoolExecutor(a.workers) as pool:
        test_images = dict(zip([(r['id'], act) for r, act in test_jobs],
                               pool.map(lambda job: load_rgb(data.output(*job)), test_jobs)))
    with torch.inference_mode():
        for r in test:
            native = data.mask_native(r)
            for action in a.pool:
                img = test_images[(r['id'], action)]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(to_tensor(img).to(device))['out'].float()
                logits = F.interpolate(logits, size=native.shape, mode='bilinear', align_corners=False)
                pred = logits[0].argmax(0).cpu().numpy().astype(np.uint8)
                cm = confusion(native, pred)
                row = dict(id=r['id'], group_id=r['group_id'], action=action, iou=miou(cm), confusion=cm.ravel().tolist())
                if CLASSES == 2:
                    row['fg_iou'] = fg_iou(cm)
                rows.append(row)
    with (out / 'eval.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    ident = [x for x in rows if x['action'] == 'identity']
    total_cm = sum(np.array(x['confusion']).reshape(CLASSES, CLASSES) for x in ident)
    majority = int(np.argmax(counts))
    maj_cm = sum(confusion(data.mask_native(r), np.full(data.mask_native(r).shape, majority, np.uint8)) for r in test)
    by_group = {}
    for x in ident:
        by_group.setdefault(x['group_id'], []).append(x['iou'])
    group_iou = float(np.mean([np.mean(v) for v in by_group.values()]))
    maj_group = {}
    for r in test:
        n = data.mask_native(r)
        maj_group.setdefault(r['group_id'], []).append(miou(confusion(n, np.full(n.shape, majority, np.uint8))))
    maj_iou = float(np.mean([np.mean(v) for v in maj_group.values()]))
    predicted_classes = int((total_cm.sum(0) > 0).sum())
    summary = dict(at_utc=now(), run=run_name, arch=a.arch, regime=a.regime, fold=a.fold, train_images=len(train),
                   test_images=len(test), updates=step, seconds=round(time.time() - start, 1),
                   identity_group_iou=group_iou, majority_group_iou=maj_iou, predicted_classes=predicted_classes,
                   capability_gate=dict(iou_at_least_0_25=group_iou >= 0.25, beats_majority_by_0_05=group_iou - maj_iou >= 0.05,
                                        at_least_3_classes=predicted_classes >= 3),
                   final_sha256=hashlib.sha256((out / 'final.pt').read_bytes()).hexdigest(), torch=torch.__version__,
                   bf16_autocast=True, bitwise_determinism_claimed=False)
    summary['capability_passed'] = all(summary['capability_gate'].values())
    (out / 'complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
