"""Score the locked TUB holdout selectors. Fits nothing on holdout labels.

Task values are the mean foreground IoU of the five saved raw LR-ASPP folds.
The probe score also uses the five raw DeepLab folds. Ridge and the CNN are the
averages of the five development-fold models.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import lsui_router_features as features
import run_ivc_b3_seg as seg
import run_ivc_b_finish_dev as finish
from run_ivc_b5_cnn import ACTIONS
from torchvision.models import mobilenet_v3_small

ROOT = Path('/root/ivcb/code')
STORE = Path('/root/ivcb/out/b6_holdout/store')
QUALITY = Path('/root/ivcb/out/b6_holdout/quality')
B3 = Path('/root/ivcb/out/tub_b3')


def build_masks(records):
    masks = {}
    for rec in records:
        label = ROOT / 'data/tub/labels' / (rec['id'] + '.txt')
        with Image.open(ROOT / rec['input']) as image:
            width, height = image.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for line in label.read_text().splitlines():
            vals = line.split()
            if len(vals) < 7:
                continue
            pts = np.array(vals[1:], dtype=float).reshape(-1, 2) * [width, height]
            draw.polygon([tuple(p) for p in pts], fill=1)
        masks[rec['id']] = np.asarray(mask, dtype=np.uint8)
    return masks


def output_file(rec, action):
    base = STORE / action / rec['id'].replace('/', '__')
    png = base.with_suffix('.png')
    return png if png.exists() else base.with_suffix('.webp')


def forward_all(arch, records, masks):
    seg.CLASSES = 2
    device = torch.device('cuda')
    values = defaultdict(list)
    confidence = defaultdict(list)
    for fold in range(5):
        model, _ = seg.build(arch)
        ck = torch.load(B3 / f'{arch}__raw__fold{fold}' / 'final.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'])
        model = model.to(device).eval()
        jobs = []
        for rec in records:
            for action in ACTIONS:
                image = np.asarray(Image.open(output_file(rec, action)).convert('RGB'), dtype=np.uint8)
                jobs.append((rec, action, image))
        groups = {}
        for item in jobs:
            groups.setdefault(item[2].shape, []).append(item)
        with torch.inference_mode():
            for group in groups.values():
                for start in range(0, len(group), 8):
                    chunk = group[start:start + 8]
                    batch = np.stack([item[2] for item in chunk])
                    logits = model(seg.to_tensor(batch, device))['out'].float()
                    for n, (rec, action, _) in enumerate(chunk):
                        native = masks[rec['id']]
                        up = F.interpolate(logits[n:n + 1], size=native.shape, mode='bilinear', align_corners=False)
                        pred = up[0].argmax(0).cpu().numpy().astype(np.uint8)
                        cm = seg.confusion(native, pred)
                        values[(rec['id'], action)].append(seg.fg_iou(cm))
                        prob = torch.softmax(up[0], dim=0)
                        confidence[(rec['id'], action)].append(float(prob.max(0).values.mean()))
        del jobs, groups, model
        torch.cuda.empty_cache()
        print('scored', arch, fold, flush=True)
    return ({k: float(np.mean(v)) for k, v in values.items()},
            {k: float(np.mean(v)) for k, v in confidence.items()})


def quality_table(records):
    table = {}
    for action in ACTIONS:
        for line in (QUALITY / f'quality_{action}.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                table[(row['id'], action)] = (row['ssim'], row['lpips'], row['group_id'])
    return table


def cnn_choice_averaged(records):
    device = torch.device('cuda')
    import ivc_b_actions as actions_module
    model_dir = Path('/root/ivcb/out/b5_cnn_tub')
    totals = {}
    actions = None
    for fold in range(5):
        state = torch.load(model_dir / f'fold{fold}.pt', map_location='cpu', weights_only=True)
        actions = state['actions']
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(actions) * 3)
        model.load_state_dict(state['model'])
        model = model.to(device).eval()
        with torch.inference_mode():
            for rec in records:
                working, _ = actions_module.load_working(ROOT / rec['input'])
                rgb = np.asarray(working, dtype=np.uint8)
                pred = model(seg.to_tensor(rgb, device)).view(len(actions), 3).detach().cpu().numpy()
                totals[rec['id']] = totals.get(rec['id'], 0) + pred
        del model
        torch.cuda.empty_cache()
    return {i: actions[int(pred.sum(1).argmax())] for i, pred in totals.items()}


def ridge_choice(records):
    dev = json.loads((ROOT / 'outputs/ivc_b_tub_v1/records.json').read_text(encoding='utf-8'))
    dev_ids = [r['id'] for r in dev['records']]
    by_id = {r['id']: r for r in dev['records']}
    import ivc_b_compat as C
    ids, _, scores = C.load(str(ROOT / 'outputs/ivc_b2_tub_v1'), str(B3), 'lraspp', 'raw', finish.ACTIONS, 'fg_iou')
    gains = {}
    anchor = finish.ACTIONS.index('identity')
    id_of = {i: n for n, i in enumerate(ids)}
    for i in ids:
        table = scores[id_of[i]] - scores[id_of[i], anchor]
        table[:, 1] *= -1
        gains[i] = table
    stats = {}
    for i in ids:
        stats[i] = features.extract_features(np.asarray(Image.open(ROOT / by_id[i]['input']).convert('RGB'), dtype=np.uint8))['stats52']
    fold_of = {}
    for fold in range(5):
        for line in (B3 / f'lraspp__raw__fold{fold}' / 'eval.jsonl').read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                fold_of[row['id']] = fold
    common = [i for i in ids if i in stats and i in fold_of]
    x_all = np.stack([stats[i] for i in common])
    y_all = np.stack([gains[i] for i in common])
    folds = [fold_of[i] for i in common]
    models = []
    for fold in range(5):
        train = np.array([f != fold for f in folds])
        mu, sd = x_all[train].mean(0), x_all[train].std(0)
        sd = np.where(sd < 1e-8, 1, sd)
        xt = (x_all[train] - mu) / sd
        yt = y_all[train].reshape(train.sum(), -1)
        coef = np.linalg.solve(xt.T @ xt + np.eye(xt.shape[1]), xt.T @ yt)
        models.append((mu, sd, coef))
    import ivc_b_actions as actions_module
    chosen = {}
    for rec in records:
        rgb = np.asarray(Image.open(ROOT / rec['input']).convert('RGB'), dtype=np.uint8)
        x = features.extract_features(rgb)['stats52']
        acc = None
        for mu, sd, coef in models:
            pred = (((x - mu) / sd) @ coef).reshape(len(finish.ACTIONS), 3)
            acc = pred if acc is None else acc + pred
        chosen[rec['id']] = finish.ACTIONS[int((acc / len(models)).sum(1).argmax())]
    return chosen


def group_macro(records, chosen, quality, task):
    per = defaultdict(list)
    for rec in records:
        action = chosen[rec['id']]
        s, l, group = quality[(rec['id'], action)]
        s0, l0, _ = quality[(rec['id'], 'identity')]
        per[group].append((s - s0, l0 - l, task[(rec['id'], action)] - task[(rec['id'], 'identity')]))
    means = [np.mean(v, axis=0) for v in per.values()]
    macro = np.mean(means, axis=0)
    return dict(scenes=len(per), ssim_gain=float(macro[0]), lpips_reduction=float(macro[1]),
                fg_iou_gain=float(macro[2]),
                effective=bool(macro[0] >= 0.001 and macro[1] >= 0 and macro[2] >= 0.001))


def main():
    records = json.loads(Path('/root/ivcb/out/b6_holdout/records.json').read_text(encoding='utf-8'))['records']
    quality = quality_table(records)
    masks = build_masks(records)
    task, conf_l = forward_all('lraspp', records, masks)
    _, conf_d = forward_all('deeplabv3', records, masks)
    probe = {}
    for rec in records:
        scores = []
        for action in ACTIONS:
            scores.append(0.5 * (conf_l[(rec['id'], action)] + conf_d[(rec['id'], action)]))
        probe[rec['id']] = ACTIONS[int(np.argmax(scores))]
    summary = {
        'identity': group_macro(records, {r['id']: 'identity' for r in records}, quality, task),
        'cnn': group_macro(records, cnn_choice_averaged(records), quality, task),
        'ridge': group_macro(records, ridge_choice(records), quality, task),
        'probe': group_macro(records, probe, quality, task),
    }
    Path('/root/ivcb/out/b6_holdout/confirmation.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print('CONFIRMATION', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
