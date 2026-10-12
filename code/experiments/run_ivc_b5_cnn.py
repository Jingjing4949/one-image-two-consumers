"""One development CNN that predicts per-action quality and task gains from the input.

Five existing content folds, eight fixed epochs, one seed. The best fixed action
is chosen on the training folds only. TUB holdout records are rejected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import ivc_b_actions as actions_module
import run_ivc_b3_seg as seg

ACTIONS = ['identity', 'gray_world_mean', 'funie_s20260913', 'funie_s20261013', 'lite_s20260913',
           'lite_s20261013', 'semi_uir', 'phaseformer_uieb', 'deep_wavenet_euvp', 'uir_polykernel',
           'shallow_uwnet', 'syreanet', 'ushape_lsui', 'tacl', 'uiec2net']


def load_joined(records_path, quality_dir, b3):
    payload = json.loads(Path(records_path).read_text(encoding='utf-8'))
    holdout = set(payload.get('holdout_experiments') or [])
    if any(r.get('experiment') in holdout for r in payload['records']):
        raise SystemExit('holdout records present')
    records = {r['id']: r for r in payload['records']}
    quality = {}
    for action in ACTIONS:
        path = Path(quality_dir) / f'quality_{action}.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                quality[(row['id'], action)] = (row['ssim'], row['lpips'])
    task, fold_of = {}, {}
    for fold in range(5):
        path = Path(b3) / f'lraspp__raw__fold{fold}' / 'eval.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['action'] in ACTIONS:
                task[(row['id'], row['action'])] = row['fg_iou'] if 'fg_iou' in row else row['iou']
                fold_of[row['id']] = fold
    ids = [i for i in records if i in fold_of and all((i, a) in quality and (i, a) in task for a in ACTIONS)]
    gains = {}
    for i in ids:
        base_s, base_l = quality[(i, 'identity')]
        base_t = task[(i, 'identity')]
        gains[i] = np.array([[quality[(i, a)][0] - base_s, base_l - quality[(i, a)][1], task[(i, a)] - base_t]
                             for a in ACTIONS], dtype=np.float32)
    return records, ids, gains, fold_of


def batch_images(root, records, ids):
    images = {}
    for i in ids:
        image, _ = actions_module.load_working(Path(root) / records[i]['input'])
        images[i] = np.asarray(image, dtype=np.uint8)
    return images


def forward_ids(model, images, ids, device):
    rows = []
    groups = {}
    for i in ids:
        groups.setdefault(images[i].shape, []).append(i)
    with torch.inference_mode():
        for group in groups.values():
            for start in range(0, len(group), 8):
                part = group[start:start + 8]
                x = seg.to_tensor(np.stack([images[i] for i in part]), device)
                pred = model(x).float().view(len(part), len(ACTIONS), 3).cpu().numpy()
                rows.extend(zip(part, pred))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--quality-dir', required=True)
    ap.add_argument('--b3', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261011)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        print('already complete', out, flush=True)
        return
    records, ids, gains, fold_of = load_joined(a.records, a.quality_dir, a.b3)
    print('images', len(ids), flush=True)
    images = batch_images(a.root, records, ids)
    device = torch.device('cuda')
    summary = []
    for fold in range(5):
        train = [i for i in ids if fold_of[i] != fold]
        test = [i for i in ids if fold_of[i] == fold]
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(ACTIONS) * 3)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        order = np.array(train)
        for epoch in range(a.epochs):
            rng = np.random.default_rng(a.seed + fold * 100 + epoch)
            rng.shuffle(order)
            total = 0.0
            seen = 0
            model.train()
            buckets = {}
            for i in order:
                buckets.setdefault(tuple(images[i].shape), []).append(i)
            for group in buckets.values():
                for start in range(0, len(group), 8):
                    part = group[start:start + 8]
                    x = seg.to_tensor(np.stack([images[i] for i in part]), device)
                    y = torch.tensor(np.stack([gains[i] for i in part]), device=device)
                    pred = model(x).view(len(part), len(ACTIONS), 3)
                    loss = loss_fn(pred, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += float(loss) * len(part)
                    seen += len(part)
            print(f'fold {fold} epoch {epoch + 1} mse {total / max(seen, 1):.6f}', flush=True)
        model.eval()
        chosen_fixed = int(np.mean([gains[i] for i in train], axis=0).sum(axis=1).argmax())
        pred_rows = forward_ids(model, images, test, device)
        joint_pred, joint_fixed, joint_oracle = [], [], []
        lines = []
        for i, pred in pred_rows:
            true = gains[i]
            pick = int(pred.sum(axis=1).argmax())
            joint_pred.append(float(true[pick].sum()))
            joint_fixed.append(float(true[chosen_fixed].sum()))
            joint_oracle.append(float(true.sum(axis=1).max()))
            lines.append(json.dumps(dict(id=i, group_id=records[i]['group_id'], fold=fold,
                                         chosen=ACTIONS[pick], fixed=ACTIONS[chosen_fixed],
                                         pred=pred.tolist(), true=true.tolist())) + '\n')
        (out / f'fold{fold}.jsonl').write_text(''.join(lines), encoding='utf-8')
        torch.save(dict(fold=fold, epochs=a.epochs, seed=a.seed, actions=ACTIONS,
                        model=model.state_dict()), out / f'fold{fold}.pt')
        summary.append(dict(fold=fold, test_images=len(test), fixed_action=ACTIONS[chosen_fixed],
                            mean_joint_pred=float(np.mean(joint_pred)),
                            mean_joint_fixed=float(np.mean(joint_fixed)),
                            mean_joint_oracle=float(np.mean(joint_oracle))))
        print('fold_done', fold, summary[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    (out / 'summary.json').write_text(json.dumps(dict(images=len(ids), epochs=a.epochs, seed=a.seed,
                                                      task_endpoint='lraspp raw', folds=summary), indent=2),
                                      encoding='utf-8')
    print('CNN_DONE', flush=True)


if __name__ == '__main__':
    main()
