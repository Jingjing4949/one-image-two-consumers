"""Server CPU timings for the locked selector paths. Sixteen development images, three rounds.

Images are chosen by the hash of their id before any timing. Holdout files are not opened.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import ivc_b_actions as actions_module
import lsui_router_features as features
import run_ivc_b3_seg as seg
from run_ivc_b5_cnn import ACTIONS
from torchvision.models import mobilenet_v3_small


def choose(records, k=16):
    ranked = sorted(records, key=lambda r: hashlib.sha256(('IVC-B-latency-20261011:' + r['id']).encode()).hexdigest())
    return ranked[:k]


def percentile(values, p):
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--cnn', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    payload = json.loads(Path(a.records).read_text(encoding='utf-8'))
    holdout = set(payload.get('holdout_experiments') or [])
    if any(r.get('experiment') in holdout for r in payload['records']):
        raise SystemExit('holdout records present')
    picked = choose(payload['records'])
    device = torch.device('cpu')
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(ACTIONS) * 3)
    state = torch.load(Path(a.cnn) / 'fold0.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    runners = {}
    rows = []
    for round_id in range(3):
        for rec in picked:
            t0 = time.perf_counter()
            image, _ = actions_module.load_working(Path(a.root) / rec['input'])
            rgb = np.asarray(image, dtype=np.uint8)
            t1 = time.perf_counter()
            features.extract_features(rgb)
            t2 = time.perf_counter()
            with torch.inference_mode():
                pred = model(seg.to_tensor(rgb, device)).view(len(ACTIONS), 3)
                choice = int(pred.sum(1).argmax())
            t3 = time.perf_counter()
            enhance_ms = None
            try:
                if ACTIONS[choice] not in runners:
                    runners[ACTIONS[choice]] = actions_module.Actions([ACTIONS[choice]], 'cpu')
                t4 = time.perf_counter()
                runners[ACTIONS[choice]].run(ACTIONS[choice], image)
                enhance_ms = (time.perf_counter() - t4) * 1000
            except Exception as exc:
                enhance_ms = None
                error = str(exc)
            else:
                error = ''
            rows.append(dict(round=round_id, id=rec['id'], load_ms=(t1 - t0) * 1000,
                             features_ms=(t2 - t1) * 1000, cnn_ms=(t3 - t2) * 1000,
                             enhance_ms=enhance_ms, error=error,
                             total_ms=(time.perf_counter() - t0) * 1000, chosen=ACTIONS[choice]))
            print('timed', rec['id'], round(rows[-1]['total_ms'], 1), flush=True)
    def pack(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return dict(p50=percentile(vals, 50), p95=percentile(vals, 95), max=max(vals), n=len(vals))
    summary = dict(device='cpu', threads=4, images=len(picked), rounds=3,
                   load=pack('load_ms'), features=pack('features_ms'), cnn=pack('cnn_ms'),
                   enhance=pack('enhance_ms'), total=pack('total_ms'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1), encoding='utf-8')
    print('LATENCY_DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
