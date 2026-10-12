"""B-series enhancement action registry.

Every action maps an RGB uint8 image at working size to an RGB uint8 image of the
same size. Built-in actions reuse the A-series weights; third-party actions are
registered after the B1 weight audit.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import lsui_cpu_deployment as deploy  # noqa: E402

LITE_SOURCE = ROOT / 'work/baseline_sources/dsp_candidate_audit_20260926/LiteEnhanceNet/model.py'
SHORT_SIDE = 512
MULTIPLE = 32

BUILTIN_WEIGHTS = {
    'funie_s20260913': ('funie', 20260913, 'outputs/lsui_e1/20260918T045144Z-df599454/payload/results/funie_e1_s20260913/final_bn_model.pt'),
    'funie_s20261013': ('funie', 20261013, 'outputs/lsui_e1/20260918T045144Z-df599454/payload/results/funie_e1_s20261013/final_bn_model.pt'),
    'lite_s20260913': ('lite', 20260913, 'outputs/ivc_lite_l1/20260927T131210Z-a105edfe/payload/results/lite_s20260913/checkpoints/epoch_200.pt'),
    'lite_s20261013': ('lite', 20261013, 'outputs/ivc_lite_l1/20260927T131210Z-a105edfe/payload/results/lite_s20261013/checkpoints/epoch_200.pt'),
}
BUILTIN = ['identity', 'gray_world_mean', *BUILTIN_WEIGHTS]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def working_size(width, height):
    scale = SHORT_SIDE / min(width, height)
    w, h = width * scale, height * scale
    return max(MULTIPLE, int(round(w / MULTIPLE)) * MULTIPLE), max(MULTIPLE, int(round(h / MULTIPLE)) * MULTIPLE)


def load_working(path):
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        size = working_size(*rgb.size)
        return rgb.resize(size, Image.Resampling.BICUBIC), list(rgb.size)


def to_uint8(array01):
    return np.clip(np.rint(np.asarray(array01, dtype=np.float64) * 255), 0, 255).astype(np.uint8)


class Actions:
    def __init__(self, names, device='cpu'):
        self.device = torch.device(device)
        self.names = list(names)
        self.models = {}
        self.weights = {}
        lite_module = None
        for name in self.names:
            if name in BUILTIN_WEIGHTS:
                family, seed, rel = BUILTIN_WEIGHTS[name]
                path = ROOT / rel
                ck = torch.load(path, map_location='cpu', weights_only=True)
                if family == 'funie':
                    assert ck['epoch'] == 120 and ck['run']['seed'] == seed
                    net = deploy.construct('funie_generator')
                else:
                    assert ck['epoch'] == 200 and ck['updates'] == 61600 and ck['seed'] == seed
                    if lite_module is None:
                        spec = importlib.util.spec_from_file_location('ivc_b_lite', LITE_SOURCE)
                        lite_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(lite_module)
                    net = lite_module.Mynet()
                net.load_state_dict(ck['model'], strict=True)
                self.models[name] = net.float().eval().requires_grad_(False).to(self.device)
                self.weights[name] = dict(path=rel, sha256=sha(path))
            elif name not in ('identity', 'gray_world_mean'):
                import ivc_b_third_party as third
                if name not in third.BUILDERS:
                    raise KeyError('Unregistered action ' + name)
                fn, info = third.BUILDERS[name](self.device)
                self.models[name] = fn
                self.weights[name] = dict(path=str(Path(info['weight']).relative_to(ROOT)), sha256=sha(info['weight']),
                                          deviation=info['deviation'])

    @torch.inference_mode()
    def run(self, name, image):
        x = np.asarray(image, dtype=np.uint8)
        if name == 'identity':
            return x.copy()
        if name == 'gray_world_mean':
            v = x.astype(np.float64) / 255
            means = v.mean((0, 1))
            return to_uint8(np.clip(v * (means.mean() / np.maximum(means, 1e-6))[None, None, :], 0, 1))
        net = self.models[name]
        if name not in BUILTIN:
            out = net(x)
            assert out.shape == x.shape and out.dtype == np.uint8
            return out
        if name.startswith('funie'):
            y = deploy.output(net, deploy.tensor(image).to(self.device))
        else:
            t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0).to(self.device)
            y = net(t).clamp(0, 1)
        out = y[0].permute(1, 2, 0).float().cpu().numpy()
        assert out.shape == x.shape and np.isfinite(out).all()
        return to_uint8(out)
