"""Official-weight wrappers for third-party enhancers (B1).

Each builder returns f(uint8 HxWx3 RGB) -> uint8 HxWx3 RGB at the same size and a
record of the weight file and any deviation from the authors' inference script.
Load only one third-party method per process: several repositories share module
names (model, models, utils).
"""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'work/baseline_sources/ivc_b1_20261010'
WEIGHTS = SRC / '_weights'


def _path(repo):
    p = str(SRC / repo)
    if p not in sys.path:
        sys.path.insert(0, p)


def _t01(x, device):
    return torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(2, 0, 1)[None].float().div(255)


def _u8(t):
    return np.clip(np.rint(t[0].permute(1, 2, 0).float().cpu().numpy() * 255), 0, 255).astype(np.uint8)


def _strip(sd):
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}


def _shim_mmcv():
    """mmcv pieces used by the repos, mapped to torch/torchvision equivalents."""
    import torchvision.ops as tvops
    mmcv = types.ModuleType('mmcv')
    ops = types.ModuleType('mmcv.ops')
    cnn = types.ModuleType('mmcv.cnn')

    def modulated_deform_conv2d(x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups):
        return tvops.deform_conv2d(x, offset, weight, bias, stride=stride, padding=padding, dilation=dilation, mask=mask)

    def build_norm_layer(cfg, num_features):
        assert cfg['type'] in ('SyncBN', 'BN', 'BN2d')
        return 'bn', nn.BatchNorm2d(num_features)

    ops.modulated_deform_conv2d = modulated_deform_conv2d
    cnn.build_norm_layer = build_norm_layer
    mmcv.ops, mmcv.cnn = ops, cnn
    sys.modules.update({'mmcv': mmcv, 'mmcv.ops': ops, 'mmcv.cnn': cnn})


def semi_uir(device):
    _shim_mmcv()
    _path('Semi-UIR')
    import cv2
    from model import AIMnet
    w = SRC / 'Semi-UIR/pretrained/model.pth'
    net = AIMnet()
    net.load_state_dict(_strip(torch.load(w, map_location='cpu', weights_only=False)['state_dict']))
    net = net.to(device).eval()

    def luminance(img):
        ill = np.ones_like(img).astype(np.float32)
        for s in (15, 60, 90):
            ill = ill + np.clip(np.log10(cv2.GaussianBlur(img, (0, 0), s) + 1e-8), 0, 255)
        ill = ill / 3
        return np.uint8((ill - ill.min()) / (ill.max() - ill.min() + 1e-6) * 255)

    def f(x):
        out, _ = net(_t01(x, device), _t01(luminance(x), device))
        return _u8(out.clamp(0, 1))
    return f, dict(weight=w, deviation='mmcv modulated_deform_conv2d replaced by torchvision.ops.deform_conv2d (same DCNv2 op)')


def phaseformer(device):
    _path('Phaseformer')
    w = SRC / 'Phaseformer/checkpoints/UIEB/best.pth'
    net = torch.load(w, map_location='cpu', weights_only=False).to(device).eval()

    def f(x):
        h, w_ = x.shape[:2]
        t = F.interpolate(_t01(x, device), size=(512, 512), mode='bicubic', align_corners=False).clamp(0, 1) * 2 - 1
        out = ((net(t)[1] + 1) / 2).clamp(0, 1)
        return _u8(F.interpolate(out, size=(h, w_), mode='bicubic', align_corners=False).clamp(0, 1))
    return f, dict(weight=w, deviation='authors resize test inputs to 512x512; output resized back to working size')


def deep_wavenet(device):
    _path('Deep-WaveNet/uie_euvp')
    from models import CC_Module
    w = SRC / 'Deep-WaveNet/uie_euvp/ckpts/netG_17.pt'
    net = CC_Module()
    net.load_state_dict(torch.load(w, map_location='cpu', weights_only=False)['model_state_dict'])
    net = net.to(device).eval()
    return (lambda x: _u8(net(_t01(x, device)).clamp(0, 1))), dict(weight=w, deviation=None)


def polykernel(device):
    _shim_mmcv()
    _path('UIR-PolyKernel')
    from models.model import UIR_PolyKernel
    w = WEIGHTS / 'UIR-PolyKernel/weight.pth'
    net = UIR_PolyKernel()
    net.load_state_dict(_strip(torch.load(w, map_location='cpu', weights_only=False)['state_dict']))
    net = net.to(device).eval()
    return (lambda x: _u8(net(_t01(x, device)).clamp(0, 1))), dict(
        weight=w, deviation='mmcv build_norm_layer(SyncBN) mapped to BatchNorm2d (identical in eval mode)')


def shallow_uwnet(device):
    _path('Shallow-UWnet')
    importlib.import_module('model')
    w = WEIGHTS / 'Shallow-UWnet/model.ckpt'
    net = torch.load(w, map_location='cpu', weights_only=False).to(device).eval()
    return (lambda x: _u8(net(_t01(x, device)).clamp(0, 1))), dict(
        weight=w, deviation='authors resize to 256x256 by default; run at working size')


def syreanet(device):
    _path('SyreaNet')
    import yaml
    from models import SyreaNet
    cfg = yaml.safe_load((SRC / 'SyreaNet/configs/syreanet_test.yaml').read_text())['Network']
    w = WEIGHTS / 'SyreaNet/pretrained.pth'
    net = SyreaNet(cfg)
    sd = {k: v.clone() for k, v in torch.load(w, map_location='cpu', weights_only=False).items()}
    net.load_state_dict(sd, assign=True)
    net = net.to(device).eval()
    return (lambda x: _u8(net(_t01(x, device))[0].clamp(0, 1))), dict(weight=w, deviation=None)


def ushape(device):
    _path('U-shape')
    from net.Ushape_Trans import Generator
    w = WEIGHTS / 'U-shape/saved_models/G/generator_795.pth'
    net = Generator()
    net.load_state_dict(torch.load(w, map_location='cpu', weights_only=False))
    net = net.to(device).eval()

    def f(x):
        h, w_ = x.shape[:2]
        t = F.interpolate(_t01(x, device), size=(256, 256), mode='bicubic', align_corners=False).clamp(0, 1)
        out = net(t)[3]
        out = (out - out.min()) / (out.max() - out.min() + 1e-12)
        out = F.interpolate(out, size=(h, w_), mode='bicubic', align_corners=False).clamp(0, 1)
        return _u8(out)
    return f, dict(weight=w, deviation='fixed 256x256 transformer: input resized to 256, output min-max normalised '
                                       '(authors save_image normalize=True) and resized back to working size')


def tacl(device):
    _path('TACL')
    from models.networks import define_G
    w = WEIGHTS / 'TACL/weights/latest_net_G_A.pth'
    net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, gpu_ids=[])
    net.load_state_dict(torch.load(w, map_location='cpu', weights_only=False))
    net = net.to(device).eval()
    return (lambda x: _u8(((net(_t01(x, device) * 2 - 1) + 1) / 2).clamp(0, 1))), dict(
        weight=w, deviation='CycleGAN G_A (underwater->enhanced) run in eval mode without dropout')


def uiec2net(device):
    _path('UWEnhancement')
    from core.Models.UWModels.UIEC2Net import UIEC2Net
    w = WEIGHTS / 'UWEnhancement/UIEC2Net.pth'
    net = UIEC2Net()
    ck = torch.load(w, map_location='cpu', weights_only=False)
    net.load_state_dict(_strip(ck.get('state_dict', ck)))
    net = net.to(device).eval()
    return (lambda x: _u8(net(_t01(x, device)).clamp(0, 1))), dict(weight=w, deviation=None)


BUILDERS = {
    'tacl': tacl,
    'uiec2net': uiec2net,
    'semi_uir': semi_uir,
    'phaseformer_uieb': phaseformer,
    'deep_wavenet_euvp': deep_wavenet,
    'uir_polykernel': polykernel,
    'shallow_uwnet': shallow_uwnet,
    'syreanet': syreanet,
    'ushape_lsui': ushape,
}
