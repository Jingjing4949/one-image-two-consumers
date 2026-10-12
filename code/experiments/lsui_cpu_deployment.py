"""Frozen-source CPU inference adapters for T1. No training or quality labels."""
import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "work/lsui_t1/third_party"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def construct(name):
    if name == "funie_generator":
        return module("t1_funie", THIRD / "funie/funiegan.py").GeneratorFunieGAN()
    if name == "dehazeformer_t":
        return module("t1_df", THIRD / "dehazeformer.py").dehazeformer_t()
    if name == "ushape_generator":
        sys.path.insert(0, str(THIRD / "ushape"))
        from net.Ushape_Trans import Generator
        return Generator()
    raise ValueError(name)


def load_model(info):
    model = construct(info["model"])
    checkpoint = torch.load(ROOT / info["weight"], map_location="cpu", weights_only=True)
    if (checkpoint["epoch"] != 40 or checkpoint["run"]["seed"] != info["seed"]
            or checkpoint["run"]["model"] != info["model"]
            or checkpoint["plan_sha256"] != hashlib.sha256((ROOT / "work/lsui_p3b/training_plan.json").read_bytes()).hexdigest()):
        raise RuntimeError("Unexpected checkpoint identity")
    model.load_state_dict(checkpoint["model"], strict=True)
    if info.get("bn_buffers"):
        buffers = torch.load(ROOT / info["bn_buffers"], map_location="cpu", weights_only=True)
        allowed = {n + "." + k for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for k in ("running_mean", "running_var", "num_batches_tracked")}
        if set(buffers) != allowed or len(allowed) != 21:
            raise RuntimeError("Unexpected BN overlay")
        state = model.state_dict()
        state.update(buffers)
        model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def canonical(path):
    data = Path(path).read_bytes()
    with Image.open(BytesIO(data)) as im:
        return im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)


def tensor(image):
    array = np.asarray(image)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1).unsqueeze(0)


def output(model, x):
    raw = x if model is None else model(x)
    if isinstance(raw, (tuple, list)):
        raw = raw[-1]
    return ((raw + 1) / 2).clamp(0, 1)


def request(model, path, feature_function=None):
    image = canonical(path)
    features = None if feature_function is None else feature_function(image)
    x = tensor(image)
    y = output(model, x)
    result = np.ascontiguousarray(y[0].permute(1, 2, 0).numpy()).copy()
    return result, features
