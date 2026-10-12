"""Input-only feature prototypes, independent of labels and filesystem paths.

These conventional descriptors are candidate baselines, not a novelty claim.
No preprocessing statistics are fitted here. Later scalers must use training
folds only, with model/feature choices frozen before calibration and final test.
"""
import numpy as np
from PIL import Image

FEATURE_VERSION = "lsui_input_only_v1"
RGB_NAMES = ["r", "g", "b"]
STAT_NAMES = ([c + "_mean" for c in RGB_NAMES] + [c + "_std" for c in RGB_NAMES]
              + [c + "_q" + str(q) for q in (10,25,50,75,90) for c in RGB_NAMES]
              + [c + "_hist" + str(b) for c in ("h","s","v") for b in range(8)]
              + ["gray_mean", "gray_std", "forward_gradient_mean", "forward_gradient_std",
                 "forward_gradient_q90", "dark_fraction", "bright_fraction"])


def specification():
    return dict(version=FEATURE_VERSION, input="uint8 RGB array; no paths or quality labels",
                canonical_resize=dict(width=256,height=256,interpolation="PIL bicubic"),
                stats52=dict(side=64,names=STAT_NAMES,quantile_method="numpy linear",histogram_bins=8,
                             histogram_range=[0,1],histogram_normalization="fraction of all pixels",
                             grayscale_coefficients=[.299,.587,.114],dark_threshold=.05,bright_threshold=.95,
                             gradient="forward differences on the shared 63x63 interior"),
                hsv_pixels768=dict(side=16,flatten_order="HWC row-major",normalization="PIL uint8 HSV / 255"),
                fitted_transform=False,physical_water_parameters=False,
                novelty_claim=False,reference_or_candidate_output_required=False)


def canonical_image(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 RGB array")
    if min(rgb.shape[:2]) < 2:
        raise ValueError("At least two rows and columns required")
    bicubic = getattr(Image,"Resampling",Image).BICUBIC
    return Image.fromarray(np.ascontiguousarray(rgb)).resize((256,256),bicubic)


def from_canonical_image(image):
    if image.mode != "RGB" or image.size != (256,256):
        raise ValueError("Use the frozen RGB 256x256 input transformation")
    bicubic = getattr(Image,"Resampling",Image).BICUBIC
    small = image.resize((64,64),bicubic)
    rgb = np.asarray(small,dtype=np.float64)/255
    pixels = rgb.reshape(-1,3)
    hsv = np.asarray(small.convert("HSV"),dtype=np.float64).reshape(-1,3)/255
    gray = rgb @ np.array([.299,.587,.114])
    gx = np.diff(gray,axis=1)[:-1,:]
    gy = np.diff(gray,axis=0)[:,:-1]
    gradient = np.hypot(gx,gy)
    pieces = [pixels.mean(axis=0),pixels.std(axis=0),np.quantile(pixels,[.1,.25,.5,.75,.9],axis=0).flatten()]
    pieces += [np.histogram(hsv[:,c],bins=8,range=(0,1))[0]/len(hsv) for c in range(3)]
    pieces += [np.array([gray.mean(),gray.std(),gradient.mean(),gradient.std(),np.quantile(gradient,.9),
                        np.mean(gray<.05),np.mean(gray>.95)])]
    stats = np.concatenate(pieces)
    hsv_pixels = np.asarray(image.resize((16,16),bicubic).convert("HSV"),dtype=np.float64).reshape(-1)/255
    if stats.shape != (52,) or hsv_pixels.shape != (768,) or not np.isfinite(stats).all() or not np.isfinite(hsv_pixels).all():
        raise RuntimeError("Feature shape or finite-value check failed")
    return {"stats52":stats,"hsv_pixels768":hsv_pixels}


def extract_features(rgb):
    return from_canonical_image(canonical_image(rgb))


def verify_analytic_inputs():
    checks = {}
    for name,value in (("black",0),("gray",128),("white",255)):
        result = extract_features(np.full((73,81,3),value,dtype=np.uint8))["stats52"]
        checks[name+"_mean"] = bool(np.allclose(result[:3],value/255,atol=1e-12,rtol=0))
        checks[name+"_constant_gradient"] = bool(np.allclose(result[47:50],0,atol=1e-12,rtol=0))
        checks[name+"_histograms_normalized"] = bool(np.allclose(result[21:45].reshape(3,8).sum(axis=1),1))
    red = np.zeros((64,64,3),dtype=np.uint8)
    red[:,:,0]=255
    feature = extract_features(red)
    checks["rgb_channel_order"] = bool(np.array_equal(feature["stats52"][:3], [1,0,0]))
    checks["hsv_channel_order"] = bool(np.array_equal(feature["hsv_pixels768"].reshape(16,16,3)[0,0],[0,1,1]))
    checks["determinism"] = all(np.array_equal(feature[k],extract_features(red.copy())[k]) for k in feature)
    ramp = np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3))
    checks["gradient_detects_ramp"] = bool(extract_features(ramp)["stats52"][47]>.01)
    checks["rejects_float_input"] = False
    try:
        extract_features(red.astype(float))
    except ValueError:
        checks["rejects_float_input"] = True
    if not all(checks.values()):
        raise AssertionError(str(checks))
    return checks


if __name__ == "__main__":
    import json
    print(json.dumps(verify_analytic_inputs(),indent=2))
