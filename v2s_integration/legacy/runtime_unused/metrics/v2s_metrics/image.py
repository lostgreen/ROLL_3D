"""Paired RGB metrics. No resizing, exposure fitting or automatic alignment."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .config import MetricConfig
from .result import metric


def rgb(value):
    if isinstance(value, (str, Path)):
        from PIL import Image
        with Image.open(value) as im:
            if im.mode not in ("RGB", "L"):
                raise ValueError("images must be opaque RGB or L; composite alpha explicitly")
            a = np.asarray(im.convert("RGB"))
    else:
        a = np.asarray(value)
    if a.ndim != 3 or a.shape[2] != 3 or not a.shape[0] or not a.shape[1]:
        raise ValueError("expected nonempty H x W x 3 RGB")
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    elif np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    else:
        raise ValueError("RGB must be uint8 or floating point in [0,1]")
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("RGB values must be finite and in [0,1]")
    return a


@lru_cache(maxsize=2)
def _lpips_model(allow_download):
    try:
        import torch
        import lpips
        from torchvision.models import AlexNet_Weights
    except ImportError as exc:
        raise ImportError("LPIPS requires torch, torchvision and lpips") from exc
    from urllib.parse import urlparse
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(AlexNet_Weights.IMAGENET1K_V1.url).path).name
    if not allow_download and not checkpoint.is_file():
        raise ImportError("LPIPS AlexNet backbone is not cached; prepare weights or explicitly allow downloads")
    model = lpips.LPIPS(net="alex", version="0.1", verbose=False).cpu().eval()
    return model, torch


def lpips_distance(a, b, allow_download=False):
    model, torch = _lpips_model(allow_download)
    def tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32))[None] * 2 - 1
    with torch.inference_mode():
        return float(model(tensor(a), tensor(b)).item())


def compare_images(prediction, reference, config=None):
    config = config or MetricConfig()
    a, b = rgb(prediction), rgb(reference)
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} vs {b.shape}; no implicit resize")
    mse = float(np.mean((a - b) ** 2))
    out = {}
    for name in config.image_metrics:
        if name == "mse":
            out[name] = metric(mse)
        elif name == "psnr_db":
            psnr = -10 * np.log10(mse) if mse else float("inf")
            out[name] = metric(min(float(psnr), config.psnr_cap_db), direction="higher", unit="dB",
                               capped=bool(psnr >= config.psnr_cap_db), cap_db=config.psnr_cap_db)
        elif name == "ssim":
            if min(a.shape[:2]) < config.ssim_window:
                out[name] = metric(direction="higher", status="invalid_input", reason="image smaller than SSIM window")
                continue
            try:
                from skimage.metrics import structural_similarity
                value = structural_similarity(a, b, channel_axis=2, data_range=1.0,
                                              win_size=config.ssim_window, gaussian_weights=False,
                                              use_sample_covariance=True)
                out[name] = metric(value, direction="higher")
            except ImportError:
                out[name] = metric(direction="higher", status="dependency_unavailable", reason="install scikit-image")
        else:
            if min(a.shape[:2]) < 64:
                out[name] = metric(status="invalid_input", reason="LPIPS AlexNet requires images >= 64 pixels per side in this protocol")
                continue
            try:
                out[name] = metric(lpips_distance(a, b, config.lpips_allow_download), backbone="alex", lpips_version="0.1")
            except ImportError as exc:
                out[name] = metric(status="dependency_unavailable", reason=str(exc)[:240])
            except (RuntimeError, OSError) as exc:
                out[name] = metric(status="backend_error", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
    return out
