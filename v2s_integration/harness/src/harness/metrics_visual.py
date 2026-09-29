"""
metrics_visual.py — Task5 visual metrics (local, offline, no external API).

Input: a list of paired image paths (agent_render_i, gt_render_i), i indexing multiple views.
The two sets are rendered with the **same render pipeline and same camera** (GT glb and agent
glb), eliminating render-style differences so pixel/perceptual/semantic metrics are reliable.

Metrics:
- MSSIM   ^  multi-scale structural similarity (scikit-image, better than PSNR)
- PSNR    ^  pixel peak signal-to-noise ratio (reference)
- PL      v  LPIPS perceptual loss (AlexNet, local weights)
- CLIP    ^  CLIP image-feature cosine similarity
- N-CLIP  v  normalized CLIP distance = 1 - CLIP (smaller is better, same direction as reconstruction error)

Dependencies: torch / torchvision / open_clip_torch / lpips / scikit-image / numpy / PIL (all local).
"""
import numpy as np
from PIL import Image

_clip = None
_lpips = None


def _load_clip():
    global _clip
    if _clip is None:
        import open_clip, torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()
        _clip = (model, preprocess, torch)
    return _clip


def _load_lpips():
    global _lpips
    if _lpips is None:
        import lpips, torch
        _lpips = (lpips.LPIPS(net='alex'), torch)
    return _lpips


def _rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size)
    return im


def mssim_psnr(a_path, g_path, size=256):
    """MSSIM + PSNR (scikit-image)."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    a = np.asarray(_rgb(a_path, (size, size)), float) / 255.0
    g = np.asarray(_rgb(g_path, (size, size)), float) / 255.0
    s = float(ssim(a, g, channel_axis=2, data_range=1.0))
    p = float(psnr(g, a, data_range=1.0))
    return {"mssim": s, "psnr": p}


def clip_sim(a_path, g_path):
    """CLIP image-feature cosine similarity, N-CLIP = 1 - cos."""
    model, preprocess, torch = _load_clip()
    with torch.no_grad():
        a = model.encode_image(preprocess(_rgb(a_path)).unsqueeze(0))
        g = model.encode_image(preprocess(_rgb(g_path)).unsqueeze(0))
        a = a / a.norm(dim=-1, keepdim=True)
        g = g / g.norm(dim=-1, keepdim=True)
        cos = float((a @ g.T).item())
    return {"clip": cos, "n_clip": 1.0 - cos}


def lpips_pl(a_path, g_path, size=256):
    """LPIPS perceptual loss (PL), smaller means more similar."""
    model, torch = _load_lpips()
    def t(p):
        x = np.asarray(_rgb(p, (size, size)), np.float32) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1))[None] * 2 - 1
        return x
    with torch.no_grad():
        d = float(model(t(a_path), t(g_path)).item())
    return {"pl_lpips": d}


def visual_compare(pairs, want=("mssim", "clip", "lpips")):
    """Compute visual metrics on multi-view paired images and average over views.
    pairs: [(agent_png, gt_png), ...]. Returns per-metric means + per-view detail.

    Note: SigLIP-2 visual similarity was tried but had insufficient discriminative power on
    whole-scene reconstruction (two unrelated bedrooms still scored 0.92 similarity; it mostly
    recognizes "is this a room photo" rather than "is it built correctly"), so it was dropped.
    The T5 primary metric stays with geometry (obj_f@5%_matched)."""
    per_view = []
    for a, g in pairs:
        r = {"agent": a, "gt": g}
        if "mssim" in want:
            r.update(mssim_psnr(a, g))
        if "clip" in want:
            r.update(clip_sim(a, g))
        if "lpips" in want:
            r.update(lpips_pl(a, g))
        per_view.append(r)
    keys = [k for k in ("mssim", "psnr", "clip", "n_clip", "pl_lpips")
            if per_view and k in per_view[0]]
    agg = {f"mean_{k}": float(np.mean([v[k] for v in per_view])) for k in keys}
    agg["per_view"] = per_view
    agg["n_views"] = len(per_view)
    return agg
