"""
metrics_3d.py — Task5 3D semantic metric (local, offline).

Uses OpenShape's PointBERT (aligned-trained on the OpenCLIP ViT-B/32 text-image-point-cloud
tri-modal space) to encode a point cloud into a 512-d vector, then computes the cosine
similarity between the agent reconstruction and the GT.

Why it is needed. Geometric metrics (Chamfer/F-score) only look at point distances and
visual CLIP only looks at 2D renders; neither captures "3D semantics". If an agent builds
a "bed" as a flat block, the geometric point distance may still be acceptable but the
semantics are entirely wrong. PointBERT embeddings are sensitive to "shape semantics" and
fill exactly this gap.

Pixal3D (arXiv 2605.10922) treats Uni3D/ULIP-2 as its core metric on in-the-wild test sets;
OpenShape PointBERT is in the same family (all aligned with OpenCLIP), so the embedding
cosine has consistent meaning.

Dependencies: torch / einops / torch_redstone (local package, see _openshape/); weights are
auto-downloaded from HF on first use.
Model: openshape-pointbert-vitb32-rgb (~50MB, ~30ms/call on Mac CPU).
"""
import os
import sys
import numpy as np

_model = None


def _load(name="openshape-pointbert-vitb32-rgb"):
    """Lazy-load PointBERT (B32 = smallest that suffices). Downloaded from HF on first use, then cached."""
    global _model
    if _model is None:
        sys.path.insert(0, os.path.dirname(__file__))
        import _openshape as openshape
        import torch
        _model = (openshape.load_pc_encoder(name), torch)
    return _model


def _to_input(points, n=10000, rng=None):
    """(K,3) numpy point cloud -> (1,6,N) tensor: xyz normalized to the unit sphere, rgb filled with gray.

    OpenShape normalizes training inputs to the origin + unit max radius; without normalization
    (our agent reconstructions have arbitrary scale) the embedding drifts off the training
    distribution and the cosine is distorted. Here we apply the same normalization as the official code:
        xyz' = (xyz - centroid) / max_dist
    """
    rng = rng or np.random.default_rng(0)
    P = np.asarray(points, np.float32)
    if len(P) > n:
        P = P[rng.choice(len(P), n, False)]
    elif len(P) < n:
        # Fewer than n points: pad by resampling with replacement (PointBERT requires a fixed N)
        idx = rng.choice(len(P), n, True)
        P = P[idx]
    centroid = P.mean(0)
    P = P - centroid
    m = float(np.linalg.norm(P, axis=1).max() + 1e-9)
    P = P / m
    rgb = np.full_like(P, 0.5, dtype=np.float32)        # neutral gray
    pc = np.concatenate([P, rgb], axis=1)               # (N, 6)
    _, torch = _load()
    return torch.from_numpy(pc).T.unsqueeze(0)          # (1, 6, N)


def encode(points, n=10000, rng=None):
    """Point cloud -> 512-d embedding (numpy)."""
    model, torch = _load()
    x = _to_input(points, n=n, rng=rng)
    with torch.no_grad():
        v = model(x)
    return v[0].cpu().numpy()


def cosine(a, b):
    """Cosine similarity of two embeddings, in [-1, 1] (usually in [0, 1])."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def pointbert_compare(pred_pts, gt_pts, gt_parts=None, n=10000, rng=None,
                       aligned_pred=None):
    """Task5 3D semantic metric main entry.

    Returns:
      - scene_pointbert: cosine of the whole-scene agent vs whole-scene GT
      - per_object_pointbert: [(part_name, cos)], per object (agent points selected by each GT object box)
      - obj_pointbert: mean of the per-object cosines (missing / empty regions counted as 0)

    aligned_pred: agent point cloud already aligned to the GT world frame via sim(3) ICP. If
      provided, both per-object selection and scene-level encoding use aligned_pred, so it shares
      the same space as obj_f@5%. If not provided, the raw pred_pts is used (the PointBERT input
      is locally normalized so scene-level comparison still works, but per-object box selection
      may come up empty).
    """
    P = np.asarray(pred_pts, float)
    Pa = np.asarray(aligned_pred, float) if aligned_pred is not None else P
    out = {"scene_pointbert": cosine(encode(Pa, n=n, rng=rng),
                                      encode(gt_pts, n=n, rng=rng))}
    if not gt_parts:
        return out
    Q = np.asarray(gt_pts, float)
    diagQ = float(np.linalg.norm(Q.max(0) - Q.min(0)))
    margin = 0.10 * diagQ
    per = []
    for name, gv in gt_parts:
        gv = np.asarray(gv, float)
        lo, hi = gv.min(0) - margin, gv.max(0) + margin
        mask = np.all((Pa >= lo) & (Pa <= hi), axis=1)
        if mask.sum() < 100:                # missing / almost no agent points in this region
            per.append({"part": name, "n_pred_in": int(mask.sum()), "pointbert": 0.0})
            continue
        cos = cosine(encode(Pa[mask], n=n, rng=rng), encode(gv, n=n, rng=rng))
        per.append({"part": name, "n_pred_in": int(mask.sum()), "pointbert": cos})
    out["per_object_pointbert"] = per
    out["obj_pointbert"] = float(np.mean([p["pointbert"] for p in per])) if per else 0.0
    return out
