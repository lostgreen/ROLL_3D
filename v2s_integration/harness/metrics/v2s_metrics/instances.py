"""GT-count-independent instance matching, missing/extra penalties and pose."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import MetricConfig
from .geometry import compare_points, points, threshold_key
from .result import metric


def rotation(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or not np.allclose(a.T @ a, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(a), 1, atol=1e-5):
        raise ValueError("rotation must be a finite, proper orthogonal 3 x 3 matrix")
    return a


@dataclass
class Instance:
    id: str
    points: np.ndarray
    category: str | None = None
    rotation: np.ndarray | None = None
    symmetries: tuple = ()

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("instance id must be a nonempty string")
        if self.category is not None and not isinstance(self.category, str):
            raise ValueError("category must be a string or null")
        self.points = points(self.points, allow_empty=False)
        if self.rotation is not None:
            self.rotation = rotation(self.rotation)
        self.symmetries = tuple(rotation(r) for r in self.symmetries)

    @property
    def bounds(self):
        return np.array([self.points.min(axis=0), self.points.max(axis=0)])

    @property
    def center(self):
        return self.bounds.mean(axis=0)


def aabb_iou(a, b):
    intersection = float(np.prod(np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]))))
    union = float(np.prod(a[1] - a[0]) + np.prod(b[1] - b[0]) - intersection)
    return intersection / union if union > 0 else None


def pose_metrics(p, g):
    extent_p, extent_g = np.diff(p.bounds, axis=0)[0], np.diff(g.bounds, axis=0)[0]
    iou = aabb_iou(p.bounds, g.bounds)
    out = {"center_error_m": metric(np.linalg.norm(p.center - g.center), unit="m"),
           "aabb_iou": metric(iou, direction="higher", status="ok" if iou is not None else "not_applicable"),
           "aabb_extent_relative_error": metric(float(np.mean(np.abs(extent_p - extent_g) / extent_g)))
                if np.all(extent_g > 1e-12) else metric(status="not_applicable", reason="GT AABB has zero extent"),
           "rotation_error_deg": metric(unit="deg", status="not_applicable", reason="semantic rotations not supplied")}
    if p.rotation is not None and g.rotation is not None:
        angles = []
        for symmetry in (np.eye(3), *g.symmetries):
            relative = p.rotation.T @ (g.rotation @ symmetry)
            angles.append(float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))))
        out["rotation_error_deg"] = metric(min(angles), unit="deg")
    return out


def compare_instances(prediction, reference, config=None):
    config = config or MetricConfig()
    for group in (prediction, reference):
        if len({i.id for i in group}) != len(group):
            raise ValueError("instance ids must be unique within each set")
    npred, ngt = len(prediction), len(reference)
    matches = []
    if npred and ngt:
        # Augmented assignment with explicit unmatched alternatives. Large penalty
        # makes cardinality primary; distance + AABB cost resolve valid assignments.
        size = npred + ngt
        penalty = float(size + 1)
        cost = np.full((size, size), penalty)
        cost[npred:, ngt:] = 0.0
        allowed = np.zeros((npred, ngt), dtype=bool)
        for i, p in enumerate(prediction):
            for j, g in enumerate(reference):
                distance = np.linalg.norm(p.center - g.center)
                compatible = not config.match_category or p.category is None or g.category is None or p.category == g.category
                allowed[i, j] = compatible and distance <= config.match_max_distance_m
                iou = aabb_iou(p.bounds, g.bounds)
                cost[i, j] = (distance / config.match_max_distance_m + 0.25 * (1 - (iou or 0))) if allowed[i, j] else 4 * penalty * size
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if i < npred and j < ngt and allowed[i, j]:
                p, g = prediction[i], reference[j]
                matches.append({"prediction_id": p.id, "reference_id": g.id,
                    "world": compare_points(p.points, g.points, config),
                    "centered_shape": compare_points(p.points - p.center, g.points - g.center, config),
                    "pose": pose_metrics(p, g)})
    matched_p = {m["prediction_id"] for m in matches}
    matched_g = {m["reference_id"] for m in matches}
    k = len(matches)
    detection = {"precision": metric(k / npred if npred else (0.0 if ngt else None), direction="higher",
                                      status="ok" if npred or ngt else "not_applicable"),
                 "recall": metric(k / ngt if ngt else None, direction="higher", status="ok" if ngt else "not_applicable"),
                 "f1": metric(2 * k / (npred + ngt) if npred + ngt else None, direction="higher",
                               status="ok" if npred + ngt else "not_applicable")}
    macro = {}
    for tau in config.distance_thresholds_m:
        key = threshold_key(tau)
        macro[key] = {space: metric(sum(m[space]["thresholds"][key]["fscore"]["value"] for m in matches) / ngt if ngt else None,
                                   direction="higher", status="ok" if ngt else "not_applicable")
                      for space in ("world", "centered_shape")}
    return {"status": "ok", "n_prediction": npred, "n_reference": ngt, "n_matched": k,
            "matching": {"max_center_distance_m": config.match_max_distance_m,
                         "category_gate": config.match_category, "unknown_category_policy": "allow",
                         "cost": "distance/max_distance + 0.25*(1-AABB_IoU); maximize allowed matches first"},
            "shape_alignment": "bbox-center translation only; preserve scale and orientation",
            "detection": detection, "macro_fscore_missing_as_zero": macro,
            "missing_reference_ids": [g.id for g in reference if g.id not in matched_g],
            "extra_prediction_ids": [p.id for p in prediction if p.id not in matched_p],
            "matches": matches}
