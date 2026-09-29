"""Unsquared Euclidean surface distances in meters; no implicit alignment."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .config import MetricConfig
from .result import metric


def points(value, allow_empty=True):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("points must be an N x 3 array (use shape (0,3) for empty)")
    if not np.isfinite(a).all():
        raise ValueError("points contain NaN or infinity")
    if not allow_empty and not len(a):
        raise ValueError("reference/instance point cloud must not be empty")
    return a


def sample_surface(vertices, faces, count=20000, seed=0):
    """Area-weighted triangle sampling; independent of mesh vertex density."""
    v = points(vertices, allow_empty=False)
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("faces must be integer T x 3 triangles")
    if not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError("invalid or empty triangle indices")
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("sample count must be positive and seed nonnegative integers")
    triangles = v[f]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    valid = np.isfinite(area) & (area > 0)
    if not valid.any():
        raise ValueError("mesh has no nondegenerate surface")
    triangles, area = triangles[valid], area[valid]
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), size=count, p=area / area.sum())]
    u = np.sqrt(rng.random(count))
    w = rng.random(count)
    return ((1 - u[:, None]) * selected[:, 0] + (u * (1 - w))[:, None] * selected[:, 1]
            + (u * w)[:, None] * selected[:, 2])


def load_points(path, config=None):
    """NPY contains pre-sampled world points; mesh node transforms are applied."""
    config = config or MetricConfig()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    if path.suffix.lower() == ".npy":
        return points(np.load(path, allow_pickle=False))
    if path.suffix.lower() not in (".glb", ".gltf", ".obj", ".ply", ".stl"):
        raise ValueError("geometry must be .npy points or a supported triangle mesh")
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("mesh loading requires trimesh; .npy inputs do not") from exc
    scene = trimesh.load(str(path), force="scene", process=False)
    vertices, faces, offset = [], [], 0
    for node in sorted(scene.graph.nodes_geometry):
        transform, key = scene.graph[node]
        mesh = scene.geometry[key]
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("mesh artifact includes non-triangle geometry; supply sampled .npy for point clouds")
        v = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
        vertices.append(v)
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(v)
    if not vertices:
        return np.empty((0, 3))
    return sample_surface(np.vstack(vertices), np.vstack(faces), config.surface_samples, config.seed)


def threshold_key(tau):
    return f"{float(tau)}m"


def compare_points(prediction, reference, config=None):
    config = config or MetricConfig()
    p, q = points(prediction), points(reference, allow_empty=False)
    out = {"alignment": "none", "units": "m", "n_prediction": len(p), "n_reference": len(q),
           "cd_definition": "0.5 * (mean_pred_to_ref + mean_ref_to_pred); unsquared Euclidean",
           "status": "ok" if len(p) else "prediction_empty", "thresholds": {}}
    if not len(p):
        for name in ("cd_mean_m", "accuracy_m", "completeness_m"):
            out[name] = metric(unit="m", status="prediction_empty", reason="no predicted surface")
        for tau in config.distance_thresholds_m:
            out["thresholds"][threshold_key(tau)] = {
                name: metric(0, direction="higher", status="prediction_empty", reason="empty prediction penalized as zero")
                for name in ("precision", "recall", "fscore")}
        return out
    d_p = cKDTree(q).query(p)[0]
    d_q = cKDTree(p).query(q)[0]
    out.update(cd_mean_m=metric((d_p.mean() + d_q.mean()) / 2, unit="m"),
               accuracy_m=metric(d_p.mean(), unit="m"), completeness_m=metric(d_q.mean(), unit="m"))
    for tau in config.distance_thresholds_m:
        precision, recall = float(np.mean(d_p < tau)), float(np.mean(d_q < tau))
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out["thresholds"][threshold_key(tau)] = {name: metric(value, direction="higher")
                    for name, value in (("precision", precision), ("recall", recall), ("fscore", f))}
    return out
