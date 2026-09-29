"""Manifest-driven evaluation. Reads frozen files; never touches Blender state."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from .config import MetricConfig
from .geometry import compare_points, load_points
from .instances import Instance, compare_instances
from .result import VERSION, error_record
from .views import compare_views


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant, object_pairs_hook=unique_pairs)


def _keys(obj, allowed, required=()):
    if not isinstance(obj, dict) or set(obj) - set(allowed) or set(required) - set(obj):
        raise ValueError(f"expected required keys {sorted(required)} and allowed keys {sorted(allowed)}")


def _identity(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _versions():
    result = {}
    for name in ("numpy", "scipy", "Pillow", "scikit-image", "trimesh", "torch", "torchvision", "lpips"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def evaluate_manifest(path, config_override=None):
    path = Path(path).resolve()
    manifest = read_json(path)
    _keys(manifest, ("schema_version", "protocol", "config", "views", "geometry", "instances"), ("schema_version", "protocol"))
    if manifest["schema_version"] != 1:
        raise ValueError("only manifest schema_version=1 is supported")
    protocol = manifest["protocol"]
    _keys(protocol, ("units", "alignment", "render_config_id"), ("units", "alignment"))
    if protocol["units"] != "m" or protocol["alignment"] != "none":
        raise ValueError("v0.1 requires meter world coordinates and alignment=none; prepare transforms explicitly")
    config_data = dict(manifest.get("config", {}))
    if config_override:
        config_data.update(config_override)
    config = MetricConfig(**config_data)
    sources = {str(path): {"sha256": _identity(path)}}

    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError("artifact paths must be nonempty strings")
        resolved = (path.parent / value).resolve()
        if str(resolved) not in sources:
            try:
                sources[str(resolved)] = {"sha256": _identity(resolved)}
            except OSError:
                sources[str(resolved)] = {"sha256": None, "status": "unreadable"}
        return resolved

    def load_instances(rows):
        if not isinstance(rows, list):
            raise ValueError("instances must be a list")
        result = []
        for row in rows:
            _keys(row, ("id", "path", "category", "rotation", "symmetries"), ("id", "path"))
            result.append(Instance(row["id"], load_points(resolve(row["path"]), config),
                                   row.get("category"), row.get("rotation"), tuple(row.get("symmetries", ()))))
        return result

    report = {"schema_version": 1, "metrics_version": VERSION, "config": config.to_dict(),
              "protocol": protocol, "dependencies": _versions(), "inputs": sources}
    active = 0
    if "views" in manifest:
        if not isinstance(manifest["views"], list) or not manifest["views"]:
            raise ValueError("views must be a nonempty list when requested")
        if not isinstance(protocol.get("render_config_id"), str) or not protocol["render_config_id"]:
            raise ValueError("image evaluation requires a shared render_config_id declaration")
        views = []
        for view in manifest["views"]:
            _keys(view, ("id", "split", "camera_id", "prediction", "reference"),
                  ("id", "split", "camera_id", "prediction", "reference"))
            for key in ("id", "camera_id"):
                if not isinstance(view[key], str) or not view[key]:
                    raise ValueError("view id and camera_id must be nonempty strings")
            views.append({**view, "prediction": resolve(view["prediction"]), "reference": resolve(view["reference"])})
        report["views"] = compare_views(views, config)
        active += 1
    if "geometry" in manifest:
        pair = manifest["geometry"]
        _keys(pair, ("prediction", "reference"), ("prediction", "reference"))
        try:
            report["geometry"] = compare_points(load_points(resolve(pair["prediction"]), config),
                                                load_points(resolve(pair["reference"]), config), config)
        except (OSError, ValueError, ImportError) as exc:
            report["geometry"] = error_record(exc)
        active += 1
    if "instances" in manifest:
        pair = manifest["instances"]
        _keys(pair, ("prediction", "reference"), ("prediction", "reference"))
        try:
            report["instances"] = compare_instances(load_instances(pair["prediction"]), load_instances(pair["reference"]), config)
        except (OSError, ValueError, ImportError) as exc:
            report["instances"] = error_record(exc)
        active += 1
    if not active:
        raise ValueError("manifest must request at least one evaluation block")
    statuses = [r["status"] for r in report.get("views", {}).get("per_view", [])]
    statuses.extend(report[key]["status"] for key in ("geometry", "instances") if key in report)
    report["evaluation_status"] = "complete" if all(s in ("ok", "prediction_empty") for s in statuses) else "incomplete"
    return report
