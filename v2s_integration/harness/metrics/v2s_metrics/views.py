"""Per-view evaluation with explicit failure coverage and split isolation."""
import math
from collections import Counter
import numpy as np
from .config import MetricConfig
from .image import compare_images
from .result import error_record, metric


def aggregate_views(records, names, worst_fraction=0.2):
    if not 0 < worst_fraction <= 1:
        raise ValueError("worst_fraction must be in (0,1]")
    result = {}
    for name in names:
        values = []
        failures = Counter()
        direction = "higher" if name in ("psnr_db", "ssim") else "lower"
        for row in records:
            m = row.get("metrics", {}).get(name, {})
            if m.get("status") == "ok" and m.get("value") is not None:
                values.append(m["value"])
            else:
                failures[m.get("status", row.get("status", "missing"))] += 1
        count = len(values)
        summary = {"status": "ok" if count == len(records) and count else "partial" if count else "unavailable",
                   "n_expected": len(records), "n_valid": count,
                   "coverage": count / len(records) if records else None,
                   "failure_counts": dict(failures), "direction": direction,
                   "mean": None, "median": None, "std": None, "worst": None,
                   "worst_fraction_mean": None, "worst_fraction": worst_fraction,
                   "aggregation": "valid_views_only; read coverage before comparing"}
        if count:
            x = np.asarray(values, dtype=float)
            ordered = np.sort(x)
            if direction == "lower":
                ordered = ordered[::-1]
            k = max(1, math.ceil(worst_fraction * count))
            summary.update(mean=float(x.mean()), median=float(np.median(x)), std=float(x.std()),
                           worst=float(ordered[0]), worst_fraction_mean=float(ordered[:k].mean()), n_worst=k)
        result[name] = summary
    return result


def compare_views(views, config=None):
    config = config or MetricConfig()
    ids = [v["id"] for v in views]
    if len(ids) != len(set(ids)):
        raise ValueError("view ids must be unique")
    records = []
    for view in views:
        split = view.get("split", "observed")
        if split not in ("observed", "feedback", "hidden"):
            raise ValueError("split must be observed, feedback or hidden")
        row = {"id": view["id"], "split": split, "camera_id": view.get("camera_id")}
        try:
            row["metrics"] = compare_images(view["prediction"], view["reference"], config)
            row["status"] = "ok" if all(m["status"] == "ok" for m in row["metrics"].values()) else "partial"
        except (OSError, ValueError, ImportError) as exc:
            row.update(error_record(exc))
            row["metrics"] = {name: metric(status=row["status"], reason=row["reason"],
                                  direction="higher" if name in ("psnr_db", "ssim") else "lower")
                              for name in config.image_metrics}
        records.append(row)
    return {"per_view": records,
            "splits": {split: aggregate_views([r for r in records if r["split"] == split],
                                                config.image_metrics, config.worst_fraction)
                       for split in ("observed", "feedback", "hidden")
                       if any(r["split"] == split for r in records)}}
