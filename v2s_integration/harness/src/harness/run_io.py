"""Shared utilities: write a single run.py result into runs/<model>/<task_type>/<task_id>/
+ recompute runs/<model>/<task_type>/_summary.json

Unified output layout of each sample directory (aligned across the 6 tasks):
  score.json              score + metadata + headline (paper table header uses headline.primary)
  steps.json              agent step trajectory
  task.json               original task config (for postprocess / reproduction)
  agent_scene.glb         T1/T2/T5 exported scene the agent placed/built
  agent_camera.json       T3 exported camera reported by the agent
  agent_frames/*.glb      T4 per-frame glb
  views/                  T2/T5 (agent, gt) comparison images rendered for the visual metric
  postprocess/            T4 rendered video / compare / strip
"""
import json, os, statistics, re

# repo root = ../../.. relative to this file
# (src/harness/run_io.py -> dirname=src/harness -> dirname=src -> dirname=repo)
_HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
RUNS = os.path.join(ROOT, "runs")

TASK_TYPES = ("task1_single", "task2_multi", "task3_camera",
              "task4_anim", "task5_recon", "task5_retrieval", "task5_layout_derived",
              "task6_anim", "task7_anim")


# ---- headline fields: the "primary score + key diagnostics" of each task. Paper table header reads this layer directly. ----
# primary    single value, used for ranking (arrow encoded in the name: down = smaller is better, up = larger is better)
# secondary  up to 3-4 key diagnostic metrics
HEADLINE_SPEC = {
    "task1_single":  {"primary_key": "mean_add_s",       "primary_dir": "down",
                       "secondary_keys": ["chamfer", "placement_acc",
                                            "mean_pe", "scene_success"]},
    "task2_multi":   {"primary_key": "mean_add_s",       "primary_dir": "down",
                       "secondary_keys": ["chamfer", "placement_acc",
                                            "mean_pe", "scene_success"]},
    "task3_camera":  {"primary_key": "cam_pos_error",    "primary_dir": "down",
                       "secondary_keys": ["cam_angle_error_deg"]},
    "task4_anim":    {"primary_key": "worst_part_err",   "primary_dir": "down",
                       "secondary_keys": ["mean_part_err", "type_mismatch_rate",
                                            "reverse_dir_rate", "movable_recall"]},
    "task5_recon":   {"primary_key": "f@5%", "primary_dir": "up",
                       "secondary_keys": ["f@5%_noalign", "f@5%_sim3",
                                            "obj_f@5%", "obj_f@5%_matched",
                                            "mean_pos_err", "match_rate",
                                            "obj_pointbert", "scene_pointbert"]},
    # T6: trajectory.worst_vehicle_err is nested, kept at the outer level so _mean_at can read it
    "task6_anim":    {"primary_key": "worst_vehicle_err", "primary_dir": "down",
                       "secondary_keys": ["mean_vehicle_err", "movable_recall",
                                            "mover_count_err", "direction_error_rate",
                                            "path_shape_err", "heading_err", "scale_error",
                                            "size_error", "mover_size_err", "layout_err"]},
    # T7: same task/scoring as T6, only the reference frames are photorealistic renders (visual-domain ablation). headline reuses T6.
    "task7_anim":    {"primary_key": "worst_vehicle_err", "primary_dir": "down",
                       "secondary_keys": ["mean_vehicle_err", "movable_recall",
                                            "mover_count_err", "direction_error_rate",
                                            "path_shape_err", "heading_err", "scale_error",
                                            "size_error", "mover_size_err", "layout_err"]},
}

# Derived canonical-layout smoke tasks use the same scorer and diagnostics but
# must remain a separate task type so they cannot enter native tables.
HEADLINE_SPEC["task5_layout_derived"] = dict(HEADLINE_SPEC["task5_recon"])
# Retrieval-enabled reconstruction uses the same scorer, but stays separate
# from native and derived runs for three-way protocol comparisons.
HEADLINE_SPEC["task5_retrieval"] = dict(HEADLINE_SPEC["task5_recon"])


# ---- Overall composite score: each task normalized to [0,100] with fixed reference upper bounds, independent of the evaluated model set ----
# The convention must exactly match the upper bounds in figures/gen_main_table.py on the paper side.
# !!! changing the constants here requires the same change in gen_main_table.py, otherwise harness score != paper score !!!
# Each sample first: score = clip(1 - e/ub, 0, 1)*100 (dir='down');
# "up" items compute clip(e/ub,0,1)*100 first. Invalid samples count as 0, then averaged across samples.
# When a task has multiple sub-items, average within the sample first. T2/T7 are ablations, excluded from Overall.
TASK_NORM = {
    "task1_single": [("mean_add_s", 4.0, "down")],            # Layout: room diagonal ~4m
    "task3_camera": [("cam_pos_error", 4.0, "down"),          # Camera: mean of pos + angle
                     ("cam_angle_error_deg", 90.0, "down")],
    "task4_anim":   [("worst_part_err", 1.0, "down")],        # Articulated (already normalized)
    "task5_recon":  [("f@5%", 1.0, "up")],                    # Reconstruction (F-score)
    "task6_anim":   [("worst_vehicle_err", 1.0, "down"),      # Dynamic: trajectory + static layout
                     ("layout_err", 1.0, "down")],
}
OVERALL_NORM_SPEC = "fixed-ref v2 clip-then-mean (Layout 4m; Camera 4m/90deg; Artic/Recon/Dyn 1.0)"
TASK_EXPECTED_N = {
    "task1_single": 100,
    "task3_camera": 100,
    "task4_anim": 100,
    "task5_recon": 100,
    "task6_anim": 10,
}

# Keep derived runs out of the cross-task benchmark composite. They have their
# own directory and headline but are not upstream-native measurements.


def build_headline(task_type: str, score: dict) -> dict:
    """Extract headline from the score dict: primary single value + a few secondary key diagnostics."""
    spec = HEADLINE_SPEC.get(task_type, {})
    pk = spec.get("primary_key")
    out = {
        "primary_key": pk,
        "primary_dir": spec.get("primary_dir"),
        "primary": score.get(pk) if pk else None,
    }
    sec = {}
    for k in spec.get("secondary_keys", []):
        if k in score:
            sec[k] = score[k]
    out["secondary"] = sec
    return out


def infer_task_type(task: dict) -> str:
    """Prefer the task_type in the task json, otherwise guess from the id name."""
    t = task.get("task_type")
    if t in TASK_TYPES:
        return t
    tid = task.get("id", "")
    for tt in TASK_TYPES:
        if tt in tid:
            return tt
    return "unknown"


def slugify_model(name: str) -> str:
    """Turn a model name into a directory name: replace unsafe characters."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def sample_dir(model: str, task_type: str, task_id: str) -> str:
    d = os.path.join(RUNS, slugify_model(model), task_type, task_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_run_outputs(model: str, task: dict, run_record: dict):
    """Write a single run.py result to disk (unified schema: meta + headline + score).

    Top-level structure of score.json:
      model, task, task_type, finished, error, elapsed_sec, num_steps,
      headline: {primary_key, primary_dir, primary, secondary},
      score:    each task's own full metric dict
    """
    tt = infer_task_type(task)
    if tt == "unknown":
        return None
    d = sample_dir(model, tt, task["id"])
    sc = dict(run_record.get("score") or {})
    protocol = task.get("protocol")
    protocol_meta = None
    if protocol:
        refs = task.get("references") or []
        protocol_meta = {
            "name": protocol,
            "n_views": len(refs) if isinstance(refs, list) else 0,
            "camera_provided": bool(task.get("camera")),
            "alignment": sc.get("alignment"),
        }
    meta = {
        "model": model,
        "task": task["id"],
        "task_type": tt,
        "finished": run_record.get("finished"),
        "error": run_record.get("error"),
        "elapsed_sec": run_record.get("elapsed_sec"),
        "num_steps": run_record.get("num_steps"),
        "protocol": protocol_meta,
        "headline": build_headline(tt, sc),
        "score": sc,
    }
    with open(os.path.join(d, "score.json"), "w") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(d, "steps.json"), "w") as handle:
        json.dump(run_record.get("steps") or [], handle, ensure_ascii=False)
    # Store a verbatim copy of the task config (for postprocess / reproduction)
    with open(os.path.join(d, "task.json"), "w") as handle:
        json.dump(task, handle, indent=2, ensure_ascii=False)
    return d


def _mean_at(rows, key):
    import math
    vs = []
    for r in rows:
        sc = r.get("score") or {}
        v = sc.get(key)
        if isinstance(v, (int, float)) and not math.isnan(v):
            vs.append(v)
    return round(statistics.mean(vs), 4) if vs else None


def rebuild_summary(model: str, task_type: str) -> dict | None:
    """Scan runs/<model>/<task_type>/*/score.json and recompute _summary.json.
    Aggregation uses the primary + secondary fields from HEADLINE_SPEC, a unified interface across the 5 tasks."""
    base = os.path.join(RUNS, slugify_model(model), task_type)
    if not os.path.isdir(base):
        return None
    rows = []
    for sid in sorted(os.listdir(base)):
        if sid.startswith("_"):
            continue
        sp = os.path.join(base, sid, "score.json")
        if os.path.exists(sp):
            try:
                rows.append(json.load(open(sp)))
            except Exception:
                pass
    if not rows:
        return None
    n = len(rows)
    finished = sum(1 for r in rows if r.get("finished"))
    summ = {"model": model, "task_type": task_type, "n": n,
            "finished_rate": round(finished / n, 4)}
    spec = HEADLINE_SPEC.get(task_type, {})
    pk = spec.get("primary_key")
    if pk:
        summ[f"mean_{pk}"] = _mean_at(rows, pk)
        summ["primary_dir"] = spec.get("primary_dir")
    for k in spec.get("secondary_keys", []):
        # Boolean fields (scene_success) become a success rate; numeric fields become a mean
        bool_count = sum(1 for r in rows if (r.get("score") or {}).get(k) is True)
        if bool_count > 0 and isinstance((rows[0].get("score") or {}).get(k), bool):
            summ[f"{k}_rate"] = round(bool_count / n, 4)
        else:
            summ[f"mean_{k}"] = _mean_at(rows, k)
    json.dump(summ, open(os.path.join(base, "_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    return summ


def _clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _task_score(model: str, task_type: str):
    """Clip-normalize per sample, count invalid samples as 0, then take the task mean."""
    import math

    spec = TASK_NORM.get(task_type)
    if not spec:
        return None
    base = os.path.join(RUNS, slugify_model(model), task_type)
    if not os.path.isdir(base):
        return None
    rows = []
    for sid in sorted(os.listdir(base)):
        if sid.startswith("_"):
            continue
        spath = os.path.join(base, sid, "score.json")
        if not os.path.exists(spath):
            continue
        try:
            rows.append(json.load(open(spath)))
        except Exception:
            pass

    expected = TASK_EXPECTED_N.get(task_type)
    if expected is not None and len(rows) < expected:
        return None
    if expected is not None:
        rows = rows[:expected]

    sample_scores = []
    for row in rows:
        score = row.get("score") or {}
        parts = []
        valid = True
        for key, ub, direction in spec:
            value = score.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                valid = False
                break
            part = _clip01(value / ub) if direction == "up" else _clip01(1 - value / ub)
            parts.append(part)
        sample_scores.append(statistics.mean(parts) if valid else 0.0)
    return round(100 * statistics.mean(sample_scores), 2) if sample_scores else None


def rebuild_overall(model: str) -> dict | None:
    """Compute normalized scores for the 5 scored tasks -> runs/<model>/_overall.json.
    overall is given only when all 5 tasks are present; otherwise overall=None, complete=False, listing only computed items."""
    per = {tt: _task_score(model, tt) for tt in TASK_NORM}
    have = {k: v for k, v in per.items() if v is not None}
    complete = len(have) == len(TASK_NORM)
    out = {
        "model": model,
        "task_scores": {k: have[k] for k in TASK_NORM if k in have},
        "overall": round(statistics.mean(have.values()), 2) if complete else None,
        "n_tasks": len(have),
        "complete": complete,
        "norm_spec": OVERALL_NORM_SPEC,
    }
    mdir = os.path.join(RUNS, slugify_model(model))
    os.makedirs(mdir, exist_ok=True)
    json.dump(out, open(os.path.join(mdir, "_overall.json"), "w"),
              indent=2, ensure_ascii=False)
    return out
