"""Benchmark runner: load models + tasks, run each (model x task), dump logs.

Usage (from repo root):
  export ANTHROPIC_API_KEY=...   ZHIPU_API_KEY=...   DEEPSEEK_API_KEY=...
  python src/harness/run.py --models models.json --task tasks/stack-two-boxes.json
  python src/harness/run.py --models models.json --all   # every task in tasks/

Scoring is intentionally left as a stub — the goal here is to get the agent
running end-to-end against Blender. Hook your transform-vs-ground-truth scorer
into score_run() later.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# allow `import adapters` etc. regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

# Load non-secret local settings for backward compatibility. API credentials are
# loaded separately from secrets/api.env after CLI parsing.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

from adapters import build_adapter           # noqa: E402
from credential_store import load_credentials  # noqa: E402
from agent import run_agent, RunResult      # noqa: E402
from run_io import write_run_outputs, rebuild_summary, rebuild_overall, infer_task_type  # noqa: E402
from environment.session import RunSession
from environment.task_spec import TaskSpec

# repo root = two levels up from src/harness/run.py
ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "tasks"
RUNS = ROOT / "runs"
LOGS = RUNS / "_logs"
BLENDER_MCP_DIR = ROOT / "blender-mcp"

DEFAULT_SYSTEM = "Reconstruct the task using the available scene tools. Verify the result against the references."


def resolve_env(cfg: dict) -> dict:
    """Replace $ENV_VAR (optionally followed by a suffix) with the env value.
    Applies to api_key and base_url.
      '$WOA_APP_CRED'                         -> <cred>
      '$WOA_APP_CRED?provider=yuewen&model=x' -> <cred>?provider=yuewen&model=x
    Pass-through gateways (e.g. StepFun/yuewen) require a query string after the
    key in Authorization, so suffix concatenation is supported."""
    import re as _re
    for field in ("api_key", "base_url", "user_key", "biz_scene"):
        v = cfg.get(field)
        if isinstance(v, str) and v.startswith("$"):
            m = _re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)(.*)$", v, _re.DOTALL)
            if m:
                cfg[field] = os.environ.get(m.group(1), "") + m.group(2)
    if not cfg.get("api_key"):
        raise RuntimeError(f"missing api_key for model {cfg.get('name')}")
    return cfg


def reset_scene(mcp, task: dict):
    """Reset Blender to a clean state and import the task's GLB(s).

    Expects task['setup_code'] to contain bpy code that clears the scene and
    imports the relevant GLB files (paths are task-defined). Kept as plain
    execute_blender_code so tasks stay self-describing.
    """
    setup = task.get("setup_code")
    if setup:
        out = mcp.call_tool("execute_blender_code", {"code": setup})
        print(f"  [setup] {out[:200]}")


def load_image_b64(path: str, bg=(255, 255, 255)) -> dict | None:
    """Read an image as a base64 dict {data, mime} for injection into the dialog.

    Fairness note: reference images are RGBA transparent webp (Blender exports
    with a black background + alpha matte), so the underlying RGB in transparent
    regions is black. Sending them as-is to each vendor's vision API means the
    server flattens RGBA->RGB inconsistently:
      - drops alpha (gpt/gemini/minimax) -> transparent regions collapse to black,
        dark furniture blends into the black background
      - composites onto white (claude/qwen/doubao) -> transparent regions turn
        white, furniture stays clear
    The same image looks different to the 6 models, a systematic unfairness.
    Here on the harness side we composite alpha onto a fixed white background,
    convert to RGB (drop alpha), and uniformly encode as PNG, ensuring all models
    receive identical 3-channel pixels and eliminating server-side flatten ambiguity.
    """
    import base64, io
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image
    except ImportError:
        # Without PIL, fall back to sending as-is (may trigger the unfairness
        # above, but at least does not crash)
        ext = os.path.splitext(path)[1].lower()
        mime = {".webp": "image/webp", ".png": "image/png",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
        with open(path, "rb") as f:
            return {"data": base64.b64encode(f.read()).decode(), "mime": mime}
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, tuple(bg) + (255,))
        im = Image.alpha_composite(canvas, rgba).convert("RGB")
    else:
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"data": base64.b64encode(buf.getvalue()).decode(), "mime": "image/png"}


def load_video_b64(path: str) -> dict | None:
    """Read a video as a base64 dict {data, mime, path} for video-capable adapter injection."""
    import base64
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {".mp4": "video/mp4", ".mov": "video/quicktime",
            ".webm": "video/webm"}.get(ext, "video/mp4")
    with open(path, "rb") as f:
        return {"data": base64.b64encode(f.read()).decode(),
                "mime": mime, "path": path}


def task_reference_media(task: dict, adapter) -> tuple[list, list]:
    """Pick the modality based on adapter capability:
      - adapter.supports_video=True and task has reference_video -> use video
      - otherwise use frame sequences (reference_image / references[] / reference_frames[])
    Returns (images, videos); either may be empty.
    """
    if not getattr(adapter, "supports_vision", False):
        return [], []
    # Video path (only taken when adapter is video-capable)
    if getattr(adapter, "supports_video", False):
        vp = task.get("reference_video")
        if vp and os.path.isfile(vp):
            v = load_video_b64(vp)
            if v:
                return [], [v]
    # Frame path (default): compatible with the old fields reference_image +
    # references[], plus the new field reference_frames[] (task4 multi-frame case)
    paths = []
    if task.get("reference_image"):
        paths.append(task["reference_image"])
    for r in task.get("references", []) or []:
        if isinstance(r, dict) and r.get("image"):
            paths.append(r["image"])
    for p in task.get("reference_frames", []) or []:
        paths.append(p)
    paths = _dedupe_media_paths(paths)
    imgs = [load_image_b64(p) for p in paths]
    return [im for im in imgs if im], []


def task_reference_images(task: dict) -> list:
    """Backward compatibility (old callers); returns only the image list. Prefer task_reference_media."""
    paths = []
    if task.get("reference_image"):
        paths.append(task["reference_image"])
    for r in task.get("references", []) or []:
        if isinstance(r, dict) and r.get("image"):
            paths.append(r["image"])
    for p in task.get("reference_frames", []) or []:
        paths.append(p)
    paths = _dedupe_media_paths(paths)
    imgs = [load_image_b64(p) for p in paths]
    return [im for im in imgs if im]


def _dedupe_media_paths(paths: list) -> list:
    """Deduplicate reference paths while preserving their task order."""
    out, seen = [], set()
    for value in paths:
        if not isinstance(value, str) or not value:
            continue
        key = os.path.realpath(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def score_run(mcp, task: dict, result, model_name: str | None = None,
              checkpoint: bool = False) -> dict:
    """Score by task_type: layout tasks read object poses and compute ADD-S etc.;
    camera tasks compute pose error.

    Unified directory convention (each sample lands in runs/<model>/<task_type>/<task_id>/):
      - T1/T2/T5: agent_scene.glb (export all mesh the agent placed/built)
      - T2/T5:    views/ (visual metric comparison images; only T5 uses it now,
                  T2 reserved for extension)
      - T3:       agent_camera.json (export the camera the agent reported)
      - T4:       agent_frames/ (already set by run_one before running the agent)

    When checkpoint=True, scoring must be read-only observation:
      - skip visual rendering that would mutate the live Blender scene;
      - T4/T6/T7 are given checkpoint-specific artifact paths by the caller;
      - do not write checkpoint products back to the sample top level.
    """
    scene_dir = task.get("scene_dir")
    sample_dir = task.get("sample_dir")
    import metrics as _m
    ttype = task.get("task_type", "") or infer_task_type(task)
    # sample top-level directory: runs/<model>/<task_type>/<task_id>/
    sample_top = None
    if model_name:
        from run_io import sample_dir as _sd
        sample_top = _sd(model_name, ttype, task["id"])
    # Also allow env var EXPORT_SCENE_DIR to force an override (old demo compat)
    export_dir = os.environ.get("EXPORT_SCENE_DIR") or sample_top
    # T1/T2/T5: export agent_scene.glb to the sample top level (drop the task_id
    # prefix, unify the file name)
    if export_dir and ttype in ("task1_single", "task2_multi", "task5_recon", "task5_retrieval", "task5_layout_derived"):
        try:
            os.makedirs(export_dir, exist_ok=True)
            out_glb = os.path.join(export_dir, "agent_scene.glb")
            code = (
                "import bpy\n"
                "_prev_sel=[o for o in bpy.context.selected_objects]\n"
                "_prev_active=bpy.context.view_layer.objects.active\n"
                "try:\n"
                "    bpy.ops.object.select_all(action='DESELECT')\n"
                "    for o in bpy.data.objects: o.select_set(o.type=='MESH')\n"
                f"    bpy.ops.export_scene.gltf(filepath={out_glb!r}, use_selection=True, export_format='GLB')\n"
                "finally:\n"
                "    bpy.ops.object.select_all(action='DESELECT')\n"
                "    for o in _prev_sel:\n"
                "        if o.name in bpy.data.objects: o.select_set(True)\n"
                "    if _prev_active and _prev_active.name in bpy.data.objects:\n"
                "        bpy.context.view_layer.objects.active=_prev_active\n"
                "print('EXPORTED')\n"
            )
            mcp.call_tool("execute_blender_code", {"code": code})
            print(f"  [export] agent_scene.glb -> {out_glb}")
        except Exception as e:
            print(f"  [export] failed: {e!r}")
    # T6/T7: harness fallback that force-exports the animated agent_scene.glb.
    # Even if the agent hits max_steps without exporting, as long as it built the
    # scene + animation in Blender, the harness can export it for scoring (aligns
    # with the T1/T2/T5 force-export strategy, avoiding N/A).
    # Note: the check is "does not exist OR <2KB empty remnant". On a rerun, runs
    # may still hold a 132B empty glb from the previous round; checking only
    # `not isfile` would skip the export and leave the old empty file (this once
    # made t6/t7 unrecoverable across reruns: the agent clearly built the scene
    # but the fallback saw the old empty file and skipped).
    if ttype in ("task6_anim", "task7_anim"):
        agent_glb = task.get("agent_glb_path")
        _need_export = agent_glb and (
            not os.path.isfile(agent_glb) or os.path.getsize(agent_glb) < 2048)
        if _need_export:
            try:
                os.makedirs(os.path.dirname(agent_glb), exist_ok=True)
                code = (
                    "import bpy\n"
                    "_prev_sel=[o for o in bpy.context.selected_objects]\n"
                    "_prev_active=bpy.context.view_layer.objects.active\n"
                    "try:\n"
                    "    bpy.ops.object.select_all(action='DESELECT')\n"
                    "    for o in bpy.data.objects:\n"
                    "        o.select_set(o.type in ('MESH','ARMATURE','EMPTY'))\n"
                    f"    bpy.ops.export_scene.gltf(filepath={agent_glb!r}, use_selection=True, "
                    "export_format='GLB', export_animations=True, export_frame_range=True)\n"
                    "finally:\n"
                    "    bpy.ops.object.select_all(action='DESELECT')\n"
                    "    for o in _prev_sel:\n"
                    "        if o.name in bpy.data.objects: o.select_set(True)\n"
                    "    if _prev_active and _prev_active.name in bpy.data.objects:\n"
                    "        bpy.context.view_layer.objects.active=_prev_active\n"
                    "print('EXPORTED_T6')\n"
                )
                mcp.call_tool("execute_blender_code", {"code": code})
                if os.path.isfile(agent_glb):
                    print(f"  [export] T6/T7 fallback export agent_scene.glb -> {agent_glb}")
            except Exception as e:
                print(f"  [export] T6/T7 fallback export failed: {e!r}")
    # T4: harness fallback frame export. When the agent hits max_steps without
    # exporting all 32 frames (or even a single frame), the harness exports the
    # "current state" from the live Blender session to fill missing frame numbers,
    # avoiding no-agent-frames (N/A).
    # Aligns with the T1/T2/T5/T6/T7 force-export strategy: reflects the scene the
    # agent built; a static fallback scores low on animation but is still valid.
    if ttype == "task4_anim":
        frames_dir = task.get("agent_frames_dir")
        n_frames = task.get("n_frames", 32)
        if frames_dir:
            try:
                os.makedirs(frames_dir, exist_ok=True)
                have = {f for f in os.listdir(frames_dir)
                        if f.startswith("agent_frame_") and f.endswith(".glb")}
                missing = [i for i in range(n_frames)
                           if f"agent_frame_{i:04d}.glb" not in have]
                if len(have) < n_frames:
                    # Export each missing frame number as the current session state (static fallback)
                    for i in missing:
                        fp = os.path.join(frames_dir, f"agent_frame_{i:04d}.glb")
                        code = (
                            "import bpy\n"
                            f"bpy.ops.export_scene.gltf(filepath={fp!r}, "
                            "use_selection=False, export_format='GLB')\n"
                            "print('EXPORTED_T4_FRAME')\n"
                        )
                        mcp.call_tool("execute_blender_code", {"code": code})
                    got = len([f for f in os.listdir(frames_dir)
                               if f.startswith("agent_frame_") and f.endswith(".glb")])
                    print(f"  [export] T4 fallback frame export: had {len(have)} -> filled to {got}/{n_frames} "
                          f"(static fallback {len(missing)} frames)")
            except Exception as e:
                print(f"  [export] T4 fallback frame export failed: {e!r}")
    try:
        if ttype == "task3_camera":
            # Task3 = camera pose error (primary) + visual comparison (agent camera
            # render vs GT camera render, same pipeline)
            # checkpoint only reads the camera pose. Visual rendering temporarily
            # uses the GT camera and must not run in a live Blender session where
            # the agent will keep running.
            render_dir = None if checkpoint else export_dir
            score = _m.score_camera_from_blender(mcp, task, render_dir=render_dir)
            # Save a copy of the camera the agent reported as agent_camera.json
            camera_out_dir = export_dir or sample_top
            if camera_out_dir:
                try:
                    os.makedirs(camera_out_dir, exist_ok=True)
                    cam_data = {k: score.get(k) for k in
                                ("agent_position", "agent_look_dir",
                                 "agent_matrix_world", "agent_fov_x_deg")
                                if k in score}
                    if cam_data:
                        camera_path = os.path.join(camera_out_dir, "agent_camera.json")
                        with open(camera_path, "w") as handle:
                            json.dump(
                                cam_data, handle, indent=2, ensure_ascii=False
                            )
                except Exception as e:
                    print(f"  [export] agent_camera.json failed: {e!r}")
        elif ttype in ("task5_recon", "task5_retrieval", "task5_layout_derived"):
            # Task5 = reconstruct from scratch: read all mesh vertices the agent
            # built (merged) and compare against the GT furniture point cloud
            import numpy as _np
            verts = _m.read_predicted_verts(mcp, sample=4000)
            allv = [v for v in verts.values() if len(v)]
            if not allv:
                score = {"error": "no reconstructed mesh"}
            else:
                P = _np.vstack(allv)
                score = _m.score_reconstruction_scene(scene_dir, P, alignment="sim3_icp")
                # Keep the historical aligned score under f@5% while exposing
                # an explicit no-align diagnostic for protocol comparisons.
                score["f@5%_sim3"] = score.get("f@5%")
                try:
                    noalign = _m.score_reconstruction_scene(scene_dir, P, alignment="none")
                    score["f@5%_noalign"] = noalign.get("f@5%")
                    score["chamfer_noalign"] = noalign.get("chamfer")
                except Exception as e:
                    score["noalign_error"] = repr(e)[:200]
                # recon_visual_compare clears the live scene and imports the GT.
                # Run only in the final scoring after the agent has finished;
                # checkpoint computes geometry metrics only.
                if export_dir and not checkpoint:
                    try:
                        score["visual"] = _m.recon_visual_compare(
                            mcp, scene_dir, os.path.join(export_dir, "views"))
                    except Exception as e:
                        score["visual_error"] = repr(e)
        elif ttype == "task4_anim":
            # Task4 = articulated animation reproduction: the agent has exported
            # per-frame glb to agent_frames_dir. The scorer aligns using a private
            # partmap (invisible to the agent), pure-geometry keyframe with three metrics.
            import metrics_anim as _ma
            frames_dir = task.get("agent_frames_dir")
            score = _ma.score_animation_from_dir(sample_dir, frames_dir)
            # After scoring: pack the 32 frame glb into a single animated glb
            # (~30x compression) and delete the original frames
            try:
                if frames_dir and os.path.isdir(frames_dir):
                    sd_top = os.path.dirname(os.path.abspath(frames_dir))
                    import sys as _sys
                    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
                    from pack_t4_animation import pack as _pack
                    _pack(sd_top, delete_frames=True)
            except Exception as e:
                print(f"  [pack] T4 frame merge failed (original frames kept): {e!r}")
        elif ttype in ("task6_anim", "task7_anim"):
            # Task6 = dynamic scene understanding (low-poly reference frames);
            # Task7 = same task but photorealistic reference frames (visual-domain ablation).
            # Scoring is identical (same GT / same metric), only the input reference
            # frame style differs.
            import metrics_t6 as _mt6
            agent_glb = task.get("agent_glb_path")
            if not agent_glb or not os.path.isfile(agent_glb):
                score = {"error": f"agent_glb_path does not exist: {agent_glb}"}
            else:
                score = _mt6.evaluate_t6(agent_glb, sample_dir)
        else:
            if not scene_dir or not os.path.isdir(scene_dir):
                scene = mcp.call_tool("get_scene_info", {})
                return {"finished": result.finished, "error": result.error,
                        "note": "no scene_dir; snapshot only", "scene_snapshot": scene[:2000]}
            pred = _m.read_predicted_verts(mcp, sample=4000)
            score = _m.score_layout(scene_dir, pred)
        score["finished"] = result.finished
        score["error"] = result.error
        return score
    except Exception as e:
        return {"finished": result.finished, "error": result.error,
                "scoring_error": repr(e)}


def _prepare_checkpoint_task(task: dict, ttype: str, checkpoint_dir: str) -> dict:
    """Return a task copy whose mutable outputs are isolated to one checkpoint.

    The checkpoint scorer runs synchronously inside the live agent session.
    It must therefore never delete the agent's accumulated T4 frames or reuse
    a stale T6/T7 export from an earlier checkpoint.
    """
    checkpoint_task = dict(task)

    if ttype == "task4_anim":
        source = task.get("agent_frames_dir")
        snapshot = os.path.join(checkpoint_dir, "agent_frames")
        shutil.rmtree(snapshot, ignore_errors=True)
        if source and os.path.isdir(source):
            shutil.copytree(source, snapshot)
        else:
            os.makedirs(snapshot, exist_ok=True)
        checkpoint_task["agent_frames_dir"] = snapshot

    elif ttype in ("task6_anim", "task7_anim"):
        snapshot = os.path.join(checkpoint_dir, "agent_scene.glb")
        # A duplicate callback for the same step must still capture current state.
        try:
            os.remove(snapshot)
        except FileNotFoundError:
            pass
        checkpoint_task["agent_glb_path"] = snapshot

    return checkpoint_task


def build_t6_frame_tool(task: dict) -> dict:
    """Build the read_reference_frames synthetic tool for task6:
    the agent passes a list of frame numbers -> return the actual reference frame
    images directly (JPEG, scaled to 512px), without rendering PNGs onto a Blender
    plane and calling render_scene_view (which costs ~80s round-trip per frame)."""
    sample_dir = task.get("sample_dir", "")
    n_frames = task.get("n_frames", 144)
    # T6 uses low-poly frames reference/; T7 ablation uses photorealistic frames
    # reference_real/ (same GT, same camera)
    ref_dir = os.path.join(sample_dir, task.get("ref_subdir", "reference"))

    def handler(args: dict) -> dict:
        frames = args.get("frames") or args.get("frame_numbers") or []
        if isinstance(frames, (int, float)):
            frames = [int(frames)]
        # Cap at 16 frames per call (control body size)
        frames = [int(f) for f in frames][:16]
        if not frames:
            # Default to 8 evenly-spaced frames
            step = max(1, n_frames // 8)
            frames = list(range(1, n_frames + 1, step))[:8]
        imgs, got = [], []
        for f in frames:
            p = os.path.join(ref_dir, f"r_{f:04d}.png")
            if not os.path.isfile(p):
                continue
            im = _downscale_jpeg_b64(p, max_px=512)
            if im:
                imgs.append(im); got.append(f)
        txt = (f"Reference frames {got} (of {n_frames} total, {task.get('fps',24)} fps). "
               f"Frame 1 = start, frame {n_frames} = end.")
        return {"text": txt, "images": imgs}

    schema = {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array", "items": {"type": "integer"},
                "description": (f"Reference frame numbers to view (1..{n_frames}). "
                                "Max 16 per call. Omit to get 8 evenly-spaced frames."),
            }
        },
    }
    return {
        "read_reference_frames": {
            "description": (
                f"Return the actual reference video frames as images (1..{n_frames}). "
                "Use this to SEE the scene and vehicle motion directly — do NOT render "
                "PNGs onto planes. Pass a list of frame numbers (max 16/call)."),
            "schema": schema,
            "handler": handler,
        }
    }


def _downscale_jpeg_b64(path: str, max_px: int = 512, bg=(255, 255, 255)) -> dict | None:
    """Read PNG -> scale to max_px -> JPEG base64 dict {data, mime}.
    Images with alpha are composited onto a white background first, then converted
    to RGB (see the fairness note in load_image_b64) to avoid transparent regions
    collapsing to the black base color."""
    import base64, io
    try:
        from PIL import Image
    except ImportError:
        return load_image_b64(path)  # fallback: original image
    try:
        im = Image.open(path)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, tuple(bg) + (255,))
            im = Image.alpha_composite(canvas, rgba).convert("RGB")
        else:
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_px / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return {"data": base64.b64encode(buf.getvalue()).decode(), "mime": "image/jpeg"}
    except Exception:
        return None


TASK_MAX_STEPS = {
    "task4_anim": 60,
    "task5_recon": 35,
    "task5_retrieval": 35,
    "task5_layout_derived": 35,
    "task6_anim": 80,
    "task7_anim": 80,
}


def run_one(adapter_cfg: dict, task: dict, mcp, max_steps: int, restart_blender=None,
            checkpoint_every: int = 0) -> dict:
    print(f"\n=== model={adapter_cfg['name']}  task={task['id']} ===")
    # per-sample cache_task_id: WOA gateway task-level affinity. md5(model+task_id)
    # keeps it globally unique per task and reproducible (rerunning the same sample
    # yields the same id). All steps of one sample share this adapter/client -> same
    # cache_task_id -> per-step prompt cache hits (reference frames + system prompt +
    # history not reprocessed) for a big speedup and fewer timeouts; different samples
    # get different ids -> fanned out across healthy backend accounts to raise success.
    _ctid = hashlib.md5(f"{adapter_cfg['name']}_{task['id']}".encode()).hexdigest()
    adapter_cfg = {**adapter_cfg, "cache_task_id": _ctid}
    adapter = build_adapter(adapter_cfg)

    # task-type-aware max_steps: ensure it is not below the recommended value
    ttype = task.get("task_type") or infer_task_type(task)
    effective_steps = max(max_steps, TASK_MAX_STEPS.get(ttype, 30))
    task_spec = TaskSpec.from_task(task, effective_steps)
    effective_steps = task_spec.budget.max_agent_steps
    if task_spec.tool_profile == "adaptive" and checkpoint_every:
        raise ValueError("adaptive runs evaluate only after completion; checkpoints are legacy-only")
    if effective_steps != max_steps:
        print(f"  [max_steps] {max_steps} -> {effective_steps} (task_type={ttype})")

    # task4-specific: replace agent_frames_dir with this run's standard path so
    # the prompt/task config follows the model switch and lands automatically in
    # runs/<model>/task4_anim/<task_id>/agent_frames/, safe for multi-model /
    # multi-concurrency. Unified directory name (renamed old frames/ to agent_frames/).
    if ttype == "task4_anim":
        from run_io import sample_dir as _sd
        task_output_dir = _sd(adapter_cfg["name"], "task4_anim", task["id"])
        target = os.path.join(task_output_dir, "agent_frames")
        # A rerun must not inherit frames or a packed animation from an older run.
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        try:
            os.remove(os.path.join(task_output_dir, "agent_animation.glb"))
        except FileNotFoundError:
            pass
        task = dict(task)
        old = task.get("agent_frames_dir", "")
        task["agent_frames_dir"] = target
        # Replace the old path in the prompt too
        if old and old in (task.get("prompt") or ""):
            task["prompt"] = task["prompt"].replace(old, target)
        # Also replace those using a placeholder
        if "{AGENT_FRAMES_DIR}" in (task.get("prompt") or ""):
            task["prompt"] = task["prompt"].replace("{AGENT_FRAMES_DIR}", target)
    elif ttype in ("task6_anim", "task7_anim"):
        # task6/7: agent exports the whole scene glb to
        # runs/<model>/<ttype>/<task_id>/agent_scene.glb
        from run_io import sample_dir as _sd
        target = os.path.join(_sd(adapter_cfg["name"], ttype, task["id"]),
                              "agent_scene.glb")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # Prevent final scoring from accepting a valid-looking GLB left by a
        # previous run when the current agent never exports.
        try:
            os.remove(target)
        except FileNotFoundError:
            pass
        task = dict(task)
        old = task.get("agent_glb_path", "")
        task["agent_glb_path"] = target
        if old and old in (task.get("prompt") or ""):
            task["prompt"] = task["prompt"].replace(old, target)
        if "{AGENT_GLB_PATH}" in (task.get("prompt") or ""):
            task["prompt"] = task["prompt"].replace("{AGENT_GLB_PATH}", target)

    reset_scene(mcp, task)

    images, videos = task_reference_media(task, adapter) if adapter_cfg.get("vision") else ([], [])
    if videos:
        print(f"  [ref] injecting 1 reference video ({videos[0].get('path','?')})")
    elif images:
        print(f"  [ref] injecting {len(images)} reference image(s)")

    # task6/7: give the agent a synthetic tool to read reference frames directly
    # (no need to detour through Blender PNG rendering)
    extra_tools = {}
    if ttype in ("task6_anim", "task7_anim") and adapter_cfg.get("vision"):
        extra_tools = build_t6_frame_tool(task)

    # No-cap curve experiment: score with the current scene every checkpoint_every
    # steps and record into the curve.
    # At the main experiment step point (this task type's original main cap),
    # additionally snapshot products to main_step/ for the main ranking.
    curve = []
    main_step = TASK_MAX_STEPS.get(ttype, 30)   # main experiment step count for this task type
    from run_io import sample_dir as _sd_cp
    _sample_top = _sd_cp(adapter_cfg["name"], ttype, task["id"])
    if checkpoint_every:
        # Never mix checkpoints from a previous trajectory into this curve.
        shutil.rmtree(os.path.join(_sample_top, "curve_steps"), ignore_errors=True)
        shutil.rmtree(os.path.join(_sample_top, "main_step"), ignore_errors=True)

    def _checkpoint(step, cp_mcp):
        # Each checkpoint's products + score are saved separately to
        # curve_steps/step_NNN/, never overwritten.
        # At the main step point (this task type's main cap), an extra copy is
        # stored in main_step/ for the main ranking to pick up directly.
        exp = os.path.join(_sample_top, "curve_steps", f"step_{step:03d}")
        os.makedirs(exp, exist_ok=True)
        is_main = (step == main_step)
        _prev = os.environ.get("EXPORT_SCENE_DIR")
        os.environ["EXPORT_SCENE_DIR"] = exp
        try:
            fake = RunResult(finished=False, steps=[None] * step,
                             error=f"checkpoint@{step}", mcp=cp_mcp)
            checkpoint_task = _prepare_checkpoint_task(task, ttype, exp)
            sc = score_run(
                cp_mcp,
                checkpoint_task,
                fake,
                model_name=adapter_cfg["name"],
                checkpoint=True,
            )
            from run_io import build_headline as _bh
            hl = _bh(ttype, dict(sc))
            # Store a full copy of each step's score in that step's directory
            checkpoint_record = {
                "step": step,
                "is_main_step": is_main,
                "headline": hl,
                "score": sc,
            }
            with open(os.path.join(exp, "score.json"), "w") as handle:
                json.dump(
                    checkpoint_record, handle, indent=2, ensure_ascii=False
                )
            curve.append({"step": step, "primary": hl.get("primary"),
                          "primary_key": hl.get("primary_key"),
                          "secondary": hl.get("secondary"), "is_main_step": is_main})
            print(f"  [checkpoint] step={step} primary={hl.get('primary')} main={is_main} -> {exp}")
            # Main step point: copy an extra one to main_step/ (for the main ranking to pick up separately)
            if is_main:
                ms = os.path.join(_sample_top, "main_step")
                if os.path.isdir(ms):
                    shutil.rmtree(ms, ignore_errors=True)
                shutil.copytree(exp, ms)
        finally:
            if _prev is None:
                os.environ.pop("EXPORT_SCENE_DIR", None)
            else:
                os.environ["EXPORT_SCENE_DIR"] = _prev

    _cp_cb = _checkpoint if checkpoint_every else None

    generation_backend = None
    references = {'ref:' + str(ref.get('view_id', i)): ref['image']
                  for i, ref in enumerate(task.get('references', [])) if ref.get('image')}
    if task.get("generation_provider") == "hunyuan3d":
        from tools.hunyuan import HunyuanHTTPBackend, HunyuanLocalBackend
        endpoint = os.environ.get("V2S_HUNYUAN_ENDPOINT")
        if task_spec.allowed_tools is None or 'asset.generate_3d_from_image' not in task_spec.allowed_tools:
            raise ValueError("Hunyuan tasks require an explicit generation tool allowlist")
        if endpoint:
            generation_backend = HunyuanHTTPBackend(endpoint)
        else:
            model_path = os.environ.get("V2S_HUNYUAN_MODEL_PATH")
            subfolder = os.environ.get("V2S_HUNYUAN_SUBFOLDER")
            if not model_path or not subfolder:
                raise ValueError("Hunyuan requires an endpoint or explicit local model path and subfolder")
            generation_backend = HunyuanLocalBackend(model_path, subfolder,
                                                    os.environ.get("V2S_HUNYUAN_DEVICE", "cuda:0"))
            if os.environ.get("V2S_HUNYUAN_PAINT_CODE"):
                from tools.hunyuan_paint import HunyuanTexturedBackend
                generation_backend = HunyuanTexturedBackend(generation_backend,
                    os.environ["V2S_HUNYUAN_PAINT_CODE"], Path(_sample_top)/"generation_intermediates",
                    python=os.environ.get("V2S_HUNYUAN_PAINT_PYTHON"))
    segmentation_backend = None
    if os.environ.get("V2S_SAM2_CHECKPOINT"):
        from tools.segmentation import SAM2Segmenter
        segmentation_backend = SAM2Segmenter(os.environ["V2S_SAM2_CHECKPOINT"],
            device=os.environ.get("V2S_SAM2_DEVICE", "cuda:0"))
    session = RunSession(task_spec, mcp, extra_tools, artifact_root=Path(_sample_top)/"artifacts",
                         generation_backend=generation_backend, reference_images=references,
                         segmentation_backend=segmentation_backend)
    snapshot_path = Path(_sample_top) / "environment.json"
    snapshot_path.write_text(json.dumps(session.snapshot(), indent=2), encoding="utf-8")

    from contracts import prompt_bundle, canonical_hash, lint_prompt
    initial_scene = "unavailable (setup observation failed)"
    try:
        observed = mcp.call_tool_rich("get_scene_info", {})
        state = json.loads(observed.get("text", "{}"))
        if isinstance(state, dict) and "object_count" in state:
            count = state["object_count"]
            initial_scene = "empty scene" if count == 0 else f"{count} objects; names: " + ", ".join(o['name'] for o in state.get('objects', []))
            if state.get('truncated'): initial_scene += " (name list truncated)"
    except Exception:
        pass
    if references: initial_scene += "; available reference image IDs: " + ", ".join(references)
    bundle = prompt_bundle(task, initial_scene)
    import subprocess
    try:
        harness_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        harness_commit = None
    environment_conditions = {"tools":session.registry.contract_hash(), "budget":task.get('budget', {}),
                              "initial_scene":initial_scene, "asset_index":task_spec.asset_index,
                              "sanitizer":"blender-sanitizer.v1", "context_compaction":"summary.v2"}
    allowed_names = [schema['name'] for schema in session.registry.schemas_for_model()]
    warnings = lint_prompt(bundle['rendered_system'] + task['prompt'], allowed_names,
                           [spec['name'] for spec in session.registry.snapshot()])
    metadata = {"task_id": task["id"], "protocol": task.get("protocol"),
                "seed": task.get("seed"), "group_id": task.get("group_id"),
                "prompt_bundle": bundle, "tool_contracts": session.registry.contracts(),
                "conditions": {"prompt_bundle_hash": canonical_hash(bundle),
                               "toolset_hash": session.registry.contract_hash(),
                               "environment_hash":canonical_hash(environment_conditions),
                               "harness_commit":harness_commit},
                "environment_conditions":environment_conditions, "prompt_lint":warnings,
                "experiment_variables":task.get("experiment_variables", []),
                "model": {"name": adapter_cfg["name"]}}
    from experiment_recording import provenance
    metadata["recording_provenance"] = provenance(ROOT, adapter, task, task_spec)
    result = run_agent(adapter, mcp, bundle["rendered_system"],
                       task["prompt"], max_steps=effective_steps,
                       images=images, videos=videos, extra_tools=extra_tools,
                       restart_blender=restart_blender,
                       on_checkpoint=_cp_cb, checkpoint_every=checkpoint_every,
                       checkpoint_at=({main_step} if checkpoint_every else None),
                       session=session, trajectory_dir=Path(_sample_top) / "trajectory",
                       episode_metadata=metadata, experiment_mode=True)
    snapshot_path.write_text(json.dumps(result.environment, indent=2), encoding="utf-8")
    # If Blender crashed and restarted during the agent run, result.mcp is a new
    # handle; use it for scoring (the old one is stale)
    if getattr(result, "mcp", None) is not None:
        mcp = result.mcp
    score = score_run(mcp, task, result, model_name=adapter_cfg["name"])

    record = {
        "model": adapter_cfg["name"],
        "task": task["id"],
        "finished": result.finished,
        "error": result.error,
        "elapsed_sec": round(result.elapsed, 1),
        "num_steps": len(result.steps),
        "steps": [s.__dict__ for s in result.steps],
        "score": score,
    }
    if checkpoint_every:
        record["curve"] = curve
    # Land in runs/<model>/<task_type>/<task_id>/{score.json, steps.json}
    try:
        write_run_outputs(adapter_cfg["name"], task, record)
        if checkpoint_every:
            json.dump(curve, open(os.path.join(_sample_top, "curve.json"), "w"),
                      indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [warn] write_run_outputs failed: {e!r}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="JSON list of model configs (e.g. models.json)")
    ap.add_argument("--secrets-file",
                    help="credential env file (default: secrets/api.env or SCENEACT_SECRETS_FILE)")
    ap.add_argument("--only-model",
                    help="if given, run only this name from the --models JSON (comma-separated to pick several)")
    ap.add_argument("--task", help="single task JSON file")
    ap.add_argument("--all", action="store_true", help="run every task in tasks dir")
    ap.add_argument("--tasks-dir", help="directory of task JSONs (default tasks/)")
    ap.add_argument("--limit", type=int, default=0, help="only first N tasks (with --all)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="no-cap curve experiment: score with the current scene every N steps and record to curve.json (0=disabled). "
                         "Use with a large --max-steps (e.g. 150)")
    ap.add_argument("--retry", type=int, default=5,
                    help="per-task failure retry count (API timeout/network/MCP), default 5")
    ap.add_argument("--resume", help="resume: path to an existing results JSON, skip completed tasks")
    ap.add_argument("--headless", action="store_true",
                    help="auto-launch a headless Blender (no GUI, no manual Connect)")
    ap.add_argument("--port", type=int, default=9876,
                    help="Blender socket port (default 9876)")
    ap.add_argument("--auto-port", action="store_true",
                    help="if --port is taken, probe upward for the first free port (for concurrent multi-instance)")
    args = ap.parse_args()

    load_credentials(args.secrets_file)
    all_models = json.loads(Path(args.models).read_text())
    if args.only_model:
        wanted = set(s.strip() for s in args.only_model.split(",") if s.strip())
        all_models = [m for m in all_models if m.get("name") in wanted]
        if not all_models:
            raise SystemExit(f"--only-model={args.only_model} found no matching name in {args.models}")
    models = [resolve_env(m) for m in all_models]

    if args.all:
        tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
        task_files = sorted(tasks_dir.glob("*.json"))
        if args.limit:
            task_files = task_files[:args.limit]
    elif args.task:
        task_files = [Path(args.task)]
    else:
        ap.error("specify --task FILE or --all")
    tasks = [json.loads(p.read_text()) for p in task_files]

    RUNS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # Optionally spawn a headless Blender ourselves. Otherwise we assume Blender
    # (GUI with "Connect" clicked, or a manually-started headless one) is already
    # listening on --port.
    sys.path.insert(0, str(ROOT / "headless"))
    port = args.port
    if args.auto_port:
        from blender_process import _port_open  # noqa: E402
        chosen = None
        for p in range(args.port, args.port + 200):
            if not _port_open("localhost", p):
                chosen = p
                break
        if chosen is None:
            raise RuntimeError(f"[auto-port] no free port in [{args.port}, {args.port+200})")
        if chosen != args.port:
            print(f"[auto-port] {args.port} busy -> using {chosen}")
        port = chosen

    headless = None
    if args.headless:
        from blender_process import HeadlessBlender  # noqa: E402
        log = RUNS / f"blender-headless-{stamp}-{port}.log"
        print(f"Launching headless Blender on port {port} (log: {log})")
        headless = HeadlessBlender(port=port, log_path=str(log)).start()
        print("Headless Blender is up.")

    # The blender-mcp server reads BLENDER_PORT to know where Blender listens.
    from mcp_client import BlenderMCPClient
    mcp = BlenderMCPClient(cwd=str(BLENDER_MCP_DIR),
                           env={**os.environ, "BLENDER_PORT": str(port)})
    mcp.start()
    print("MCP connected. Tools:", [t["name"] for t in mcp.list_tools()])

    # Mutable holder for the Blender/MCP handle (replaced on crash-restart; finally reads it too)
    _hold = {"mcp": mcp, "headless": headless}

    # results-<stamp>.json is written to _logs/ as a timestamped audit;
    # the real primary storage is runs/<model>/<task_type>/<task_id>/score.json (write_run_outputs)
    # With concurrent multi-workers (--only-model), the file name carries a model slug
    # to avoid overwriting each other's audit files
    if args.only_model:
        from run_io import slugify_model
        _msuffix = "-" + "_".join(slugify_model(s.strip())
                                  for s in args.only_model.split(",") if s.strip())
    else:
        _msuffix = ""
    out = LOGS / f"results-{stamp}{_msuffix}.json"

    # Resume: if --resume is given, load existing results and skip successful (model,task)
    all_results = []
    done = set()
    if args.resume and Path(args.resume).exists():
        try:
            all_results = json.loads(Path(args.resume).read_text())
            for r in all_results:
                if not str(r.get("error", "")).startswith("FATAL"):
                    done.add((r.get("model"), r.get("task")))
            out = Path(args.resume)  # keep writing to the same file
            print(f"Resume: {len(done)} already completed, skipping")
        except Exception as e:
            print(f"resume read failed, running from scratch: {e!r}")

    def save():
        out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    try:
        # Serial: Blender MCP is single-connection / single-threaded.
        crash_count = 0

        # Blender crash-restart callback: called by the agent loop when some step's
        # execute_code crashes Blender.
        # _hold was initialized above; after restart the new mcp/headless propagate
        # back into this scope (reused by scoring/subsequent tasks/finally).
        def restart_blender():
            if not _hold["headless"]:
                return None
            print(f"  [blender-crash] restarting Blender (port={port}) ...")
            try:
                _hold["mcp"].close()
            except Exception:
                pass
            try:
                _hold["headless"].stop()
            except Exception:
                pass
            time.sleep(3)
            from blender_process import HeadlessBlender
            _hold["headless"] = HeadlessBlender(port=port, log_path=str(
                RUNS / f"blender-headless-restart-{port}.log")).start()
            new_mcp = BlenderMCPClient(cwd=str(BLENDER_MCP_DIR),
                                       env={**os.environ, "BLENDER_PORT": str(port)})
            new_mcp.start()
            _hold["mcp"] = new_mcp
            print(f"  [blender-crash] Blender restart complete port={port}")
            return new_mcp

        for task in tasks:
            for m in models:
                if (m["name"], task["id"]) in done:
                    continue
                # Retry: on per-task failure (API timeout/network/MCP), retry up to args.retry times
                r = None
                for attempt in range(args.retry + 1):
                    try:
                        r = run_one(m, task, _hold["mcp"], args.max_steps,
                                    restart_blender=restart_blender,
                                    checkpoint_every=args.checkpoint_every)
                        mcp = _hold["mcp"]  # sync back if the agent restarted internally
                        crash_count = 0
                        break
                    except Exception as e:
                        msg = f"{type(e).__name__}: {e}"
                        # Blender crash detection: MCP disconnect/socket error -> restart Blender
                        is_blender_crash = any(x in msg.lower() for x in
                            ("connection", "broken pipe", "socket", "eof",
                             "blendermcp", "communicat"))
                        if is_blender_crash and _hold["headless"]:
                            crash_count += 1
                            print(f"  [blender-crash] {task['id']} Blender disconnected (consecutive {crash_count}), restarting...")
                            restart_blender()
                            mcp = _hold["mcp"]
                            if attempt < args.retry:
                                continue
                        # Exponential backoff: 10->30->60->90->120s
                        wait = min(10 * (2 ** attempt), 120)
                        if attempt < args.retry:
                            print(f"  [retry {attempt+1}/{args.retry}] {task['id']} failed: {msg[:120]} (waiting {wait}s)")
                            time.sleep(wait)
                        else:
                            print(f"  [FATAL] {task['id']} retries exhausted: {msg[:120]}")
                            r = {"model": m["name"], "task": task["id"],
                                 "error": f"FATAL: {msg[:300]}"}
                all_results.append(r)
                save()  # Incremental save: write to disk after each task; nothing lost if it crashes midway
    finally:
        try:
            _hold["mcp"].close()
        except Exception:
            pass
        if _hold["headless"]:
            _hold["headless"].stop()
            print("Headless Blender stopped.")

    save()
    print(f"\nWrote {out}")
    # Recompute _summary.json for each (model, task_type)
    seen = set()
    for r in all_results:
        m = r.get("model"); t = r.get("task")
        if not m or not t: continue
        # Infer task_type from task_id
        tt = None
        for cand in ("task1_single","task2_multi","task3_camera","task4_anim",
                     "task5_layout_derived","task5_retrieval","task5_recon","task6_anim","task7_anim"):
            if cand in t: tt = cand; break
        if tt and (m, tt) not in seen:
            seen.add((m, tt))
            try:
                summ = rebuild_summary(m, tt)
                if summ:
                    print(f"  [summary] {m}/{tt}: n={summ['n']} finished_rate={summ['finished_rate']}")
            except Exception as e:
                print(f"  [warn] rebuild_summary({m},{tt}) failed: {e!r}")
    # Recompute each model's cross-task Overall composite score -> runs/<model>/_overall.json
    for m in sorted({r.get("model") for r in all_results if r.get("model")}):
        try:
            ov = rebuild_overall(m)
            if ov and ov["overall"] is not None:
                print(f"  [overall] {m}: {ov['overall']}  {ov['task_scores']}")
            elif ov:
                print(f"  [overall] {m}: partial ({ov['n_tasks']}/5 tasks): {ov['task_scores']}")
        except Exception as e:
            print(f"  [warn] rebuild_overall({m}) failed: {e!r}")
    for r in all_results:
        sc = r.get("score") or {}
        adds = sc.get("mean_add_s")
        print(f"  {r['model']:18} {r['task']:42} "
              f"finished={r.get('finished')} steps={r.get('num_steps')} "
              f"add_s={adds:.3f}" if isinstance(adds, (int, float))
              else f"  {r['model']:18} {r['task']:42} err={r.get('error')}")

    # Aggregate metrics by model (layout tasks)
    print("\n=== Summary (by model) ===")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        sc = r.get("score") or {}
        if isinstance(sc.get("mean_add_s"), (int, float)):
            agg[r["model"]].append(sc)
    for model, scs in agg.items():
        import statistics as st
        madds = st.mean(s["mean_add_s"] for s in scs)
        mcd = st.mean(s["chamfer"] for s in scs)
        macc = st.mean(s["placement_acc"] for s in scs)
        ssr = st.mean(1.0 if s["scene_success"] else 0.0 for s in scs)
        print(f"  {model:18} n={len(scs):3}  mADD-S={madds:.3f}  "
              f"Chamfer={mcd:.3f}  PlaceAcc={macc:.2%}  SceneSR={ssr:.2%}")


if __name__ == "__main__":
    main()
