"""Asynchronous scene snapshot handoff and background rendering.

The live Blender session is only used to export an immutable GLB after a
mutating agent step. Rendering runs in a separate Blender process, so the next
model request does not wait for the preview render. Snapshot files are removed
after a successful or failed render unless ``keep_snapshots`` is enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


MUTATING_TOOLS = frozenset({
    "execute_blender_code",
    "scene.import_artifact",
    "scene.set_transform",
})


class SceneSnapshotManager:
    """Export and render scene snapshots without blocking the rollout on PNGs."""

    def __init__(self, recorder, mcp, *, mode: str = "mutations",
                 blender_bin: str | None = None, renderer_script: str | None = None,
                 max_pending: int = 4, keep_snapshots: bool = False,
                 render_timeout: int = 180):
        self.recorder = recorder
        self.mcp = mcp
        self.mode = mode
        self.blender_bin = blender_bin or os.environ.get("V2S_SNAPSHOT_BLENDER") or os.environ.get("BLENDER_BIN", "blender")
        default_renderer = Path(__file__).resolve().parents[2] / "scripts" / "render_scene_snapshot.py"
        self.renderer_script = Path(renderer_script or os.environ.get(
            "V2S_SNAPSHOT_RENDERER", str(default_renderer)))
        self.keep_snapshots = keep_snapshots or os.environ.get("V2S_SCENE_SNAPSHOT_KEEP", "0") == "1"
        self.render_timeout = render_timeout
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop,
                                        name="scene-snapshot-renderer", daemon=False)
        self._worker.start()

    def _should_snapshot(self, call_logs):
        names = [str(log.get("name", "")) for log in (call_logs or [])]
        if self.mode == "all":
            return True, names
        if self.mode == "none":
            return False, names
        return any(name in MUTATING_TOOLS for name in names), names

    def after_step(self, step: int, mcp, call_logs):
        """Synchronous export handoff called by the agent loop after a step."""
        should, names = self._should_snapshot(call_logs)
        if not should:
            return None
        if self._closed:
            return None
        if self._queue.full():
            self.recorder.event("scene.render", event_step=step, status="skipped",
                                reason="render_queue_full", mutation_tools=names)
            return None
        snapshot_dir = self.recorder.root / "snapshots"
        render_dir = self.recorder.root / "renders" / f"step_{step:04d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"step_{step:04d}.glb"
        started = time.time()
        # The export is deliberately the only live-session operation. The GLB
        # is immutable before it enters the queue; the PNG render is independent.
        code = (
            "import bpy\n"
            f"bpy.ops.export_scene.gltf(filepath={json.dumps(str(snapshot_path))}, "
            "export_format='GLB', use_selection=False)\n"
            "print('VIDEO2SCENE_SNAPSHOT_EXPORTED')"
        )
        try:
            result = mcp.call_tool("execute_blender_code", {"code": code})
            if not snapshot_path.exists():
                raise RuntimeError(f"Blender returned but snapshot was not created: {str(result)[:240]}")
            state_id = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()[:16]
            self.recorder.event(
                "scene.snapshot", event_step=step, status="ready",
                snapshot_path=str(snapshot_path.relative_to(self.recorder.root)),
                state_id=state_id, mutation_tools=names,
                handoff_elapsed_sec=round(time.time() - started, 3),
            )
        except Exception as exc:
            if not self.keep_snapshots:
                snapshot_path.unlink(missing_ok=True)
            self.recorder.event(
                "scene.snapshot", event_step=step, status="failed",
                mutation_tools=names, error_type=type(exc).__name__,
                error=str(exc)[:500], handoff_elapsed_sec=round(time.time() - started, 3),
            )
            return None

        job = (step, snapshot_path, render_dir, time.time(), state_id)
        try:
            self.recorder.event(
                "scene.render", event_step=step, status="queued",
                state_id=state_id,
                snapshot_path=str(snapshot_path.relative_to(self.recorder.root)),
                render_path=str(render_dir.relative_to(self.recorder.root)),
            )
            self._queue.put_nowait(job)
        except queue.Full:
            self.recorder.event(
                "scene.render", event_step=step, status="skipped",
                reason="render_queue_full", state_id=state_id,
            )
            if not self.keep_snapshots:
                snapshot_path.unlink(missing_ok=True)
        return state_id

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            step, snapshot_path, render_dir, queued_at, state_id = job
            started = time.time()
            status = "succeeded"
            error = None
            output = ""
            try:
                proc = subprocess.run(
                    [self.blender_bin, "-b", "--factory-startup", "--threads", "2",
                     "--python-exit-code", "1", "--python", str(self.renderer_script), "--",
                     str(snapshot_path), str(render_dir), str(self.recorder.root / "preview_camera.json")],
                    capture_output=True, text=True, timeout=self.render_timeout,
                )
                output = (proc.stdout or "")[-600:] + (proc.stderr or "")[-600:]
                if proc.returncode != 0:
                    status = "failed"
                    error = f"renderer_exit_{proc.returncode}"
            except FileNotFoundError:
                status, error = "failed", "blender_not_found"
            except subprocess.TimeoutExpired:
                status, error = "failed", "renderer_timeout"
            except Exception as exc:
                status, error = "failed", f"{type(exc).__name__}: {exc}"
            pngs = sorted(str(p.relative_to(self.recorder.root))
                          for p in render_dir.glob("*.png") if p.is_file())
            if status == "succeeded" and not pngs:
                status, error = "failed", "renderer_produced_no_png"
            try:
                if not self.keep_snapshots:
                    snapshot_path.unlink(missing_ok=True)
                self.recorder.event(
                    "scene.render", event_step=step, status=status, state_id=state_id,
                    snapshot_path=str(snapshot_path.relative_to(self.recorder.root)),
                    snapshot_retained=snapshot_path.exists(), observer_only=True,
                    render_files=pngs, render_path=str(render_dir.relative_to(self.recorder.root)),
                    queue_wait_sec=round(started - queued_at, 3),
                    render_elapsed_sec=round(time.time() - started, 3),
                    error=error, renderer_tail=output[-800:] if error else None,
                )
            finally:
                self._queue.task_done()

    def close(self, wait: bool = True):
        if self._closed:
            return
        self._closed = True
        def drain():
            self._queue.join()
            self._queue.put(None)
            self._worker.join()
        if wait:
            drain()
        else:
            # Return immediately; a non-daemon drainer keeps pending work alive
            # until process shutdown. It does not claim zero final drain cost.
            threading.Thread(target=drain, name="scene-render-drain", daemon=False).start()
