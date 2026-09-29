"""Fallback module: auto-scan failures, rerun, peritem isolation, zombie-process cleanup.

Usage:
  python -m fallback scan <model>              # scan failed tasks
  python -m fallback rerun <model> [--dry]     # auto-rerun failed ones
  python -m fallback peritem <model> [--port N] # peritem mode for missing T4
  python -m fallback cleanup                   # clean up zombie Blender processes
"""

import json
import glob
import os
import shutil
import subprocess
import signal
import sys
import time
from pathlib import Path

RUNS_DIR = Path("runs")
TASKS_DIR = Path("tasks")

TASK_MAX_STEPS = {
    "task4_anim": 60,
    "task5_recon": 35,
    "task5_retrieval": 35,
    "task5_layout_derived": 35,
    "task6_anim": 80,
    "task7_anim": 80,
}


def scan_failures(model: str) -> dict:
    """Scan failed tasks under runs/<model>/ and return a classified result."""
    all_tasks = {}
    for f in glob.glob(str(TASKS_DIR / "t*" / "*.json")):
        t = json.load(open(f))
        all_tasks[t["id"]] = f

    done = set()
    api_errors = []
    max_steps_reached = []

    for f in glob.glob(str(RUNS_DIR / model / "*" / "*" / "score.json")):
        tid = os.path.basename(os.path.dirname(f))
        done.add(tid)
        try:
            m = json.load(open(f))
            err = str(m.get("error", "") or "")
            if any(x in err for x in ("500", "400", "402", "504", "APIError",
                                       "FATAL", "litellm", "Corrupted", "thought_signature")):
                api_errors.append(tid)
            elif "max_steps" in err:
                max_steps_reached.append(tid)
        except Exception:
            pass

    missing = sorted(set(all_tasks.keys()) - done)

    return {
        "model": model,
        "total_tasks": len(all_tasks),
        "completed": len(done),
        "api_errors": api_errors,
        "max_steps_reached": max_steps_reached,
        "missing": missing,
        "all_tasks": all_tasks,
    }


def rerun_failures(model: str, dry: bool = False, port: int = 9900,
                   models_json: str = "models.json") -> None:
    """Auto-rerun API-failed + missing tasks. Grouped by task_type with the correct step count."""
    result = scan_failures(model)
    to_rerun = result["api_errors"] + result["missing"]

    if not to_rerun:
        print(f"{model}: nothing to rerun (0 failures, 0 missing)")
        return

    print(f"{model}: {len(to_rerun)} to rerun "
          f"({len(result['api_errors'])} API errors + {len(result['missing'])} missing)")

    if dry:
        from collections import Counter
        types = Counter()
        for tid in to_rerun:
            if tid in result["all_tasks"]:
                t = json.load(open(result["all_tasks"][tid]))
                types[t.get("task_type", "?")] += 1
        for tt, cnt in types.most_common():
            print(f"  {tt}: {cnt}")
        return

    # Delete old results of API-failed tasks
    for tid in result["api_errors"]:
        for d in glob.glob(str(RUNS_DIR / model / "*" / tid)):
            shutil.rmtree(d)

    # Group by max_steps
    from collections import defaultdict
    groups = defaultdict(list)
    for tid in to_rerun:
        if tid not in result["all_tasks"]:
            continue
        t = json.load(open(result["all_tasks"][tid]))
        tt = t.get("task_type", "")
        steps = TASK_MAX_STEPS.get(tt, 30)
        groups[steps].append(result["all_tasks"][tid])

    # Launch group by group
    for steps, files in sorted(groups.items()):
        tmp = f"/tmp/_fallback_{model}_{steps}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for tf in files:
            shutil.copy(tf, tmp)
        print(f"  launch: {len(files)} tasks @{steps} steps port={port}")

        cmd = [
            "uv", "run", "python", "src/harness/run.py",
            "--models", models_json, "--only-model", model,
            "--all", "--tasks-dir", tmp,
            "--headless", "--auto-port", "--port", str(port),
            "--max-steps", str(steps), "--retry", "10",
        ]
        log = f"runs/_logs/fallback-{model}-{steps}.log"
        with open(log, "w") as lf:
            subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        port += 5


def peritem_runner(model: str, port: int = 9950,
                   max_steps: int = 0,
                   models_json: str = "models.json") -> None:
    """peritem mode: launch a dedicated Blender per task (prevents crash contagion). Only runs missing T4."""
    result = scan_failures(model)
    missing = result["missing"]
    if not missing:
        print(f"{model}: no missing tasks")
        return

    # Only take T4 (peritem mainly solves Blender crash issues, concentrated in T4)
    t4_missing = []
    for tid in missing:
        if tid in result["all_tasks"]:
            t = json.load(open(result["all_tasks"][tid]))
            if t.get("task_type") == "task4_anim":
                t4_missing.append(result["all_tasks"][tid])

    if not t4_missing:
        print(f"{model}: no missing T4 tasks (use rerun for other missing ones)")
        return

    effective_steps = max_steps or TASK_MAX_STEPS.get("task4_anim", 60)
    print(f"{model}: peritem running {len(t4_missing)} T4 tasks @{effective_steps} steps")

    for tf in t4_missing:
        t = json.load(open(tf))
        tid = t["id"]
        score_path = RUNS_DIR / model / "task4_anim" / tid / "score.json"
        if score_path.exists():
            print(f"  [skip] {tid}")
            continue

        print(f"  [run] {tid}")
        tmp = f"/tmp/_peritem_{model}_{tid}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        shutil.copy(tf, tmp)

        cmd = [
            "uv", "run", "python", "src/harness/run.py",
            "--models", models_json, "--only-model", model,
            "--all", "--tasks-dir", tmp,
            "--headless", "--auto-port", "--port", str(port),
            "--max-steps", str(effective_steps), "--retry", "5",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        shutil.rmtree(tmp, ignore_errors=True)

        if score_path.exists():
            print(f"  [done] {tid}")
        else:
            print(f"  [fail] {tid}")


def cleanup_zombies() -> None:
    """Clean up zombie headless Blender processes."""
    import platform
    if platform.system() != "Darwin" and platform.system() != "Linux":
        print("cleanup_zombies only supports macOS/Linux")
        return

    result = subprocess.run(
        ["pgrep", "-f", "Blender.*headless"],
        capture_output=True, text=True
    )
    pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
    if not pids:
        print("no zombie Blender processes")
        return

    print(f"found {len(pids)} headless Blender processes, cleaning up...")
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"  killed PID {pid}")
        except (ProcessLookupError, PermissionError):
            pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "scan":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        if not model:
            print("usage: python -m fallback scan <model>")
            sys.exit(1)
        r = scan_failures(model)
        print(f"{model}: completed={r['completed']}/{r['total_tasks']}  "
              f"api_err={len(r['api_errors'])}  missing={len(r['missing'])}  "
              f"max_steps={len(r['max_steps_reached'])}")
    elif cmd == "rerun":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        dry = "--dry" in sys.argv
        if not model:
            print("usage: python -m fallback rerun <model> [--dry]")
            sys.exit(1)
        rerun_failures(model, dry=dry)
    elif cmd == "peritem":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        port = 9950
        for i, a in enumerate(sys.argv):
            if a == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        if not model:
            print("usage: python -m fallback peritem <model> [--port N]")
            sys.exit(1)
        peritem_runner(model, port=port)
    elif cmd == "cleanup":
        cleanup_zombies()
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
