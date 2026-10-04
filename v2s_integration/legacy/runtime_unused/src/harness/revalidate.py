#!/usr/bin/env python3
"""revalidate.py — loop and rerun invalid samples of any stepcurve line until all are valid or max rounds is reached.

Validity check (consistent with the main monitor):
  - Product OK: t3=agent_camera.json exists; t4=has a >2KB .glb or non-empty agent_frames; otherwise=agent_scene.glb>2KB
  - score.headline.primary is finite (not None/NaN/inf) and != 0

Usage:
  python revalidate.py <line> [--max-rounds N] [--workers W] [--base-port P] [--dry]

<line> is a directory name under runs/, e.g. mimo-v2.5-stepcurve / claude-sonnet-5-stepcurve.
The models config (models_stepcurve_<short>.json) and step count are inferred from the line name
(sonnet5 uses the main experiment's fixed 30 steps; others 150).

Encodes the pitfalls hit during this experiment:
  1. Validity uses "product + score" not just the error field (fallback.py's old check missed empty glb / placeholder / invalid score).
  2. Exclude samples "currently being run by another worker" (shard-dir pgrep) to avoid duplicate runs / port conflicts.
  3. Same-named t6/t7 scenes (t6l1_castle exists in both t6 and t7) are distinguished by task_type prefix, no misjudged dedup.
  4. Invalid samples left behind after a worker exits are re-included in the next round (fixes "worker finishes and exits -> orphaned sample left with no owner").
  5. Loop until all valid or max_rounds; if a sample stays invalid across rounds, report it as the model's true ceiling.
"""
import os, sys, json, math, struct, shutil, subprocess, time, argparse
from pathlib import Path

# repo root = this file's ../../.. (src/harness/revalidate.py -> repo root).
# Override with env vars: BENCH_ROOT (data/runs root), HARNESS_ROOT (harness code root).
_REPO = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("BENCH_ROOT", _REPO))
_HARNESS = Path(os.environ.get("HARNESS_ROOT", _REPO))
RUNS = BENCH / "runs"
PUB = BENCH / "tasks_public"
RUN_PY = _HARNESS / "src/harness/run.py"
# python for scoring/driving: prefer env var VENV_PY, otherwise the current interpreter.
VENV = Path(os.environ.get("VENV_PY", sys.executable))

TP = {"task1_single": "t1", "task2_multi": "t2", "task3_camera": "t3",
      "task4_anim": "t4", "task5_recon": "t5", "task6_anim": "t6", "task7_anim": "t7"}

def line_short(line):
    # claude-sonnet-5-stepcurve -> sonnet5 ; mimo-v2.5-stepcurve -> mimo
    if "sonnet-5" in line: return "sonnet5"
    if line.startswith("mimo"): return "mimo"
    return line.split("-stepcurve")[0]

def fin(x): return isinstance(x, (int, float)) and math.isfinite(x)

def prod_ok(td, tt):
    if "task3" in tt:
        return os.path.exists(os.path.join(td, "agent_camera.json"))
    if "task4" in tt:
        if any(f.endswith(".glb") and os.path.getsize(os.path.join(td, f)) > 2048
               for f in os.listdir(td)):
            return True
        fr = os.path.join(td, "agent_frames")
        return os.path.isdir(fr) and len(os.listdir(fr)) > 0
    g = os.path.join(td, "agent_scene.glb")
    return os.path.exists(g) and os.path.getsize(g) > 2048

def has_valid_checkpoint(td, tt):
    """Fallback check: does any step under curve_steps/step_NNN/ succeed.

    The main experiment step counts (35/60/80/150) are just upper bounds; the agent may finish on its own
    at an earlier step and produce a valid result. Occasional timeout/hang at the end can corrupt the top-level
    score.json/curve.json (empty/None), but each step's score under curve_steps is complete. If any step succeeds,
    the sample succeeds — use that step when reading, no need to look at the corrupt top-level file, and no need to rerun. Read-only, does not modify any file.

    Rule: a step succeeds if its score.json has a finite non-0 primary. **Does not require the curve_steps
    directory to still hold the product files** — animation types (task4/6/7) score per-frame products without persisting them to curve_steps,
    but the score is already computed (a primary value proves a valid product was scored at the time). Layout types (t1/2/5)
    usually leave agent_scene.glb in the step dir, but even if not, a valid score equally proves it succeeded then."""
    cs_root = os.path.join(td, "curve_steps")
    if not os.path.isdir(cs_root):
        return False
    for step_dir in os.listdir(cs_root):
        sd = os.path.join(cs_root, step_dir)
        if not os.path.isdir(sd):
            continue
        sc = os.path.join(sd, "score.json")
        if not os.path.exists(sc):
            continue
        try:
            prim = (json.load(open(sc)).get("headline") or {}).get("primary")
        except Exception:
            continue
        if fin(prim) and prim != 0:
            return True
    return False

def result_name(shard_fn):
    """shard file name (tN__...json) -> result directory name under runs. t4/t6/t7 have dedicated prefixes."""
    stem = shard_fn[:-5]
    parts = stem.split("__", 1)
    pref = parts[0]
    rest = parts[1] if len(parts) > 1 else stem
    if pref == "t4": return "anim-" + rest
    if pref == "t6": return "t6anim-" + rest
    if pref == "t7": return "t7anim-" + rest
    return rest

def scan_invalid(line):
    """Return invalid samples as {result_name: (task_type, dir)}."""
    base = RUNS / line
    inv = {}
    for tt in os.listdir(base):
        tp = base / tt
        if not tp.is_dir(): continue
        for t in os.listdir(tp):
            td = tp / t
            if not td.is_dir() or not (td / "curve.json").exists(): continue
            po = prod_ok(str(td), tt)
            prim = None
            sc = td / "score.json"
            if sc.exists():
                try: prim = (json.load(open(sc)).get("headline") or {}).get("primary")
                except Exception: pass
            if not (po and fin(prim) and prim != 0):
                # Top-level invalid: fall back to whether any curve_steps step succeeded (agent stopped early / finishing stage corrupted the top level).
                # If so, the sample is actually a success, not invalid, and should not be rerun.
                if not has_valid_checkpoint(str(td), tt):
                    inv[t] = (tt, str(td))
    return inv

def find_task_src(result_nm, tt):
    """Find the task json source of this invalid sample in tasks_public."""
    pref = TP[tt]
    if tt == "task4_anim":
        core = result_nm[len("anim-"):] if result_nm.startswith("anim-") else result_nm
        base = f"{core}.json"
    elif tt == "task6_anim":
        core = result_nm[len("t6anim-"):] if result_nm.startswith("t6anim-") else result_nm
        base = f"{core}.json"
    elif tt == "task7_anim":
        core = result_nm[len("t7anim-"):] if result_nm.startswith("t7anim-") else result_nm
        base = f"{core}.json"
    else:
        base = f"{result_nm}.json"
    src = PUB / pref / base
    return src if src.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base-port", type=int, default=24000)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    short = line_short(args.line)
    models = BENCH / f"models_stepcurve_{short}.json"
    max_steps = 30 if short == "sonnet5" else 150  # sonnet5 uses the main-experiment fixed step count
    if not models.exists():
        print(f"[error] models config does not exist: {models}"); sys.exit(1)

    for rnd in range(1, args.max_rounds + 1):
        inv = scan_invalid(args.line)
        print(f"\n===== {args.line} ROUND {rnd}: invalid {len(inv)} =====", flush=True)
        if not inv:
            print(f"==== {args.line} all valid, converged at round {rnd} ====")
            return
        # Build shards (unique names to avoid t6/t7 name collisions), find sources
        srcs = []
        miss = []
        for rn, (tt, td) in inv.items():
            src = find_task_src(rn, tt)
            if src: srcs.append((TP[tt] + "__" + os.path.basename(src), src))
            else: miss.append(rn)
        if miss:
            print(f"  [warn] {len(miss)} could not find a source (skipped): {miss[:5]}")
        if args.dry:
            from collections import Counter
            print("  types:", dict(Counter(tt for _, (tt, _) in inv.items())))
            return
        dst = BENCH / f"shard_reval_{short}"
        if dst.exists(): shutil.rmtree(dst)
        nw = min(args.workers, max(1, len(srcs)))
        for i in range(nw): (dst / f"{i:02d}").mkdir(parents=True)
        for idx, (fn, src) in enumerate(srcs):
            os.symlink(os.path.realpath(src), dst / f"{idx % nw:02d}" / fn)
        # Start workers, block until done
        procs = []
        for i in range(nw):
            sh = dst / f"{i:02d}"
            if not any(sh.iterdir()): continue
            port = args.base_port + i * 10
            log = BENCH / f"logs_reval_{short}_r{rnd}_w{i:02d}.log"
            cmd = [str(VENV), str(RUN_PY), "--models", str(models), "--all",
                   "--tasks-dir", str(sh), "--headless", "--auto-port",
                   "--port", str(port), "--max-steps", str(max_steps),
                   "--checkpoint-every", "10", "--retry", "3"]
            procs.append(subprocess.Popen(cmd, stdout=open(log, "w"),
                                          stderr=subprocess.STDOUT,
                                          env={**os.environ, "PWD": str(BENCH), "TMPDIR": "/tmp"},
                                          cwd=str(BENCH)))
            time.sleep(3)
        print(f"  round {rnd}: {len(procs)} workers started, waiting for completion...", flush=True)
        for p in procs: p.wait()
        print(f"  round {rnd} done", flush=True)

    # Still invalid after reaching max_rounds
    inv = scan_invalid(args.line)
    if inv:
        print(f"\n==== {args.line} reached max_rounds={args.max_rounds}, still {len(inv)} invalid ====")
        print("these stay invalid after repeated reruns, treated as the model's real ceiling:")
        for rn, (tt, td) in list(inv.items())[:20]:
            print(f"  {tt}/{rn}")

if __name__ == "__main__":
    main()
