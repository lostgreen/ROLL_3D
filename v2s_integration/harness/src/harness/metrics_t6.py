"""
metrics_t6.py —— Task 6 dynamic-scene reproduction scoring (task-agnostic).

Task: the agent sees reference frames (144 frames, read on demand) plus a
      library of component parts, then in Blender it (1) lays out the static
      scene and (2) creates per-frame animation for the movable objects, and
      exports one whole-scene glb with animation. Scoring compares agent glb vs GT.

Task-agnostic design: T6 samples have heterogeneous motion types (traffic /
  lane-change / boat / conveyor / golf / platform jumping / racetrack / train,
  8 motion semantics), but all are reduced to a "per-frame 3D position sequence":
  - GT ground truth = gt/trajectory.json (uniform format {name:[{f,loc}]}, loc
    already in Blender Z-up); no task-specific motion semantics (lane/conveyor/loop...)
    are parsed.
  - agent = world centroid of each driven node, sampled from the exported glb's
    animation tracks.

Three scoring layers:
  (1) trajectory: Hungarian matching + per-frame L2 normalized —— main score worst_vehicle_err.
  (2) motion descriptors (semantic): task-agnostic geometric quantities (main-axis
      direction / straightness / closedness / vertical range / lateral offset),
      agent vs GT —— direction_error_rate / path_shape_err.
  (3) static layout: agent non-animated object centroids vs GT layout point cloud,
      bidirectional Chamfer (no classification).

GT source: benchmark_t6_final/<sid>/{gt/trajectory.json, layout_gt.json, meta.json}
Prediction source: the agent-exported glb with animation (whole scene).

Dependencies: numpy scipy
"""
import json, struct, math, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


# ============ glTF parsing: vertices + animation sampling ============

def _read_glb_json_bin(path):
    data = open(path, "rb").read()
    clen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + clen].decode("utf-8", "replace"))
    # the second chunk is BIN
    bin_off = 20 + clen
    blob = b""
    if bin_off < len(data):
        blen = struct.unpack("<I", data[bin_off:bin_off + 4])[0]
        blob = data[bin_off + 8:bin_off + 8 + blen]
    return gltf, blob


_CT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
       5125: ("I", 4), 5126: ("f", 4)}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(gltf, blob, idx):
    """Read accessor -> (count, ncomp) ndarray."""
    a = gltf["accessors"][idx]
    bv = gltf["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    fmt, sz = _CT[a["componentType"]]
    nc = _NC[a["type"]]
    npdt = {"b": np.int8, "B": np.uint8, "h": np.int16, "H": np.uint16,
            "I": np.uint32, "f": np.float32}[fmt]
    arr = np.frombuffer(blob, npdt, a["count"] * nc, off).reshape(-1, nc).astype(np.float64)
    return arr


def _node_local_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        M = M @ np.diag(node["scale"] + [1.0])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w),     0],
            [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w),     0],
            [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y), 0],
            [0, 0, 0, 1]], float)
        M = R @ M
    if "translation" in node:
        T = np.eye(4); T[:3, 3] = node["translation"]
        M = T @ M
    return M


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),   0],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),   0],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y), 0],
        [0, 0, 0, 1]], float)


def _sample_animation(gltf, blob, fps=24.0):
    """Parse all animation channels -> {node_idx: {"T":[(t,vec3)], "R":[(t,quat)], "S":...}}.
    Returns the keyframe time series of each driven node."""
    drives = {}  # node_idx -> {"translation":(times,values),"rotation":(...),...}
    for anim in gltf.get("animations", []):
        samplers = anim["samplers"]
        for ch in anim["channels"]:
            tgt = ch["target"]
            ni = tgt.get("node")
            path = tgt["path"]  # translation/rotation/scale
            if ni is None:
                continue
            smp = samplers[ch["sampler"]]
            times = _accessor(gltf, blob, smp["input"]).reshape(-1)
            vals = _accessor(gltf, blob, smp["output"])
            drives.setdefault(ni, {})[path] = (times, vals)
    return drives


def _interp_at(times, vals, t, is_quat=False):
    """Linear interpolation at time t (quat uses nlerp)."""
    if len(times) == 1:
        return vals[0]
    if t <= times[0]:
        return vals[0]
    if t >= times[-1]:
        return vals[-1]
    i = np.searchsorted(times, t) - 1
    i = max(0, min(i, len(times) - 2))
    t0, t1 = times[i], times[i + 1]
    f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    v0, v1 = vals[i], vals[i + 1]
    if is_quat:
        if np.dot(v0, v1) < 0:
            v1 = -v1
        v = v0 * (1 - f) + v1 * f
        return v / (np.linalg.norm(v) + 1e-12)
    return v0 * (1 - f) + v1 * f


def _node_matrix_at(node, ni, drives, t):
    """Local matrix of the node at time t: use animated values to override TRS if
    animated, otherwise use static values."""
    d = drives.get(ni, {})
    # translation
    if "translation" in d:
        tr = _interp_at(d["translation"][0], d["translation"][1], t)
    else:
        tr = np.array(node.get("translation", [0, 0, 0]), float)
    # rotation
    if "rotation" in d:
        q = _interp_at(d["rotation"][0], d["rotation"][1], t, is_quat=True)
    else:
        q = np.array(node.get("rotation", [0, 0, 0, 1]), float)
    # scale
    if "scale" in d:
        sc = _interp_at(d["scale"][0], d["scale"][1], t)
    else:
        sc = np.array(node.get("scale", [1, 1, 1]), float)
    M = np.eye(4)
    M = M @ np.diag(list(sc) + [1.0])
    M = _quat_to_mat(q) @ M
    T = np.eye(4); T[:3, 3] = tr
    return T @ M


def node_world_centroids_at(gltf, blob, drives, t, node_names=None):
    """At time t, return {node_idx: world centroid} (only for nodes with a mesh, or
    nodes with a given name). Used to read "the world position of each car at time t"."""
    nodes = gltf.get("nodes", [])
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(nodes))))
    # precompute the local centroid of each mesh (mean of vertices)
    mesh_centroid_local = {}
    for mi, m in enumerate(gltf.get("meshes", [])):
        pts = []
        for pr in m["primitives"]:
            pts.append(_accessor(gltf, blob, pr["attributes"]["POSITION"]))
        if pts:
            allp = np.vstack(pts)
            mesh_centroid_local[mi] = allp.mean(0)
    result = {}  # node_idx -> world centroid (world-mean of all mesh vertices in this node subtree)

    def dfs(ni, parent_M, collect_into):
        n = nodes[ni]
        M = parent_M @ _node_matrix_at(n, ni, drives, t)
        acc = collect_into
        if "mesh" in n and n["mesh"] in mesh_centroid_local:
            c = mesh_centroid_local[n["mesh"]]
            wc = (M @ np.array([c[0], c[1], c[2], 1.0]))[:3]
            if acc is not None:
                acc.append(wc)
        for ch in n.get("children", []):
            dfs(ch, M, acc)

    # accumulate separately for each root node (treat each top-level node as one "object")
    for ri in roots:
        pts = []
        dfs(ri, np.eye(4), pts)
        if pts:
            result[ri] = np.mean(pts, axis=0)
    return result, nodes


# ============ T6 scoring main flow ============

def _node_name(nodes, ni):
    return nodes[ni].get("name", f"node_{ni}")


def _gltf_to_blender(p):
    """glTF +Y up -> Blender +Z up: (x, y, z)_gltf = (x, -z, y)_blender.
    Internally metrics uniformly use the Blender frame (Y=lane, Z=height), consistent
    with meta.json/lanes."""
    if p.ndim == 1:
        return np.array([p[0], -p[2], p[1]])
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


# ============ T6 scoring main flow (task-agnostic) ============
# Design: all mover motion is uniformly reduced to a "per-frame 3D position sequence".
#   GT ground truth = gt/trajectory.json (uniform format, no need to parse the 8 motion semantics).
#   agent = world centroid of each driven node, sampled from the exported glb's animation tracks.
# Three scoring layers: trajectory geometry (main) + task-agnostic motion descriptors
# (diagnostic) + static-layout Chamfer.

def _subtree_centroid(gltf, blob, drives, root_ni, t):
    """World centroid of a single node subtree at time t (native glTF frame, +Y up)."""
    nodes = gltf["nodes"]
    mesh_centroid_local = {}
    pts_acc = []

    def local_centroid(mi):
        if mi not in mesh_centroid_local:
            ps = [_accessor(gltf, blob, pr["attributes"]["POSITION"])
                  for pr in gltf["meshes"][mi]["primitives"]]
            mesh_centroid_local[mi] = np.vstack(ps).mean(0) if ps else np.zeros(3)
        return mesh_centroid_local[mi]

    def dfs(ni, parent_M):
        n = nodes[ni]
        M = parent_M @ _node_matrix_at(n, ni, drives, t)
        if "mesh" in n:
            c = local_centroid(n["mesh"])
            pts_acc.append((M @ np.array([c[0], c[1], c[2], 1.0]))[:3])
        for ch in n.get("children", []):
            dfs(ch, M)
    dfs(root_ni, np.eye(4))
    return np.mean(pts_acc, axis=0) if pts_acc else np.zeros(3)


# ---- GT / agent trajectory extraction ----

def load_gt_trajectories(traj_json_path, n_frames):
    """Read gt/trajectory.json -> {name: (n_frames,3)} (Blender frame, loc already Z-up).
    Keep only keys whose items contain f+loc (filter out non-mover entries like the
    platformer coins)."""
    raw = json.load(open(traj_json_path))
    out = {}
    for name, items in raw.items():
        if not isinstance(items, list) or not items:
            continue
        it0 = items[0]
        if not (isinstance(it0, dict) and "f" in it0 and "loc" in it0):
            continue  # non-mover (e.g. coins)
        arr = np.zeros((n_frames, 3))
        last = None
        filled = [False] * n_frames
        for it in items:
            f = int(it["f"]) - 1  # 1-based -> 0-based
            if 0 <= f < n_frames:
                arr[f] = np.array(it["loc"], float)
                filled[f] = True
                last = arr[f]
        # fill holes: forward fill (sparse frames), head uses the first valid value
        first_valid = next((i for i, v in enumerate(filled) if v), None)
        if first_valid is None:
            continue
        for i in range(n_frames):
            if not filled[i]:
                arr[i] = arr[i-1] if i > 0 else arr[first_valid]
        out[name] = arr
    return out


def extract_agent_trajectories(agent_glb, fps=24.0, n_frames=144):
    """Extract the per-frame world centroid of each animation-driven mover from the
    agent-exported glb.

    Key point: a mover (e.g. a car) is a nested hierarchy in the glb (car_pickup -> ...
    -> car door puertaizq). The agent may also add animation to a car's *sub-parts*
    (door / body group), producing multiple driven nodes for one car. All driven nodes
    must be merged into the *top-level scene object root* they belong to, so each object
    produces only one trajectory (whole-car subtree centroid); otherwise the door is
    treated as an independent mover and pollutes the score.

    Top-level object root = walk up from the driven node and take the highest ancestor
    beneath the scene roots. Sample that object root's subtree centroid and convert to
    the Blender frame. Returns {root_name: (n_frames,3)}.
    """
    gltf, blob = _read_glb_json_bin(agent_glb)
    drives = _sample_animation(gltf, blob, fps)
    nodes = gltf.get("nodes", [])
    if not drives:
        return {}
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    scene_roots = set(scene.get("nodes", list(range(len(nodes)))))
    # walk each driven node up to its top-level object root:
    #   if some ancestor is a scene root -> that scene root is the object root;
    #   otherwise walk up to the topmost ancestor (no parent).
    obj_roots = set()
    for ni in drives:
        top = ni
        x = ni
        while True:
            if x in scene_roots:
                top = x
                break
            p = parent.get(x)
            if p is None:
                top = x
                break
            x = p
        obj_roots.add(top)
    trajs = {}
    for ni in obj_roots:
        pts = [_gltf_to_blender(_subtree_centroid(gltf, blob, drives, ni, f / fps))
               for f in range(n_frames)]
        trajs[nodes[ni].get("name", f"node_{ni}")] = np.array(pts)
    return trajs


# ---- scene scale (task-agnostic, per-sample) ----

def compute_scene_scale(layout_gt, gt_trajs):
    """Merge static object positions + GT trajectory points, take the XY bounding-box
    diagonal; floor 5m (avoid being too strict on small scenes)."""
    pts = []
    for o in (layout_gt or {}).get("objects", []):
        if "location" in o:
            pts.append(o["location"][:3])
    for tr in gt_trajs.values():
        pts.extend(tr.tolist())
    if not pts:
        return 40.0
    P = np.array(pts)
    span_xy = np.linalg.norm(P[:, :2].max(0) - P[:, :2].min(0))
    return max(float(span_xy), 5.0)


# ---- task-agnostic motion descriptors ----

def _heading_series(P, scene_scale, min_step=None):
    """Per-frame heading series (uses velocity direction, independent of node rotation):
    heading(f)=atan2(Dy,Dx). Only taken on frames with large enough displacement (heading
    is noisy on slow frames, skip them). Returns [(frame_idx, heading_rad), ...].
    Using velocity direction as a uniform convention -> agent/GT both compute it this way,
    automatically canceling the offset from "different definitions of the vehicle front".

    Drop teleport frames: conveyor / loop trajectories (box reaching the end and jumping
    back to start) have a single-frame displacement far above normal, which would be
    mistaken for a 180deg+180deg=360deg "turn". Skip frames whose displacement > 8x the
    normal median step to avoid a false-positive heading_err. A real turn has smooth step
    length and never jumps 8x, so it is unaffected."""
    if min_step is None:
        min_step = 0.005 * scene_scale
    d = P[1:] - P[:-1]
    step = np.hypot(d[:, 0], d[:, 1])
    moving = step[step >= min_step]
    tele = (8.0 * float(np.median(moving))) if len(moving) else np.inf
    out = []
    for i, dv in enumerate(d):
        s = step[i]
        if s >= min_step and s <= tele:     # skip slow noise frames + teleport jump frames
            out.append((i, float(np.arctan2(dv[1], dv[0]))))
    return out


def _total_turning(P, scene_scale):
    """Total turning of the trajectory (rad): sum |heading diff of adjacent valid frames|.
    A full circle ~2pi, a straight line ~0. Cancels the absolute heading offset (only looks
    at how much it turned), measuring "turning / steering" behavior."""
    hs = _heading_series(P, scene_scale)
    if len(hs) < 2:
        return 0.0
    tot = 0.0
    for (_, h0), (_, h1) in zip(hs[:-1], hs[1:]):
        dh = h1 - h0
        # wrap to [-pi, pi]
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        tot += abs(dh)
    return float(tot)


def motion_descriptors(P, scene_scale):
    """Compute task-agnostic geometric descriptors from one trajectory P(N,3, Blender).
    Computed the same way for GT/agent."""
    eps = 1e-9
    d = P[1:] - P[:-1]
    seg = np.linalg.norm(d, axis=1)
    # path_len: ignore segments > 0.3*scene_scale of teleport (conveyor loop teleport)
    seg_clean = seg[seg <= 0.3 * scene_scale]
    path_len = float(seg_clean.sum())
    net = P[-1] - P[0]
    net_disp = float(np.linalg.norm(net))
    straightness = net_disp / (path_len + eps)
    moving = path_len > 0.05 * scene_scale
    closed = bool(moving and net_disp < 0.10 * path_len)
    # main axis: PCA first principal component, sign aligned to the net direction
    axis_dir = np.zeros(3)
    if moving:
        Q = P - P.mean(0)
        try:
            _, _, Vt = np.linalg.svd(Q, full_matrices=False)
            axis_dir = Vt[0]
            if np.dot(axis_dir, net) < 0:
                axis_dir = -axis_dir
            nrm = np.linalg.norm(axis_dir)
            axis_dir = axis_dir / nrm if nrm > eps else np.zeros(3)
        except np.linalg.LinAlgError:
            axis_dir = net / (net_disp + eps)
    vertical_range = float(P[:, 2].max() - P[:, 2].min())
    # lateral: max perpendicular distance from the path to the start->end chord
    lateral = 0.0
    if net_disp > eps:
        u = net / net_disp
        rel = P - P[0]
        proj = np.outer(rel @ u, u)
        perp = np.linalg.norm(rel - proj, axis=1)
        lateral = float(perp.max())
    return {
        "net_disp_n": net_disp / scene_scale,
        "path_len_n": path_len / scene_scale,
        "straightness": straightness,
        "closed": closed,
        "moving": moving,
        "axis_dir": axis_dir.tolist(),
        "vertical_range_n": vertical_range / scene_scale,
        "lateral_n": lateral / scene_scale,
    }


# ---- agent<->GT global alignment (only ground translation, no scale/rotation) ----

def align_xy_translation(agent_trajs, gt_trajs):
    """Apply a single *XY translation* to the whole agent trajectory to align it to GT.
    Returns (aligned agent_trajs, scale_est).

    Design trade-off (given that the prompt already provides a reference camera):
    - *Apply translation*: the agent's absolute back-projected origin still has reasonable
      error; absorb it so an origin offset is not over-penalized.
    - *No scale*: the prompt gives a reference camera + native component sizes, so the agent
      should be able to back-project the true scale; building too big/small is a genuine
      capability error and must be penalized -> not absorbed. The estimated scale factor is
      also returned as a diagnostic (scale_error = |log(s)|, 0=scale correct).
    - *No rotation*: heading is pinned by the prompt basis convention (main axis +X); and
      applying rotation would absorb the rotation error of symmetric scenes.

    Estimate globally with all mover points across all frames (centroid diff after a coarse
    matching), not per-car (which would absorb relative-layout errors).
    """
    ag = list(agent_trajs.items())
    gt = list(gt_trajs.items())
    if not ag or not gt:
        return agent_trajs, 1.0, (0.0, 0.0, 0.0)
    C = np.zeros((len(ag), len(gt)))
    for i, (_, ta) in enumerate(ag):
        for j, (_, tg) in enumerate(gt):
            m = min(len(ta), len(tg))
            C[i, j] = np.linalg.norm(ta[:m] - tg[:m], axis=1).mean()
    ri, cj = linear_sum_assignment(C)
    A_pts, G_pts = [], []
    for i, j in zip(ri, cj):
        ta, tg = ag[i][1], gt[j][1]
        m = min(len(ta), len(tg))
        A_pts.append(ta[:m]); G_pts.append(tg[:m])
    if not A_pts:
        return agent_trajs, 1.0, (0.0, 0.0, 0.0)
    A = np.vstack(A_pts); G = np.vstack(G_pts)
    # estimate scale (diagnostic only, not applied): XY isotropic
    Axy, Gxy = A[:, :2], G[:, :2]
    ca, cg = Axy.mean(0), Gxy.mean(0)
    A0, G0 = Axy - ca, Gxy - cg
    denom = float((A0 * A0).sum())
    s_est = float((A0 * G0).sum() / denom) if denom > 1e-9 else 1.0
    s_est = max(min(s_est, 10.0), 0.1)
    # translation only (XY + Z centroid alignment), no scaling
    t_xy = cg - ca
    dz = float(np.median(G[:, 2] - A[:, 2]))
    shift = (float(t_xy[0]), float(t_xy[1]), dz)
    out = {}
    for k, v in agent_trajs.items():
        w = v.copy()
        w[:, 0] += t_xy[0]; w[:, 1] += t_xy[1]; w[:, 2] += dz
        out[k] = w
    return out, s_est, shift


# old-name compatibility (scale alignment disabled)


# old-name compatibility (scale alignment disabled)
def align_xy_scale(agent_trajs, gt_trajs):
    out, _, _ = align_xy_translation(agent_trajs, gt_trajs)
    return out


# ---- three-layer scoring ----

def score_trajectory(agent_trajs, gt_trajs, scene_scale=40.0):
    """Trajectory geometry score: Hungarian (whole-trajectory mean L2 cost) matching +
    per-frame L2 normalized. Missing GT movers are counted with the worst err=1.0 (not
    silently dropped)."""
    ag = list(agent_trajs.items())
    gt = list(gt_trajs.items())
    n_gt = len(gt)
    if not gt:
        return {"worst_vehicle_err": 1.0, "mean_vehicle_err": 1.0,
                "movable_recall": 0.0, "mover_count_err": 1.0, "matches": []}
    if not ag:
        return {"worst_vehicle_err": 1.0, "mean_vehicle_err": 1.0,
                "movable_recall": 0.0,
                "mover_count_err": 1.0, "matches": []}
    # cost matrix: whole-trajectory per-frame mean L2 (more stable than start-point-only
    # distance; avoids mismatching loop/circular trajectories when start points are close)
    C = np.zeros((len(ag), n_gt))
    for i, (_, ta) in enumerate(ag):
        for j, (_, tg) in enumerate(gt):
            m = min(len(ta), len(tg))
            C[i, j] = np.linalg.norm(ta[:m] - tg[:m], axis=1).mean()
    ri, cj = linear_sum_assignment(C)
    matches, errs = [], []
    matched_gt = set()
    for i, j in zip(ri, cj):
        ta, tg = ag[i][1], gt[j][1]
        m = min(len(ta), len(tg))
        per = np.linalg.norm(ta[:m] - tg[:m], axis=1)
        e = float(per.mean() / scene_scale)
        errs.append(e)
        matched_gt.add(j)
        matches.append({"agent": ag[i][0], "gt": gt[j][0],
                        "traj_err_norm": round(e, 4),
                        "traj_err_m": round(float(per.mean()), 3)})
    # missing GT movers: each counted with the worst err=1.0
    n_missing = n_gt - len(matched_gt)
    errs.extend([1.0] * n_missing)
    recall = len(matched_gt) / n_gt
    mover_count_err = abs(len(ag) - n_gt) / n_gt
    return {"worst_vehicle_err": round(max(errs), 4),
            "mean_vehicle_err": round(float(np.mean(errs)), 4),
            "movable_recall": round(recall, 3),
            "mover_count_err": round(mover_count_err, 3),
            "n_gt": n_gt, "n_agent": len(ag),
            "matches": matches}


def score_motion_descriptors(agent_trajs, gt_trajs, matches, scene_scale):
    """Task-agnostic motion diagnostics: main-axis direction + path shape + steering
    (heading) consistency."""
    per = []
    dir_err = 0
    n_dir = 0
    shape_errs = []
    heading_errs = []
    for mt in matches:
        Ap = agent_trajs[mt["agent"]]; Gp = gt_trajs[mt["gt"]]
        A = motion_descriptors(Ap, scene_scale)
        G = motion_descriptors(Gp, scene_scale)
        # direction: only counted when GT actually moves *and is not a closed loop*.
        # A closed loop (loop/circular, e.g. a train going around) has net displacement ~0,
        # so the PCA main-axis direction is unstable and meaningless.
        if G["moving"] and not G["closed"]:
            n_dir += 1
            av, gv = np.array(A["axis_dir"]), np.array(G["axis_dir"])
            cosang = float(np.clip(np.dot(av, gv), -1, 1)) if (av.any() and gv.any()) else 1.0
            ang_deg = float(np.degrees(np.arccos(cosang)))
            direction_match = ang_deg < 30.0
            if not direction_match:
                dir_err += 1
        else:
            ang_deg = 0.0
            direction_match = True
        shape = float(np.mean([
            abs(A["net_disp_n"] - G["net_disp_n"]),
            abs(A["path_len_n"] - G["path_len_n"]),
            abs(A["straightness"] - G["straightness"]),
            abs(A["vertical_range_n"] - G["vertical_range_n"]),
            abs(A["lateral_n"] - G["lateral_n"]),
        ]))
        shape_errs.append(shape)
        # steering (heading): compare agent vs GT "total turning" (how much it turned),
        # normalized to 2pi. Catches "position right but no steering" (e.g. agent slides
        # through a curve without turning the vehicle front). Only counted when GT moves.
        g_turn = _total_turning(Gp, scene_scale)
        a_turn = _total_turning(Ap, scene_scale)
        if G["moving"]:
            h_err = min(abs(a_turn - g_turn) / (2 * np.pi), 1.0)
            heading_errs.append(h_err)
        else:
            h_err = 0.0
        per.append({"gt": mt["gt"], "agent": mt["agent"],
                    "axis_angle_deg": round(ang_deg, 1),
                    "direction_match": direction_match,
                    "closed_match": A["closed"] == G["closed"],
                    "shape_err": round(shape, 4),
                    "heading_err": round(h_err, 4),
                    "gt_turning_deg": round(np.degrees(g_turn), 1),
                    "agent_turning_deg": round(np.degrees(a_turn), 1),
                    "gt_desc": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in G.items() if k != "axis_dir"},
                    "agent_desc": {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in A.items() if k != "axis_dir"}})
    return {
        "direction_error_rate": round(dir_err / n_dir, 3) if n_dir else 0.0,
        "path_shape_err": round(float(np.mean(shape_errs)), 4) if shape_errs else 1.0,
        "heading_err": round(float(np.mean(heading_errs)), 4) if heading_errs else 0.0,
        "per_vehicle": per,
    }


def _agent_static_centroids(agent_glb):
    """Subtree centroids of all *non-animated* top-level nodes in the agent glb (Blender
    frame). Animated parts (movers) are not counted as static. Returns (K,3)."""
    gltf, blob = _read_glb_json_bin(agent_glb)
    drives = _sample_animation(gltf, blob)
    nodes = gltf.get("nodes", [])
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(nodes))))
    driven = set(drives.keys())

    def subtree_has_driven(ni):
        if ni in driven:
            return True
        return any(subtree_has_driven(c) for c in nodes[ni].get("children", []))

    pts = []

    def walk(ni):
        # entire subtree has no animation -> take its centroid as one static object
        if not subtree_has_driven(ni):
            c = _subtree_centroid(gltf, blob, drives, ni, 0.0)
            if np.any(c):
                pts.append(_gltf_to_blender(c))
            return
        # subtree contains animation -> drill down, collect static child nodes separately
        for ch in nodes[ni].get("children", []):
            walk(ch)
    for r in roots:
        walk(r)
    return np.array(pts) if pts else np.zeros((0, 3))


def _static_scene_bbox(glb_path):
    """Measure the glb scene bbox size (Blender frame X,Y,Z), excluding the ground / large
    background slab. Used for "overall scene size" comparison —— if the agent builds
    buildings/objects too short/big, the overall bbox size (especially Z height) is off.
    Uses the t=0 static pose; it contains movers but their footprint is small and does not
    affect the overall size magnitude.

    Key point (task-agnostic): the agent often adds a 60x60 ground slab that blows up the
    bbox (most GT has no full-coverage ground). Collect per-node bboxes and treat a slab
    whose "XY span is far larger than the group median and Z is extremely thin" as ground
    and remove it, so the "scene size" convention is consistent between agent/GT (measuring
    the real distribution range of buildings/machinery, not the ground area)."""
    gltf, blob = _read_glb_json_bin(glb_path)
    drives = _sample_animation(gltf, blob)
    nodes = gltf.get("nodes", [])
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(nodes))))
    mesh_local = {}

    def local_pts(mi):
        if mi not in mesh_local:
            ps = [_accessor(gltf, blob, pr["attributes"]["POSITION"])
                  for pr in gltf["meshes"][mi]["primitives"]]
            mesh_local[mi] = np.vstack(ps) if ps else np.zeros((0, 3))
        return mesh_local[mi]

    # collect per-node world bbox (min, max), not the whole vertex pile -> easy to drop
    # per-node outliers
    boxes = []   # [(mn(3), mx(3))] Blender frame

    def dfs(ni, parentM):
        n = nodes[ni]
        M = parentM @ _node_matrix_at(n, ni, drives, 0.0)
        if "mesh" in n:
            v = local_pts(n["mesh"])
            if len(v):
                vh = np.c_[v, np.ones(len(v))]
                w = (M @ vh.T).T[:, :3]
                wb = _gltf_to_blender(w)
                boxes.append((wb.min(0), wb.max(0)))
        for ch in n.get("children", []):
            dfs(ch, M)
    for r in roots:
        dfs(r, np.eye(4))
    if not boxes:
        return np.zeros(3)

    # each node's XY diagonal span + Z span
    xy_span = np.array([math.hypot(mx[0]-mn[0], mx[1]-mn[1]) for mn, mx in boxes])
    z_span = np.array([mx[2]-mn[2] for mn, mx in boxes])
    keep = np.ones(len(boxes), dtype=bool)
    if len(boxes) >= 4:
        med_xy = float(np.median(xy_span))
        for i in range(len(boxes)):
            # ground-slab criterion: XY span > 5x group median, and Z extremely thin
            # (< 1/8 of XY) -> background ground
            if med_xy > 1e-6 and xy_span[i] > 5.0 * med_xy and z_span[i] < 0.125 * xy_span[i]:
                keep[i] = False
        if not keep.any():        # if everything is dropped (anomaly), do not drop, fall back to all
            keep[:] = True

    mns = np.array([boxes[i][0] for i in range(len(boxes)) if keep[i]])
    mxs = np.array([boxes[i][1] for i in range(len(boxes)) if keep[i]])
    return mxs.max(0) - mns.min(0)   # (sx, sy, sz) Blender


def score_size(agent_glb, gt_glb):
    """Overall scene size comparison: agent vs GT static scene bbox size (X,Y,Z).
    size_error = mean|log(agent_size / gt_size)| over the three axes, 0=size all correct.
    Catches size errors like "buildings too short / objects too big" that geometric metrics
    (which only look at centroid positions) miss."""
    if not gt_glb or not os.path.isfile(gt_glb):
        return {"size_error": None, "note": "no gt_scene glb"}
    a = _static_scene_bbox(agent_glb)
    g = _static_scene_bbox(gt_glb)
    errs = []
    per_axis = {}
    for i, ax in enumerate("xyz"):
        if g[i] > 0.1 and a[i] > 0.1:
            r = a[i] / g[i]
            errs.append(abs(math.log(r)))
            per_axis[ax] = {"agent": round(float(a[i]), 2),
                            "gt": round(float(g[i]), 2), "ratio": round(float(r), 2)}
    return {
        "size_error": round(float(np.mean(errs)), 4) if errs else None,
        "per_axis": per_axis,
    }


def _mover_bbox_sizes_blender(glb_path, names):
    """Use headless Blender to measure the world bbox size of the named objects (together
    with descendant meshes). Used for movers containing skin/armature (pure parsing handles
    skinned vertex transforms inaccurately and underestimates). Blender correctly computes
    the evaluated world bounding box of a skinned mesh. Returns {name: np.array([sx,sy,sz])}."""
    import subprocess, tempfile, json as _json, os as _os
    blender = _os.environ.get("BLENDER", "blender")
    script = '''
import bpy, sys, json
from mathutils import Vector
glb, names_json, out_json = sys.argv[sys.argv.index("--")+1:][:3]
names = json.load(open(names_json))
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for blk in (bpy.data.meshes, bpy.data.objects):
    for b in list(blk):
        try: blk.remove(b)
        except: pass
bpy.ops.import_scene.gltf(filepath=glb)
def descendants(o):
    r=[o]
    for c in bpy.data.objects:
        if c.parent is o: r+=descendants(c)
    return r
out={}
for nm in names:
    o=bpy.data.objects.get(nm)
    if o is None:
        # fuzzy: take the first one containing this name
        cand=[x for x in bpy.data.objects if nm in x.name]
        o=cand[0] if cand else None
    if o is None: continue
    mn=[1e9]*3; mx=[-1e9]*3; got=False
    for ob in descendants(o):
        if ob.type=='MESH':
            for v in ob.bound_box:
                w=ob.matrix_world @ Vector(v)
                for i in range(3): mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
            got=True
    if got: out[nm]=[mx[i]-mn[i] for i in range(3)]
json.dump(out, open(out_json,"w"))
'''
    with tempfile.TemporaryDirectory() as td:
        sp = _os.path.join(td, "s.py"); nj = _os.path.join(td, "n.json"); oj = _os.path.join(td, "o.json")
        open(sp, "w").write(script); _json.dump(list(names), open(nj, "w"))
        try:
            subprocess.run([blender, "--background", "--python", sp, "--", glb_path, nj, oj],
                           check=True, capture_output=True, timeout=180)
            raw = _json.load(open(oj))
            return {k: np.array(v, float) for k, v in raw.items()}
        except Exception:
            return {}


def _mover_bbox_sizes(glb_path, names):
    """Measure the subtree bbox size (Blender frame X,Y,Z) of the named nodes in the glb,
    at the t=0 static pose. Returns {name: np.array([sx,sy,sz])}. Used for per-mover size
    comparison (box size, car size). Exact match on node.name; missing ones are skipped.

    A glb with skin (skinned characters) is measured with Blender (pure parsing handles
    bone transforms inaccurately and underestimates); other rigid-body movers use pure
    parsing (fast, verified accurate)."""
    gltf, blob = _read_glb_json_bin(glb_path)
    if gltf.get("skins"):
        b = _mover_bbox_sizes_blender(glb_path, names)
        if b:
            return b
        # if Blender is unavailable, fall back to pure parsing (may underestimate, but some is better than none)
    drives = _sample_animation(gltf, blob)
    nodes = gltf.get("nodes", [])
    name2ni = {}
    for i, n in enumerate(nodes):
        nm = n.get("name")
        if nm and nm not in name2ni:
            name2ni[nm] = i
    mesh_local = {}

    def local_pts(mi):
        if mi not in mesh_local:
            ps = [_accessor(gltf, blob, pr["attributes"]["POSITION"])
                  for pr in gltf["meshes"][mi]["primitives"]]
            mesh_local[mi] = np.vstack(ps) if ps else np.zeros((0, 3))
        return mesh_local[mi]

    def subtree_bbox(root_ni):
        acc = []

        def dfs(ni, pM):
            n = nodes[ni]
            M = pM @ _node_matrix_at(n, ni, drives, 0.0)
            if "mesh" in n:
                v = local_pts(n["mesh"])
                if len(v):
                    vh = np.c_[v, np.ones(len(v))]
                    acc.append((M @ vh.T).T[:, :3])
            for ch in n.get("children", []):
                dfs(ch, M)
        dfs(root_ni, np.eye(4))
        if not acc:
            return None
        P = _gltf_to_blender(np.vstack(acc))
        return P.max(0) - P.min(0)

    out = {}
    for nm in names:
        ni = name2ni.get(nm)
        if ni is not None:
            d = subtree_bbox(ni)
            if d is not None:
                out[nm] = d
    return out


def score_mover_size(agent_glb, gt_scene_glb, matches):
    """Per-mover size comparison: for each matched mover pair, compare the agent vs GT bbox
    size on the three axes.
    mover_size_err = mean|log(agent_dim/gt_dim)| per mover, then averaged, 0=size all correct.
    Catches "agent builds all boxes the same size, missing GT's size diversity" —— completely
    invisible from centroid trajectories. GT sizes are measured from gt_scene.glb by mover
    name (Box_0..); agent from agent_glb by matched name."""
    if not gt_scene_glb or not os.path.isfile(gt_scene_glb):
        return {"mover_size_err": None, "note": "no gt_scene glb"}
    if not matches:
        return {"mover_size_err": None, "note": "no matched movers"}
    gt_names = [m["gt"] for m in matches]
    ag_names = [m["agent"] for m in matches]
    gt_sz = _mover_bbox_sizes(gt_scene_glb, gt_names)
    ag_sz = _mover_bbox_sizes(agent_glb, ag_names)
    per, errs = [], []
    for m in matches:
        g = gt_sz.get(m["gt"]); a = ag_sz.get(m["agent"])
        if g is None or a is None:
            continue
        axis_logs = []
        for i in range(3):
            if g[i] > 0.05 and a[i] > 0.05:
                axis_logs.append(abs(math.log(a[i] / g[i])))
        if not axis_logs:
            continue
        e = float(np.mean(axis_logs))
        errs.append(e)
        per.append({"gt": m["gt"], "agent": m["agent"], "size_err": round(e, 4),
                    "agent_dim": [round(float(x), 2) for x in a],
                    "gt_dim": [round(float(x), 2) for x in g]})
    return {
        "mover_size_err": round(float(np.mean(errs)), 4) if errs else None,
        "per_mover": per,
    }


def score_layout(agent_glb, layout_gt, scene_scale=40.0, agent_shift=(0.0, 0.0, 0.0)):
    """Static layout score (task-agnostic): whole point-cloud bidirectional Chamfer + object
    count error. No classification (data constraint: the layout-type vocabulary != component
    file names, so cross-agent/GT category matching is unreliable). Uses each object's
    location in layout_gt as GT points, and the agent's non-animated top-level node centroids
    as agent points. agent_shift: the same translation as the trajectory, applied to the agent
    static points so both main scores share the reference frame (absorbs absolute-origin
    ambiguity, but does not absorb scale —— building too big/small still shows in the Chamfer).
    Compatible with both objects / static_objects keys in layout_gt."""
    objs = (layout_gt or {}).get("objects") or (layout_gt or {}).get("static_objects") or []
    # compatible field names: location (most) / loc (crossroad)
    G = np.array([(o.get("location") or o.get("loc"))[:3] for o in objs
                  if o.get("location") or o.get("loc")])
    A = _agent_static_centroids(agent_glb)
    if len(A):
        A = A + np.asarray(agent_shift, float)   # same translation as the trajectory
    n_gt, n_ag = len(G), len(A)
    if n_gt == 0:
        return {"layout_err": None, "note": "no GT layout objects"}
    count_err = abs(n_ag - n_gt) / n_gt
    if n_ag == 0:
        return {"layout_err": 1.0, "chamfer_norm": 1.0,
                "gt_count": n_gt, "agent_count": 0,
                "count_ratio": 0.0, "layout_count_err": 1.0}
    d_a = cKDTree(G).query(A)[0].mean()
    d_g = cKDTree(A).query(G)[0].mean()
    chamfer = float((d_a + d_g) / 2 / scene_scale)
    return {
        "layout_err": round(chamfer, 4),
        "chamfer_norm": round(chamfer, 4),
        "gt_count": n_gt, "agent_count": n_ag,
        "count_ratio": round(n_ag / n_gt, 2),
        "layout_count_err": round(count_err, 3),
    }


def evaluate_t6(agent_glb, sample_dir):
    """T6 main scoring entry (task-agnostic).
    agent_glb: the agent-exported whole-scene glb with animation.
    sample_dir: benchmark_t6_final/<sid>/ (contains gt/trajectory.json, layout_gt.json, meta.json).
    GT ground truth = gt/trajectory.json (per-frame positions, uniform format)."""
    import os
    meta = json.load(open(os.path.join(sample_dir, "meta.json")))
    n_frames = meta.get("n_frames", 144)
    fps = meta.get("fps", 24)

    # GT trajectory (ground truth)
    traj_json = os.path.join(sample_dir, "gt", "trajectory.json")
    if not os.path.isfile(traj_json):
        return {"error": f"missing GT trajectory: {traj_json}"}
    gt_trajs = load_gt_trajectories(traj_json, n_frames)

    # static layout GT
    layout_gt = {}
    lg = os.path.join(sample_dir, "layout_gt.json")
    if os.path.isfile(lg):
        layout_gt = json.load(open(lg))

    scene_scale = compute_scene_scale(layout_gt, gt_trajs)

    # agent trajectory
    agent_trajs_raw = extract_agent_trajectories(agent_glb, fps, n_frames)
    # global alignment: only apply XY translation (absorb absolute-origin ambiguity), no scale/rotation.
    #   scale: the prompt gives a reference camera + native component sizes, building too big/small is a
    #          genuine error -> penalize, output scale_error as a diagnostic.
    #   rotation: pinned by the prompt basis convention (main axis +X); applying rotation would miss the
    #          rotation error of symmetric scenes.
    agent_trajs, s_est, shift = align_xy_translation(agent_trajs_raw, gt_trajs)
    scale_error = round(abs(math.log(s_est)), 4) if s_est > 0 else None

    traj = score_trajectory(agent_trajs, gt_trajs, scene_scale)
    sem = score_motion_descriptors(agent_trajs, gt_trajs, traj["matches"], scene_scale)
    try:
        # layout uses the same translation as the trajectory (same reference frame), no scale
        layout = score_layout(agent_glb, layout_gt, scene_scale, agent_shift=shift)
    except Exception as e:
        layout = {"layout_err": None, "error": repr(e)}
    # static scene size comparison (catches "buildings too short / objects too big" —— geometric
    # metrics that only look at centroid position miss it)
    gt_scene_glb = os.path.join(sample_dir, "gt", "gt_scene.glb")
    if not os.path.isfile(gt_scene_glb):
        gt_scene_glb = os.path.join(sample_dir, "gt_scene.glb")
    try:
        size = score_size(agent_glb, gt_scene_glb)
    except Exception as e:
        size = {"size_error": None, "error": repr(e)}
    # per-mover size (catches "all boxes built the same size, missing GT size diversity")
    try:
        mover_size = score_mover_size(agent_glb, gt_scene_glb, traj["matches"])
    except Exception as e:
        mover_size = {"mover_size_err": None, "error": repr(e)}

    return {
        "sample": meta.get("sample_id"),
        "scene_scale": round(scene_scale, 3),
        # main scores + diagnostics flattened to the top level (for run_io HEADLINE_SPEC / _mean_at)
        "worst_vehicle_err": traj["worst_vehicle_err"],
        "mean_vehicle_err": traj["mean_vehicle_err"],
        "movable_recall": traj["movable_recall"],
        "mover_count_err": traj["mover_count_err"],
        "direction_error_rate": sem["direction_error_rate"],
        "path_shape_err": sem["path_shape_err"],
        "heading_err": sem["heading_err"],     # whether mover steering (how much it turned) is reproduced, 0=correct
        "scale_error": scale_error,        # mover trajectory scale |log|: 0=correct
        "size_error": size.get("size_error"),  # static scene bbox size |log|: catches buildings too short / objects too big
        "mover_size_err": mover_size.get("mover_size_err"),  # per-mover size |log|: catches all boxes same size
        "layout_err": layout.get("layout_err"),
        # layout_count_err stays nested in layout as a diagnostic (granularity: agent glb subtree count != GT object count, not in main score)
        # full nested diagnostics
        "size": size,
        "mover_size": mover_size,
        "trajectory": traj,
        "semantic": sem,
        "layout": layout,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        r = evaluate_t6(sys.argv[1], sys.argv[2])
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("usage: metrics_t6.py <agent_glb> <sample_dir>")
