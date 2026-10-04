"""
metrics_anim.py -- Task 4 Articulated Animation scoring (keyframes, GT-free, verified).

Task: the agent sees an anonymized input.glb (mesh names carry no semantics) + a
      reference open/close video (reference frames), reproduces the open/close animation
      in Blender, and exports it frame by frame as glb. The agent reports no joint
      parameters and cannot read any GT.

Design basis: METRICS_TASK4_ANIMATION.md section 10 (fundamental revision).
  - agent deliverable = per-frame glb (the moved geometry); the geometry itself is the
    answer. **Joint parameters are not scored** (Axis/Pivot/Direction/Range removed).
  - part assignment uses the *scoring-side-private* _scoring_partmap.json (vertex index
    -> pid), invisible to the agent.
      vertex-order aligned (the agent drives with our glb and preserves vertex order;
      measured Blender round-trip maxdiff=0) -> direct indexing, exact.
      vertices reordered / count changed -> fall back to *spatial assignment* (build a
      KDTree on the GT closed frame for nearest-neighbor voting, still scoring-side-private GT).
  - frame count not enforced: **keyframe semantic alignment** (each side finds its most
    closed/most open frame, not relying on frame number) handles agent frame count != GT.
  - three main-board items are pure geometry (section 9.2/10.4, five-sample + full-library verified):
      (1) per-part local ADD-S (keyframe full-open + closed) -- main, captures motion fidelity.
      (2) global ADD-S/Chamfer (keyframe full-open)     -- auxiliary, captures "moving part
          drifting out of place" (more sensitive than local for tall/short objects).
      (3) mover recall / false-positive rate            -- captures missed/extra detections
          (geometry dilutes them, must be reported independently).

GT source: benchmark_s2o_final/<sample>/{meta.json, _scoring_partmap.json, gt/gt_XXXX.glb}
Prediction source: the agent's per-frame glb directory (agent_frame_XXXX.glb).

Dependencies: numpy scipy
"""
import json, os, struct, glob
import numpy as np
from scipy.spatial import cKDTree


# ---------- glb vertex parsing (does not go through the Blender importer, guaranteeing coords = the file's original frame) ----------

def _node_matrix(node):
    """The gltf node's local 4x4 matrix (column-major matrix or TRS)."""
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T  # gltf column-major -> row-major
    M = np.eye(4)
    if "scale" in node:
        S = np.diag(node["scale"] + [1.0])
        M = M @ S
    if "rotation" in node:  # quaternion [x,y,z,w]
        x, y, z, w = node["rotation"]
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w),     0],
            [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w),     0],
            [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y), 0],
            [0, 0, 0, 1]], float)
        M = R @ M  # note: T*R*S order, here scale first then rotation
    if "translation" in node:
        T = np.eye(4); T[:3, 3] = node["translation"]
        M = T @ M
    return M


def glb_verts(path):
    """Parse glb -> (N,3) all vertices. **Accumulate node world transforms along the gltf
    scene/node tree**, so motion is read correctly whether it is baked into vertices (GT
    frames) or stored as a node transform (agent export).
    Order: depth-first traversal, consistent with the mesh appearance order (to match the
    private partmap vertex indices)."""
    data = open(path, "rb").read()
    clen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + clen].decode("utf-8", "replace"))
    blob = data[20 + clen + 8:]
    nodes = gltf.get("nodes", [])

    def read_mesh(mesh_idx, world):
        chunks = []
        for pr in gltf["meshes"][mesh_idx]["primitives"]:
            a = gltf["accessors"][pr["attributes"]["POSITION"]]
            bv = gltf["bufferViews"][a["bufferView"]]
            off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            v = np.frombuffer(blob, np.float32, a["count"] * 3, off).reshape(-1, 3).astype(np.float64)
            vh = np.c_[v, np.ones(len(v))]                  # homogeneous
            chunks.append((world @ vh.T).T[:, :3])
        return chunks

    out = []
    # if there is a node tree, DFS-accumulate transforms from the scene roots; otherwise degrade to reading meshes directly (local)
    if nodes:
        scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
        roots = scene.get("nodes", list(range(len(nodes))))
        # but mesh order must match gltf["meshes"] -> use (mesh_idx -> that node's world matrix)
        mesh_world = {}

        def dfs(ni, parent):
            n = nodes[ni]
            world = parent @ _node_matrix(n)
            if "mesh" in n:
                mesh_world[n["mesh"]] = world
            for ch in n.get("children", []):
                dfs(ch, world)
        for ri in roots:
            dfs(ri, np.eye(4))
        # output in mesh-index order (consistent with the partmap segment order)
        for mi in range(len(gltf["meshes"])):
            w = mesh_world.get(mi, np.eye(4))
            out.extend(read_mesh(mi, w))
    else:
        for mi in range(len(gltf["meshes"])):
            out.extend(read_mesh(mi, np.eye(4)))
    return np.vstack(out) if out else np.zeros((0, 3))


def add_s(P, Q):
    """Mean one-way nearest-point distance (meters): each point of P to its nearest point
    in Q. Symmetric-tolerant (GT same-shape point cloud)."""
    if len(P) == 0 or len(Q) == 0:
        return float("nan")
    return float(cKDTree(Q).query(P)[0].mean())


def chamfer(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return float("nan")
    return float(cKDTree(Q).query(P)[0].mean() + cKDTree(P).query(Q)[0].mean())


def _fit_rigid(P, Q):
    """SVD fit of the rigid transform P->Q, returns (rotation angle, signed rotation axis,
    centroid displacement). Angle near 0 -> pure translation; translation magnitude near 0
    -> pure rotation; both significant -> mixed. The rotation axis is extracted from the
    antisymmetric matrix R-R^T, preserving direction sign (right-hand rule), not collapsed
    by SVD."""
    if len(P) < 3 or len(P) != len(Q):
        return 0.0, np.zeros(3), np.zeros(3)
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    cosa = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang_deg = float(np.degrees(np.arccos(cosa)))
    # extract axis from the antisymmetric part (right-hand rule, direction preserved): vec(R - R^T) = 2 sin(theta) . axis
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(ax))
    axn = (ax / n) if n > 1e-9 else np.zeros(3)
    return ang_deg, axn, (cQ - cP)


def _diagnose_joint(a_close, a_open, g_close, g_open, gt_joint):
    """Diagnose the agent joint vs GT truth: type consistency + axis/direction consistency.
    gt_joint: meta['joints'][pid] dict, containing type, axis_world, rangeMax, etc.
    Key point: use the "actual motion axis" fit from the GT frames themselves as truth,
    not meta's axis_world (which may be inconsistent with the GT frames).

    Bug-fix(2026-06-19): when geometry cov is high but SVD solves a small angle/small
    displacement, do not directly judge static -- use the geometric displacement magnitude
    as a fallback (ratio of max vertex displacement to GT full displacement) to judge
    whether the agent moved.
    mixed is also no longer directly judged type_mismatch: if the main component agrees with GT it counts as a match.
    """
    a_ang, a_ax, a_t = _fit_rigid(a_close, a_open)
    g_ang, g_ax, g_t = _fit_rigid(g_close, g_open)
    a_t_norm = float(np.linalg.norm(a_t))
    g_t_norm = float(np.linalg.norm(g_t))
    gt_type = gt_joint["type"]
    rng = gt_joint.get("rangeMax") or 0.5

    # geometric displacement magnitude: whether the agent actually moved (independent of the SVD-fit angle/displacement)
    # use the vertex-level displacement distribution -- vertex mean displacement / vertex max displacement, both compared to the same GT part
    if len(a_close) == len(a_open) and len(g_close) == len(g_open) and len(a_close) > 0:
        a_disp_mean = float(np.linalg.norm(a_open - a_close, axis=1).mean())
        g_disp_mean = float(np.linalg.norm(g_open - g_close, axis=1).mean())
    else:
        a_disp_mean = (float(np.linalg.norm(a_open.mean(0) - a_close.mean(0)))
                       if len(a_close) and len(a_open) else 0.0)
        g_disp_mean = (float(np.linalg.norm(g_open.mean(0) - g_close.mean(0)))
                       if len(g_close) and len(g_open) else 0.0)
    # geom_moved: an agent vertex displacement >= 25% of the GT vertex displacement counts as having performed an action
    geom_moved = a_disp_mean > max(0.05, 0.25 * g_disp_mean)

    # infer the motion type the agent actually did (purely geometric, not relying on GT)
    # Bug-fix(2026-06-19): a rotation about an axis far from the centroid is equivalent to "rotation + large translation"; the SVD
    #   |t| is almost certainly above threshold, and the old logic directly judged mixed -> almost all hinge rotations judged mixed -> false positives.
    #   Fix: judge rotation only by angle, translation only by displacement; when they overlap, decide by GT type and do not count as mixed.
    has_rot = a_ang > 15.0
    has_trans = a_t_norm > 0.1 * (rng if gt_type == "translation" else 0.5)
    if has_rot:
        # rotation-dominated: even if |t| is large, that is a by-product of rotating about an axis far from the centroid, not an independent translation
        agent_type = "rotation"
    elif has_trans:
        agent_type = "translation"
    elif geom_moved:
        # SVD solves double-zero, but geometry did move -- the agent did non-rigid / reverse deformation / fragment motion
        agent_type = "non_rigid"
    else:
        agent_type = "static"

    # type_match rules:
    #  - agent_type == gt_type             -> True
    #  - mixed: the agent made a pure type "mixed" (GT is pure translation but the agent also rotated / vice versa)
    #           -> mismatch (human feel: "should pull out but flipped up", joint kinematics wrong)
    #  - non_rigid: SVD cannot fit a rigid body, but agent vertices moved. May be "over-range / clipping / fragments",
    #               not necessarily a joint-type error (in many cases it "moved but the pose is bad"). Not counted as mismatch,
    #               instead reflected by the geometric main score (worst_part_err) for "moved correctly or not".
    #  - static                            -> this part performed no action, reflected by recall, not counted as mismatch
    type_match = (agent_type == gt_type) or (agent_type in ("non_rigid", "static"))

    # axis/direction consistency: use the actual motion direction fit from the GT frames as truth, do not trust meta (may be reversed)
    # Bug-fix(2026-06-19): translation uses the centroid displacement vector (most sensitive to reversal, not confused by SVD
    #   rotation/translation), no longer relies on a_t (which is the SVD-fit "displacement component" and in reversed scenes is
    #   wrongly attributed to rotation, making |a_t| too small).
    if gt_type == "rotation":
        gt_truth_dir = g_ax           # rotation axis fit from the GT frames (signed)
        cos = float(np.dot(a_ax, gt_truth_dir)) if a_ang > 1.0 and np.linalg.norm(gt_truth_dir) > 0.5 else 0.0
        gt_motion = g_ang
        agent_motion = a_ang
    else:  # translation: directly use the centroid displacement vector, not the SVD a_t
        if len(a_close) and len(a_open):
            a_disp_vec = a_open.mean(0) - a_close.mean(0)
        else:
            a_disp_vec = np.zeros(3)
        if len(g_close) and len(g_open):
            g_disp_vec = g_open.mean(0) - g_close.mean(0)
        else:
            g_disp_vec = np.zeros(3)
        a_dn = float(np.linalg.norm(a_disp_vec))
        g_dn = float(np.linalg.norm(g_disp_vec))
        cos = float(np.dot(a_disp_vec / (a_dn + 1e-9),
                           g_disp_vec / (g_dn + 1e-9))) if a_dn > 1e-3 and g_dn > 1e-3 else 0.0
        gt_motion = g_dn
        agent_motion = a_dn
    dir_sign = "same" if cos > 0.7 else ("reverse" if cos < -0.7 else "off-axis")

    return {
        "gt_type": gt_type,
        "agent_type_inferred": agent_type,
        "type_match": bool(type_match),
        "geom_moved": bool(geom_moved),                # vertex-level displacement judgment: whether the agent actually moved
        "axis_cos": round(cos, 3),                     # cosine with the GT frames' actual direction
        "dir_sign": dir_sign,
        "gt_motion": round(gt_motion, 3),
        "agent_motion": round(agent_motion, 3),
        "agent_rot_deg": round(a_ang, 1),
        "agent_trans_m": round(a_t_norm, 3),
        "agent_disp_mean": round(a_disp_mean, 3),
        "gt_disp_mean": round(g_disp_mean, 3),
    }


class AnimScorer:
    """Scoring side: holds GT + the private partmap. The agent only submits per-frame vertices."""

    def __init__(self, sample_dir):
        self.dir = sample_dir
        self.meta = json.load(open(os.path.join(sample_dir, "meta.json")))
        self.n = self.meta["n_frames"]
        pm = json.load(open(os.path.join(sample_dir, "_scoring_partmap.json")))
        self.total_verts = pm["total_verts"]
        self.lab = np.empty(self.total_verts, int)          # vertex index -> pid (private)
        for s in pm["segments"]:
            self.lab[s["v_start"]:s["v_start"] + s["v_count"]] = s["pid"]
        self.movable = sorted({s["pid"] for s in pm["segments"] if s["movable"]})
        # GT per-frame vertices (file's original frame)
        self.gt = [glb_verts(os.path.join(sample_dir, "gt", f"gt_{t:04d}.glb"))
                   for t in range(self.n)]
        self.gt_close = self.gt[0]
        self.gt_open = self.gt[self.n // 2]                 # GT full-open = the middle frame (verified open to rangeMax)
        self._sp_tree = cKDTree(self.gt_close)              # for spatial assignment

    # ---- part assignment ----
    def _labels_for(self, V):
        """Return the pid of each vertex of V. If vertex order is aligned, use the private
        index; otherwise nearest-neighbor spatial vote."""
        if len(V) == self.total_verts:
            return self.lab                                 # vertex order aligned (exact)
        _, idx = self._sp_tree.query(V)                     # reordered / count changed -> spatial assignment
        return self.lab[idx]

    # ---- keyframe semantic alignment (not by frame number) ----
    def _keyframes(self, frames):
        """Each side finds its 'most full-open' / 'most closed' frame: openness is defined
        by the mean displacement relative to the first frame."""
        base = frames[0]
        disp = []
        for f in frames:
            if len(f) == len(base):
                disp.append(float(np.linalg.norm(f - base, axis=1).mean()))
            else:
                disp.append(float(self._sp_tree.query(f)[0].mean()))
        return int(np.argmax(disp)), int(np.argmin(disp))   # (full-open idx, closed idx)

    # ---- main scoring ----
    def score(self, agent_frames):
        """
        agent_frames: list[np.ndarray(Ni,3)]  agent per-frame whole-object vertices (file's original frame).

        Core: leverage that input.glb and the agent-exported glb vertices correspond one-to-one
              (the agent drives with our glb and preserves vertex order), using *per-vertex L2*
              (each vertex vs its corresponding GT vertex) rather than ADD-S (nearest point) --
              ADD-S has blind spots for thin plates/symmetric parts (door flip / in-plane rotation
              are undetectable; see the metric doc for measurements), while per-vertex L2 knows
              "where each point should go" and has no symmetry blind spot.
        Reported per-part, main score = worst-part (worst moving part), not averaged -- missing
        one moving part must not be diluted by good parts.
        If the agent reorders vertices (vertex-order mismatch), fall back to ADD-S + spatial
        assignment (has blind spots, marked degraded=True).
        """
        if not agent_frames:
            return {"error": "no agent frames"}
        m = len(agent_frames)
        pk_a, cl_a = self._keyframes(agent_frames)

        aligned = (len(agent_frames[pk_a]) == self.total_verts
                   and len(agent_frames[cl_a]) == self.total_verts)

        def err_part(Pverts, Gverts):
            """Per-vertex L2 (vertex aligned) or degrade to symmetric ADD-S."""
            if aligned and len(Pverts) == len(Gverts):
                return float(np.linalg.norm(Pverts - Gverts, axis=1).mean())
            return max(add_s(Pverts, Gverts), add_s(Gverts, Pverts))

        la_open = self.lab if aligned else self._labels_for(agent_frames[pk_a])
        la_close = self.lab if aligned else self._labels_for(agent_frames[cl_a])

        # ---- align by "openness" (removing timing/speed/frame-count differences), catch pose errors + use coverage to catch not-opened-far-enough ----
        # core distinction (answers "what if the agent's frame != the same-progress GT but ~= another GT frame"):
        #   do not align by time/frame progress, align by *openness* -- each agent frame finds the GT frame with the *same openness* to compare geometry.
        #   (1) timing difference (fast/slow/phase/different frame count, but all poses traversed are correct) -> every frame matches a same-openness GT -> L2~=0, not penalized.
        #   (2) pose error (random turning in the middle / over-opening = a pose GT does not have at that openness) -> large L2, penalized.
        #   (3) not opened far enough (only half open) -> coverage (agent max openness) < 1, penalized.
        # openness = the part centroid's displacement relative to closed / GT full displacement.
        per_part = {}
        for pid in self.movable:
            mask = self.lab == pid
            gv_close = self.gt_close[mask]
            gv_open = self.gt_open[mask]
            # Openness definition changed to "vertex displacement distribution" (also sensitive to a door rotating about its own edge):
            #   each frame op = mean( ||this frame's vertices - its own closed-frame vertices|| ) / GT full-open of the same quantity
            # The old "centroid displacement" definition fails for rotation about the part's own axis (centroid does not move but openness is already large).
            g_full = max(float(np.linalg.norm(gv_open - gv_close, axis=1).mean()), 1e-6)
            gt_half = list(range(0, self.n // 2 + 1))
            def _vert_open(verts):
                return float(np.linalg.norm(verts - gv_close, axis=1).mean()) / g_full
            gt_op = [(_vert_open(self.gt[g][mask]), g) for g in gt_half]
            # agent closed-frame vertices of this part (as the baseline of the agent's own openness)
            la_close_v = self.lab if aligned else self._labels_for(agent_frames[cl_a])
            av_close = agent_frames[cl_a][la_close_v == pid]

            def nearest_gt(op):
                return min(gt_op, key=lambda t: abs(t[0] - min(max(op, 0.0), 1.0)))[1]

            # each agent frame -> its own openness (mean vertex displacement); find the same-openness GT frame and compare per-vertex L2
            pose_errs, max_open = [], 0.0
            for ai in range(len(agent_frames)):
                la = self.lab if aligned else self._labels_for(agent_frames[ai])
                ap = agent_frames[ai][la == pid]
                if len(ap) == len(av_close):
                    op = float(np.linalg.norm(ap - av_close, axis=1).mean())
                    op = op / g_full if g_full > 0 else 0.0
                elif len(ap) > 0 and len(av_close) > 0:
                    op = float(np.linalg.norm(ap.mean(0) - av_close.mean(0))) / g_full
                else:
                    op = 0.0
                max_open = max(max_open, op)
                pose_errs.append(err_part(ap, self.gt[nearest_gt(op)][mask]))
            pose_err = float(np.nanmax(pose_errs))          # max error over the poses traversed (random turning / over-opening)
            coverage = min(max_open, 1.0)                    # coverage (whether it reached full-open)
            cover_deficit = (1.0 - coverage) * g_full        # gap of not opening far enough (meters)
            err = max(pose_err, cover_deficit)               # part error

            eo = err_part(agent_frames[pk_a][la_open == pid], gv_open)
            ec = err_part(agent_frames[cl_a][la_close == pid], gv_close)
            a_disp = float(np.linalg.norm(agent_frames[pk_a][la_open == pid].mean(0)
                                          - av_close.mean(0))) if (la_open == pid).any() else 0.0
            moved = bool(g_full > 1e-3 and a_disp > 0.2 * g_full)
            # joint diagnosis: whether type/direction match the GT truth
            diag = {}
            try:
                gt_joint = self.meta.get("joints", {}).get(str(pid))
                if gt_joint:
                    diag = _diagnose_joint(
                        agent_frames[cl_a][la_close == pid],
                        agent_frames[pk_a][la_open == pid],
                        gv_close, gv_open, gt_joint)
            except Exception:
                diag = {}
            per_part[pid] = {"err_open": round(eo, 4), "err_close": round(ec, 4),
                             "pose_err": round(pose_err, 4),       # pose error over poses traversed (random turning / over-opening)
                             "coverage": round(coverage, 3),       # openness coverage (1=opened to full-open)
                             "err": round(err, 4),                 # part error
                             "agent_disp": round(a_disp, 3), "gt_disp": round(g_full, 3),
                             "moved": moved,
                             **({"joint_diag": diag} if diag else {})}

        part_errs = [v["err"] for v in per_part.values()
                     if v.get("err") is not None and not np.isnan(v["err"])]
        # main score: worst-part (worst moving part) -- misses / a fully-wrong part are not diluted
        # defense: GT has no movable parts / all nan -> main score None (skipped in aggregation), do not return nan
        worst_part_err = float(np.max(part_errs)) if part_errs else None
        mean_part_err = float(np.mean(part_errs)) if part_errs else None

        # mover recall (proportion of movers that moved into place) + false positives (spuriously moving static parts)
        recall = sum(v["moved"] for v in per_part.values()) / max(len(self.movable), 1)
        # joint-diagnosis summary: cross-mover statistics of type-mismatch rate / reverse rate
        # Bug-fix(2026-06-19): the denominator of mismatch_rate / reverse_dir_rate counts only parts where the agent
        #   "actually performed a rigid action" (geom_moved=True). Parts that did not move should be reflected in recall, not
        #   counted as type errors; otherwise a sample with 8 unmoved parts has mismatch=1.0, redundant info that crushes the recall signal.
        diag_total = 0; type_mis = 0; reverse_dir = 0
        for v in per_part.values():
            d = v.get("joint_diag")
            if d and d.get("geom_moved"):
                diag_total += 1
                if not d["type_match"]: type_mis += 1
                if d["dir_sign"] == "reverse": reverse_dir += 1
        type_mismatch_rate = type_mis / max(diag_total, 1) if diag_total else 0.0
        reverse_dir_rate = reverse_dir / max(diag_total, 1) if diag_total else 0.0
        static_pids = sorted(set(self.lab.tolist()) - set(self.movable))
        false_moved = 0
        # Bug-fix(2026-06-19): on a static part (e.g. base), a small clump of vertices is wrongly moved by the agent (typical:
        #   a "divider" inside the base misidentified as movable), the overall centroid barely moves, but local vertex displacement
        #   is large. Averaging over the centroid dilutes it. Use the top-1% mean of vertex displacement -> catches "a few vertices
        #   moving violently" without being dominated by a single-point outlier.
        #   When vertex counts do not match, use the top-1% mean of the close->open bidirectional nearest-neighbor distance.
        for pid in static_pids:
            if not (la_open == pid).any() or not (la_close == pid).any():
                continue
            ap_o = agent_frames[pk_a][la_open == pid]
            ap_c = agent_frames[cl_a][la_close == pid]
            if len(ap_o) == len(ap_c) and len(ap_o) > 0:
                disps = np.linalg.norm(ap_o - ap_c, axis=1)
            elif len(ap_o) > 0 and len(ap_c) > 0:
                # vertex count mismatch -> close vertices find open nearest neighbor + open vertices find close nearest neighbor, max
                d1, _ = cKDTree(ap_o).query(ap_c)
                d2, _ = cKDTree(ap_c).query(ap_o)
                disps = np.concatenate([d1, d2])
            else:
                disps = np.array([0.0])
            # top-1% mean: catches "a small local clump of vertices wrongly moved", robust to single-point outliers
            if len(disps) >= 100:
                k = max(1, len(disps) // 100)
                hi_disp = float(np.partition(disps, -k)[-k:].mean())
            else:
                hi_disp = float(disps.max()) if len(disps) else 0.0
            if hi_disp > 0.05:
                false_moved += 1
        false_rate = false_moved / max(len(static_pids), 1) if static_pids else 0.0

        # global (auxiliary reference)
        global_open = add_s(agent_frames[pk_a], self.gt_open)

        return {
            "sample": self.meta["sample_id"],
            "category": self.meta.get("category"),
            "n_gt_frames": self.n, "n_agent_frames": m,
            "vert_aligned": aligned,
            "degraded_to_adds": not aligned,     # True=vertex mismatch, degraded to ADD-S (has blind spots)
            "agent_keyframes": {"open": pk_a, "close": cl_a},
            # main metric: per-vertex L2, per-part
            "worst_part_err": (round(worst_part_err, 4) if worst_part_err is not None else None),   # main score
            "mean_part_err": (round(mean_part_err, 4) if mean_part_err is not None else None),
            "per_part": {str(k): v for k, v in per_part.items()},
            # mover identification
            "movable_recall": round(recall, 3),
            "false_move_rate": round(false_rate, 3),
            "type_mismatch_rate": round(type_mismatch_rate, 3),
            "reverse_dir_rate": round(reverse_dir_rate, 3),
            # auxiliary
            "global_adds_open": round(global_open, 4),
        }


# ---------- convenience entry: score from the agent's per-frame glb directory ----------

def score_animation_from_dir(sample_dir, agent_frames_dir, pattern="agent_frame_*.glb"):
    """Read the agent per-frame glb (sorted by file name) and score."""
    scorer = AnimScorer(sample_dir)
    files = sorted(glob.glob(os.path.join(agent_frames_dir, pattern)))
    if not files:
        return {"error": f"no agent frames in {agent_frames_dir} ({pattern})"}
    frames = [glb_verts(f) for f in files]
    out = scorer.score(frames)
    out["agent_frames_dir"] = agent_frames_dir
    out["n_agent_files"] = len(files)
    return out


def score_animation(sample_dir, pred):
    """Compatibility entry: pred = {"frames": [ndarray...]} or {"frames_dir": path}."""
    if "frames_dir" in pred:
        return score_animation_from_dir(sample_dir, pred["frames_dir"],
                                        pred.get("pattern", "agent_frame_*.glb"))
    scorer = AnimScorer(sample_dir)
    return scorer.score(pred["frames"])
