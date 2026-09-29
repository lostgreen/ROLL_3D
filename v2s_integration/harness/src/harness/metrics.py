"""
metrics.py -- scoring for the GLB placement benchmark (Task 1 & 2 layout tasks).

Core (validated on real data, see BENCHMARK_DATA_AND_METRIC.md sec 4.18):
  - use the exact vertex correspondence of the GT mesh, with ADD-S (the gold
    standard for pose estimation) rather than bbox-IoU
  - primary metrics: mean ADD-S (per object) + Chamfer (scene level, match-free)
  - object matching: Hungarian (cost = ADD-S), scheme (2) free agent placement
  - diagnostics: PE/RE; success rate: ADD-S < 10% of object diameter, Scene Success

Ground truth source: the scene directory's gt.json (own-frame vertices + layout
poses) + own-frame glb vertices.
Prediction source: after the agent places objects, read each object_NN's world
vertices from Blender (execute_blender_code).

Coordinate frame: unified Blender Z-up. GT object world vertices = own-frame
vertices transformed by (rotate about Z) + (translate lay2bl(location)).
lay2bl(x,y,z)=(x,-z,y).

Dependencies: numpy scipy trimesh
"""
import json
import os
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

try:
    import trimesh
except Exception:
    trimesh = None


# ---------- vertex reading ----------

def canonical_world_verts(scene_dir, gt_obj, sample=4000, rng=None):
    """GT object world vertices (Blender Z-up), exactly matching the Blender-side
    setup + placement.

    The own-frame glb is in a neutral orientation (build: centered + rotate about
    Y by -rotation + xs, trimesh Y-up).
    Restoring to the scene pose = Blender import (auto Y->Z up) + rotate about Z by
    rotation + translate.
    Equivalent numpy implementation (verified own-frame rotate-about-Y + rot ==
    original scene glb orientation):
      1. trimesh vertices (Y-up) Y->Z up: (x,y,z)->(x,-z,y)
      2. rotate about Blender Z by rotation
      3. translate lay2bl(location) = (lx, -lz, ly)
    """
    rng = rng or np.random.default_rng(0)
    path = os.path.join(scene_dir, "canonical", gt_obj["anon_id"] + ".glb")
    scene = trimesh.load(path)
    merged = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    V = np.asarray(merged.vertices, dtype=np.float64)   # own-frame, trimesh Y-up
    V = np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])   # Y-up -> Z-up (matches Blender import, measured ADD-S=0)
    # GT placement = Blender matrix_world = Translation(lay2bl(loc)) @ Rotation(yaw, 'Z').
    # numpy replica: first y2z, then rotate about Z by +yaw, then translate. Measured
    # per-point ADD-S=0 against correct Blender placement.
    # Note: the Blender side must place with obj.matrix_world = T @ R; obj.rotation_euler
    # does not take effect in the MCP execute_blender_code context (depsgraph not
    # refreshed) (known pitfall).
    yaw = gt_obj["rotation"]
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # about Blender Z (+yaw)
    V = V @ Rz.T
    loc = gt_obj["location"]
    V = V + np.array([loc[0], -loc[2], loc[1]])         # lay2bl translation
    if len(V) > sample:
        V = V[rng.choice(len(V), sample, False)]
    return V


# bpy code: export the world vertices of every mesh in the scene (downsampled), return JSON
READ_VERTS_CODE = r"""
import bpy, json, random
random.seed(0)
SAMPLE = {sample}
out = {{}}
for ob in bpy.data.objects:
    if ob.type != 'MESH':
        continue
    mw = ob.matrix_world
    vs = ob.data.vertices
    idx = range(len(vs))
    if len(vs) > SAMPLE:
        idx = random.sample(range(len(vs)), SAMPLE)
    pts = []
    for i in idx:
        co = mw @ vs[i].co
        pts.append([round(co.x, 5), round(co.y, 5), round(co.z, 5)])
    out[ob.name] = pts
print("VERTS_JSON_BEGIN")
print(json.dumps(out))
print("VERTS_JSON_END")
"""


def read_predicted_verts(mcp, sample=4000):
    """Run bpy to read all mesh world vertices in the scene. Returns {object_name: ndarray(N,3)}."""
    code = READ_VERTS_CODE.format(sample=sample)
    text = mcp.call_tool("execute_blender_code", {"code": code})
    s = text.find("VERTS_JSON_BEGIN")
    e = text.find("VERTS_JSON_END")
    if s < 0 or e < 0:
        raise RuntimeError(f"read verts failed: {text[:300]}")
    payload = text[s + len("VERTS_JSON_BEGIN"):e].strip()
    raw = json.loads(payload)
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items() if v}


# ---------- metrics ----------

def add_s(Vp, Vg, sample=2000, rng=None):
    """ADD-S: symmetric tolerant nearest-neighbor vertex distance (meters)."""
    rng = rng or np.random.default_rng(0)
    if len(Vp) > sample:
        Vp = Vp[rng.choice(len(Vp), sample, False)]
    if len(Vg) > sample:
        Vg = Vg[rng.choice(len(Vg), sample, False)]
    d, _ = cKDTree(Vg).query(Vp)
    return float(d.mean())


def chamfer(P, Q, sample=20000, rng=None):
    """Scene-level bidirectional Chamfer (meters)."""
    rng = rng or np.random.default_rng(0)
    if len(P) > sample:
        P = P[rng.choice(len(P), sample, False)]
    if len(Q) > sample:
        Q = Q[rng.choice(len(Q), sample, False)]
    d1, _ = cKDTree(Q).query(P)
    d2, _ = cKDTree(P).query(Q)
    return float(d1.mean() + d2.mean())


def _center(V):
    return (V.max(0) + V.min(0)) / 2.0


# ---------- Task5: single-image reconstruction (sim(3) alignment + Chamfer + F-score) ----------

def umeyama_sim3(P, Q):
    """Find the similarity transform (s, R, t) aligning P to Q, minimizing
    ||sRP+t - Q||^2 (requires one-to-one correspondence between P and Q).
    Returns (s, R(3x3), t(3)). Classic Umeyama closed-form solution."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    muP, muQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - muP, Q - muQ
    Sigma = Qc.T @ Pc / len(P)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    varP = (Pc ** 2).sum() / len(P)
    s = float(np.trace(np.diag(D) @ S) / varP) if varP > 1e-12 else 1.0
    t = muQ - s * R @ muP
    return s, R, t


def icp_sim3(P, Q, iters=30, sample=4000, rng=None):
    """Point cloud alignment (ICP): **scale fixed** to the ratio of the two point
    cloud diagonals (prevents ICP free-scaling collapse); iterations optimize only
    rotation + translation (rigid). Multiple initial rotations (about Z 0/90/180/270
    deg), keep the best. Returns the aligned P_aligned and the reference Q."""
    rng = rng or np.random.default_rng(0)
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if len(P) > sample: P = P[rng.choice(len(P), sample, False)]
    if len(Q) > sample: Q = Q[rng.choice(len(Q), sample, False)]

    def diag(X): return np.linalg.norm(X.max(0) - X.min(0))
    s_fixed = diag(Q) / (diag(P) + 1e-9)        # fixed scale = diagonal ratio, no longer iterated
    cQ = _center(Q)
    treeQ = cKDTree(Q)

    def rigid(A, B):
        """Find the rotation + translation aligning A to B (scale=1), Umeyama with
        scaling removed. Requires correspondence between A and B."""
        muA, muB = A.mean(0), B.mean(0)
        H = (A - muA).T @ (B - muB)
        U, _, Vt = np.linalg.svd(H)
        S = np.eye(3)
        if np.linalg.det(Vt.T @ U.T) < 0:
            S[2, 2] = -1
        R = Vt.T @ S @ U.T
        t = muB - R @ muA
        return R, t

    def run_from(deg):
        a = np.radians(deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Pa = (s_fixed * (Rz @ (P - _center(P)).T).T) + cQ   # init: center + fixed scale + pre-rotate about Z
        prev = 1e18
        for _ in range(iters):
            _, idx = treeQ.query(Pa)
            R, t = rigid(Pa, Q[idx])              # rotation + translation only, scale unchanged
            Pa = (R @ Pa.T).T + t
            d, _ = treeQ.query(Pa)
            if abs(prev - d.mean()) < 1e-5:
                break
            prev = d.mean()
        return Pa, prev

    best, best_err = None, 1e18
    for deg in (0, 90, 180, 270):
        Pa, err = run_from(deg)
        if err < best_err:
            best, best_err = Pa, err
    return best, Q



def f_score(P, Q, tau):
    """F-score@tau: harmonic mean of the fraction of P->Q nearest distances < tau
    (precision) and Q->P (recall)."""
    dPQ, _ = cKDTree(Q).query(P)
    dQP, _ = cKDTree(P).query(Q)
    prec = float((dPQ < tau).mean())
    rec = float((dQP < tau).mean())
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"fscore": f, "precision": prec, "recall": rec}


def score_reconstruction(pred_verts, gt_verts, gt_parts=None, sample=4000, rng=None,
                         alignment="sim3_icp"):
    """Task5 scoring with an explicit alignment contract.

    ``sim3_icp`` preserves the historical diagnostic. ``none`` evaluates the
    submitted world in its native coordinates and is required for pose/scale
    diagnostics. Both paths share the same point sampling and object metrics.
    - pred_verts / gt_verts: (N,3) world vertices (arbitrary scale/pose); gt_verts
      = the merged point cloud of all GT objects.
    - gt_parts: optional [(name, (M,3)), ...] vertices per GT object. If given, also
      compute a "per-object partition" metric (spatially select aligned agent points
      by each GT object's bounding box), solving the problem where large objects
      dilute small ones in multi-object scenes and missing objects cannot be
      located/detected. Independent of how the agent partitions objects.
    """
    if alignment not in {"none", "sim3_icp"}:
        raise ValueError("alignment must be none or sim3_icp")
    rng = rng or np.random.default_rng(0)
    P = np.asarray(pred_verts, float); Q = np.asarray(gt_verts, float)
    if len(P) < 10 or len(Q) < 10:
        return {"error": "too few points", "n_pred": len(P), "n_gt": len(Q)}
    if alignment == "sim3_icp":
        Pa, Qs = icp_sim3(P, Q, sample=sample, rng=rng)
    else:
        Pa, Qs = P.copy(), Q.copy()
    cd = chamfer(Pa, Qs, sample=sample, rng=rng)
    diagQ = float(np.linalg.norm(Qs.max(0) - Qs.min(0)))
    # Global f-score also reports precision/recall. Note: precision is easily
    # inflated by "sprayed scatter points" (in a furniture-dense room, random points
    # happen to lie near some GT surface); recall (fraction of GT surface covered)
    # is harder to game.
    # Therefore the **global f is diagnostic only**, not a primary metric (see the
    # per-object headline below).
    fglob = {}
    for p in (0.02, 0.05, 0.1):
        r = f_score(Pa, Qs, p * diagQ)
        fglob[f"f@{int(p*100)}%"] = r["fscore"]
        fglob[f"recall@{int(p*100)}%"] = r["recall"]
        fglob[f"prec@{int(p*100)}%"] = r["precision"]
    out = {"alignment": alignment, "chamfer_aligned": cd,
           "chamfer": cd, **fglob,
           "n_pred": len(P), "n_gt": len(Q), "gt_diag": diagQ}

    # ---- per-object partition (honest primary metric: prevents "brick shards
    # sprayed into the box count as built") ----
    # Key fix: covered is no longer "there are points in the box" (gameable by
    # spraying), but a **quality threshold**: the object's local f@5% must meet
    # COVER_TAU to count as genuinely reconstructed.
    # headline uses the per-object f-score mean (missing/not-meeting scores 0),
    # consistent with visual impression.
    COVER_F5 = 0.30          # per-object f@5% >= 0.30 to count as "reconstructed that object"
    if gt_parts:
        per = []
        margin = 0.10 * diagQ
        for name, gv in gt_parts:
            gv = np.asarray(gv, float)
            lo, hi = gv.min(0) - margin, gv.max(0) + margin
            mask = np.all((Pa >= lo) & (Pa <= hi), axis=1)
            n_in = int(mask.sum())
            part_diag = float(np.linalg.norm(gv.max(0) - gv.min(0)))
            tau = 0.05 * part_diag
            if n_in >= 10:
                Pp = Pa[mask]
                pcd = chamfer(Pp, gv, sample=sample, rng=rng)
                fr = f_score(Pp, gv, tau)
                pf = fr["fscore"]; prec = fr["precision"]; rec = fr["recall"]
            else:
                # Almost no agent points in this object region = missing build
                pcd, pf, prec, rec = float("nan"), 0.0, 0.0, 0.0
            covered = pf >= COVER_F5          # quality threshold, no longer "any point counts"
            per.append({"part": name, "n_pred_in": n_in, "chamfer": pcd,
                        "f@5%": pf, "precision": prec, "recall": rec,
                        "covered": bool(covered)})
        cov = [p for p in per if p["covered"]]
        out["per_object"] = per
        # Primary metric: per-object f-score mean (missing scores 0) -- honestly
        # reflects "how well each object is built"
        out["obj_f@5%"] = float(np.mean([p["f@5%"] for p in per]))
        out["obj_recall@5%"] = float(np.mean([p["recall"] for p in per]))  # per-object surface coverage
        out["mean_part_chamfer"] = float(np.mean([p["chamfer"] for p in cov])) if cov else float("nan")
        out["mean_part_f@5%"] = out["obj_f@5%"]            # compat with old field name
        # coverage: fraction of objects meeting the quality threshold (detects both
        # missing and badly built, neither counts as covered)
        out["object_coverage"] = len(cov) / len(per)

        # ---- per-object matched version (position/shape decoupled) ----
        # Global ICP is lopsided (the whole agent point pile is pushed to the densest
        # region, all other objects misaligned -> f=0).
        # Fix: DBSCAN cuts the aligned agent point cloud into clusters -> Hungarian
        # matches clusters to GT objects by centroid distance -> each pair then does
        # a local rigid alignment (translate to GT centroid) and computes f@5% /
        # pBERT, **reporting position error separately**.
        # This way objects with "right shape, slightly wrong position" are not scored
        # 0 due to misalignment.
        try:
            from sklearn.cluster import DBSCAN
            from scipy.optimize import linear_sum_assignment
            target_K = len(gt_parts)
            avg_part_diag = float(np.mean([np.linalg.norm(np.asarray(g).max(0) -
                                                          np.asarray(g).min(0))
                                           for _, g in gt_parts]))
            # 1) Clustering: eps adaptive -- scan from small to large, pick the one
            #    whose cluster count is closest to the GT object count
            best_pts, best_eps, best_diff = None, None, 1e9
            for ratio in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
                eps_try = max(0.04, ratio * avg_part_diag)
                lab = DBSCAN(eps=eps_try, min_samples=10).fit(Pa).labels_
                cls = [Pa[lab == c] for c in set(lab) if c != -1 and (lab == c).sum() >= 30]
                diff = abs(len(cls) - target_K)
                # Prefer cluster count >= target (can later be filtered by Hungarian),
                # avoid gluing multiple objects into one blob
                penalty = diff + (0.5 if len(cls) < target_K else 0)
                if penalty < best_diff:
                    best_pts, best_eps, best_diff = cls, eps_try, penalty
                    if diff == 0 and len(cls) >= target_K: break
            cluster_pts = best_pts or []
            n_cl = len(cluster_pts)

            per_m = []
            if n_cl == 0:
                for name, gv in gt_parts:
                    per_m.append({"part": name, "matched": False,
                                   "f@5%_matched": 0.0, "pos_err": float("nan"),
                                   "n_cluster_pts": 0})
            else:
                gt_centers = np.array([np.asarray(g).mean(0) for _, g in gt_parts])
                cl_centers = np.array([c.mean(0) for c in cluster_pts])
                C = np.linalg.norm(gt_centers[:, None, :] - cl_centers[None, :, :], axis=-1)
                MAX_MATCH_DIST = 0.6 * diagQ
                C_pad = C.copy()
                C_pad[C_pad > MAX_MATCH_DIST] = MAX_MATCH_DIST + 100.0
                row, col = linear_sum_assignment(C_pad)
                used = {}
                for r, c in zip(row, col):
                    if C[r, c] <= MAX_MATCH_DIST:
                        used[r] = c
                for gi, (name, gv) in enumerate(gt_parts):
                    gv = np.asarray(gv, float)
                    part_diag = float(np.linalg.norm(gv.max(0) - gv.min(0)))
                    tau = 0.05 * part_diag
                    if gi not in used:
                        per_m.append({"part": name, "matched": False,
                                       "f@5%_matched": 0.0, "pos_err": float("nan"),
                                       "n_cluster_pts": 0})
                        continue
                    cl_pts = cluster_pts[used[gi]]
                    pos_err = float(np.linalg.norm(cl_pts.mean(0) - gv.mean(0)))
                    shifted = cl_pts - cl_pts.mean(0) + gv.mean(0)
                    fr = f_score(shifted, gv, tau)
                    per_m.append({"part": name, "matched": True,
                                   "f@5%_matched": fr["fscore"],
                                   "precision_matched": fr["precision"],
                                   "recall_matched": fr["recall"],
                                   "pos_err": pos_err,
                                   "n_cluster_pts": int(len(cl_pts))})
            out["per_object_matched"] = per_m
            out["n_clusters"] = n_cl
            out["match_eps"] = best_eps if best_eps else 0.0
            out["obj_f@5%_matched"] = float(np.mean([p["f@5%_matched"] for p in per_m]))
            matched_pos = [p["pos_err"] for p in per_m if p["matched"]]
            out["mean_pos_err"] = float(np.mean(matched_pos)) if matched_pos else float("nan")
            out["match_rate"] = sum(1 for p in per_m if p["matched"]) / max(len(per_m), 1)
        except Exception as e:
            out["match_error"] = repr(e)[:200]

    # ---- 3D semantic metric (PointBERT, OpenShape ViT-B/32 aligned version) ----
    # Geometry metrics miss "semantic errors" (build a bed as a flat block, point
    # distance is okay but semantics are wrong); PointBERT embedding cosine
    # complements: scene-level + per-object, using the aligned Pa (same space as obj_f@5%).
    # Same family as Pixal3D's Uni3D/ULIP-2 metric (all OpenCLIP-aligned point cloud encoders).
    try:
        import metrics_3d as _m3
        rng3 = np.random.default_rng(0)
        sem = _m3.pointbert_compare(P, Q, gt_parts=gt_parts, aligned_pred=Pa, rng=rng3)
        out.update(sem)
    except Exception as e:
        out["pointbert_error"] = repr(e)[:200]
    return out


def load_gt_furniture(scene_dir):
    """Read the GT furniture point cloud from full.glb (globally merged +
    per-object parts). Used for Task5 reconstruction scoring.
    Returns (Q_all (N,3), parts [(name,(M,3)),...]). Furniture only, excluding
    walls/floor/ceiling."""
    import glob as _glob
    cand = _glob.glob(os.path.join(scene_dir, "scene", "*_full.glb"))
    if not cand:
        return None, None
    scene = trimesh.load(cand[0])
    keep = ("Bed", "Cabinet", "Lighting", "Chair", "Table", "Sofa", "Wardrobe",
            "Desk", "Shelf", "Stool", "Nightstand", "Pendant", "Dressing", "TV",
            "Wine", "Bookshelf", "Lamp", "Couch", "Stand")
    parts = []
    if isinstance(scene, trimesh.Scene):
        for node in scene.graph.nodes_geometry:
            tf, gn = scene.graph[node]
            if any(k in gn for k in keep):
                v = trimesh.transformations.transform_points(
                    np.asarray(scene.geometry[gn].vertices), tf)
                parts.append((gn[:24], np.asarray(v, float)))
    if not parts:
        m = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
        parts = [("scene", np.asarray(m.vertices, float))]
    Q_all = np.vstack([v for _, v in parts])
    return Q_all, parts


def score_reconstruction_scene(scene_dir, pred_verts, sample=4000, rng=None,
                               alignment="sim3_icp"):
    """Task5 entry: read the GT furniture point cloud + the agent reconstructed
    point cloud, compute reconstruction metrics."""
    Q, parts = load_gt_furniture(scene_dir)
    if Q is None:
        return {"error": "no GT full.glb"}
    return score_reconstruction(pred_verts, Q, gt_parts=parts, sample=sample,
                                rng=rng, alignment=alignment)


def recon_visual_compare(mcp, scene_dir, render_dir, n_views=4):
    """Task5 visual: export the agent's current reconstructed scene, render it and
    the GT full.glb with the same EEVEE pipeline under n_views orbiting cameras,
    and compute MSSIM/CLIP/LPIPS. Returns a visual metrics dict.
    In Blender: first export the agent scene glb, then import agent / GT separately and render."""
    import glob as _glob
    os.makedirs(render_dir, exist_ok=True)
    agent_glb = os.path.join(render_dir, "agent_recon.glb")
    gt_glb = next(iter(_glob.glob(os.path.join(scene_dir, "scene", "*_full.glb"))), None)
    if not gt_glb:
        return {"visual_error": "no GT full.glb"}
    # 1) Export the agent's current scene
    exp = ("import bpy\n"
           "for o in bpy.data.objects: o.select_set(o.type=='MESH')\n"
           f"bpy.ops.export_scene.gltf(filepath={agent_glb!r}, use_selection=True, export_format='GLB')\n"
           "print('EXP_OK')\n")
    mcp.call_tool("execute_blender_code", {"code": exp})
    if not os.path.exists(agent_glb):
        return {"visual_error": "export agent failed"}

    # 2) Blender-internal function: import a glb, render n views orbiting the object
    #    (shared camera orbit, based on that glb's own bounding box)
    #    To keep agent / GT at the same viewpoints, the orbit radius/center uses GT's
    #    bounding box (both already at reasonable scale, rendering only cares about composition)
    #    Rendering pipeline: three-point area lights (key+fill+rim) + neutral gray
    #    background + default gray BSDF fallback, ensuring both images are "visible"
    #    and not swallowed by EEVEE's default black scene (without which MSSIM is inflated).
    #    When rendering GT, furniture_only=True filters out walls/floor/ceiling/windows,
    #    rendering furniture only, so agent (which only builds furniture) and GT compare fairly.
    def render_glb(glb, tag, furniture_only=False):
        # Keep the same furniture whitelist as load_gt_furniture, ensuring the
        # geometry metric and visual metric look at the same set of objects
        keep = ("Bed", "Cabinet", "Lighting", "Chair", "Table", "Sofa", "Wardrobe",
                "Desk", "Shelf", "Stool", "Nightstand", "Pendant", "Dressing", "TV",
                "Wine", "Bookshelf", "Lamp", "Couch", "Stand")
        keep_str = repr(keep)
        filter_block = (
            f"keep_kw={keep_str}\n"
            "for o in list(meshes):\n"
            "    if not any(k in o.name for k in keep_kw):\n"
            "        bpy.data.objects.remove(o, do_unlink=True)\n"
            "meshes=[o for o in bpy.data.objects if o.type=='MESH']\n"
        ) if furniture_only else ""
        code = (
            "import bpy, mathutils, math, json\n"
            "bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)\n"
            "for m in list(bpy.data.meshes): bpy.data.meshes.remove(m)\n"
            "for lt in list(bpy.data.lights): bpy.data.lights.remove(lt)\n"
            f"bpy.ops.import_scene.gltf(filepath={glb!r})\n"
            "meshes=[o for o in bpy.data.objects if o.type=='MESH']\n"
            f"{filter_block}"
            "cs=[o.matrix_world@mathutils.Vector(v) for o in meshes for v in o.bound_box]\n"
            "xs=[c.x for c in cs];ys=[c.y for c in cs];zs=[c.z for c in cs]\n"
            "ctr=mathutils.Vector(((max(xs)+min(xs))/2,(max(ys)+min(ys))/2,(max(zs)+min(zs))/2))\n"
            "ext=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs))+1e-3\n"
            "R=ext*1.6\n"
            # default gray BSDF (fallback only for mesh without material, does not override agent-set materials)
            "default_mat=bpy.data.materials.new('eval_default'); default_mat.use_nodes=True\n"
            "bsdf=next((nd for nd in default_mat.node_tree.nodes if nd.type=='BSDF_PRINCIPLED'),None)\n"
            "if bsdf:\n"
            "    bsdf.inputs['Base Color'].default_value=(0.72,0.72,0.75,1)\n"
            "    if 'Roughness' in bsdf.inputs: bsdf.inputs['Roughness'].default_value=0.65\n"
            "for o in meshes:\n"
            "    if not o.data.materials: o.data.materials.append(default_mat)\n"
            # three-point lighting (area light, key/fill/rim, avoid EEVEE default black scene)
            "def add_light(name, loc, energy, sz):\n"
            "    bpy.ops.object.light_add(type='AREA', location=loc)\n"
            "    lt=bpy.context.object\n"
            "    lt.name=name; lt.data.energy=energy; lt.data.size=ext*sz\n"
            "    lt.rotation_euler=(ctr-mathutils.Vector(loc)).to_track_quat('-Z','Y').to_euler()\n"
            "add_light('Key',  (ctr.x+R*0.9, ctr.y-R*0.7, ctr.z+ext*1.0), 1200, 0.9)\n"
            "add_light('Fill', (ctr.x-R*0.6, ctr.y-R*0.4, ctr.z+ext*0.5),  400, 1.2)\n"
            "add_light('Rim',  (ctr.x,        ctr.y+R*0.9, ctr.z+ext*0.9),  600, 0.7)\n"
            "s=bpy.context.scene\n"
            "try: s.render.engine='BLENDER_EEVEE_NEXT'\n"
            "except Exception: s.render.engine='BLENDER_EEVEE'\n"
            # neutral dark gray background (aligned with 3DCodeBench rendering spec), no longer relying on world env light
            "w=s.world or bpy.data.worlds.new('W'); s.world=w; w.use_nodes=True\n"
            "for nd in list(w.node_tree.nodes): w.node_tree.nodes.remove(nd)\n"
            "bg=w.node_tree.nodes.new('ShaderNodeBackground')\n"
            "wo=w.node_tree.nodes.new('ShaderNodeOutputWorld')\n"
            "bg.inputs[0].default_value=(0.05,0.05,0.06,1); bg.inputs[1].default_value=0.4\n"
            "w.node_tree.links.new(bg.outputs[0], wo.inputs[0])\n"
            "cd=bpy.data.cameras.new('C'); cam=bpy.data.objects.new('C',cd); s.collection.objects.link(cam); s.camera=cam\n"
            "cd.lens=50\n"
            "s.render.resolution_x=384; s.render.resolution_y=384; s.render.film_transparent=False\n"
            f"N={n_views}\n"
            "outs=[]\n"
            "for i in range(N):\n"
            "    a=2*math.pi*i/N\n"
            "    cam.location=ctr+mathutils.Vector((R*math.cos(a),R*math.sin(a),R*0.6))\n"
            "    cam.rotation_euler=(ctr-cam.location).to_track_quat('-Z','Y').to_euler()\n"
            f"    p={render_dir!r}+'/'+{tag!r}+'_v'+str(i)+'.png'\n"
            "    s.render.filepath=p; bpy.ops.render.render(write_still=True); outs.append(p)\n"
            "print('VIEWS',json.dumps(outs))\n"
        )
        out = mcp.call_tool("execute_blender_code", {"code": code})
        j = out.find("VIEWS")
        return json.loads(out[j+5:].strip().splitlines()[0]) if j >= 0 else []

    a_views = render_glb(agent_glb, "agent", furniture_only=False)
    g_views = render_glb(gt_glb, "gt", furniture_only=True)
    if not a_views or len(a_views) != len(g_views):
        return {"visual_error": "render mismatch"}
    import metrics_visual as _v
    return _v.visual_compare(list(zip(a_views, g_views)))


def score_layout(scene_dir, pred_verts, sample=2000):
    """Primary scoring: Hungarian matching + ADD-S/PE + Chamfer + success rate.
    scene_dir: scene directory containing gt.json and canonical/
    pred_verts: {object_name: ndarray}  agent-placed world vertices (building
        structure must be filtered out)
    """
    gt = json.load(open(os.path.join(scene_dir, "gt.json")))
    rng = np.random.default_rng(0)

    # GT world vertices per object
    gt_objs = gt["objects"]
    gt_V = [canonical_world_verts(scene_dir, o, sample, rng) for o in gt_objs]
    gt_diam = [float(np.linalg.norm(V.max(0) - V.min(0))) for V in gt_V]

    # Prediction: take only object_NN (agent-placed), filter out building/camera/light
    pred = {k: v for k, v in pred_verts.items() if k.startswith("object_")}
    pred_names = list(pred.keys())
    pred_V = [pred[k] for k in pred_names]

    np_, ng = len(pred_V), len(gt_V)
    if np_ == 0:
        return {"error": "no predicted objects", "n_gt": ng}

    # Hungarian matching (cost = ADD-S), rectangular is fine
    C = np.zeros((np_, ng))
    for i in range(np_):
        for j in range(ng):
            C[i, j] = add_s(pred_V[i], gt_V[j], sample, rng)
    ri, cj = linear_sum_assignment(C)

    per_obj = []
    for i, j in zip(ri, cj):
        pe = float(np.linalg.norm(_center(pred_V[i]) - _center(gt_V[j])))
        pe_xz = float(np.linalg.norm(_center(pred_V[i])[[0, 1]] - _center(gt_V[j])[[0, 1]]))
        pe_y = float(abs(_center(pred_V[i])[2] - _center(gt_V[j])[2]))
        adds = float(C[i, j])
        per_obj.append({
            "pred": pred_names[i], "gt_category": gt_objs[j]["category"],
            "add_s": adds, "pe": pe, "pe_xz": pe_xz, "pe_y": pe_y,
            "success": bool(adds < 0.1 * gt_diam[j]),
        })

    mean_adds = float(np.mean([o["add_s"] for o in per_obj]))
    mean_pe = float(np.mean([o["pe"] for o in per_obj]))
    n_succ = sum(o["success"] for o in per_obj)
    scene_success = bool(n_succ == ng and np_ == ng)

    # Scene-level Chamfer (all predicted points vs all GT points)
    P = np.vstack(pred_V); Q = np.vstack(gt_V)
    cd = chamfer(P, Q, rng=rng)

    return {
        "n_gt": ng, "n_pred": np_,
        "mean_add_s": mean_adds,          # primary metric (per object)
        "chamfer": cd,                     # primary metric (scene level)
        "mean_pe": mean_pe,
        "placement_acc": n_succ / ng if ng else 0.0,
        "scene_success": scene_success,
        "per_object": per_obj,
    }


# ---------- Task3: camera pose error ----------

def camera_pose_error(pred_pos, pred_look, gt_pos, gt_look):
    """Camera pose error: position Euclidean distance (meters) + orientation angle (degrees).
    pred/gt are each position[3] and look_dir[3]."""
    pp = np.asarray(pred_pos, float); gp = np.asarray(gt_pos, float)
    pos_err = float(np.linalg.norm(pp - gp))
    pl = np.asarray(pred_look, float); gl = np.asarray(gt_look, float)
    pl /= (np.linalg.norm(pl) + 1e-9); gl /= (np.linalg.norm(gl) + 1e-9)
    ang = float(np.degrees(np.arccos(np.clip(pl @ gl, -1, 1))))
    return {"cam_pos_error": pos_err, "cam_angle_error_deg": ang}


def score_camera(task, pred_pos, pred_look):
    """Task3 scoring: agent camera vs gt_camera. Image similarity (LPIPS/CLIP)
    needs images separately, reported separately."""
    gt = task.get("gt_camera")
    if not gt:
        return {"error": "no gt_camera"}
    return camera_pose_error(pred_pos, pred_look, gt["position"], gt["look_dir"])


def _render_current_scene(mcp, cam_matrix_world, fov_x, out_path, res=384):
    """Render the Blender current scene with the given camera pose (same EEVEE
    pipeline), saving to out_path.
    cam_matrix_world: 4x4 list (row-major); fov_x: radians.

    Rendering is observational: the active camera and render/world settings are
    restored before control returns to the agent.
    """
    mw = cam_matrix_world
    code = (
        "import bpy, mathutils, math\n"
        "s=bpy.context.scene\n"
        "_prev_camera=s.camera\n"
        "_prev_engine=s.render.engine\n"
        "_prev_res_x=s.render.resolution_x\n"
        "_prev_res_y=s.render.resolution_y\n"
        "_prev_filepath=s.render.filepath\n"
        "_prev_transparent=s.render.film_transparent\n"
        "_prev_world=s.world\n"
        "_world_created=s.world is None\n"
        "w=s.world or bpy.data.worlds.new('W'); s.world=w\n"
        "_prev_use_nodes=w.use_nodes\n"
        "w.use_nodes=True\n"
        "_bg=w.node_tree.nodes.get('Background') if w.node_tree else None\n"
        "_prev_bg_strength=_bg.inputs[1].default_value if _bg else None\n"
        "cam=bpy.data.objects.get('_EVALCAM')\n"
        "_cam_created=cam is None\n"
        "if cam is None:\n"
        "    cd=bpy.data.cameras.new('_EVALCAM'); cam=bpy.data.objects.new('_EVALCAM',cd)\n"
        "    s.collection.objects.link(cam)\n"
        "_prev_cam_matrix=cam.matrix_world.copy()\n"
        "_prev_cam_angle=cam.data.angle_x\n"
        "try:\n"
        "    try: s.render.engine='BLENDER_EEVEE_NEXT'\n"
        "    except Exception: s.render.engine='BLENDER_EEVEE'\n"
        "    if _bg: _bg.inputs[1].default_value=1.5\n"
        "    s.camera=cam\n"
        f"    cam.matrix_world=mathutils.Matrix({mw})\n"
        f"    cam.data.angle_x={fov_x}\n"
        f"    s.render.resolution_x={res}\n"
        f"    s.render.resolution_y={res}\n"
        "    s.render.film_transparent=False\n"
        f"    s.render.filepath={out_path!r}\n"
        "    bpy.ops.render.render(write_still=True)\n"
        "finally:\n"
        "    s.camera=_prev_camera\n"
        "    try: s.render.engine=_prev_engine\n"
        "    except Exception: pass\n"
        "    s.render.resolution_x=_prev_res_x\n"
        "    s.render.resolution_y=_prev_res_y\n"
        "    s.render.filepath=_prev_filepath\n"
        "    s.render.film_transparent=_prev_transparent\n"
        "    if _bg and _prev_bg_strength is not None:\n"
        "        _bg.inputs[1].default_value=_prev_bg_strength\n"
        "    w.use_nodes=_prev_use_nodes\n"
        "    if _world_created:\n"
        "        s.world=_prev_world\n"
        "        if w.users==0: bpy.data.worlds.remove(w)\n"
        "    if _cam_created:\n"
        "        bpy.data.objects.remove(cam, do_unlink=True)\n"
        "    else:\n"
        "        cam.matrix_world=_prev_cam_matrix\n"
        "        cam.data.angle_x=_prev_cam_angle\n"
        "print('RENDER_OK')\n"
    )
    out = mcp.call_tool("execute_blender_code", {"code": code})
    return os.path.exists(out_path)


def score_camera_from_blender(mcp, task, render_dir=None):
    """Task3 scoring: (1) camera pose error (primary, geometry); (2) visual
    comparison (agent camera render vs GT camera render, same pipeline same scene,
    eliminating rendering style differences). If render_dir is given, render both
    and compute MSSIM/CLIP/LPIPS."""
    code = (
        "import bpy, json\n"
        "c=bpy.context.scene.camera\n"
        "mw=c.matrix_world\n"
        "pos=[mw[0][3],mw[1][3],mw[2][3]]\n"
        "look=[-mw[0][2],-mw[1][2],-mw[2][2]]\n"
        "M=[[mw[r][col] for col in range(4)] for r in range(4)]\n"
        "fov=c.data.angle_x\n"
        "print('CAM_JSON',json.dumps({'pos':pos,'look':look,'M':M,'fov':fov}))\n"
    )
    text = mcp.call_tool("execute_blender_code", {"code": code})
    i = text.find("CAM_JSON")
    if i < 0:
        return {"error": f"no camera read: {text[:200]}"}
    d = json.loads(text[i + len("CAM_JSON"):].strip().splitlines()[0])
    res = score_camera(task, d["pos"], d["look"])
    # agent camera info (for run.py to export agent_camera.json)
    res["agent_position"] = d["pos"]
    res["agent_look_dir"] = d["look"]
    res["agent_matrix_world"] = d["M"]
    res["agent_fov_x_deg"] = round(math.degrees(d.get("fov", 0.69)), 2)

    # Visual: render one from the agent camera + one from the GT camera (same scene
    # same pipeline), compare the images
    gt = task.get("gt_camera")
    if render_dir and gt and gt.get("matrix_world"):
        try:
            os.makedirs(render_dir, exist_ok=True)
            sid = task["id"]
            a_png = os.path.join(render_dir, sid + "__agent_view.png")
            g_png = os.path.join(render_dir, sid + "__gt_view.png")
            fov = d.get("fov", math.radians(gt.get("fov_x_deg", 39.6)))
            ok_a = _render_current_scene(mcp, d["M"], fov, a_png)        # agent camera
            ok_g = _render_current_scene(mcp, gt["matrix_world"], fov, g_png)  # GT camera
            if ok_a and ok_g:
                import metrics_visual as _v
                res["visual"] = _v.visual_compare([(a_png, g_png)])
                res["agent_view"], res["gt_view"] = a_png, g_png
        except Exception as e:
            res["visual_error"] = repr(e)
    return res


# ---------- Task3: image similarity ----------

def _load_gray(path, size=256):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(im, dtype=np.float64) / 255.0


def image_similarity(pred_img_path, ref_img_path, size=256):
    """Image similarity of the agent render vs the ref image.
    Base metrics (pure numpy/PIL, always available): PSNR, SSIM, pixel MAE.
    Optional metrics (computed only if torch+lpips / clip installed): LPIPS, CLIP cosine.
    Note: the agent render style (materials/lighting) may differ from the ref;
    image metrics are affected by this and serve as auxiliary reference; the
    primary metric is still camera pose error."""
    out = {}
    P = _load_gray(pred_img_path, size); R = _load_gray(ref_img_path, size)
    mae = float(np.abs(P - R).mean())
    mse = float(((P - R) ** 2).mean())
    out["pixel_mae"] = mae
    out["psnr"] = float(10 * np.log10(1.0 / mse)) if mse > 1e-12 else 99.0
    # SSIM (grayscale, global simplified version)
    pg = P.mean(2); rg = R.mean(2)
    mp, mr = pg.mean(), rg.mean()
    vp, vr = pg.var(), rg.var()
    cov = ((pg - mp) * (rg - mr)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out["ssim"] = float(((2*mp*mr + c1)*(2*cov + c2)) /
                        ((mp**2 + mr**2 + c1)*(vp + vr + c2)))
    # Optional: LPIPS
    try:
        import torch, lpips as _lpips
        if not hasattr(image_similarity, "_lpips"):
            image_similarity._lpips = _lpips.LPIPS(net='alex')
        def t(x): return torch.tensor(x.transpose(2,0,1)[None]*2-1, dtype=torch.float32)
        out["lpips"] = float(image_similarity._lpips(t(P), t(R)).item())
    except Exception:
        pass
    # Optional: CLIP
    try:
        import torch, clip as _clip
        if not hasattr(image_similarity, "_clip"):
            image_similarity._clip = _clip.load("ViT-B/32", device="cpu")
        model, prep = image_similarity._clip
        from PIL import Image
        with torch.no_grad():
            a = model.encode_image(prep(Image.open(pred_img_path)).unsqueeze(0))
            b = model.encode_image(prep(Image.open(ref_img_path)).unsqueeze(0))
            a /= a.norm(); b /= b.norm()
            out["clip_cosine"] = float((a @ b.T).item())
    except Exception:
        pass
    return out


def score_task3(task, pred_pos, pred_look, pred_img_path=None):
    """Task3 full scoring: camera pose error + (optional) image similarity. Each
    metric class outputs its own values."""
    gt = task.get("gt_camera")
    res = {}
    if gt:
        res.update(camera_pose_error(pred_pos, pred_look, gt["position"], gt["look_dir"]))
    if pred_img_path and task.get("reference_image"):
        res["image"] = image_similarity(pred_img_path, task["reference_image"])
    return res
