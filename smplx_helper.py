import os
import pickle
import numpy as np
import trimesh
import smplx  # pip install smplx

# from mano_v1_2.webuser.smpl_handpca_wrapper import load_model as smpl_load_model
import scipy.sparse as sp
import chumpy as ch
import trimesh
from typing import Optional, Tuple


# -------------------- SMPL UV support --------------------
# We load a canonical SMPL UV template OBJ (with `vt` and `f v/vt`) and reuse its UVs
# for any SMPL/SMPL-H mesh that shares the same topology (6890 verts, 13776 tris).
_SMPL_UV_CACHE = {}  # uv_obj_path -> (vt_list[(u,v)], face_key->v2vt{v:vt}, v_default{v:vt})


def _load_smpl_uv_template(uv_obj_path: str):
    """Parse a UV-template OBJ and return:
    vt_list: List[(u,v)] (0-based indexing when referenced)
    face_v2vt: dict[tuple(sorted(v0,v1,v2))] -> {v0:vt0, v1:vt1, v2:vt2}
    v_default: dict[v] -> some vt for that vertex (first seen)
    """
    uv_obj_path = os.path.abspath(uv_obj_path)
    if uv_obj_path in _SMPL_UV_CACHE:
        return _SMPL_UV_CACHE[uv_obj_path]

    vt_list = []
    face_v2vt = {}
    v_default = {}

    with open(uv_obj_path, "r") as f:
        for line in f:
            if line.startswith("vt "):
                parts = line.strip().split()
                if len(parts) >= 3:
                    vt_list.append((float(parts[1]), float(parts[2])))
            elif line.startswith("f "):
                toks = line.strip().split()[1:]
                if len(toks) != 3:
                    continue  # expect triangles
                vs = []
                vts = []
                for t in toks:
                    sp = t.split("/")
                    v = int(sp[0]) - 1
                    vt = int(sp[1]) - 1 if len(sp) > 1 and sp[1] != "" else None
                    vs.append(v)
                    vts.append(vt)
                    if vt is not None and v not in v_default:
                        v_default[v] = vt
                key = tuple(sorted(vs))
                face_v2vt[key] = {vs[0]: vts[0], vs[1]: vts[1], vs[2]: vts[2]}

    _SMPL_UV_CACHE[uv_obj_path] = (vt_list, face_v2vt, v_default)
    return vt_list, face_v2vt, v_default


# --------------------------------------------------------


def export_body_arm_aligned_to_mano_seam(
    save_path: str,
    *,
    smpl_models_dir: str,
    smpl_gender: str,
    hand_beta: np.ndarray,
    hand_pose: np.ndarray,
    smplh_pkl_path: str,
    mano_models_dir: str,
    smpl2mano_right_path: str,  # MANO→SMPL-H vert id map (length 778, -1 for none)
    mano_vertices: np.ndarray,  # (778,3) MANO verts for this sample
    handy_texture_img=None,  # optional: to tint body Kd from hand texture
    randomize_arm_pose: bool = False,
    wrist_jitter_deg: float = 0.0,  # jitter applied to wrist_aa before SMPL-H
    seam_twist_jitter_deg: float = 0.0,  # post-align twist about hand wrist axis
    origin_point: np.ndarray = None,
    handy_ref_verts: np.ndarray = None,  # NEW: (H,3) HandyLayer verts in the desired “standard handy” frame
    remove_mask_pkl: Optional[str] = None,
    prefer_pkl: bool = True,
    arm_color_from_wrist_ring: bool = True,
    no_wrist_rot: bool = True,  # If false, we have rot at the hand, use another plan
    R_wrist: np.ndarray = None,
    C_wrist: np.ndarray = None,
    uv_template_obj: Optional[
        str
    ] = None,  # path to canonical SMPL UV template OBJ (vt + f v/vt)
    return_wrist_info: bool = False,
) -> np.ndarray:
    """
    Export BODY+ARM ONLY OBJ, translated so its wrist seam center coincides with the
    MANO seam center computed from this sample's MANO vertices. Returns seam_center_mano (3,).
    """
    import numpy as _np, os

    # ---- 1) Build SMPL-H with your pose (wrist_aa + right-hand 45D) ----
    mano_has_wrist = hand_pose.shape[0] >= 48
    wrist_aa = (
        hand_pose[:3].astype(_np.float32)
        if mano_has_wrist
        else _np.zeros(3, _np.float32)
    )
    rh_pose_45 = (
        hand_pose[3:].astype(_np.float32)
        if mano_has_wrist
        else hand_pose.astype(_np.float32)
    )
    betas_10 = hand_beta.astype(_np.float32)[:10]

    if wrist_jitter_deg and wrist_jitter_deg > 0:
        jitter = _np.deg2rad(
            _np.random.uniform(-wrist_jitter_deg, wrist_jitter_deg, size=3)
        ).astype(_np.float32)
        wrist_aa = (wrist_aa + jitter).astype(_np.float32)

    V_smpl, F_smpl, J_smpl = _load_smplh_vertices_faces(
        smpl_models_dir=smpl_models_dir,
        smpl_gender=smpl_gender,
        right_hand_pose_45=rh_pose_45,
        betas_10=betas_10,
        randomize_arm_pose=randomize_arm_pose,
        smplh_pkl_path=smplh_pkl_path,
        mano_models_dir=mano_models_dir,
        wrist_aa=wrist_aa,
    )

    # Always compute MANO→SMPL-H mapping and the right-hand SMPL vertex set,
    # since we use these later to find the wrist boundary even on the pkl fast path.
    smpl_ids = _np.asarray(_np.load(smpl2mano_right_path))  # (778,)
    rh_smpl_verts = _np.array(
        [int(v) for v in smpl_ids.ravel() if v >= 0], dtype=_np.int64
    )

    # # ---- 2) Remove SMPL right-hand faces using MANO→SMPL map ----
    # ---- CUT PIPELINE (torso removed; right arm kept) ----
    # If a cache pkl exists (and allowed), use it to select faces directly.
    faces_keep_idx = None
    verts_keep_idx = None
    if prefer_pkl and remove_mask_pkl and os.path.exists(remove_mask_pkl):
        try:
            with open(remove_mask_pkl, "rb") as _f:
                _mask = pickle.load(_f)
            faces_keep_idx = _np.asarray(_mask.get("faces_keep", []), dtype=_np.int64)
            verts_keep_idx = _np.asarray(_mask.get("verts_keep", []), dtype=_np.int64)
            # Basic sanity
            if faces_keep_idx.size == 0:
                faces_keep_idx = None
            if verts_keep_idx.size == 0:
                verts_keep_idx = None
            if faces_keep_idx is not None:
                F_body = _np.asarray(F_smpl[faces_keep_idx], dtype=_np.int32)
            else:
                F_body = _np.asarray(F_smpl, dtype=_np.int32)
            # print(f"[SMPLX] Loaded removal mask pkl: faces_keep={0 if faces_keep_idx is None else faces_keep_idx.size}, "
            #       f"verts_keep={0 if verts_keep_idx is None else verts_keep_idx.size}")
        except Exception as _e:
            # print(f"[SMPLX] Warning: failed to read removal mask pkl '{remove_mask_pkl}': {_e}. Recomputing…")
            faces_keep_idx = None
            verts_keep_idx = None

    if faces_keep_idx is None:
        # print("[SMPLX] No removal mask pkl found or failed to read; computing from scratch…")
        # 1) Start from faces kept by the right-shoulder plane cut (remove torso/left side)
        F_body = remove_faces_toward_left_from_right_shoulder(
            V_smpl, F_smpl, J_smpl, right_shoulder_index=17, eps=0.0
        )
        # 2) Drop any face that touches the right-hand vertices (MANO→SMPL map)
        # smpl_ids = _np.asarray(_np.load(smpl2mano_right_path))  # (778,)
        # rh_smpl_verts = _np.array([int(v) for v in smpl_ids.ravel() if v >= 0], dtype=_np.int64)
        F_body = _np.asarray(
            [f for f in F_body if not any(int(v) in rh_smpl_verts for v in f)],
            dtype=_np.int32,
        )
        # Build and (optionally) save the mask so subsequent runs can skip geometry ops
        if remove_mask_pkl:
            # Map kept faces back to indices of F_smpl
            # (exact row match; SMPL-H faces are unique so a set works well)
            _set = {tuple(map(int, r)) for r in F_body.tolist()}
            faces_keep_idx = _np.array(
                [
                    i
                    for i, r in enumerate(F_smpl.tolist())
                    if tuple(map(int, r)) in _set
                ],
                dtype=_np.int64,
            )
            verts_keep_idx = _np.unique(F_body.reshape(-1)).astype(_np.int64)
            try:
                with open(remove_mask_pkl, "wb") as _f:
                    pickle.dump(
                        {
                            "faces_keep": faces_keep_idx,
                            "verts_keep": verts_keep_idx,
                            "meta": {
                                "V_smpl": int(V_smpl.shape[0]),
                                "F_smpl": int(F_smpl.shape[0]),
                                "kept_faces": int(faces_keep_idx.size),
                                "kept_verts": int(verts_keep_idx.size),
                                "note": "Indices refer to the canonical SMPL-H topology before compaction.",
                            },
                        },
                        _f,
                    )
                # print(f"[SMPLX] Saved removal mask to '{remove_mask_pkl}' "
                #       f"(faces_keep={faces_keep_idx.size}, verts_keep={verts_keep_idx.size}).")
            except Exception as _e:
                print(
                    f"[SMPLX] Warning: failed to write removal mask pkl '{remove_mask_pkl}': {_e}"
                )

    # 3) IMPORTANT: no vertex compaction/reindexing
    V_body = V_smpl.copy()

    # ---- 3) Compute SMPL wrist seam center (body side) ----
    seam_ids = _smpl_wrist_boundary_loop(F_smpl, _np.asarray(rh_smpl_verts))
    seam_center_smpl = (
        V_smpl[seam_ids].mean(axis=0).astype(_np.float32)
        if seam_ids.size > 0
        else V_smpl.mean(0)
    )

    # ---- 4) Compute MANO seam center using inverse map on this sample's MANO verts ----
    # Take MANO indices whose mapped SMPL id lies on the seam; average MANO verts there.
    smpl_id_set = set(map(int, seam_ids.tolist())) if seam_ids.size > 0 else set()
    mano_mask = _np.array(
        [(int(smpl_ids[k]) in smpl_id_set) for k in range(smpl_ids.shape[0])],
        dtype=bool,
    )
    seam_center_mano = mano_vertices[mano_mask].mean(axis=0).astype(_np.float32)

    # ---- 4b) Build local wrist frames from seam geometry (PCA) ----
    def _frame_from_ring(pts):
        # pts: (N,3) centered around its mean
        U, S, Vt = _np.linalg.svd(
            pts, full_matrices=False
        )  # Vt rows are orthonormal axes
        # Largest variance axis -> tangent in seam plane; smallest -> plane normal
        t1 = Vt[0]  # principal tangent
        n = Vt[-1]  # normal (smallest variance)
        t2 = _np.cross(n, t1)
        # Re-orthonormalize to be safe
        t1 /= _np.linalg.norm(t1) + 1e-9
        n /= _np.linalg.norm(n) + 1e-9
        t2 /= _np.linalg.norm(t2) + 1e-9
        return _np.stack([t1, t2, n], axis=1)  # 3x3, columns = axes

    smpl_ring = V_smpl[seam_ids] - seam_center_smpl[None, :]
    mano_ring = mano_vertices[mano_mask] - seam_center_mano[None, :]

    # R_body = _frame_from_ring(smpl_ring)   # SMPL wrist frame
    # R_hand = _frame_from_ring(mano_ring)   # MANO wrist frame

    # # Rotation that maps body frame → hand frame
    # R_fix = R_hand @ R_body.T

    # --- build frames from rings (already in your code) ---
    R_body = _frame_from_ring(smpl_ring).astype(np.float32)
    R_hand = _frame_from_ring(mano_ring).astype(np.float32)

    t1_b, t2_b, n_b = R_body[:, 0], R_body[:, 1], R_body[:, 2]
    t1_h, t2_h, n_h = R_hand[:, 0], R_hand[:, 1], R_hand[:, 2]

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # set [0,0,1] if using Z-up

    # Step 1: map body wrist axis to the opposite of hand wrist axis (body sits on other side)
    target_n = -n_h
    R1 = _rot_between(n_b, target_n)

    # Step 2: rotate about target_n so the seam tangent matches.
    # Try both signs for t1_h and pick the one that maximizes +Y.
    def twist_to(target_t1):
        t1_b_p = R1 @ t1_b
        num = float(np.dot(target_n, np.cross(t1_b_p, target_t1)))
        den = float(np.dot(t1_b_p, target_t1))
        ang = np.arctan2(num, den)
        return _rodrigues(target_n, ang)

    R2_a = twist_to(t1_h)  # maps t1_b -> +t1_h
    R2_b = twist_to(-t1_h)  # maps t1_b -> -t1_h

    # Evaluate which yields an "up" closer to +Y
    up_a = (R2_a @ R1 @ R_body)[:, 1]  # mapped body's up
    up_b = (R2_b @ R1 @ R_body)[:, 1]

    R_fix = (
        R2_a @ R1
        if float(np.dot(up_a, world_up)) >= float(np.dot(up_b, world_up))
        else R2_b @ R1
    )

    # ---- 5) Translate BODY so its seam matches MANO seam; then center to that seam ----
    V_body = V_body + (seam_center_mano - seam_center_smpl)  # bring seam→MANO seam
    V_body = V_body - seam_center_mano  # put origin at seam (shared with hand)

    # ---- 5b) Rotate body around the seam center to match hand orientation ----
    V_body = (R_fix @ V_body.T).T

    up_after = (R_fix @ R_body)[:, 1]
    if float(np.dot(up_after, world_up)) < 0.0:
        Rflip = _rodrigues(n_h, np.pi)
        V_body = (Rflip @ V_body.T).T
        R_fix = Rflip @ R_fix

    y_min = float(np.min(V_body[:, 1]))
    y_max = float(np.max(V_body[:, 1]))
    if y_max < y_min:  # fallback guard; see note below
        Rflip = _rodrigues(n_h, np.pi)
        V_body = (Rflip @ V_body.T).T

    # ---- 5c) Side-consistency fix: ensure BODY is on the opposite side of the seam vs HAND ----
    # # Hand wrist frame axes:
    t1_hand = R_hand[:, 0]  # seam tangent
    n_hand = R_hand[:, 2]  # seam plane normal (along wrist axis)

    x_mean = float(np.mean(V_body[:, 0]))
    # print(f"[SMPLX] Mean X of body mesh is {x_mean:.4f}")
    if x_mean < 0:
        # print(" < 0 → flipping body")
        R180 = _rodrigues(t1_hand, np.pi)
        V_body = (R180 @ V_body.T).T

    # ---- 5e) FINAL TRANSLATION USING PRECISE HANDY VERTICES (if provided) ----
    # At this point V_body is expressed in the seam-centered wrist frame of the hand.
    # If the caller passes precise HandyLayer verts (handy_ref_verts), use their
    # centroid as the *authoritative* origin to place BODY so that subsequent
    # HAND export (centered similarly) will coincide pixel-perfectly.
    if handy_ref_verts is not None and isinstance(handy_ref_verts, np.ndarray):
        # print("[SMPLX] Using HandyLayer reference verts to center BODY.")
        ref_c = handy_ref_verts.astype(np.float32).mean(axis=0)
        # V_body is seam-centered at 0; move it so that the origin is the Handy centroid.
        # (Equivalent to replacing the old origin_point logic with a vertex-accurate anchor.)
        V_body = (
            V_body
            + (seam_center_mano - ref_c.astype(np.float32))
            + origin_point[None, :]
        )

    # ---- 5d) Remove now-unused vertices and reindex faces (kept_idx maps new->old ids) ----
    V_body, F_body, kept_idx, old2new = _compact_vertices(V_body, F_body)

    # ---- 6) Compute coloring for BODY vertices ----
    # Default uniform gray if no texture provided.
    kd = (0.8, 0.8, 0.8)
    per_vertex_rgb01 = None  # when set, we will write OBJ with vertex colors

    if handy_texture_img is not None and (uv_template_obj is None):
        uv_template = _get_hand_uv_template()
        tex_u8 = _np.asarray(handy_texture_img, dtype=_np.uint8)[..., :3]
        try:
            if (
                uv_template is not None
                and mano_mask.any()
                and arm_color_from_wrist_ring
            ):
                # (a) sample per-ring colors (K,3) in [0,1]
                wrist_uvs = uv_template[mano_mask]  # (K,2)
                wrist_colors = _sample_texture_colors_at_uvs(  # (K,3) float in [0,1]
                    texture_img=tex_u8, uvs01=wrist_uvs, pad_px=1
                )
                # (b) compute BODY vertices *before* compaction for nearest-ring lookup
                V_body_full = V_body.copy()  # (N,3) seam-centered & rotated already
                # Transform SMPL seam vertices the same way V_body was transformed
                seam_pts = V_smpl[seam_ids].copy()
                seam_pts = seam_pts + (seam_center_mano - seam_center_smpl)
                seam_pts = (R_fix @ (seam_pts - seam_center_mano).T).T
                # If we flipped later to enforce side consistency, reflect that too
                # (R180 around t1_hand) might have been applied; detect by sign mismatch and replicate:
                # We check s_body sign logic we used above:
                t1_hand = R_hand[:, 0]
                n_hand = R_hand[:, 2]
                s_body_tmp = _np.median(V_body_full @ n_hand)
                if _np.sign(s_body_tmp) != 1.0:
                    R180 = _rodrigues(t1_hand, _np.pi)
                    V_body_full = (R180 @ V_body_full.T).T
                    seam_pts = (R180 @ seam_pts.T).T

                # (c) nearest-neighbor on ring for each BODY vertex
                # distance matrix (N,K) – K ~ ring size (O(100)); N ~ arm verts (few k) → fine
                diff = V_body_full[:, None, :] - seam_pts[None, :, :]
                nn_idx = _np.argmin(_np.sum(diff * diff, axis=2), axis=1)  # (N,)
                per_vertex_rgb01_full = wrist_colors[nn_idx]  # (N,3) in [0,1]

                # (d) compact vertices (kept_idx) and gather matching colors
                V_body, F_body, kept_idx, old2new = _compact_vertices(
                    V_body_full, F_body
                )
                per_vertex_rgb01 = per_vertex_rgb01_full[kept_idx]

                # For tools that still consult Kd, also provide mean as material color
                kd = tuple(_np.clip(per_vertex_rgb01.mean(axis=0), 0.0, 1.0).tolist())
                # print(f"[SMPLX] Painted arm with per-vertex colors from wrist ring; Kd fallback={kd}.")
            else:
                # Fallback: mean wrist ring color (previous behavior)
                # print("[SMPLX] Warning: no UV template or wrist ring mask; using global mean color.")
                if uv_template is not None and mano_mask.any():
                    wrist_uvs = uv_template[mano_mask]
                    wrist_mean_rgb01 = _sample_texture_at_uvs(
                        texture_img=tex_u8, uvs01=wrist_uvs, pad_px=1
                    )
                    kd = tuple(float(x) for x in wrist_mean_rgb01)
                else:
                    kd = tuple(
                        (tex_u8.reshape(-1, 3).mean(axis=0) / 255.0).astype(float)
                    )
                # keep using V_body/F_body from earlier path (already compacted once below)
        except Exception as _e:
            # Last-resort fallback to global mean
            kd = tuple((tex_u8.reshape(-1, 3).mean(axis=0) / 255.0).astype(float))
            # print(f"[SMPLX][warn] Wrist-ring per-vertex paint failed: {_e}. Falling back to mean color {kd}.")

    # If we didn't run the per-vertex path above, ensure we compact once.
    # If we didn't run the per-vertex path above, we already compacted at step 5d.
    if per_vertex_rgb01 is None:
        pass

    out_dir = os.path.dirname(save_path)
    os.makedirs(out_dir, exist_ok=True)
    mtl_path = os.path.splitext(save_path)[0] + "_body.mtl"
    with open(mtl_path, "w") as mtl:
        mtl.write("newmtl body\n")
        mtl.write(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n")

    if not no_wrist_rot:
        V_body = (R_wrist @ (V_body - C_wrist).T).T + C_wrist

    with open(save_path, "w") as obj:
        obj.write(f"mtllib {os.path.basename(mtl_path)}\n")

        # If we have a canonical SMPL UV template, write vt + face v/vt so textures (e.g. SMPLitex) work.
        use_uv = False
        vt_list = None
        face_v2vt = None
        v_default = None
        if uv_template_obj is not None and os.path.exists(uv_template_obj):
            vt_list, face_v2vt, v_default = _load_smpl_uv_template(uv_template_obj)
            use_uv = vt_list is not None and len(vt_list) > 0

        # Prefer texture over vertex colors when UVs are enabled.
        if use_uv:
            per_vertex_rgb01 = None

        if per_vertex_rgb01 is not None:
            # OBJ with vertex colors (Assimp/Blender/MeshLab commonly support this extension)
            for v, rgb in zip(V_body, per_vertex_rgb01):
                obj.write(
                    f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n"
                )
        else:
            for v in V_body:
                obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if use_uv:
            for u, v in vt_list:
                obj.write(f"vt {u:.6f} {v:.6f}\n")

        obj.write("usemtl body\n")

        if use_uv:
            # Faces are indexed in the compacted mesh (V_body/F_body); UV lookup is done in the *canonical*
            # vertex id space using kept_idx (new->old).
            for f in F_body:
                i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
                ov0, ov1, ov2 = int(kept_idx[i0]), int(kept_idx[i1]), int(kept_idx[i2])
                key = tuple(sorted([ov0, ov1, ov2]))
                mm = face_v2vt.get(key, None) if face_v2vt is not None else None

                if mm is None:
                    # Fallback: pick any vt per vertex (may cause seams to look worse, but avoids crashing)
                    vt0 = int(v_default.get(ov0, 0))
                    vt1 = int(v_default.get(ov1, 0))
                    vt2 = int(v_default.get(ov2, 0))
                else:
                    vt0 = int(mm.get(ov0, v_default.get(ov0, 0)))
                    vt1 = int(mm.get(ov1, v_default.get(ov1, 0)))
                    vt2 = int(mm.get(ov2, v_default.get(ov2, 0)))

                a, b, c = i0 + 1, i1 + 1, i2 + 1
                ua, ub, uc = vt0 + 1, vt1 + 1, vt2 + 1
                obj.write(f"f {a}/{ua} {b}/{ub} {c}/{uc}\n")
        else:
            for f in F_body:
                a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
                obj.write(f"f {a} {b} {c}\n")

    if return_wrist_info:
        return seam_center_mano, seam_ids, mano_mask

    return seam_center_mano


def _sample_texture_colors_at_uvs(
    texture_img: np.ndarray, uvs01: np.ndarray, pad_px: int = 1
) -> np.ndarray:
    """
    Return per-UV average colors (K,3) in [0,1] by sampling a (2*pad_px+1)^2 patch per UV.
    """
    tex = np.asarray(texture_img, dtype=np.uint8)[..., :3]
    H, W = tex.shape[:2]
    us = np.clip((uvs01[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
    vs = np.clip((uvs01[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    K = uvs01.shape[0]
    out = np.zeros((K, 3), dtype=np.float32)
    for i, (u, v) in enumerate(zip(us, vs)):
        u0, u1 = max(0, u - pad_px), min(W - 1, u + pad_px)
        v0, v1 = max(0, v - pad_px), min(H - 1, v + pad_px)
        patch = tex[v0 : v1 + 1, u0 : u1 + 1, :].reshape(-1, 3)
        if patch.size:
            out[i] = patch.mean(axis=0).astype(np.float32) / 255.0
        else:
            out[i] = tex[v, u].astype(np.float32) / 255.0
    return out


def _compact_vertices(V: np.ndarray, F: np.ndarray):
    """
    Keep only vertices referenced by F and remap faces.
    Returns:
      V_new: (K,3)
      F_new: (M,3) with indices into V_new
      kept_idx: (K,) old vertex ids kept
     old2new: (N,) map, -1 for dropped verts
    """
    F = np.asarray(F, dtype=np.int64)
    V = np.asarray(V, dtype=np.float32)
    used = np.unique(F.reshape(-1))
    old2new = -np.ones(V.shape[0], dtype=np.int64)
    old2new[used] = np.arange(used.size, dtype=np.int64)
    V_new = V[used]
    F_new = old2new[F]
    return (
        V_new.astype(np.float32),
        F_new.astype(np.int32),
        used.astype(np.int64),
        old2new.astype(np.int64),
    )


def remove_faces_toward_left_from_right_shoulder(
    V: np.ndarray,
    F: np.ndarray,
    J_body: np.ndarray,
    right_shoulder_index: int = 17,
    left_shoulder_index: int = 16,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Remove faces whose all three vertices lie on the 'left-of-right-shoulder' side,
    i.e., toward the torso, using a plane at the right shoulder orthogonal to the
    right→left shoulder vector. Vertices are kept (faces only).
    """
    p = J_body[right_shoulder_index].astype(
        np.float32
    )  # point on plane (right shoulder)
    n = (J_body[left_shoulder_index] - p).astype(np.float32)  # normal: right→left
    n /= np.linalg.norm(n) + 1e-9

    signed = (V - p[None, :]) @ n  # >0 means 'left of the right shoulder'
    face_signed = signed[F]  # (M,3)
    faces_all_left = (face_signed > eps).all(axis=1)  # all three on torso/left side
    return F[~faces_all_left]


def _get_hand_uv_template() -> Optional[np.ndarray]:
    """
    Retrieve the per-vertex UV template (in [0,1]^2) for the Handy/MANO right hand.
    Tries a few common locations/APIs and returns (V,2) or None if unavailable.
    """
    # 1) Try to pull from HandyLayer if available (preferred; stable in many repos)
    try:
        from handylayer import HandyLayer

        _tmp = HandyLayer(
            use_pca=False,
            flat_hand_mean=False,
            ncomps=45,
            handy_assets_root="./manotorch/assets/handy",
            mano_assets_root="./manotorch/assets/mano",
        )
        if hasattr(_tmp, "uv_template") and _tmp.uv_template is not None:
            uv = np.asarray(_tmp.uv_template, dtype=np.float32)
            if uv.ndim == 2 and uv.shape[1] == 2:
                return uv
        if hasattr(_tmp, "get_uv_template"):
            uv = np.asarray(_tmp.get_uv_template(), dtype=np.float32)
            if uv.ndim == 2 and uv.shape[1] == 2:
                return uv
    except Exception:
        pass

    # 2) Fallback: look for a cached UV template file (optional, ignore if missing)
    for cand in [
        "assets/handy_uv_template_right.npy",
        "assets/models/handy_uv_template_right.npy",
        "manotorch/assets/handy/uv_template_right.npy",
    ]:
        if os.path.exists(cand):
            try:
                uv = np.load(cand).astype(np.float32)
                if uv.ndim == 2 and uv.shape[1] == 2:
                    return uv
            except Exception:
                pass
    return None


def _sample_texture_at_uvs(
    texture_img: np.ndarray, uvs01: np.ndarray, pad_px: int = 1
) -> np.ndarray:
    """
    texture_img: (H,W,3) uint8
    uvs01: (K,2) in [0,1] (u,v)
    pad_px: neighborhood half-size in pixels (1 => 3x3)
    Returns mean RGB (3,) in [0,1].
    """
    tex = np.asarray(texture_img, dtype=np.uint8)[..., :3]
    H, W = tex.shape[:2]
    us = np.clip((uvs01[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
    vs = np.clip((uvs01[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    samples = []
    for u, v in zip(us, vs):
        u0, u1 = max(0, u - pad_px), min(W - 1, u + pad_px)
        v0, v1 = max(0, v - pad_px), min(H - 1, v + pad_px)
        patch = tex[v0 : v1 + 1, u0 : u1 + 1, :].reshape(-1, 3)
        if patch.size:
            samples.append(patch)
    if not samples:
        return tex.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
    arr = np.concatenate(samples, axis=0)
    return arr.mean(axis=0).astype(np.float32) / 255.0


def _rodrigues(axis, theta):
    axis = axis.astype(np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    I = np.eye(3, dtype=np.float32)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rot_between(a, b):
    """Minimal rotation R that sends unit vector a -> unit vector b."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a /= np.linalg.norm(a) + 1e-9
    b /= np.linalg.norm(b) + 1e-9
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)  # already aligned
        # 180°: choose any axis ⟂ a
        tmp = np.array([1, 0, 0], dtype=np.float32)
        if abs(np.dot(tmp, a)) > 0.99:
            tmp = np.array([0, 1, 0], dtype=np.float32)
        axis = tmp - np.dot(tmp, a) * a
        axis /= np.linalg.norm(axis) + 1e-9
        return _rodrigues(axis, np.pi)
    axis = v / (s + 1e-9)
    ang = np.arctan2(s, c)
    return _rodrigues(axis, ang)


def _smpl_wrist_boundary_loop(
    F_smpl: np.ndarray, hand_vertex_ids: np.ndarray
) -> np.ndarray:
    """
    Return the set of SMPL vertex ids that lie on the open boundary between the kept arm mesh
    and the removed hand. `hand_vertex_ids` should be the set of SMPL vertices that belong to the hand.
    """
    hand_set = set(int(v) for v in hand_vertex_ids.tolist())
    boundary = set()
    # scan each face; for each edge, if it crosses hand↔nonhand, mark both end verts
    for f in F_smpl.astype(np.int64):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        e01 = (a in hand_set) ^ (b in hand_set)
        e12 = (b in hand_set) ^ (c in hand_set)
        e20 = (c in hand_set) ^ (a in hand_set)
        if e01:
            boundary.update([a, b])
        if e12:
            boundary.update([b, c])
        if e20:
            boundary.update([c, a])
    return np.array(sorted(boundary), dtype=np.int64)


def _load_smplh_no_pca_via_obman(
    smplh_pkl_path: str,
    mano_models_dir: str,
    right_hand_pose_45: np.ndarray,
    wrist_aa: np.ndarray,
    betas_10: np.ndarray,
    randomize_arm_pose: bool = True,
    flat_hand_mean: bool = False,
):
    """
    Load SMPL-H vertices/faces WITHOUT PCA using the ObMan-style ops:
    fullpose = [ body(66) ; left_hand(45) ; right_hand(45) ].
    - left hand = mean (or zeros if flat_hand_mean=True)
    - right hand = your MANO right-hand 45D axis-angle
    - betas = your hand betas (first 10)
    Returns V (6890,3), F (Fx3)
    """
    # ---- load SMPLH dict (pickle) ----
    with open(smplh_pkl_path, "rb") as f:
        smpl_data = pickle.load(f, encoding="latin1")
    # Some NPZ→PKL conversions leave np.lib.npyio.NpzFile-like dicts; force a plain dict
    if not isinstance(smpl_data, dict):
        smpl_data = dict(smpl_data)

    # ---- load MANO LEFT/RIGHT to get hand means (not PCA) ----
    with open(os.path.join(mano_models_dir, "MANO_LEFT.pkl"), "rb") as f:
        hand_l = pickle.load(f, encoding="latin1")
    with open(os.path.join(mano_models_dir, "MANO_RIGHT.pkl"), "rb") as f:
        hand_r = pickle.load(f, encoding="latin1")
    hands_meanl = (
        np.zeros(45, dtype=np.float32)
        if flat_hand_mean
        else hand_l["hands_mean"].astype(np.float32)
    )
    hands_meanr = (
        np.zeros(45, dtype=np.float32)
        if flat_hand_mean
        else hand_r["hands_mean"].astype(np.float32)
    )

    body_pose_dofs = 66
    body_pose = np.zeros((body_pose_dofs,), dtype=np.float32)
    if randomize_arm_pose:

        def _aa(deg):
            return np.deg2rad(np.random.uniform(-deg, deg, size=3)).astype(np.float32)

        body_pose[17 * 3 : 17 * 3 + 3] = _aa(15)  # right shoulder
        body_pose[19 * 3 : 19 * 3 + 3] = _aa(25)  # right elbow

    # Apply MANO wrist at right wrist
    if wrist_aa is not None:
        body_pose[21 * 3 : 21 * 3 + 3] = wrist_aa

    # right hand pose comes from MANO: 45D axis-angle (we assume you pass only the 45D vector)
    rh = right_hand_pose_45.astype(np.float32)
    lh = hands_meanl  # left stays at mean (or zeros)

    fullpose = np.concatenate([body_pose, lh, rh]).astype(np.float32)  # (66+45+45=156)

    # ---- chumpy prep (adapted from ObMan) ----
    dd = dict(smpl_data)  # shallow copy
    # Ensure required SMPL-H fields exist
    dd.setdefault("bs_type", "lrotmin")
    dd.setdefault("bs_style", "lbs")
    # Wrap only the arrays that are used in chumpy ops.
    # Keep kintree_table and faces as numpy (verts_core expects ints, not chumpy Selects).
    for k in ["v_template", "weights", "posedirs", "J", "shapedirs"]:
        if k in dd and not hasattr(dd[k], "dterms"):
            dd[k] = ch.array(dd[k])
    # Ensure kintree_table is numpy int array of shape (2, 24)
    if "kintree_table" in dd:
        dd["kintree_table"] = np.asarray(dd["kintree_table"]).astype(np.int64)
    # Faces should stay numpy
    if "f" in dd and hasattr(dd["f"], "dterms"):
        dd["f"] = np.asarray(dd["f"])
    # J_regressor can be ndarray or any scipy sparse format → convert to CSR
    if "J_regressor" in dd:
        Jreg = dd["J_regressor"]
        if sp.issparse(Jreg):
            dd["J_regressor"] = Jreg.tocsr()
        elif isinstance(Jreg, np.ndarray):
            # If already dense matrix of shape (24, 6890), wrap as CSR
            dd["J_regressor"] = sp.csr_matrix(Jreg)
    # Prepare betas
    if "betas" in dd:
        b = np.zeros(int(np.asarray(dd["shapedirs"]).shape[-1]), dtype=np.float32)
        b[: min(10, b.shape[0])] = betas_10[: min(10, b.shape[0])]
        dd["betas"] = ch.array(b)
    else:
        dd["betas"] = ch.zeros(dd["shapedirs"].shape[-1])
    # set user betas (first 10)
    if "betas" in dd:
        b = np.zeros_like(np.asarray(dd["betas"]).ravel())
        b[: min(10, b.shape[0])] = betas_10[: min(10, b.shape[0])]
        dd["betas"] = ch.array(b)

    # build v_shaped, J, and v_posed using fullpose for posedirs
    shapedirs = dd["shapedirs"]
    v_template = dd["v_template"]
    v_shaped = shapedirs.dot(dd["betas"]) + v_template
    dd["v_shaped"] = v_shaped
    Jx = ch.MatVecMult(dd["J_regressor"], v_shaped[:, 0])
    Jy = ch.MatVecMult(dd["J_regressor"], v_shaped[:, 1])
    Jz = ch.MatVecMult(dd["J_regressor"], v_shaped[:, 2])
    dd["J"] = ch.vstack((Jx, Jy, Jz)).T

    # posedirs expects bs_type mapping; ObMan used posemap internally; here we feed axis-angle directly
    from mano_v1_2.webuser.posemapper import posemap

    dd["v_posed"] = v_shaped + dd["posedirs"].dot(
        posemap(dd["bs_type"])(ch.array(fullpose))
    )

    # verts_core
    from mano_v1_2.webuser.verts import verts_core

    result_prev, meta = verts_core(
        pose=ch.array(fullpose),
        v=dd["v_posed"],
        J=dd["J"],
        weights=dd["weights"],
        kintree_table=dd["kintree_table"],
        xp=ch,
        want_Jtr=True,
        bs_style=dd["bs_style"],
    )
    result = result_prev  # no global trans
    V = np.asarray(result, dtype=np.float32)
    F = np.asarray(dd.get("f", smpl_data.get("f")), dtype=np.int32)
    J = np.asarray(dd["J"], dtype=np.float32)
    return V, F, J


def _load_smplh_vertices_faces(
    smpl_models_dir: str,
    smpl_gender: str,
    right_hand_pose_45: np.ndarray,
    betas_10: np.ndarray,
    randomize_arm_pose: bool,
    smplh_pkl_path: str,
    mano_models_dir: str,
    wrist_aa: np.ndarray,
):
    # Try smplx (preferred)

    # Resolve likely on-disk model files up-front so we never reference before assignment.
    smplh_path = None
    try:
        gender = (smpl_gender or "neutral").lower()
    except Exception:
        gender = "neutral"
    smplh_npz = os.path.join(smpl_models_dir, "smplh", gender, "model.npz")
    smplh_pkl_guess = os.path.join(
        smpl_models_dir, "smplh", f"SMPLH_{gender.upper()}.pkl"
    )
    if os.path.isfile(smplh_npz):
        smplh_path = smplh_npz
    elif os.path.isfile(smplh_pkl_guess):
        smplh_path = smplh_pkl_guess

    try:
        exp_hint = (
            smplh_path
            if smplh_path is not None
            else "smplh/<gender>/model.npz or smplh/SMPLH_<GENDER>.pkl"
        )
        # print(f"[SMPLX] model_path='{smpl_models_dir}', gender='{smpl_gender}' (expects {exp_hint})")
        model = smplx.create(
            model_path=smpl_models_dir,  # parent containing "smplh"
            model_type="smplh",
            gender=smpl_gender,
            num_betas=10,
            use_pca=False,
            num_pca_comps=0,
            flat_hand_mean=False,  # you asked for False
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
        )

        # body_pose = np.zeros((21, 3), dtype=np.float32)
        # body_pose[18] = wrist_aa  # keep MANO wrist

        # if randomize_arm_pose:
        #     def _aa(deg): return np.deg2rad(np.random.uniform(-deg, deg, size=3)).astype(np.float32)
        #     body_pose[16] = _aa(15)   # shoulder
        #     body_pose[17] = _aa(25)   # elbow

        # ----- build body_pose with the correct right-arm joints -----
        # Try to get the actual body joint count from the model
        num_body = 21
        try:
            # smplx models expose NUM_BODY_JOINTS
            num_body = int(getattr(model, "NUM_BODY_JOINTS", 21))
        except Exception:
            pass
        body_pose = np.zeros((num_body, 3), dtype=np.float32)

        # Right arm indices across common layouts:
        #   22-joint layout: shoulder=17, elbow=19, wrist=21
        #   21-joint layout: shoulder=17, elbow=19, wrist=20
        RIGHT_SHOULDER = 17
        RIGHT_ELBOW = 19
        RIGHT_WRIST_22 = 21
        RIGHT_WRIST_21 = 20

        def _set_joint(jidx, aa):
            if 0 <= jidx < body_pose.shape[0]:
                body_pose[jidx] = aa

        # Randomize right shoulder/elbow if requested
        if randomize_arm_pose:

            def _aa(deg):
                return np.deg2rad(np.random.uniform(-deg, deg, size=3)).astype(
                    np.float32
                )

            _set_joint(RIGHT_SHOULDER, _aa(15))
            _set_joint(RIGHT_ELBOW, _aa(25))

        # Apply MANO wrist axis-angle to the right wrist
        # Prefer 22-layout index; fall back to 21-layout if needed
        if wrist_aa is not None:
            if RIGHT_WRIST_22 < body_pose.shape[0]:
                _set_joint(RIGHT_WRIST_22, wrist_aa)
            else:
                _set_joint(RIGHT_WRIST_21, wrist_aa)

        out = model(
            betas=betas_10.reshape(1, -1).astype(np.float32),
            body_pose=body_pose.reshape(1, -1),
            global_orient=np.zeros((1, 3), dtype=np.float32),
            left_hand_pose=np.zeros((1, 45), dtype=np.float32),
            right_hand_pose=right_hand_pose_45.reshape(1, -1).astype(np.float32),
            pose2rot=True,
        )
        V = out.vertices.detach().cpu().numpy()[0]
        F = model.faces.astype(np.int32)
        J = out.joints.detach().cpu().numpy()[0]  # SMPL-H joints
        return V, F, J

    except Exception as e:
        # Any failure in smplx (missing files, PCA-key mismatch, etc.) → fallback
        # print(f"[SMPLX] load failed: {type(e).__name__}: {e}\n[SMPLX] Falling back to ObMan no-PCA loader…")
        if not smplh_pkl_path or not mano_models_dir:
            raise RuntimeError(
                "Fallback requires both 'smplh_pkl_path' (SMPLH_*.pkl) and 'mano_models_dir' "
                "(contains MANO_LEFT.pkl / MANO_RIGHT.pkl)."
            )
        return _load_smplh_no_pca_via_obman(
            smplh_pkl_path=smplh_pkl_path,
            mano_models_dir=mano_models_dir,
            right_hand_pose_45=right_hand_pose_45,
            betas_10=betas_10,
            randomize_arm_pose=randomize_arm_pose,
            flat_hand_mean=False,
            wrist_aa=wrist_aa,
        )
