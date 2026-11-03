# create_sync_data_graspxl.py
"""
GraspXL-compatible data generator.
- Samples MANO *sequences* (frames) from GraspXL instead of single poses.
- Exports Handy right-hand mesh (+ optional SMPL-X body/arm, identical to current sync pipeline).
- Loads a grasped *object* mesh (OBJ, no texture) and assigns a random solid color.
- Hand root rotation ('rot') is zeroed at render time; the object is rigidly re-expressed
  relative to the hand to preserve the grasp geometry:
    R_obj' = R_hand^{-1} * R_obj
    t_obj' = R_hand^{-1} * (t_obj - t_hand)

Outputs mirror the base create_sync_data.py structure, but add an object into the scene.
This script intentionally reuses helpers from `sync_data_utils_graspxl.py` and
`sync_data_utils.py` (lighting / background / envmaps / zip-face helpers).

Assumptions
-----------
- Directory layout follows the GraspXL documentation the user provided:
  object_dataset/<size>/<object_id>/<object_id>.obj
  mano_dataset/<size>/<object_id>/<seq_id>.npy
- The Hydra config provides:
  cfg.generate_mesh.grasp_xl_mano_dir
  cfg.generate_mesh.grasp_xl_obj_dir
  cfg.simulate.n_frames  (top-K closest frames to object per sequence)
  and all existing knobs used by create_sync_data.py (intrinsics, n_images, etc.)

"""

from __future__ import annotations
import os, json, random, math
from scipy.spatial.transform import Rotation as R
import numpy as np
import hydra
from omegaconf import DictConfig

from sync_data_utils_graspxl import (
    get_topk_close_frames,
    axis_to_quaternion,
    axis_angle_to_matrix,
    colorize_obj_to_tmp,
    run_simulation_sapien_graspxl,
    ensure_textured_objaverse_obj,
    transform_obj_keep_materials,
)

# Reuse background utilities and arm-zip helpers from your existing utils
from sync_data_utils import (
    get_random_background,
    render_envmap_to_ldr,
    apply_zip_faces_to_body,
    assign_random_arm_texture,
    augment_skin_tone_random,
    alpha_mask_from_bg,
    is_skin_like_rgb_samples,
    lab_shift,
    resolve_obj_map_kd,
    skin_mask_ycrcb_rgb,
)

from moge_utils import (
    warmup_moge,
    moge_depth_from_rgb,
    adjust_bg_depth_layering,
    compose_hand_bg_depth,
)


def _iter_object_seq_pairs(root_dir: str):
    """Yield (object_id, seq_id, path_to_npy) from all size subfolders."""
    for size_dir in os.listdir(root_dir):  # e.g., small, medium, large
        size_path = os.path.join(root_dir, size_dir)
        if not os.path.isdir(size_path):
            continue
        for object_id in os.listdir(size_path):
            obj_path = os.path.join(size_path, object_id)
            if not os.path.isdir(obj_path):
                continue
            for file_name in os.listdir(obj_path):
                if file_name.endswith(".npy"):
                    seq_id = file_name[:-4]  # strip ".npy"
                    yield size_dir, object_id, seq_id, os.path.join(obj_path, file_name)


def generate_beta_batch(
    npz_path: str,
    N: int,
):
    """
    Returns: torch.FloatTensor of shape (N, 48): [0,0,0] + sampled 45D pose
    """
    import numpy as np
    import torch

    def load_poses(npz_path: str) -> np.ndarray:
        data = np.load(npz_path)
        poses = data["poses"]  # (N, 48)
        print(f"Loaded poses shape: {poses.shape}")  # (N, 48)
        poses_45 = poses[:, 3:].astype(
            np.float32, copy=False
        )  # drop global (root) rotation
        betas = data["betas"]
        print(f"Loaded betas shape: {betas.shape}")  # (N, 10)
        return poses_45, betas

    rng = np.random.default_rng()
    X, X_beta = load_poses(npz_path)  # (M, 48)
    M, D = X.shape

    idx = rng.integers(0, M, size=N)
    sampled_beta = torch.from_numpy(X_beta[idx])  # (N, 10)
    M, D = sampled_beta.shape
    assert D == 10, f"Expected 10-dim betas, got {D}"

    return sampled_beta


def transform_B_to_A(tA, rA_axis_angle, tB, rB_axis_angle):
    """
    Express object B's pose in object A's coordinate frame for a sequence.

    Args:
        tA : (T, 3) translations of A in world frame
        rA_axis_angle : (T, 3) axis-angle rotations of A in world frame (radians)
        tB : (T, 3) translations of B in world frame
        rB_axis_angle : (T, 3) axis-angle rotations of B in world frame (radians)

    Returns:
        tB_in_A : (T, 3) translations of B expressed in A's frame
        rB_in_A_axis_angle : (T, 3) axis-angle rotations of B expressed in A's frame
    """
    # to arrays and validate shapes
    tA = np.asarray(tA, dtype=np.float64).reshape(-1, 3)
    rA_axis_angle = np.asarray(rA_axis_angle, dtype=np.float64).reshape(-1, 3)
    tB = np.asarray(tB, dtype=np.float64).reshape(-1, 3)
    rB_axis_angle = np.asarray(rB_axis_angle, dtype=np.float64).reshape(-1, 3)

    if not (tA.shape == rA_axis_angle.shape == tB.shape == rB_axis_angle.shape):
        raise ValueError("All inputs must have identical shape (T, 3).")

    # Rotations of A and B in world frame (batched)
    RA = R.from_rotvec(rA_axis_angle)  # (T,)
    RB = R.from_rotvec(rB_axis_angle)  # (T,)

    # Rotation of B in A's frame: R_A^T * R_B  (RA.inv() is R_A^T)
    RA_inv = RA.inv()
    R_B_in_A = RA_inv * RB

    # Translation of B in A's frame: R_A^T * (tB - tA)
    tB_in_A = RA_inv.apply(tB - tA)  # (T, 3)

    # Back to axis-angle for rotation
    rB_in_A_axis_angle = R_B_in_A.as_rotvec()  # (T, 3)

    return tB_in_A, rB_in_A_axis_angle


@hydra.main(
    version_base=None, config_path="configs", config_name="create_sync_data_graspxl"
)
def main(cfg: DictConfig):
    print("starting GraspXL data generation...")
    generate_data_graspxl(cfg)


def generate_data_graspxl(cfg: DictConfig):
    # Heavy imports inside for faster module import in wrappers
    import torch
    import h5py
    from tqdm import tqdm
    from manotorch.manolayer import ManoLayer
    from handylayer import HandyLayer
    from PIL import Image

    ROOT_DIR = cfg.generate_mesh.root_dir

    # --- I/O dirs
    save_dir = cfg.generate_mesh.mesh_save_dir
    temp_dir = cfg.generate_mesh.mesh_temp_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # --- MANO/Handy
    beta_dim = cfg.generate_mesh.mano_betas_size
    pose_dim = cfg.generate_mesh.mano_poses_size

    mano_layer = ManoLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=pose_dim,
        mano_assets_root=ROOT_DIR + "/" + "manotorch/assets/mano",
    )
    handy_layer = HandyLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=pose_dim,
        shape_ncomps=beta_dim,
        handy_assets_root=ROOT_DIR + "/" + "manotorch/assets/handy",
        mano_assets_root=ROOT_DIR + "/" + "manotorch/assets/mano",
    )

    mano_layer_FLAT_HAND_MEAN_FALSE = ManoLayer(
        use_pca=False,
        flat_hand_mean=False,
        ncomps=pose_dim,
        mano_assets_root=ROOT_DIR + "/" + "manotorch/assets/mano",
    )
    FLAT_HAND_MEAN = (
        mano_layer_FLAT_HAND_MEAN_FALSE.th_hands_mean.squeeze(0).cpu().numpy()
    )
    # print(FLAT_HAND_MEAN.shape, FLAT_HAND_MEAN)

    # --- scene counts
    n_frames_per_seq = cfg.simulate.n_frames_per_seq
    n_images = cfg.simulate.n_images
    RGB_width = cfg.generate_mesh.RGB_width
    RGB_height = cfg.generate_mesh.RGB_height

    # --- build the sampling pool (object_id, seq_id, path_to_npy)
    mano_dir = ROOT_DIR + "/" + cfg.generate_mesh.grasp_xl_mano_dir
    obj_dir = ROOT_DIR + "/" + cfg.generate_mesh.grasp_xl_obj_dir
    pool = [(r, a, b, p) for r, a, b, p in _iter_object_seq_pairs(mano_dir)]
    if not pool:
        raise RuntimeError(f"No GraspXL sequences found under {mano_dir}")

    # total scenes to generate & batch size (one scene = one frame from a sequence)
    N_total = int(cfg.generate_mesh.num_generate)
    M = int(cfg.generate_mesh.batch_size)
    n_batches = int(np.ceil(N_total / M))
    # sample indices WITH replacement
    sampled_idx = np.random.randint(0, len(pool), size=N_total)

    print(
        f"Generating {N_total} scenes from {len(pool)} GraspXL sequences in {n_batches} batches of {M}"
    )

    # --- backgrounds
    background_dir = ROOT_DIR + "/" + cfg.simulate.background.background_dir
    env_map_dir = ROOT_DIR + "/" + cfg.simulate.background.env_map_dir
    if cfg.simulate.background.env_map_dir != "":
        with open(os.path.join(env_map_dir, "configs.json")) as file:
            env_map_data = json.load(file)
            env_map_list = env_map_data["env_maps"]

        env_map_selected = random.choices(env_map_list, k=N_total)

        BG_FLAG = "env_map"
    elif cfg.simulate.background.background_dir != "":
        bg_files = [
            os.path.join(background_dir, f)
            for f in os.listdir(background_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))
        ]
        BG_FLAG = "bg_image"
    else:
        BG_FLAG = None

    use_moge = bool(
        getattr(cfg.simulate, "moge_bg_depth", None) and cfg.simulate.moge_bg_depth.flag
    )
    if use_moge:
        warmup_moge(model_name=str(cfg.simulate.moge_bg_depth.model_name))

    # --- process in batches (match create_sync_data.py save cadence)
    for batch_id in range(n_batches):
        start_idx = batch_id * M
        B = min(M, N_total - start_idx)

        print(
            f"\n=== Batch {batch_id + 1}/{n_batches} "
            f"({start_idx} ... {start_idx + B - 1}) ==="
        )

        # per-batch buffers (same shapes/order as create_sync_data.py)
        image_rgb_data = np.zeros((B, n_images, RGB_width, RGB_height, 3), np.uint8)
        depth_data = np.zeros((B, n_images, RGB_width, RGB_height), np.float16)
        camera_poses = np.zeros((B, n_images, 4, 4), np.float16)
        bbox_data = np.zeros((B, n_images, 4, 2), np.float16)
        fov_rads_data = np.zeros((B, n_images), np.float16)
        mask_data = np.zeros((B, n_images, RGB_width, RGB_height), np.float16)

        keep_hand = use_moge and getattr(
            cfg.simulate.moge_bg_depth, "keep_hand_depth", False
        )
        if keep_hand:
            depth_hand_only_data = np.zeros(
                (B, n_images, RGB_width, RGB_height), np.float32
            )

        pose_batch = np.zeros((B, pose_dim + 3), np.float32)
        save_verts = getattr(cfg.generate_mesh, "save_verts", False)
        if save_verts:
            verts_data = np.zeros((B, 778, 3), np.float32)
        beta_batch = generate_beta_batch(
            npz_path=ROOT_DIR + "/" + cfg.generate_mesh.real_pose_npz_path, N=B
        )
        obj_ids_batch = [""] * B
        seq_ids_batch = [""] * B
        frame_batch = np.zeros(B, dtype=np.int32)

        for i_local, idx in enumerate(
            tqdm(
                sampled_idx[start_idx : start_idx + B],
                desc=f"Batch {batch_id + 1}/{n_batches}",
            )
        ):
            size_folder, obj_id, seq_id, cfg_path = pool[idx]
            data = np.load(cfg_path, allow_pickle=True).item()

            hand_trans_all = np.asarray(data["right_hand"]["trans"])  # (T,3)
            hand_rot_all = np.asarray(data["right_hand"]["rot"])  # (T,3) axis-angle
            hand_pose_all = (
                np.asarray(data["right_hand"]["pose"]) + FLAT_HAND_MEAN
            )  # (T,45)
            obj_trans_all = np.asarray(data[obj_id]["trans"])  # (T,3)
            obj_rot_all = np.asarray(data[obj_id]["rot"])  # (T,3)

            # Re-express object ◀in hand coordinates▶ (vectorized over frames)
            # obj_trans_all, obj_rot_all = transform_B_to_A(
            #     hand_trans_all, hand_rot_all, obj_trans_all, obj_rot_all
            # )

            obj_trans_all -= hand_trans_all

            # pick ONE frame from top-K closest hand↔object frames
            topk = max(1, int(n_frames_per_seq))
            sel = get_topk_close_frames(obj_trans_all, k=topk)
            t = int(random.choice(sel))
            # t = 40

            # constant random handy skin texture per (obj,seq)
            orig_texture = None  # RGB uint8; needed later for hand↔arm seam sync
            if getattr(cfg.generate_mesh, "n_textures", 0) > 0:
                from create_sync_data import load_texture  # reuse your existing loader

                tex_idx = np.random.randint(cfg.generate_texture.n_textures)
                orig_texture = load_texture(cfg, tex_idx)  # RGB uint8 (unmodified)

                # Match create_sync_data.py behavior:
                # build a possibly-augmented texture for initial hand OBJ export,
                # but keep orig_texture around for seam sampling + later sync overwrite.
                if (
                    getattr(cfg.generate_mesh, "texture_color_shift", None)
                    and cfg.generate_mesh.texture_color_shift.flag
                ):
                    # augment_skin_tone_random expects BGR
                    aug_bgr = augment_skin_tone_random(
                        orig_texture[:, :, ::-1],
                        p=float(cfg.generate_mesh.texture_color_shift.percent),
                    )
                    texture_image = Image.fromarray(aug_bgr[:, :, ::-1])
                else:
                    texture_image = Image.fromarray(orig_texture)
            else:
                texture_image = None

            # # prepare OBJ (single mesh + random color)
            # raw_obj_path = os.path.join(obj_dir, f"{size_folder}/{obj_id}/{obj_id}.obj")
            # out_obj_path = os.path.join(temp_dir, f"{i_local}_obj.obj")
            # colored_obj_path = colorize_obj_to_tmp(raw_obj_path, out_obj_path=out_obj_path, \
            #                                        obj_trans=obj_trans_all[t], obj_rot=obj_rot_all[t])

            # prepare object mesh to render

            raw_obj_path = os.path.join(obj_dir, f"{size_folder}/{obj_id}/{obj_id}.obj")

            use_textured = bool(
                getattr(cfg.generate_mesh, "use_textured_objaverse", False)
            )
            if use_textured:
                # 1) ensure cached rescaled textured OBJ exists next to simplified mesh
                textured_cached = ensure_textured_objaverse_obj(
                    obj_root_dir=obj_dir,
                    size_folder=size_folder,
                    obj_id=obj_id,
                    simplified_obj_path=raw_obj_path,
                )
                # 2) bake this frame's rigid transform into the OBJ while preserving materials/UVs
                obj_mesh_for_render = transform_obj_keep_materials(
                    src_obj_path=textured_cached,
                    out_obj_path=os.path.join(temp_dir, f"{i_local}_obj_tx.obj"),
                    obj_trans=obj_trans_all[t],
                    obj_rot=obj_rot_all[t],
                )
            else:
                # original solid-color path (also bakes the transform)
                obj_mesh_for_render = colorize_obj_to_tmp(
                    raw_obj_path,
                    out_obj_path=os.path.join(temp_dir, f"{i_local}_obj.obj"),
                    obj_trans=obj_trans_all[t],
                    obj_rot=obj_rot_all[t],
                )

            # # ----- MANO params (for selected frame t)
            # beta = np.random.randn(beta_dim).astype(np.float32) * float(cfg.generate_mesh.beta_std)

            # rot *will be zeroed* for geometry export
            hand_rot = hand_rot_all[t].astype(np.float32)
            hand_pose = hand_pose_all[t].astype(np.float32)

            # store GT (rot+pose) into H5 for traceability
            pose_full = np.concatenate([hand_rot, hand_pose], axis=0)
            # beta_batch[i_local] = beta
            pose_batch[i_local] = pose_full
            obj_ids_batch[i_local] = obj_id
            seq_ids_batch[i_local] = seq_id
            frame_batch[i_local] = t

            beta_b = beta_batch[i_local].unsqueeze(0)
            with torch.no_grad():
                mano_out = mano_layer(torch.from_numpy(pose_full).unsqueeze(0), beta_b)
                # convert MANO→Handy topology
                handy_verts = (
                    handy_layer.mano_to_handy(mano_out.verts)[0].detach().cpu().numpy()
                )
                if save_verts:
                    verts_data[i_local] = mano_out.verts[0].detach().cpu().numpy()

            with torch.no_grad():
                pose_zero_root = np.concatenate([np.zeros(3), hand_pose])
                mano_out_zero_root = mano_layer(
                    torch.from_numpy(pose_zero_root).unsqueeze(0).float(), beta_b
                )
                handy_verts_zero_root = (
                    handy_layer.mano_to_handy(mano_out_zero_root.verts)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                joint_zero_root = mano_out_zero_root.joints[0].cpu().numpy()

            R_wrist = R.from_rotvec(hand_rot).as_matrix()
            # print(R_wrist)

            # export hand mesh OBJ for this frame
            hand_obj_path = os.path.join(temp_dir, f"{i_local}_hand.obj")
            # hand_obj_path_zr = os.path.join(temp_dir, f"{i_local}_hand_zr.obj")
            if texture_image is not None:
                # use the already-initialized handy_layer to write a textured obj
                _ = handy_layer.to_textured_obj(
                    handy_verts, texture_image, save_path=hand_obj_path
                )
                # _ = handy_layer.to_textured_obj(handy_verts_zero_root, texture_image, save_path=hand_obj_path_zr)
            else:
                import trimesh

                m = trimesh.Trimesh(
                    vertices=handy_verts,
                    faces=mano_layer.th_faces.numpy(),
                    process=False,
                )
                m.export(hand_obj_path)
                # m = trimesh.Trimesh(vertices=handy_verts_zero_root, faces=mano_layer.th_faces.numpy(), process=False)
                # m.export(hand_obj_path_zr)

            # optional: body/arm mesh aligned to this hand (same method as base)
            body_obj_path = None
            if getattr(cfg.generate_mesh, "add_arm", None) and getattr(
                cfg.generate_mesh.add_arm, "flag", False
            ):
                from smplx_helper import export_body_arm_aligned_to_mano_seam

                hand_origin = handy_verts_zero_root.mean(axis=0).astype(np.float32)
                body_obj_path = os.path.join(temp_dir, f"{i_local}_body.obj")

                # Needed for short-sleeve seam sampling + sync
                seam_ids_smpl = None
                mano_wrist_mask = None

                # NEW: enable UVs on BODY by mapping from canonical SMPL UV template (for SMPLitex textures)
                uv_template_obj = getattr(
                    cfg.generate_mesh.add_arm, "uv_template_obj", None
                )
                if uv_template_obj and (not os.path.isabs(uv_template_obj)):
                    uv_template_obj = os.path.join(ROOT_DIR, uv_template_obj)

                seam_center_mano, seam_ids_smpl, mano_wrist_mask = (
                    export_body_arm_aligned_to_mano_seam(
                        save_path=body_obj_path,
                        smpl_models_dir=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.smpl_models_dir,
                        smpl_gender=getattr(
                            cfg.generate_mesh.add_arm, "smpl_gender", "neutral"
                        ),
                        hand_beta=beta_batch[i_local].detach().cpu().numpy(),
                        hand_pose=pose_zero_root,
                        smplh_pkl_path=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.smplh_pkl_path,
                        mano_models_dir=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.mano_models_dir,
                        smpl2mano_right_path=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.smpl2mano_right_path,
                        mano_vertices=mano_out_zero_root.verts[0]
                        .detach()
                        .cpu()
                        .numpy(),
                        handy_texture_img=texture_image,
                        randomize_arm_pose=getattr(
                            cfg.generate_mesh.add_arm, "randomize_arm_pose", False
                        ),
                        wrist_jitter_deg=getattr(
                            cfg.generate_mesh.add_arm, "wrist_jitter_deg", 0.0
                        ),
                        seam_twist_jitter_deg=getattr(
                            cfg.generate_mesh.add_arm, "seam_twist_jitter_deg", 0.0
                        ),
                        origin_point=hand_origin,  # NEW: recenter BODY to hand centroid
                        handy_ref_verts=handy_verts_zero_root,
                        remove_mask_pkl=ROOT_DIR + "/" + "keep_right_arm.pkl",
                        prefer_pkl=True,
                        no_wrist_rot=False,
                        R_wrist=R_wrist,
                        C_wrist=joint_zero_root[0],
                        uv_template_obj=uv_template_obj,
                        return_wrist_info=True,
                    )
                )
                # Optional face stitching (same as base)
                zip_pkl_path = ROOT_DIR + "/" + "zip_arm.pkl"
                if os.path.exists(zip_pkl_path):
                    try:
                        # Optionally write a separate file; here we overwrite BODY in place.
                        patched_path = apply_zip_faces_to_body(
                            body_obj_path=body_obj_path,
                            hand_obj_path=hand_obj_path,
                            zip_pkl_path=zip_pkl_path,
                            out_path=body_obj_path,  # overwrite; or f"{body_obj_path[:-4]}_zip.obj"
                        )
                        print(
                            f"[zip] Applied {os.path.basename(zip_pkl_path)} → {patched_path}"
                        )
                    except Exception as e:
                        print(
                            f"[zip][warn] Failed to apply zip faces: {type(e).__name__}: {e}"
                        )
                else:
                    print("[zip] No zip_hand.pkl found; skipping face graft.")

                # NEW: If enabled, stamp a realistic arm texture onto BODY MTL (e.g., SMPLitex)
                rar = getattr(cfg.generate_mesh.add_arm, "realistic_arm_texture", None)
                if rar and getattr(rar, "flag", False):
                    tex_dir = getattr(rar, "textures_dir", None)
                    tex_zip = getattr(rar, "textures_zip", None)

                    if tex_dir and (not os.path.isabs(tex_dir)):
                        tex_dir = os.path.join(ROOT_DIR, tex_dir)
                    if tex_zip and (not os.path.isabs(tex_zip)):
                        tex_zip = os.path.join(ROOT_DIR, tex_zip)

                    # Copy chosen PNG into temp_dir so the renderer can resolve map_Kd
                    copy_dir = temp_dir
                    try:
                        used_png = assign_random_arm_texture(
                            obj_path=body_obj_path,
                            textures_dir=tex_dir,
                            textures_zip=tex_zip,
                            copy_to_dir=copy_dir,
                        )

                        # --- NEW: short-sleeve heuristic + skin-tone synchronization (from create_sync_data.py) ---
                        try:
                            # Default: ON (matches create_sync_data.py behavior when no cfg is provided)
                            do_sync = True
                            sync_cfg = getattr(rar, "sync_hand_arm_skin", None)
                            if sync_cfg is not None:
                                do_sync = bool(getattr(sync_cfg, "flag", False))

                            # Need: hand texture available + seam ids available + actual PNGs exist
                            if (
                                do_sync
                                and used_png
                                and (orig_texture is not None)
                                and (seam_ids_smpl is not None)
                                and (mano_wrist_mask is not None)
                            ):
                                import cv2
                                from smplx_helper import (
                                    _get_hand_uv_template,
                                    _load_smpl_uv_template,
                                    _sample_texture_colors_at_uvs,
                                )

                                hand_map_kd = resolve_obj_map_kd(hand_obj_path)
                                if (
                                    hand_map_kd
                                    and os.path.exists(hand_map_kd)
                                    and os.path.exists(used_png)
                                ):
                                    # base hand texture: ORIGINAL, so we align first then mutate jointly
                                    hand_rgb_base = np.asarray(
                                        orig_texture, dtype=np.uint8
                                    )  # RGB
                                    arm_rgb = np.asarray(
                                        Image.open(used_png).convert("RGB"),
                                        dtype=np.uint8,
                                    )

                                    # sample wrist colors (hand: MANO UV)
                                    hand_uv = _get_hand_uv_template()[mano_wrist_mask]
                                    if hand_uv.size == 0:
                                        raise ValueError("empty MANO wrist UV set")
                                    hand_wrist_u8 = (
                                        _sample_texture_colors_at_uvs(
                                            hand_rgb_base, hand_uv, pad_px=1
                                        )
                                        * 255.0
                                    ).astype(np.uint8)

                                    # Heuristic thresholds (optional config knobs)
                                    frac_thr = (
                                        float(getattr(sync_cfg, "frac_thr", 0.25))
                                        if sync_cfg
                                        else 0.25
                                    )
                                    arm_skin_frac_thr = (
                                        float(
                                            getattr(sync_cfg, "arm_skin_frac_thr", 0.12)
                                        )
                                        if sync_cfg
                                        else 0.12
                                    )
                                    min_skin_px = (
                                        int(getattr(sync_cfg, "min_skin_px", 64))
                                        if sync_cfg
                                        else 64
                                    )
                                    max_samples = (
                                        int(getattr(sync_cfg, "max_samples", 4096))
                                        if sync_cfg
                                        else 4096
                                    )
                                    rng_local = np.random.default_rng()

                                    # Decide "short-sleeve / bare skin" on the arm texture.
                                    # Prefer seam sampling if uv_template_obj exists; otherwise fall back to global skin fraction.
                                    short_sleeve = False
                                    arm_samples_u8 = None

                                    mask_arm = skin_mask_ycrcb_rgb(
                                        arm_rgb
                                    )  # 0/255 mask in RGB space
                                    arm_skin = arm_rgb[mask_arm.astype(bool)]
                                    arm_skin_frac = float((mask_arm > 0).mean())

                                    if uv_template_obj and os.path.exists(
                                        uv_template_obj
                                    ):
                                        vt_list, _, v_default = _load_smpl_uv_template(
                                            uv_template_obj
                                        )
                                        vt_ids = [
                                            v_default.get(int(v), None)
                                            for v in seam_ids_smpl
                                        ]
                                        vt_ids = [x for x in vt_ids if x is not None]
                                        if len(vt_ids) > 0:
                                            arm_uv = np.asarray(
                                                [vt_list[x] for x in vt_ids],
                                                dtype=np.float32,
                                            )
                                            arm_wrist_u8 = (
                                                _sample_texture_colors_at_uvs(
                                                    arm_rgb, arm_uv, pad_px=1
                                                )
                                                * 255.0
                                            ).astype(np.uint8)
                                            arm_samples_u8 = arm_wrist_u8
                                            short_sleeve = is_skin_like_rgb_samples(
                                                arm_wrist_u8, frac_thr=frac_thr
                                            )

                                    # Fallback: if enough skin exists anywhere on the arm texture, treat as short-sleeve.
                                    if (
                                        (not short_sleeve)
                                        and (arm_skin.shape[0] >= min_skin_px)
                                        and (arm_skin_frac >= arm_skin_frac_thr)
                                    ):
                                        short_sleeve = True
                                        if arm_skin.shape[0] > max_samples:
                                            idxs = rng_local.choice(
                                                arm_skin.shape[0],
                                                size=max_samples,
                                                replace=False,
                                            )
                                            arm_samples_u8 = arm_skin[idxs]
                                        else:
                                            arm_samples_u8 = arm_skin

                                    if short_sleeve and (arm_samples_u8 is not None):
                                        # masks: skin only (and avoid hand texture background)
                                        hand_bgr = hand_rgb_base[:, :, ::-1].copy()
                                        mask_hand = cv2.bitwise_and(
                                            alpha_mask_from_bg(hand_bgr),
                                            skin_mask_ycrcb_rgb(hand_rgb_base),
                                        )

                                        # align hand seam mean -> arm seam mean (LAB)
                                        hand_wrist_lab = (
                                            cv2.cvtColor(
                                                hand_wrist_u8.reshape(1, -1, 3)[
                                                    :, :, ::-1
                                                ],
                                                cv2.COLOR_BGR2LAB,
                                            )
                                            .reshape(-1, 3)
                                            .astype(np.float32)
                                        )
                                        arm_wrist_lab = (
                                            cv2.cvtColor(
                                                arm_samples_u8.reshape(1, -1, 3)[
                                                    :, :, ::-1
                                                ],
                                                cv2.COLOR_BGR2LAB,
                                            )
                                            .reshape(-1, 3)
                                            .astype(np.float32)
                                        )
                                        delta = arm_wrist_lab.mean(
                                            axis=0
                                        ) - hand_wrist_lab.mean(axis=0)

                                        hand_bgr = lab_shift(
                                            hand_bgr,
                                            mask=mask_hand,
                                            dL=float(delta[0]),
                                            da=float(delta[1]),
                                            db=float(delta[2]),
                                            sL=1.0,
                                            gamma=1.0,
                                        )

                                        # apply SAME random mutation to BOTH (skin only)
                                        if (
                                            getattr(
                                                cfg.generate_mesh,
                                                "texture_color_shift",
                                                None,
                                            )
                                            and cfg.generate_mesh.texture_color_shift.flag
                                        ):
                                            hand_bgr, params = augment_skin_tone_random(
                                                hand_bgr,
                                                p=float(
                                                    cfg.generate_mesh.texture_color_shift.percent
                                                ),
                                                return_params=True,
                                                mask=mask_hand,
                                            )
                                            if params is not None:
                                                arm_bgr = arm_rgb[:, :, ::-1].copy()
                                                arm_bgr = augment_skin_tone_random(
                                                    arm_bgr,
                                                    p=1.0,
                                                    params=params,
                                                    mask=mask_arm,
                                                )
                                                Image.fromarray(
                                                    arm_bgr[:, :, ::-1]
                                                ).save(used_png)

                                        # overwrite the hand texture referenced by the hand OBJ
                                        Image.fromarray(hand_bgr[:, :, ::-1]).save(
                                            hand_map_kd
                                        )
                        except Exception as e:
                            print(
                                f"[armtex][warn] skin sync failed: {type(e).__name__}: {e}"
                            )
                    except Exception as e:
                        print(
                            f"[armtex][warn] failed to assign arm texture: {type(e).__name__}: {e}"
                        )

            # background choice per frame
            if BG_FLAG == "env_map":
                env_map_idx = start_idx + i_local
                env_map_path = (
                    env_map_selected[env_map_idx].replace("./", ROOT_DIR + "/")
                    if env_map_selected[env_map_idx].startswith("./")
                    else os.path.join(env_map_dir, env_map_selected[env_map_idx])
                )
                bg_img_array = render_envmap_to_ldr(
                    env_map_path=env_map_path,
                    width=cfg.simulate.intrinsics.height * 1.5,
                    height=cfg.simulate.intrinsics.width * 1.2,
                )
            elif BG_FLAG == "bg_image":
                bg_img_array = get_random_background(
                    bg_files,
                    target_H=cfg.simulate.intrinsics.height,
                    target_W=cfg.simulate.intrinsics.width,
                )
            else:
                bg_img_array = None

            bg_depth_scene = None
            if use_moge and (bg_img_array is not None):
                # bg_img_array is already uint8 RGB patch (envmap rendered or picked BG)
                bg_depth_scene = moge_depth_from_rgb(bg_img_array)

            # ---- SAPIEN render with object
            (
                rgba_images,
                depth_images,
                camera_poses_all,
                fov_rads_all,
                bbox_all,
                mask_all,
                union_mask_all,
                bg_crop_xy_all,
            ) = run_simulation_sapien_graspxl(
                hand_mesh_path=hand_obj_path,
                body_mesh_path=body_obj_path,
                object_mesh_path=obj_mesh_for_render,
                object_rot=obj_rot_all[t],
                object_trans=obj_trans_all[t],
                simulation_cfg=cfg.simulate,
                bg_img_array=bg_img_array,
                lowres_hand_flag=cfg.simulate.lowres_hand.flag,
                lowres_downscale=cfg.simulate.lowres_hand.downscale_factor,
                edge_soften_flag=cfg.simulate.edge_soften.flag,
                edge_soften_radius_px=cfg.simulate.edge_soften.radius_px,
                edge_soften_shrink_px=cfg.simulate.edge_soften.shrink_px,
                use_theme_lighting=cfg.simulate.lighting.theme_from_bg.flag,
                theme_ambient_boost=cfg.simulate.lighting.theme_from_bg.ambient_boost,
                theme_key_align_prob=cfg.simulate.lighting.theme_from_bg.key_align_prob,
                theme_key_tint_strength=cfg.simulate.lighting.theme_from_bg.key_tint_strength,
                theme_saturation_boost=cfg.simulate.lighting.theme_from_bg.saturation_boost,
            )

            # RGB + separate mask (alpha→mask), like the base generator
            image_rgb_data[i_local] = rgba_images
            mask_data[i_local] = mask_all
            camera_poses[i_local] = camera_poses_all
            fov_rads_data[i_local] = fov_rads_all
            bbox_data[i_local] = bbox_all

            if use_moge and (bg_depth_scene is not None):
                H, W = depth_images.shape[1], depth_images.shape[2]
                depth_full = np.zeros_like(depth_images, dtype=np.float32)
                for j in range(depth_images.shape[0]):
                    # SAPIEN depth is in mm in our pipeline → convert to meters to match MoGe
                    hand = depth_images[j].astype(np.float32) / 1000.0
                    m01 = (union_mask_all[j] > 0.5).astype(
                        np.uint8
                    )  # hand ∪ arm ∪ object

                    # Crop the MoGe scene depth to the same BG window (cy, cx) used for RGB compositing
                    cy, cx = map(int, bg_crop_xy_all[j])
                    cy = max(0, min(cy, bg_depth_scene.shape[0] - H))
                    cx = max(0, min(cx, bg_depth_scene.shape[1] - W))
                    bg_crop = bg_depth_scene[cy : cy + H, cx : cx + W]

                    # Optional boundary align/de-align to control the hand–BG gap
                    bg_adj = adjust_bg_depth_layering(
                        bg_depth=bg_crop,
                        hand_depth=hand,
                        mask01=m01,
                        mode=cfg.simulate.moge_bg_depth.align_mode,
                        target_gap=float(cfg.simulate.moge_bg_depth.target_gap),
                        ring_px=int(cfg.simulate.moge_bg_depth.ring_px),
                        fit_scale=bool(cfg.simulate.moge_bg_depth.fit_scale),
                    )
                    depth_full[j] = compose_hand_bg_depth(
                        hand_depth=hand, bg_depth=bg_adj, mask01=m01
                    )

                depth_data[i_local] = depth_full.astype(np.float16)
                if keep_hand:
                    depth_hand_only_data[i_local] = depth_images.astype(np.float32)
            else:
                # Fallback: keep SAPIEN hand-only depth
                depth_data[i_local] = depth_images

        # ---------- persist this batch (H5 + PNGs/masks + JSON meta) ----------
        existing = [
            f
            for f in os.listdir(save_dir)
            if f.startswith(cfg.generate_mesh.file_prefix) and f.endswith(".h5")
        ]
        next_id = len(existing)
        out_h5 = os.path.join(
            save_dir, f"{cfg.generate_mesh.file_prefix}_{next_id:05d}.h5"
        )

        print("Saving H5 ...")
        with h5py.File(out_h5, "w") as hf:
            hf.attrs["start_idx"] = start_idx
            hf.attrs["end_idx"] = start_idx + B  # exclusive

            hf.create_dataset(
                "beta", data=beta_batch, dtype="float32", compression="gzip"
            )
            hf.create_dataset(
                "pose", data=pose_batch, dtype="float32", compression="gzip"
            )
            hf.create_dataset(
                "image_rgba", data=image_rgb_data, dtype="uint8", compression="gzip"
            )
            hf.create_dataset(
                "depth", data=depth_data, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "mask", data=mask_data, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "camera_poses", data=camera_poses, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "fov_rads", data=fov_rads_data, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "bbox", data=bbox_data, dtype="float16", compression="gzip"
            )

            if save_verts:
                hf.create_dataset(
                    "verts", data=verts_data, dtype="float32", compression="gzip"
                )

            # (NEW) Optionally persist hand-only depth (original SAPIEN z) like base pipeline
            if use_moge and getattr(
                cfg.simulate.moge_bg_depth, "keep_hand_depth", False
            ):
                hf.create_dataset(
                    "depth_hand_only",
                    data=depth_hand_only_data.astype(np.float16),
                    dtype="float16",
                    compression="gzip",
                )

        print(f"✔ H5 saved {out_h5}")

        # --------------------------------------------------------------
        # 6 · free RAM *before* next batch
        # --------------------------------------------------------------

        if not cfg.generate_mesh.leave_tmp_files:
            for f in os.listdir(temp_dir):  # purge temp meshes
                os.remove(os.path.join(temp_dir, f))

    print("Generation Completed!")


if __name__ == "__main__":
    main()
