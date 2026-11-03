#!/usr/bin/env python
# batch_h5_to_tar.py
"""
Recursively convert every *.h5 hand-pose file found under --source_dir into the
sharded *.tar layout used by MiNAFs / Freihand loaders.

A single H5 (512 scenes × 8 views) -> typically one shard (4096 images),
but --shard_len can be changed.

Example
-------
python batch_h5_to_tar.py \
        --source_dir  data/mesh \
        --tmp_dir   /dev/shm/extract_tmp \
        --tar_dir   dataset_tars \
        --depth_scale 1.0
"""

import random
import argparse, tarfile, pickle, shutil, sys
from pathlib import Path

import cv2
import h5py, numpy as np
from PIL import Image
from tqdm.auto import tqdm

import os, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, atexit

# ────────────────────────────────────────────────────────────────────────────
# MANO & projection helpers (unchanged from the previous script)
# ────────────────────────────────────────────────────────────────────────────
import torch
from manotorch.manolayer import ManoLayer

# mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False, ncomps=45,
#                        mano_assets_root="./manotorch/assets/mano")


def get_keypoints_3d_batch(mano_layer, beta, pose):
    """
    Given a beta as np.array (N, 10) and a pose as np.array (N, 45), return 3D joints as np.array (N, 21,3)
    """
    beta_tensor = torch.tensor(beta)
    pose_tensor = torch.tensor(pose)

    mano_output = mano_layer(pose_tensor, beta_tensor)
    joints_loc_3d = mano_output.joints.cpu().numpy()

    return joints_loc_3d


def get_keypoints_3d_single(mano_layer, beta, pose):
    """
    Given a beta as np.array (N, 10) and a pose as np.array (N, 45), return 3D joints as np.array (N, 21,3)
    """
    beta_tensor = torch.tensor(beta).unsqueeze(0)
    pose_tensor = torch.tensor(pose).unsqueeze(0)

    mano_output = mano_layer(pose_tensor, beta_tensor)
    joints_loc_3d = mano_output.joints[0].cpu().numpy()

    return joints_loc_3d


def get_verts_3d_single(mano_layer, beta, pose):
    """
    Given beta (10,) and pose (48,), return MANO vertices (778, 3).
    """
    beta_tensor = torch.tensor(beta).unsqueeze(0)
    pose_tensor = torch.tensor(pose).unsqueeze(0)

    mano_output = mano_layer(pose_tensor, beta_tensor)
    return mano_output.verts[0].cpu().numpy()


def project_joints_cv_to_image(
    points_cv: np.ndarray, fov_rad: float, width: int, height: int
) -> np.ndarray:
    """
    Project 3D points expressed in the **OpenCV camera frame**:
      X = right, Y = down, Z = forward (Z>0)
    onto image pixels (u, v).
    """
    pts = np.asarray(points_cv, dtype=np.float32)
    assert pts.shape[1] == 3

    # intrinsics from *vertical* FOV (same as the existing pipeline)
    fy = (height * 0.5) / np.tan(fov_rad * 0.5)
    fx = fy * width / height
    cx = width * 0.5
    cy = height * 0.5

    X = pts[:, 0]  # right
    Y = pts[:, 1]  # down
    Z = pts[:, 2]  # forward

    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    mask = Z > 0
    uv[mask, 0] = cx + fx * (X[mask] / Z[mask])
    uv[mask, 1] = cy + fy * (Y[mask] / Z[mask])
    return uv


# Fixed basis change from SAPIEN cam frame (X_fwd, Y_left, Z_up)
# to OpenCV cam frame (X_right, Y_down, Z_fwd).
M_CV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)

# handy constant
I3 = np.eye(3, dtype=np.float32)


def project_joints_to_image(
    joints_3d: np.ndarray, cam_pose: np.ndarray, fov_rad: float, width: int, height: int
) -> np.ndarray:
    """
    Project 3-D world points onto a pin-hole image plane using SAPIEN 's camera
    convention (columns = forward +X, left +Y, up +Z).

    Args
    ----
    joints_3d : (N, 3) float32/64
        3-D points in world coordinates.
    cam_pose  : (4, 4) float
        Camera pose matrix returned by SAPIEN (rotation in the first 3 * 3 block,
        position in the last column).
    fov_rad   : float
        *Vertical* field-of-view in radians.
    width     : int
        Image width  in pixels.
    height    : int
        Image height in pixels.

    Returns
    -------
    pixels : (N, 2) float32
        Pixel coordinates (u, v).  Values can be outside [0, width/height] if a
        point lies outside the view frustum.  If a point is behind the camera
        (depth ≤ 0), the corresponding (u, v) entry is set to np.nan.
    """
    # --------------------------------------------------------------------- #
    # 1. split pose into rotation (R) and translation (t)
    # --------------------------------------------------------------------- #
    R = cam_pose[:3, :3]  # world → camera axes
    t = cam_pose[:3, 3]  # camera position in world space

    # --------------------------------------------------------------------- #
    # 2. transform points into camera-space (column 0 is the depth axis)
    #      X_cam = +depth  (optical axis)
    #      Y_cam = +left
    #      Z_cam = +up
    # --------------------------------------------------------------------- #
    # (p_w − t) brings points into camera origin; Rᵀ aligns them
    p_cam = (joints_3d - t) @ R  # shape (N, 3); faster than R.T @ …

    x_cam = p_cam[:, 0]  # depth
    y_cam = p_cam[:, 1]  # image-left  (+Y shrinks u)
    z_cam = p_cam[:, 2]  # image-up    (+Z shrinks v)

    # --------------------------------------------------------------------- #
    # 3. intrinsics: focal length from vertical FOV
    # --------------------------------------------------------------------- #
    fy = (height * 0.5) / np.tan(fov_rad * 0.5)
    fx = fy * width / height
    cx = width * 0.5
    cy = height * 0.5

    # --------------------------------------------------------------------- #
    # 4. perspective division
    # --------------------------------------------------------------------- #
    #    u = cx – fx * (y / x)
    #    v = cy – fy * (z / x)
    #
    # subtract (“–”) because +Y/+Z point left/up in the image
    # while pixel coordinates increase right/down
    #
    good = x_cam > 0  # only points in front of camera
    u = np.full_like(x_cam, np.nan, dtype=np.float32)
    v = np.full_like(x_cam, np.nan, dtype=np.float32)

    u[good] = cx - fx * (y_cam[good] / x_cam[good])
    v[good] = cy - fy * (z_cam[good] / x_cam[good])

    return np.stack([u, v], axis=1)


def get_keypoints_2d(keypoints_3d, cam_pose, fov_rad, im_width, im_height):
    """
    Given a keypoints 3d of shape (21,3), project it on the image.
    Return as a (21, 2) array.
    """
    keypoints_2d = project_joints_to_image(
        joints_3d=keypoints_3d,
        cam_pose=cam_pose,
        fov_rad=fov_rad,
        width=im_width,
        height=im_height,
    )

    return keypoints_2d


def bbox_to_center_scale(bbox: np.ndarray):
    """
    Args
    ----
    bbox : (4, 2) float32 array
        Corner points (x, y) of the hand bounding box in pixel space.

    Returns
    -------
    center : (2,) float32
        (cx, cy) in pixels.

    scale  : (2,) float32
        (width, height) **in pixels** - *not* divided by 200 here.
        `ImageDataset` will divide by 200 when it loads the .tar shard,
        giving the normalised scale it needs for cropping.
    """
    # Safety: ensure float32 and compute “tight” AABB
    pts = bbox.astype(np.float32)

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)

    width = max(x_max - x_min, 1.0)  # avoid zero-area boxes
    height = max(y_max - y_min, 1.0)

    scale = (
        np.array([width, height], dtype=np.float32) / 200.0 * 2.5 * 1.1
    )  # scale for cropping

    return center, scale


def float16_to_uint16(arr, scale=1.0):
    return np.clip(arr * scale, 0, 65535).astype(np.uint16)


def meters_to_mm_u16(depth_m: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    """
    Convert depth in meters → uint16 millimeters.
    Invalid/negative become 0 (invalid).
    scale: multiplier applied to meters before quantization (default 1000 = mm).
    """
    d = depth_m.astype(np.float32)
    # map NaN/Inf/negatives to 0 as the invalid code
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    d = np.rint(d * scale)
    return np.clip(d, 0, 65535).astype(np.uint16)


def save_depth_vis_png(depth_f32, out_png_path, mask_u8=None):
    """
    Save a color-mapped depth visualization (uint8 PNG).
    - depth_f32: (H, W) float32/float16 depth in meters
    - mask_u8  : optional (H, W) uint8 mask (0/255). If provided, the color range
                 is computed from masked (hand) pixels for good contrast.
    """
    d = depth_f32.astype(np.float32).copy()
    valid = np.isfinite(d)
    if mask_u8 is not None:
        valid = valid & (mask_u8 > 0)

    if not np.any(valid):
        valid = np.isfinite(d)

    vmin = float(np.nanmin(d[valid]))
    vmax = float(np.nanmax(d[valid]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin + 1e-6:
        vmin, vmax = float(np.nanmin(d)), float(np.nanmax(d))
        if vmax <= vmin + 1e-6:
            vmin, vmax = 0.0, 1.0

    nrm = (d - vmin) / (vmax - vmin + 1e-8)
    nrm = np.clip(nrm, 0.0, 1.0)
    img8 = (nrm * 255.0).astype(np.uint8)

    # invalid to 0 for consistent background
    img8[~np.isfinite(d)] = 0

    # Use TURBO if available, otherwise JET
    try:
        colored = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    except Exception:
        colored = cv2.applyColorMap(img8, cv2.COLORMAP_JET)

    cv2.imwrite(str(out_png_path), colored)


# ────────────────────────────────────────────────────────────────────────────
# core utilities
# ────────────────────────────────────────────────────────────────────────────
def _tar_directory(src: Path, dst_tar: Path):
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst_tar, "w") as tf:
        # for fp in src.iterdir():
        for fp in sorted(src.iterdir(), key=lambda p: p.name):
            tf.add(fp, arcname=fp.name)


def enlarge_bbox_four_sides(
    center: np.ndarray,
    scale: np.ndarray,
    img_w: int,
    img_h: int,
    max_factor: float = 1.5,
):
    """
    Randomly expand each side of the GT bbox outward exactly once,
    guaranteeing containment of the original bbox and clamping to image bounds.

    Inputs
    ------
    center : (2,) in pixels
    scale  : (2,) stored as (w/200, h/200) — same format used elsewhere
    img_w, img_h : image size in pixels
    max_factor : sides move outwards by up to (factor-1) * (w or h)

    Returns
    -------
    new_center : (2,) in pixels
    new_scale  : (2,) in (w/200, h/200) format
    """
    cx, cy = float(center[0]), float(center[1])
    ow, oh = float(scale[0]) * 200.0, float(scale[1]) * 200.0  # pixels

    # Original tight box
    x1 = cx - ow * 0.5
    x2 = cx + ow * 0.5
    y1 = cy - oh * 0.5
    y2 = cy + oh * 0.5

    # Sample per-side factors in [1.0, max_factor]
    fw1 = random.uniform(1.0, max_factor)  # left  expansion
    fw2 = random.uniform(1.0, max_factor)  # right expansion
    fh1 = random.uniform(1.0, max_factor)  # top   expansion
    fh2 = random.uniform(1.0, max_factor)  # bottom expansion

    # Move sides outward; clamp to image bounds
    nx1 = max(0.0, x1 - ow * (fw1 - 1.0))
    nx2 = min(float(img_w - 1), x2 + ow * (fw2 - 1.0))
    ny1 = max(0.0, y1 - oh * (fh1 - 1.0))
    ny2 = min(float(img_h - 1), y2 + oh * (fh2 - 1.0))

    # Ensure valid ordering if extremely tight to borders
    nx1 = min(nx1, x1)  # containment
    ny1 = min(ny1, y1)
    nx2 = max(nx2, x2)
    ny2 = max(ny2, y2)

    nw = max(1.0, nx2 - nx1)
    nh = max(1.0, ny2 - ny1)

    new_center = np.array([(nx1 + nx2) * 0.5, (ny1 + ny2) * 0.5], dtype=np.float32)
    new_scale = np.array([nw / 200.0, nh / 200.0], dtype=np.float32)
    return new_center, new_scale


def convert_h5(
    h5_path: Path,
    tmp_root: Path,
    tar_root: Path,
    shard_len: int,
    depth_scale: float,
    shard_start: int,
    mano_layer=None,
    enlarge: bool = False,
    save_depth_png: bool = False,
):
    """
    Convert one *.h5 file.  Returns (next_shard_idx, images_written)
    """
    with h5py.File(h5_path, "r") as hf:
        start_idx = int(hf.attrs["start_idx"])
        beta, pose = hf["beta"][:], hf["pose"][:]
        imgs = hf["image_rgba"][:]
        depth, mask = hf["depth"][:], hf["mask"][:]
        cam_poses = hf["camera_poses"][:]
        fov_rads, bbox = hf["fov_rads"][:], hf["bbox"][:]
        has_verts = "verts" in hf
        verts = hf["verts"][:] if has_verts else None

    n_scenes, n_views = imgs.shape[:2]
    assert n_views * n_scenes == beta.shape[0] * n_views, "shape mismatch"

    img_offset = start_idx * n_views
    shard_idx = shard_start
    shard_dir = tmp_root / f"{shard_idx:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    written_in_shard = 0

    for scene_i in tqdm(range(n_scenes), desc=h5_path.name, leave=False):
        for view_i in range(n_views):
            global_id = img_offset + scene_i * n_views + view_i
            name = f"{global_id:08d}"

            rgb = imgs[scene_i, view_i, ..., :3]
            H, W = rgb.shape[:2]
            Image.fromarray(rgb).save(shard_dir / f"{name}.jpg", quality=95)

            # m = (mask[scene_i, view_i] > 0.5).astype(np.uint8) * 255
            # # Image.fromarray(m).save(shard_dir / f"{name}.mask.png")

            # # d16 = float16_to_uint16(depth[scene_i, view_i], depth_scale)
            # # Image.fromarray(d16, mode="I;16").save(shard_dir / f"{name}.depth.png")

            # d_raw = depth[scene_i, view_i]  # typically float16 in your H5
            # np.save(shard_dir / f"{name}.depth.npy", d_raw)

            # if save_depth_png:
            #     # use the hand mask to set the dynamic range (gives clean contrast)
            #     save_depth_vis_png(d_raw.astype(np.float32),
            #                        shard_dir / f"{name}.depth.png",
            #                        mask_u8=None)
            m = (mask[scene_i, view_i] > 0.5).astype(np.uint8) * 255
            Image.fromarray(m).save(shard_dir / f"{name}.mask.png")

            # --- save RAW DEPTH as 16-bit PNG in millimeters ---
            d_m = depth[scene_i, view_i].astype(np.float32)  # meters
            d_u16 = meters_to_mm_u16(d_m, scale=depth_scale)  # mm → uint16
            Image.fromarray(d_u16, mode="I;16").save(shard_dir / f"{name}.depth.png")

            # --- optional color-mapped preview for quick visual debugging ---
            if save_depth_png:
                # note: do NOT overwrite the raw 16-bit file
                save_depth_vis_png(
                    d_m, shard_dir / f"{name}.depth_vis.png", mask_u8=None
                )

            # Compute GT bbox center/scale (in current pipeline's units)
            ctr, scl = bbox_to_center_scale(bbox[scene_i, view_i])
            gt_ctr = ctr.copy()
            gt_scl = scl.copy()

            # Optional randomized outward expansion of the four sides
            if enlarge:
                ctr, scl = enlarge_bbox_four_sides(
                    ctr, scl, img_w=W, img_h=H, max_factor=1.5
                )

            #############################################################################################

            # ───────────────────────────────────────────────────────────
            # 1) Camera extrinsics (World→Cam) and conversion to CV frame
            #    T_wc maps world coords to camera coords.
            # ───────────────────────────────────────────────────────────
            T_wc = np.linalg.inv(cam_poses[scene_i, view_i].astype(np.float32)).astype(
                np.float32
            )
            R_wc_s = T_wc[:3, :3]  # world → cam (SAPIEN cam axes)
            t_wc_s = T_wc[:3, 3]  # translation (column form)
            # express both in **CV camera axes**
            R_wc_cv = M_CV @ R_wc_s
            t_wc_cv = M_CV @ t_wc_s

            # ───────────────────────────────────────────────────────────
            # 2) Rotation center C = first joint under **zero-root** pose
            #    (MANO root rotation acts around joints[0])
            # ───────────────────────────────────────────────────────────
            pose_zero = pose[scene_i].copy()
            pose_zero[:3] = 0.0  # ensure zero-root
            joints_zero = get_keypoints_3d_single(mano_layer, beta[scene_i], pose_zero)
            C = joints_zero[0].astype(np.float32)  # (3,)

            # ───────────────────────────────────────────────────────────
            # 3) Compose *existing* root rotation with the camera rotation
            #       R_new = R_cam · R0
            #    where R0 is from the original pose[:3] (may be zero or non-zero)
            # ───────────────────────────────────────────────────────────
            orig_root_aa = pose[scene_i][:3].astype(np.float32)  # axis-angle (3,)
            R0, _ = cv2.Rodrigues(orig_root_aa)  # (3,3)
            R_new = R_wc_cv @ R0  # compose in the same frame
            rot_new, _ = cv2.Rodrigues(R_new)  # back to axis-angle (3,1)

            hand_pose_view = pose[scene_i].copy().astype(np.float32)
            hand_pose_view[:3] = rot_new.flatten().astype(np.float32)

            # MANO forward with the composed root rotation
            j3d_rot = get_keypoints_3d_single(
                mano_layer, beta[scene_i], hand_pose_view
            )  # (21,3)
            v3d_rot = get_verts_3d_single(
                mano_layer, beta[scene_i], hand_pose_view
            )  # (778,3)

            # ───────────────────────────────────────────────────────────
            # 4) Translation to put rotated points into **camera coords**
            #    This depends only on R_cam (not on R0):
            #       t_corr = t_cam − (I − R_cam) · C
            # ───────────────────────────────────────────────────────────
            t_corr = t_wc_cv - (I3 - R_wc_cv) @ C  # (3,)
            j3d_cam_cv = j3d_rot + t_corr[None, :]  # (21,3) in CV camera coords
            v3d_cam_cv = v3d_rot + t_corr[None, :]  # (778,3) in CV camera coords

            # 5) 2D projection (unchanged)
            k2d = project_joints_cv_to_image(
                j3d_cam_cv, float(fov_rads[scene_i, view_i]), W, H
            ).astype(np.float32)
            k2d_full = np.hstack([k2d, np.ones((21, 1), dtype=np.float32)])

            # 6) Store meta (unchanged except that hand_pose now includes composed root)
            key3d_full = np.hstack(
                [j3d_cam_cv.astype(np.float32), np.ones((21, 1), dtype=np.float32)]
            )

            # Per-image "camera-equivalent" focal length (pixels), derived from vertical FOV.
            # We pretend a standard pinhole camera even though the image was ray-traced.
            _fov = float(fov_rads[scene_i, view_i])
            focal_length = (H * 0.5) / np.tan(_fov * 0.5)

            meta = dict(
                right=True,
                has_betas=True,
                betas=beta[scene_i],
                has_hand_pose=True,
                hand_pose=hand_pose_view.astype(np.float32),
                center=ctr,
                scale=scl,
                GT_center=gt_ctr,
                GT_scale=gt_scl,
                joints_3d=j3d_cam_cv.astype(np.float32),
                keypoints_2d=k2d_full.astype(np.float32),
                keypoints_3d=key3d_full.astype(np.float32),
                cam_t=t_corr.astype(np.float32),
                focal_length=focal_length,
                depth_unit="mm",  # stored unit of *.depth.png
                depth_scale_to_m=1.0
                / float(depth_scale),  # convert stored integer → meters
                depth_invalid_code=0,
                verts_3d=v3d_cam_cv.astype(np.float32),
            )

            with open(shard_dir / f"{name}.data.pyd", "wb") as fp:
                pickle.dump([meta], fp)

            written_in_shard += 1
            if written_in_shard == shard_len:
                # close shard
                tar_name = tar_root / f"{shard_idx:06d}.tar"
                _tar_directory(shard_dir, tar_name)
                shutil.rmtree(shard_dir)
                shard_idx += 1
                shard_dir = tmp_root / f"{shard_idx:06d}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                written_in_shard = 0

    # flush last partial shard
    if any(shard_dir.iterdir()):
        tar_name = tar_root / f"{shard_idx:06d}.tar"
        _tar_directory(shard_dir, tar_name)
        shutil.rmtree(shard_dir)
        shard_idx += 1

    total_images = n_scenes * n_views
    return shard_idx, total_images


# ────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser("Batch-convert *.h5 to tar dataset")
    p.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="folder that contains pod0/, pod1/, … with *.h5 files",
    )
    p.add_argument(
        "--tmp_dir",
        type=Path,
        default=Path("./extract_tmp"),
        help="scratch area for extracted JPG/PNG/PYD (auto-deleted)",
    )
    p.add_argument(
        "--tar_dir",
        type=Path,
        default=Path("./tars_out"),
        help="where all 000000.tar, 000001.tar, … are stored",
    )
    p.add_argument("--shard_len", type=int, default=4096, help="images per tar file")
    p.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="meters → stored integer multiplier for 16-bit depth PNGs "
        "(default 1000 = millimeters)",
    )
    p.add_argument(
        "--enlarge_bbox",
        action="store_true",
        help="Apply random enlargement to bounding box center/scale",
    )
    p.add_argument(
        "--resume_file",
        type=Path,
        default=Path("tar_resume.json"),
        help="stores conversion progress so the run can be resumed",
    )
    p.add_argument("--root_dir", type=str)
    p.add_argument(
        "--save_depth_png",
        action="store_true",
        default=False,
        help="If set, also write a color-mapped depth.png for quick visualization. "
        "Raw depth is always saved as depth.npy.",
    )
    args = p.parse_args()

    ROOT_DIR = args.root_dir
    mano_layer = ManoLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        mano_assets_root=ROOT_DIR + "/" + "manotorch/assets/mano",
    )

    # ------------------------------------------------------------------ #
    # 1.  read or initialize resume-state
    # ------------------------------------------------------------------ #
    state = dict(processed=[], shard_idx=0, grand_total=0)
    if args.resume_file.exists():
        with open(args.resume_file) as fp:
            state.update(json.load(fp))

    h5_list = sorted(args.source_dir.rglob("*.h5"))
    if not h5_list:
        sys.exit(f"No *.h5 found under {args.source_dir}")

    print(f"Found {len(h5_list)} h5 files ({len(state['processed'])} already done)")

    shard_idx = state["shard_idx"]
    grand_total = state["grand_total"]

    # ------------------------------------------------------------------ #
    # 2.  make sure we always flush progress even on ^C
    # ------------------------------------------------------------------ #
    def _save():
        args.resume_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.resume_file, "w") as fp:
            json.dump(state, fp, indent=2)

    atexit.register(_save)

    # ------------------------------------------------------------------ #
    # 3.  main loop with skipping + state update
    # ------------------------------------------------------------------ #
    for h5_path in h5_list:
        if str(h5_path) in state["processed"]:
            continue  # already finished in a previous run

        print(f"\n→ Processing {h5_path.relative_to(args.source_dir)}")
        try:
            new_shard_idx, n_img = convert_h5(
                h5_path,
                args.tmp_dir,
                args.tar_dir,
                args.shard_len,
                args.depth_scale,
                shard_idx,
                enlarge=args.enlarge_bbox,
                mano_layer=mano_layer,
                save_depth_png=args.save_depth_png,
            )
        except (OSError, BlockingIOError, PermissionError) as e:
            print(f"   ⚠️  Skipping {h5_path.name}: {e}")
            continue  # leave counters unchanged

        # ✅ success – update counters and persistent state
        shard_idx = new_shard_idx
        grand_total += n_img
        state["processed"].append(str(h5_path))
        state["shard_idx"] = shard_idx
        state["grand_total"] = grand_total
        _save()  # write immediately

        print(f"   done - {n_img} images ({grand_total} so far)")

    # clean up tmp_dir if empty
    if args.tmp_dir.exists() and not any(args.tmp_dir.iterdir()):
        args.tmp_dir.rmdir()

    print(f"\nAll done ✅  Total images: {grand_total}  Tars written: {shard_idx}")
    _save()  # final flush


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
