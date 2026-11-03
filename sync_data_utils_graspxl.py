import numpy as _np
import os as _os
import cv2 as _cv2
import tempfile as _tmp
import time

from typing import List, Tuple

# Reuse lighting / background / compositing helpers from your existing utils
from sync_data_utils import (
    add_random_lighting,
    composite_on_random_bg,
    process_segmentation,
    _theme_color_from_bg,
)

# -------------------- math helpers --------------------


def axis_angle_to_matrix(aa: _np.ndarray) -> _np.ndarray:
    aa = _np.asarray(aa, dtype=_np.float32)
    theta = float(_np.linalg.norm(aa))
    if theta < 1e-8:
        return _np.eye(3, dtype=_np.float32)
    k = aa / theta
    K = _np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=_np.float32
    )
    R = (
        _np.eye(3, dtype=_np.float32)
        + _np.sin(theta) * K
        + (1.0 - _np.cos(theta)) * (K @ K)
    )
    return R


def matrix_to_quaternion(R: _np.ndarray) -> _np.ndarray:
    # xyzw
    R = _np.asarray(R, dtype=_np.float32)
    t = _np.trace(R)
    if t > 0.0:
        s = _np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = _np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = _np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = _np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = _np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return _np.array([qx, qy, qz, qw], dtype=_np.float32)


def axis_to_quaternion(axis):
    """
    Convert a 3D rotation vector (axis-angle representation)
    to a quaternion [w, x, y, z].

    Parameters:
    -----------
    axis : array_like of shape (3,)
        The rotation vector where its norm is the rotation angle
        (in radians) and its direction is the rotation axis.

    Returns:
    --------
    numpy.ndarray of shape (4,)
        The quaternion representing the same rotation.
        Format: [w, x, y, z]
    """
    import numpy as np

    axis = np.array(axis, dtype=float)
    theta = np.linalg.norm(axis)

    # When theta is very small, avoid division by zero:
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])

    half_theta = theta / 2.0
    w = np.cos(half_theta)
    sin_half_theta = np.sin(half_theta)

    # Normalize the axis to get the rotation axis
    axis_normalized = axis / theta
    xyz = axis_normalized * sin_half_theta

    return np.concatenate(([w], xyz))


# -------------------- pose-convention helpers --------------------
def _xyzw_to_wxyz(q_xyzw: _np.ndarray) -> _np.ndarray:
    q_xyzw = _np.asarray(q_xyzw, dtype=_np.float32)
    return _np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=_np.float32)


def _quat_xyzw_to_matrix(q: _np.ndarray) -> _np.ndarray:
    q = _np.asarray(q, dtype=_np.float32)
    x, y, z, w = q
    # normalize to be safe
    n = _np.linalg.norm(q)
    if n == 0:
        return _np.eye(3, dtype=_np.float32)
    x /= n
    y /= n
    z /= n
    w /= n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return _np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=_np.float32,
    )


def _cv_to_sapien_axes_matrix() -> _np.ndarray:
    # Map OpenCV camera coords (x right, y down, z forward)
    # → SAPIEN world coords (x forward, y left, z up)
    # v_sp = M @ v_cv
    return _np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=_np.float32)


def _convert_cv_to_sapien_pose(
    t_cv: _np.ndarray, q_xyzw_cv: _np.ndarray
) -> tuple[_np.ndarray, _np.ndarray]:
    M = _cv_to_sapien_axes_matrix()
    # position
    t_sp = (M @ _np.asarray(t_cv, dtype=_np.float32).reshape(3)).astype(_np.float32)
    # rotation
    R_cv = _quat_xyzw_to_matrix(q_xyzw_cv)
    R_sp = M @ R_cv @ M.T
    q_xyzw_sp = matrix_to_quaternion(R_sp)  # still xyzw
    q_wxyz_sp = _xyzw_to_wxyz(q_xyzw_sp)
    return t_sp, q_wxyz_sp


# -------------------- GraspXL helpers --------------------


def get_topk_close_frames(obj_locs: _np.ndarray, k: int) -> List[int]:
    """
    Return indices of the K frames where the object is closest to the origin (0,0,0).
    """
    obj_locs = _np.asarray(obj_locs, dtype=_np.float32)
    d = _np.linalg.norm(obj_locs, axis=1)
    idx = _np.argsort(d)[:k]
    return idx.tolist()


def _write_solid_color_mtl(mtl_path: str, name: str, rgb: Tuple[float, float, float]):
    r, g, b = rgb
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {name}\n")
        f.write(f"Kd {r:.5f} {g:.5f} {b:.5f}\n")
        f.write("Ka 0 0 0\nKs 0 0 0\nNs 1\n")


# def colorize_obj_to_tmp(src_obj_path: str, out_dir: str) -> str:
#     """Create a temp OBJ+MTL that paints the geometry with a single random color.
#     - Strips any existing mtllib/usemtl from the source and injects our own.
#     - Keeps all geometry (v/vt/vn/f) intact.
#     """
#     _os.makedirs(out_dir, exist_ok=True)
#     base = _os.path.splitext(_os.path.basename(src_obj_path))[0]
#     dst_obj = _os.path.join(out_dir, f"{base}_colored.obj")
#     dst_mtl = _os.path.join(out_dir, f"{base}_colored.mtl")

#     # random pleasant color in [0.2, 0.9]
#     rgb = tuple(float(x) for x in _np.random.uniform(0.2, 0.9, size=3))

#     with open(src_obj_path, "r", errors="ignore") as si, open(dst_obj, "w") as so:
#         so.write(f"mtllib {_os.path.basename(dst_mtl)}\n")
#         so.write("usemtl single_color\n")
#         for line in si:
#             if line.startswith("mtllib") or line.startswith("usemtl"):
#                 continue
#             so.write(line)

#     _write_solid_color_mtl(dst_mtl, "single_color", rgb)
#     return dst_obj


def _axis_angle_to_rotmat(aa):
    """Rodrigues: 3D axis-angle (radians) -> 3x3 rotation matrix."""
    aa = _np.asarray(aa, dtype=_np.float32).reshape(3)
    theta = float(_np.linalg.norm(aa))
    if theta < 1e-12:
        return _np.eye(3, dtype=_np.float32)
    k = aa / theta
    K = _np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=_np.float32
    )
    I = _np.eye(3, dtype=_np.float32)
    return I + _np.sin(theta) * K + (1.0 - _np.cos(theta)) * (K @ K)


def colorize_obj_to_tmp(
    src_obj_path: str, out_obj_path: str, obj_trans, obj_rot
) -> str:
    """Create a temp OBJ+MTL, paint with a single random color, and APPLY a rigid transform.
    - obj_rot: 3D axis-angle (radians)
    - obj_trans: 3D translation
    - Strips existing mtllib/usemtl; injects our own.
    - Preserves topology (v/vt/vn/f). Transforms v and vn.
    """
    base = _os.path.splitext(_os.path.basename(src_obj_path))[0]
    dst_mtl = out_obj_path.replace(".obj", ".mtl")

    # random pleasant color in [0.2, 0.9]
    rgb = tuple(float(x) for x in _np.random.uniform(0.2, 0.9, size=3))

    # Build transform
    R = _axis_angle_to_rotmat(obj_rot)  # (3,3)
    t = _np.asarray(obj_trans, dtype=_np.float32).reshape(3)

    with open(src_obj_path, "r", errors="ignore") as si, open(out_obj_path, "w") as so:
        so.write(f"mtllib {_os.path.basename(dst_mtl)}\n")
        so.write("usemtl single_color\n")
        for line in si:
            # remove any prior material references
            if line.startswith("mtllib") or line.startswith("usemtl"):
                continue

            if line.startswith("v "):  # vertex
                parts = line.strip().split()
                # handle possible extra components (e.g., w) by transforming only xyz
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    v = _np.array([x, y, z], dtype=_np.float32)
                    vt = R @ v + t
                    extras = "".join(f" {p}" for p in parts[4:])
                    so.write(f"v {vt[0]:.6f} {vt[1]:.6f} {vt[2]:.6f}{extras}\n")
                else:
                    # malformed v line; pass through
                    so.write(line)

            elif line.startswith("vn "):  # normal
                parts = line.strip().split()
                if len(parts) >= 4:
                    nx, ny, nz = map(float, parts[1:4])
                    n = _np.array([nx, ny, nz], dtype=_np.float32)
                    nr = R @ n
                    # re-normalize to be safe
                    ln = float(_np.linalg.norm(nr))
                    if ln > 0:
                        nr /= ln
                    so.write(f"vn {nr[0]:.6f} {nr[1]:.6f} {nr[2]:.6f}\n")
                else:
                    so.write(line)

            elif line.startswith("vt "):  # texcoord (leave as is)
                so.write(line)

            else:
                # faces, groups, objects, comments, etc.
                so.write(line)

    _write_solid_color_mtl(dst_mtl, "single_color", rgb)
    return out_obj_path


# -------------------- Textured object (Objaverse) helpers --------------------


def _resize_mesh_to_match_bbox(
    large_mesh_addr: str, small_mesh_addr: str, output_file: str
) -> str:
    """
    Rescale + recenter LARGE (e.g., Objaverse GLB/OBJ) to SMALL (the GraspXL simplified OBJ)
    by matching their AABB max extent and center; export as OBJ (with .mtl if present).
    """
    import trimesh as _trimesh

    _os.makedirs(_os.path.dirname(output_file), exist_ok=True)

    large = _trimesh.load(large_mesh_addr, force="mesh")
    small = _trimesh.load(small_mesh_addr, force="mesh")

    minL, maxL = large.bounds
    minS, maxS = small.bounds
    centerL, centerS = (minL + maxL) / 2.0, (minS + maxS) / 2.0
    scale = float(_np.max(maxS - minS) / max(1e-12, _np.max(maxL - minL)))

    large.apply_translation(-centerL)
    large.apply_scale(scale)
    large.apply_translation(centerS)

    # Export. Trimesh will emit .obj + .mtl (+ texture files) when possible.
    large.export(output_file)
    return output_file


def ensure_textured_objaverse_obj(
    obj_root_dir: str, size_folder: str, obj_id: str, simplified_obj_path: str
) -> str:
    """
    Ensure a rescaled textured mesh <obj_id>_resized.obj exists next to the simplified mesh:
      object_dataset/<size>/<obj_id>/<obj_id>_resized.obj
    If missing, download from Objaverse, place raw under the same folder, rescale to the simplified OBJ.
    Returns path to the cached/rescaled OBJ.
    """
    import shutil

    try:
        import objaverse  # heavy import; keep inside
    except Exception as e:
        raise RuntimeError(
            "objaverse is required when use_textured_objaverse=true"
        ) from e

    target_dir = _os.path.join(obj_root_dir, f"{size_folder}/{obj_id}")
    _os.makedirs(target_dir, exist_ok=True)

    cached = _os.path.join(target_dir, f"{obj_id}_resized.obj")
    if _os.path.exists(cached):
        return cached

    # Download the original (typically .glb) to the target folder
    # addr = objaverse.load_objects([obj_id])
    # glb_src = addr[obj_id]  # source GLB
    max_retries = 32
    for attempt in range(max_retries):
        try:
            addr = objaverse.load_objects([obj_id])
            glb_src = addr[obj_id]  # source GLB
            break  # success
        except Exception as e:
            print(f"[objaverse] download failed: {e}")
            if attempt < max_retries - 1:
                print(f"[objaverse] retrying download {attempt + 1}/{max_retries} ...")
                time.sleep(2**attempt)  # exponential backoff: 1s, 2s, 4s
            else:
                raise RuntimeError(
                    f"Failed to download {obj_id} from objaverse after {max_retries} attempts."
                ) from e

    glb_dst = _os.path.join(target_dir, f"{obj_id}.glb")
    if _os.path.exists(glb_dst):
        _os.remove(glb_dst)
    shutil.move(glb_src, glb_dst)

    # Rescale + recenter to match the simplified OBJ bbox; export as OBJ
    _resize_mesh_to_match_bbox(glb_dst, simplified_obj_path, cached)
    return cached


# def transform_obj_keep_materials(src_obj_path: str,
#                                  out_obj_path: str,
#                                  obj_trans,
#                                  obj_rot) -> str:
#     """
#     Like colorize_obj_to_tmp, but PRESERVES mtllib/usemtl/vt/f lines.
#     Applies the rigid transform to v and vn only, keeping all materials & UVs.
#     """
#     R = _axis_angle_to_rotmat(obj_rot)
#     t = _np.asarray(obj_trans, dtype=_np.float32).reshape(3)

#     with open(src_obj_path, "r", errors="ignore") as si, open(out_obj_path, "w") as so:
#         for line in si:
#             if line.startswith("v "):  # vertex
#                 parts = line.strip().split()
#                 if len(parts) >= 4:
#                     x, y, z = map(float, parts[1:4])
#                     v = _np.array([x, y, z], dtype=_np.float32)
#                     vt = R @ v + t
#                     extras = "".join(f" {p}" for p in parts[4:])
#                     so.write(f"v {vt[0]:.6f} {vt[1]:.6f} {vt[2]:.6f}{extras}\n")
#                 else:
#                     so.write(line)
#             elif line.startswith("vn "):  # normal
#                 parts = line.strip().split()
#                 if len(parts) >= 4:
#                     nx, ny, nz = map(float, parts[1:4])
#                     n = _np.array([nx, ny, nz], dtype=_np.float32)
#                     nr = R @ n
#                     ln = float(_np.linalg.norm(nr))
#                     if ln > 0:
#                         nr /= ln
#                     so.write(f"vn {nr[0]:.6f} {nr[1]:.6f} {nr[2]:.6f}\n")
#                 else:
#                     so.write(line)
#             else:
#                 # Keep ALL other lines intact (mtllib/usemtl/vt/f/etc)
#                 so.write(line)
#     return out_obj_path


def transform_obj_keep_materials(
    src_obj_path: str, out_obj_path: str, obj_trans, obj_rot
) -> str:
    """
    Like colorize_obj_to_tmp, but preserves mtllib/usemtl/vt/f lines.
    To avoid naming conflicts, renames material & texture files uniquely.
    Applies the rigid transform to v and vn only.
    """
    import shutil

    R = _axis_angle_to_rotmat(obj_rot)
    t = _np.asarray(obj_trans, dtype=_np.float32).reshape(3)

    src_dir = _os.path.dirname(src_obj_path)
    out_dir = _os.path.dirname(out_obj_path)
    _os.makedirs(out_dir, exist_ok=True)

    new_basename = _os.path.splitext(_os.path.basename(out_obj_path))[
        0
    ]  # e.g., i_local_obj_tx
    new_mtl_name = f"{new_basename}.mtl"
    new_mtl_path = _os.path.join(out_dir, new_mtl_name)

    # First pass: rewrite .obj with renamed materials, transformed v/vn
    old_to_new_material = {}
    with open(src_obj_path, "r", errors="ignore") as si, open(out_obj_path, "w") as so:
        for line in si:
            if line.startswith("mtllib "):
                so.write(f"mtllib {new_mtl_name}\n")
            elif line.startswith("v "):  # vertex
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    v = _np.array([x, y, z], dtype=_np.float32)
                    vt = R @ v + t
                    extras = "".join(f" {p}" for p in parts[4:])
                    so.write(f"v {vt[0]:.6f} {vt[1]:.6f} {vt[2]:.6f}{extras}\n")
                else:
                    so.write(line)
            elif line.startswith("vn "):  # normal
                parts = line.strip().split()
                if len(parts) >= 4:
                    nx, ny, nz = map(float, parts[1:4])
                    n = _np.array([nx, ny, nz], dtype=_np.float32)
                    nr = R @ n
                    ln = float(_np.linalg.norm(nr))
                    if ln > 0:
                        nr /= ln
                    so.write(f"vn {nr[0]:.6f} {nr[1]:.6f} {nr[2]:.6f}\n")
                else:
                    so.write(line)
            elif line.startswith("usemtl "):
                orig_mat = line.strip().split()[1]
                new_mat = f"{new_basename}_{orig_mat}"
                old_to_new_material[orig_mat] = new_mat
                so.write(f"usemtl {new_mat}\n")
            else:
                so.write(line)

    # Second pass: rewrite .mtl with renamed material names and copied texture paths
    orig_mtl_path = None
    with open(src_obj_path, "r", errors="ignore") as si:
        for line in si:
            if line.startswith("mtllib "):
                orig_mtl_name = line.strip().split()[1]
                orig_mtl_path = _os.path.join(src_dir, orig_mtl_name)
                break

    if orig_mtl_path and _os.path.exists(orig_mtl_path):
        with (
            open(orig_mtl_path, "r", errors="ignore") as mi,
            open(new_mtl_path, "w") as mo,
        ):
            current_mat = None
            for line in mi:
                if line.startswith("newmtl "):
                    orig_mat = line.strip().split()[1]
                    new_mat = old_to_new_material.get(
                        orig_mat, f"{new_basename}_{orig_mat}"
                    )
                    current_mat = orig_mat
                    mo.write(f"newmtl {new_mat}\n")
                elif any(
                    line.strip().startswith(key)
                    for key in [
                        "map_Kd",
                        "map_Ks",
                        "map_Ka",
                        "map_d",
                        "bump",
                        "disp",
                        "decal",
                    ]
                ):
                    parts = line.strip().split()
                    texture_file = parts[-1]
                    src_tex = _os.path.join(src_dir, texture_file)
                    new_tex = f"{new_basename}_{_os.path.basename(texture_file)}"
                    dst_tex = _os.path.join(out_dir, new_tex)
                    if _os.path.exists(src_tex):
                        shutil.copy(src_tex, dst_tex)
                    mo.write(f"{parts[0]} {new_tex}\n")
                else:
                    mo.write(line)

    return out_obj_path


# -------------------- SAPIEN rendering with object --------------------


def run_simulation_sapien_graspxl(
    hand_mesh_path: str,
    body_mesh_path: str,
    object_mesh_path: str,
    object_rot: _np.ndarray,
    object_trans: _np.ndarray,
    simulation_cfg,
    bg_img_array,
    lowres_hand_flag: bool = False,
    lowres_downscale: int = 4,
    edge_soften_flag: bool = False,
    edge_soften_radius_px: float = 1.5,
    edge_soften_shrink_px: int = 0,
    use_theme_lighting: bool = False,
    theme_ambient_boost: float = 1.0,
    theme_key_align_prob: float = 0.0,
    theme_key_tint_strength: float = 0.5,
    theme_saturation_boost: float = 1.0,
):
    """Render a set of random cameras for (hand [+body] + object).

    Returns
    -------
    rgba_images: (n_images, H, W, 4) uint8
    depth_images: (n_images, H, W) float32
    camera_poses_all: (n_images, 4, 4) float32 (world←camera)
    fov_rads_all: (n_images,) float32
    bbox_all: (n_images, 4, 2) float32  # hand bbox only
    """
    import sapien.core as sapien
    from sapien.core import Pose
    import numpy as np
    import trimesh as _trimesh

    # Set rendering parameters
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_viewer_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(
        256
    )  # less noise with higher samples
    sapien.render.set_ray_tracing_denoiser("none")  # options: "optix" or "oidn"

    H = simulation_cfg.intrinsics.height
    W = simulation_cfg.intrinsics.width

    # --- engine / scene
    # Create scene
    scene = sapien.Scene()

    # camera sampling params
    n_images = int(simulation_cfg.n_images)
    cam_dist_std = float(simulation_cfg.cam_dist_std)
    cam_loc_std = float(simulation_cfg.cam_loc_std)
    cam_dist_mean = float(simulation_cfg.cam_dist_mean)

    # lighting
    n_lights = np.random.randint(1, simulation_cfg.max_lights + 1)
    # add_random_lighting(scene, num_lights=n_lights)

    # Themed lighting from BG (if enabled & BG is provided)
    theme_col = None
    if use_theme_lighting and bg_img_array is not None:
        theme_col = _theme_color_from_bg(
            bg_img_array, saturation_boost=theme_saturation_boost
        )

    add_random_lighting(
        scene,
        num_lights=n_lights,
        theme_color=theme_col,
        ambient_boost=theme_ambient_boost,
        key_align_prob=theme_key_align_prob,
        key_tint_strength=theme_key_tint_strength,
    )

    # --- load HAND
    hand_actor_builder = scene.create_actor_builder()
    hand_actor_builder.add_convex_collision_from_file(hand_mesh_path)
    hand_actor_builder.add_visual_from_file(hand_mesh_path)
    hand_actor = hand_actor_builder.build_static(name="hand")
    hand_actor.set_pose(
        sapien.Pose(p=np.array([0.0, 0.0, 0.0]), q=np.array([1, 0, 0, 0]))
    )

    # Precompute the hand mesh AABB center in the HAND'S LOCAL frame.
    # We'll transform this to WORLD each frame using the hand actor's pose.
    try:
        _hand_tm = _trimesh.load(hand_mesh_path, force="mesh", process=False)
        hand_center_local = _hand_tm.bounds.mean(axis=0).astype(_np.float32)  # (3,)
    except Exception:
        # Fallback: if mesh can't be loaded, assume local center at origin.
        hand_center_local = _np.zeros(3, dtype=_np.float32)

    # --- optional BODY
    if body_mesh_path:
        body_builder = scene.create_actor_builder()
        body_builder.add_convex_collision_from_file(body_mesh_path)
        body_builder.add_visual_from_file(body_mesh_path)
        body_mesh = body_builder.build_static(name="body")
        body_mesh.set_pose(
            sapien.Pose(p=np.array([0.0, 0.0, 0.0]), q=np.array([1, 0, 0, 0]))
        )
        HAS_BODY = True
    else:
        body_actor = None
        HAS_BODY = False

    # --- OBJECT
    obj_builder = scene.create_actor_builder()
    obj_builder.add_convex_collision_from_file(object_mesh_path)
    obj_builder.add_visual_from_file(object_mesh_path)
    obj_actor = obj_builder.build_static(name="object")
    # Ensure SAPIEN gets wxyz; optionally convert CV→SAPIEN axes.
    # obj_actor.set_pose(Pose(p=object_trans, q=axis_to_quaternion(object_rot)))
    obj_actor.set_pose(
        sapien.Pose(p=np.array([0.0, 0.0, 0.0]), q=np.array([1, 0, 0, 0]))
    )

    rgba_images = []
    depth_images = []
    camera_poses = []
    fov_rads = []
    bbox_all = []
    mask_all = []
    union_mask_all = []
    bg_crop_xy_all = []

    fov_deg = np.random.uniform(30, 45)
    fov_rad = np.deg2rad(fov_deg)

    # --- camera (single reusable camera)
    camera = scene.add_camera(
        name="camera",
        width=W,
        height=H,
        fovy=fov_rad,
        near=simulation_cfg.intrinsics.near,
        far=simulation_cfg.intrinsics.far,
    )

    for i in range(n_images):
        distance = (
            np.random.normal(loc=cam_dist_mean, scale=cam_dist_std) / 100.0
        )  # convert to meters
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.deg2rad(np.random.uniform(-180, 180))
        cam_x = distance * np.cos(elevation) * np.cos(azimuth)
        cam_y = distance * np.cos(elevation) * np.sin(azimuth)
        cam_z = distance * np.sin(elevation)
        cam_pos = np.array([cam_x, cam_y, cam_z])
        # cam_pos += np.random.normal(scale=cam_loc_dist, size=3)

        direction = -cam_pos  # from camera to origin
        forward = direction / np.linalg.norm(direction)
        world_up = np.array([0, 0, 1])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0, 1, 0])

        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Build the 4x4 camera pose matrix.
        cam_pose = np.eye(4)
        # cam_pose[:3, :3] = np.stack([forward, left, up], axis=1)
        cam_pose[:3, :3] = np.stack([forward, -right, up], axis=1)
        # cam_pose[:3, 3] = cam_pos

        # --- recenter to 3D AABB center of the HAND (no rotation change) ---
        # Transform local AABB center to WORLD using the hand actor's pose.
        _hand_pose = hand_actor.get_pose()  # sapien.Pose
        _T_hand = _hand_pose.to_transformation_matrix()  # (4,4), world ← hand
        hand_center_world = (_T_hand[:3, :3] @ hand_center_local) + _T_hand[:3, 3]

        # Slide camera sideways so optical axis goes through the hand center.
        v = hand_center_world - cam_pos  # vector from camera to center
        depth = float(np.dot(v, forward))  # along optical axis
        lateral = v - depth * forward  # perpendicular to optical axis
        cam_pos = cam_pos + lateral  # translate only; keep orientation
        cam_pose[:3, 3] = cam_pos

        camera.entity.set_pose(sapien.Pose(cam_pose))

        # Step the scene and render.
        scene.step()
        scene.update_render()
        camera.take_picture()  # Submit rendering jobs

        # ------------------------------- #
        # Retrieve Rendered Images and Process Segmentation
        # ------------------------------- #
        rgba = camera.get_picture("Color")  # [H, W, 4] float image in [0,1]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")

        position = camera.get_picture("Position")  # [H, W, 4]
        depth = -position[..., 2]
        depth_img = depth * 1000.0

        # seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        # label_img = seg_labels[..., 0].astype(np.uint8)
        # proc_result = process_segmentation(label_img, rgba_img, cut=False)
        # mask = proc_result["mask"]

        # bbox_dict = proc_result.get("bbox", None)

        seg_labels = camera.get_picture(
            "Segmentation"
        )  # [H, W, 4]; channel 0 = instance/entity id
        label_img = seg_labels[..., 0].astype(np.int32)

        # --- masks by auto-assigned entity IDs ---
        hand_mask = (label_img == 1).astype(np.uint8)
        if HAS_BODY:
            body_mask = (label_img == 2).astype(np.uint8)
            obj_mask = (label_img == 3).astype(np.uint8)  # not used
        else:
            obj_mask = (label_img == 2).astype(np.uint8)  # not used
            body_mask = np.zeros_like(hand_mask, dtype=np.uint8)
        union_mask = ((hand_mask | body_mask | obj_mask) > 0).astype(np.uint8)

        # --- bbox on HAND ONLY ---
        coords = np.argwhere(hand_mask > 0)
        if coords.size == 0:
            bbox_dict = None
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox_dict = {
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
            }

        if bbox_dict is not None:
            # Original bbox from segmentation (x_min, y_min, x_max, y_max)
            bbox_arr = np.array(
                [
                    bbox_dict["x_min"],
                    bbox_dict["y_min"],
                    bbox_dict["x_max"],
                    bbox_dict["y_max"],
                ]
            )
        else:
            bbox_arr = np.array([-1, -1, -1, -1])

        # ------------------------------- #
        # Compute bounding box in center-size form and convert to four corners.
        # ------------------------------- #
        if bbox_arr[0] != -1:
            x_min, y_min, x_max, y_max = bbox_arr
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            box_width = x_max - x_min
            box_height = y_max - y_min
            # Represent bbox as the four corner points (shape: (4,2))
            bbox_array = np.array(
                [
                    [center_x - box_width / 2, center_y - box_height / 2],  # top-left
                    [center_x + box_width / 2, center_y - box_height / 2],  # top-right
                    [
                        center_x + box_width / 2,
                        center_y + box_height / 2,
                    ],  # bottom-right
                    [
                        center_x - box_width / 2,
                        center_y + box_height / 2,
                    ],  # bottom-left
                ],
                dtype=float,
            )
        else:
            bbox_array = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]], dtype=float)

        if lowres_hand_flag:
            H, W = rgba_img.shape[:2]
            s = max(1, int(lowres_downscale))
            # Downscale with AREA, then upscale with LINEAR for a clean "low-res" look
            small = _cv2.resize(
                rgba_img,
                (max(1, W // s), max(1, H // s)),
                interpolation=_cv2.INTER_AREA,
            )
            rgba_img = _cv2.resize(small, (W, H), interpolation=_cv2.INTER_LINEAR)

        # if bg_img_array is not None:
        #     # Union mask (hand ∪ body), then border removal is done inside compositor
        #     composite_rgb = composite_on_random_bg(
        #         rgba_img=rgba_img,
        #         mask=union_mask,
        #         bg_img_array=bg_img_array,
        #     )
        #     rgba_img = composite_rgb.astype(np.uint8)

        if bg_img_array is not None:
            # Union mask (hand ∪ body ∪ object). Request crop offsets for matching MoGe crop later.
            composite_rgb, (cy, cx) = composite_on_random_bg(
                rgba_img=rgba_img,
                mask=union_mask,
                bg_img_array=bg_img_array,
                feather_radius_px=edge_soften_radius_px if edge_soften_flag else 0.0,
                feather_shrink_px=edge_soften_shrink_px if edge_soften_flag else 0,
                return_crop=True,
            )
            rgba_img = composite_rgb.astype(np.uint8)
            bg_crop_xy_all.append((cy, cx))
        else:
            bg_crop_xy_all.append((0, 0))

        rgba_images.append(rgba_img)
        depth_images.append(depth_img)
        camera_poses.append(cam_pose)
        fov_rads.append(fov_rad)
        bbox_all.append(bbox_array)
        mask_all.append(hand_mask)
        union_mask_all.append(union_mask)

        # scene.remove_camera(camera)

    rgba_images = np.stack(rgba_images, axis=0)
    depth_images = np.stack(depth_images, axis=0)
    camera_poses_all = np.stack(camera_poses, axis=0)
    fov_rads_all = np.stack(fov_rads, axis=0)
    bbox_all = np.stack(bbox_all, axis=0)
    mask_all = np.stack(mask_all, axis=0)
    bg_crop_xy_all = np.array(bg_crop_xy_all, dtype=np.int32)

    # Clear Sapien Memory
    del scene

    return (
        rgba_images,
        depth_images,
        camera_poses_all,
        fov_rads_all,
        bbox_all,
        mask_all,
        union_mask_all,
        bg_crop_xy_all,
    )


# Keep the module import-friendly name if you drop this into its own .py file
if __name__ == "__main__":
    import sys

    print(
        "This file defines create_sync_data_graspxl.py and sync_data_utils_graspxl helpers."
    )
