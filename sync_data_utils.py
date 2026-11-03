import os
import random
from typing import Optional, Tuple, Union
import numpy as np
import sapien

import cv2

import OpenEXR
import Imath
import pickle

import torch

cv2.setNumThreads(1)


def process_segmentation(label_img, orig_image, cut=False):
    """
    Process the segmentation image to produce:
      (0) A binary mask (0: background, 1: hand/mesh)
      (1) A tight bounding box around the hand in the mask
      (2) Optionally, the cropped image and cropped mask based on the bounding box.

    Parameters:
      label_img (np.array): The segmentation image from SAPIEN (first channel; [H, W]).
      orig_image (np.array): The original rendered RGB image ([H, W, 4] or [H, W, 3]).
      cut (bool): If True, output the cropped (cut) image and mask.

    Returns:
      dict: Contains "mask", "bbox" and if cut==True, "cut_image" and "cut_mask".
    """
    # (0) Create binary mask: any nonzero pixel is considered part of the mesh.
    binary_mask = (label_img > 0).astype(np.uint8)

    # (1) Find the tightest bounding box around the nonzero (mesh) pixels.
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        bbox = None  # No mesh detected
    else:
        # rows (y) and columns (x)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
        }

    result = {"mask": binary_mask, "bbox": bbox}

    # (2) If cut is True and a valid bounding box exists, crop the original image and mask.
    if cut and bbox is not None:
        cut_image = orig_image[
            bbox["y_min"] : bbox["y_max"] + 1, bbox["x_min"] : bbox["x_max"] + 1
        ]
        cut_mask = binary_mask[
            bbox["y_min"] : bbox["y_max"] + 1, bbox["x_min"] : bbox["x_max"] + 1
        ]
        result.update({"cut_image": cut_image, "cut_mask": cut_mask})

    return result


def _theme_color_from_bg(
    bg_img_array: np.ndarray, saturation_boost: float = 1.0
) -> list[float]:
    """
    Returns an sRGB theme color in [0,1] from a BG image/patch.
    We downscale to 64x64, take the mean, optionally boost saturation a bit.
    """
    import cv2, numpy as np

    if bg_img_array is None:
        return None
    img = bg_img_array
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    # mean color in sRGB
    mean_rgb = img64.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0

    # subtle saturation boost (HSV)
    hsv = cv2.cvtColor(
        (mean_rgb[None, None, :] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
    )
    h, s, v = hsv[0, 0].astype(np.float32)
    s = np.clip(s * float(saturation_boost), 0, 255)
    hsv[0, 0] = np.array([h, s, v], dtype=np.uint8)
    boosted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32) / 255.0

    return [float(x) for x in np.clip(boosted, 0.0, 1.0)]


def add_random_lighting(
    scene,
    num_lights=5,
    color_chance=0.2,
    theme_color=None,
    ambient_boost: float = 1.0,
    key_align_prob: float = 0.0,
    key_tint_strength: float = 0.5,
):
    """
    Add random lighting to a SAPIEN scene to generate a diverse image dataset.

    This function randomly sets the ambient light color and adds several lights
    (point, directional, and spot) with randomized parameters to the scene. It
    optionally adds an area light for ray tracing with a 50% chance. The function
    does not return any value; it modifies the scene in place by adding new light entities.

    Parameters:
        scene: An instance of a SAPIEN scene that includes methods:
               - set_ambient_light(color)
               - add_point_light(position, color, shadow, shadow_near, shadow_far, shadow_map_size)
               - add_directional_light(direction, color, shadow, position, shadow_scale, shadow_near, shadow_far, shadow_map_size)
               - add_spot_light(position, direction, inner_fov, outer_fov, color, shadow, shadow_near, shadow_far, shadow_map_size)
               - add_area_light_for_ray_tracing(pose, color, half_width, half_height)

        num_lights (int, optional): The number of random lights (point, directional, or spot) to add. Default is 5.

    Returns:
        None

    Side Effects:
        - Sets a random ambient light color for the scene.
        - Adds several light entities with randomized properties to the scene.

    Usage Example:
        # Assuming 'scene' is a valid SAPIEN scene object:
        add_random_lighting(scene, num_lights=5)
    """
    colored_light_added = False

    # Set a random ambient light color
    # Ambient: from theme if given; else current neutral-ish white
    if theme_color is not None:
        amb = np.clip(
            np.array(theme_color, dtype=np.float32) * float(ambient_boost), 0.0, 1.0
        )
        ambient_color = [float(amb[0]), float(amb[1]), float(amb[2])]
    else:
        ambient_color = [
            random.uniform(0.9, 1.0),
            random.uniform(0.9, 1.0),
            random.uniform(0.8, 0.95),
        ]
    scene.set_ambient_light(ambient_color)

    for i in range(num_lights):
        # Randomly choose the light type from "point", "directional", and "spot"
        light_type = random.choice(["point", "directional", "spot"])

        if light_type == "point":
            # Generate random parameters for a point light
            position = [random.uniform(-10, 10) for _ in range(3)]
            if not colored_light_added and random.random() < color_chance:
                # Use a more vivid / colored light
                color = [random.uniform(0.0, 1.0) for _ in range(3)]
                colored_light_added = True
            else:
                color = [
                    random.uniform(0.9, 1.0),
                    random.uniform(0.9, 1.0),
                    random.uniform(0.8, 0.95),
                ]
            shadow = random.choice([True, False])
            shadow_near = random.uniform(0.1, 1.0)
            shadow_far = random.uniform(5.0, 20.0)
            scene.add_point_light(
                position,
                color,
                shadow=shadow,
                shadow_near=shadow_near,
                shadow_far=shadow_far,
                shadow_map_size=2048,  # Default shadow map size; modify as needed
            )

        elif light_type == "directional":
            # Generate random parameters for a directional light
            direction = [random.uniform(-1, 1) for _ in range(3)]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = [d / norm for d in direction]
            else:
                direction = [1, 0, 0]
            if not colored_light_added and random.random() < color_chance:
                # Use a more vivid / colored light
                color = [random.uniform(0.0, 1.0) for _ in range(3)]
                colored_light_added = True
            else:
                color = [
                    random.uniform(0.9, 1.0),
                    random.uniform(0.9, 1.0),
                    random.uniform(0.8, 0.95),
                ]
            shadow = random.choice([True, False])
            # Random origin position for the directional light (affects shadow mapping)
            position = [random.uniform(-10, 10) for _ in range(3)]
            shadow_scale = random.uniform(5.0, 20.0)
            shadow_near = random.uniform(-20, 0)
            shadow_far = random.uniform(0, 20)
            scene.add_directional_light(
                direction,
                color,
                shadow=shadow,
                position=position,
                shadow_scale=shadow_scale,
                shadow_near=shadow_near,
                shadow_far=shadow_far,
                shadow_map_size=2048,
            )

        elif light_type == "spot":
            # Generate random parameters for a spot light
            position = [random.uniform(-10, 10) for _ in range(3)]
            direction = [random.uniform(-1, 1) for _ in range(3)]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = [d / norm for d in direction]
            else:
                direction = [1, 0, 0]
            inner_fov = random.uniform(10, 30)  # Inner field of view in degrees
            outer_fov = inner_fov + random.uniform(
                10, 30
            )  # Outer field of view in degrees
            if not colored_light_added and random.random() < color_chance:
                # Use a more vivid / colored light
                color = [random.uniform(0.0, 1.0) for _ in range(3)]
                colored_light_added = True
            else:
                color = [
                    random.uniform(0.9, 1.0),
                    random.uniform(0.9, 1.0),
                    random.uniform(0.8, 0.95),
                ]
            shadow = random.choice([True, False])
            shadow_near = random.uniform(0.1, 1.0)
            shadow_far = random.uniform(5.0, 20.0)
            scene.add_spot_light(
                position,
                direction,
                inner_fov,
                outer_fov,
                color,
                shadow=shadow,
                shadow_near=shadow_near,
                shadow_far=shadow_far,
                shadow_map_size=2048,
            )

    # Optionally add an area light for ray tracing with a 50% chance
    if random.choice([True, False]):
        position = [random.uniform(-10, 10) for _ in range(3)]
        # Create a pose for the area light (using identity rotation by default)
        pose = sapien.Pose(position)
        if not colored_light_added and random.random() < color_chance:
            # Use a more vivid / colored light
            color = [random.uniform(0.0, 1.0) for _ in range(3)]
            colored_light_added = True
        else:
            color = [
                random.uniform(0.9, 1.0),
                random.uniform(0.9, 1.0),
                random.uniform(0.8, 0.95),
            ]
        half_width = random.uniform(0.5, 5.0)
        half_height = random.uniform(0.5, 5.0)
        scene.add_area_light_for_ray_tracing(pose, color, half_width, half_height)


def remove_white_border(binary_mask: np.ndarray) -> np.ndarray:
    """
    Remove white boundary pixels from a binary mask.

    A boundary pixel is defined as a white pixel that becomes black
    after applying a one-pixel-wide erosion (i.e. it neighbors a black pixel).

    Parameters:
        binary_mask (np.ndarray): Input binary mask (dtype: uint8) with values 0 or 1.

    Returns:
        np.ndarray: A new binary mask (dtype: uint8) with immediate border white pixels set to 0.
    """
    from scipy.ndimage import binary_erosion

    # Convert to boolean for erosion operation
    mask_bool = binary_mask.astype(bool)

    # Create an 8-connected structuring element (3x3 all True)
    structure = np.ones((3, 3), dtype=bool)

    # Apply erosion: white pixels at the border become False.
    eroded = binary_erosion(mask_bool, structure=structure)

    # Identify the boundary: original mask is True but eroded version is False.
    boundary = mask_bool & ~eroded

    # Create a copy so we don't modify the original mask
    updated_mask = binary_mask.copy()
    updated_mask[boundary] = 0
    return updated_mask


def composite_on_random_bg(
    rgba_img: np.ndarray,
    mask: np.ndarray,
    bg_img_array: np.ndarray,
    rotate_bg: bool = False,
    feather_radius_px: float = 0.0,
    feather_shrink_px: int = 0,
    return_crop: bool = False,
):
    """
    Paste a foreground RGBA (or RGB) image onto a randomly rotated / translated
    background image, returning an (H, W, 3) composite RGB result.

    Parameters
    ----------
    rgba_img      : (H, W, 3|4) uint8 foreground image. If RGB, alpha=opaque.
    mask          : (H, W) float / uint8  foreground mask ∈ [0,1] or {0,1}.
    bg_img_array  : (h0, w0, 3|4) uint8 background image.

    Returns
    -------
    composite_rgb : (H, W, 3) uint8 final composite with the FG pasted on BG.
    """

    from PIL import Image

    # ────────────────────────── foreground sizes ───────────────────────────
    H, W = rgba_img.shape[:2]

    # ────────────────── 1) optional random in-plane rotation ────────────────
    bg_img = bg_img_array.copy()

    if rotate_bg:
        angle = np.random.uniform(0.0, 360.0)

        h0, w0 = bg_img.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w0 / 2.0, h0 / 2.0), angle, 1.0)

        # enlarge canvas so BG is never clipped
        cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])
        new_w = int(h0 * sin + w0 * cos)
        new_h = int(h0 * cos + w0 * sin)
        rot_mat[0, 2] += (new_w / 2.0) - (w0 / 2.0)
        rot_mat[1, 2] += (new_h / 2.0) - (h0 / 2.0)

        bg_img = cv2.warpAffine(
            bg_img,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # ─────────── 2) upscale BG if still smaller than foreground ────────────
    if bg_img.shape[0] < H or bg_img.shape[1] < W:
        scale_up = max(H / bg_img.shape[0], W / bg_img.shape[1])
        new_size = (
            int(bg_img.shape[1] * scale_up),
            int(bg_img.shape[0] * scale_up),
        )
        bg_img = np.array(Image.fromarray(bg_img).resize(new_size, Image.BICUBIC))

    # ────────────────── 3) random crop (= translation) ─────────────────────
    y0 = np.random.randint(0, bg_img.shape[0] - H + 1)
    x0 = np.random.randint(0, bg_img.shape[1] - W + 1)
    bg_crop = bg_img[y0 : y0 + H, x0 : x0 + W]

    # ───────────────────── 4) alpha / channel align ────────────────────────
    if rgba_img.shape[2] == 3:  # no alpha in FG
        alpha_fg = 255 * np.ones((H, W, 1), dtype=np.uint8)
        rgba_img = np.concatenate([rgba_img, alpha_fg], axis=2)

    if bg_crop.shape[2] == 3:  # no alpha in BG
        alpha_bg = 255 * np.ones((H, W, 1), dtype=np.uint8)
        bg_crop = np.concatenate([bg_crop, alpha_bg], axis=2)

    # ───────────────────── 5) actual compositing ───────────────────────────
    # Make sure RGBA is present for FG; BG alpha handled above already.
    if rgba_img.shape[2] == 3:
        alpha_fg = 255 * np.ones((H, W, 1), dtype=np.uint8)
        rgba_img = np.concatenate([rgba_img, alpha_fg], axis=2)

    # Build a soft mask near the boundary (feather) or fall back to binary.
    if feather_radius_px > 0.0 or feather_shrink_px > 0:
        # Start from [0,1] float mask
        mask_soft = mask.astype(np.float32)
        mask_soft = np.clip(mask_soft, 0.0, 1.0)

        # Optional: shrink a bit to avoid halos from bright BG
        if feather_shrink_px > 0:
            import cv2

            kernel = np.ones((3, 3), np.uint8)
            mask_soft = cv2.erode(
                (mask_soft * 255).astype(np.uint8),
                kernel,
                iterations=int(feather_shrink_px),
            )
            mask_soft = mask_soft.astype(np.float32) / 255.0

        # Feather the edge with Gaussian blur
        import cv2

        mask_soft = cv2.GaussianBlur(
            mask_soft,
            ksize=(0, 0),
            sigmaX=float(feather_radius_px),
            sigmaY=float(feather_radius_px),
        )
        mask_soft = np.clip(mask_soft, 0.0, 1.0)
    else:
        # Old behavior: hard matte + border cleanup
        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask = remove_white_border(binary_mask)
        mask_soft = binary_mask.astype(np.float32)

    # Proper alpha compositing with pre-multiplication near edges
    fg_rgb = rgba_img[..., :3].astype(np.float32)
    fg_a = rgba_img[..., 3].astype(np.float32) / 255.0
    # Combine renderer alpha with (softened) union mask
    alpha = np.clip(fg_a * mask_soft, 0.0, 1.0)[..., None]

    bg_rgb = bg_crop[..., :3].astype(np.float32)

    composite_rgb = (fg_rgb * alpha + bg_rgb * (1.0 - alpha)).astype(np.uint8)

    if return_crop:
        return composite_rgb, (int(y0), int(x0))
    else:
        return composite_rgb

    # # ───────────────────── 5) actual compositing ───────────────────────────
    # binary_mask = (mask > 0.5).astype(np.uint8)
    # binary_mask = remove_white_border(binary_mask)
    # mask_exp = np.expand_dims(binary_mask, axis=-1).astype(np.float32)

    # composite_rgb = (
    #     rgba_img[..., :3].astype(np.float32) * mask_exp
    #     + bg_crop[..., :3].astype(np.float32) * (1.0 - mask_exp)
    # ).astype(np.uint8)

    # return composite_rgb


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


def run_simulation_sapien(
    mesh_file: str,
    mesh_file_body: str,
    max_lights: int,
    n_images: int,
    cam_dist_mean: float,
    cam_dist_std: float,
    cam_loc_dist: float,
    intrinsics: dict,
    bg_img_array: np.array = None,
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
    """
    Runs a SAPIEN ray-traced simulation for a static mesh, producing multiple rendered views
    along with depth maps, camera poses, field-of-view values, and object bounding boxes.
    Optionally composites each render onto a random crop of a provided background image.

    Args:
        mesh_file (str):
            Path to the **hand** mesh file (e.g., OBJ/PLY/STL) to load at the origin.
        mesh_file_body (str, optional):
            Path to the **body/arm** mesh file to load together with the hand at the origin.max_lights (int):
            Maximum number of randomly placed light sources; the actual count is sampled
            uniformly from [1, max_lights].
        n_images (int):
            Number of distinct camera views (and corresponding images) to generate.
        cam_dist_mean (float):
            Mean camera distance from the mesh origin, in centimeters (converted internally to meters).
        cam_dist_std (float):
            Standard deviation of the camera distance, in centimeters.
        cam_loc_dist (float):
            Standard deviation for an optional additional random translation of the camera
            (currently commented out in code).
        intrinsics (dict):
            Dictionary of camera intrinsics with keys:
            - 'width' (int): image width in pixels
            - 'height' (int): image height in pixels
            - 'near' (float): near clipping plane distance (in meters)
            - 'far' (float): far clipping plane distance (in meters)
        bg_img_array (np.ndarray, optional):
            HxWxC background image array (uint8) used to composite the rendered foreground.
            If None, no background compositing is performed.

    Returns:
        rgba_images (np.ndarray):
            Array of shape (n_images, H, W, 3) or (n_images, H, W, 4), uint8 RGB(A) renderings.
        depth_images (np.ndarray):
            Array of shape (n_images, H, W), float depth maps in millimeters.
        camera_poses_all (np.ndarray):
            Array of shape (n_images, 4, 4), world-from-camera pose matrices for each view.
        fov_rads_all (np.ndarray):
            Array of shape (n_images,), vertical field-of-view angles in radians used for each image.
        bbox_all (np.ndarray):
            Array of shape (n_images, 4, 2), pixel coordinates of the four bounding-box corners
            (top-left, top-right, bottom-right, bottom-left). If no object is detected in a view,
            all four corners will be [-1, -1].
    """

    import numpy as np
    from PIL import Image

    # Set rendering parameters
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_viewer_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(
        256
    )  # less noise with higher samples
    sapien.render.set_ray_tracing_denoiser("none")  # options: "optix" or "oidn"

    # Create scene
    scene = sapien.Scene()

    # ---------------------------------------------------------------------- #
    # Build actors at the origin (0,0,0) with identity rotation.
    # Hand is always loaded. Body is optional.
    # ---------------------------------------------------------------------- #
    hand_builder = scene.create_actor_builder()
    hand_builder.add_convex_collision_from_file(mesh_file)
    hand_builder.add_visual_from_file(mesh_file)
    hand_mesh = hand_builder.build_static(name="hand_mesh")
    hand_mesh.set_pose(
        sapien.Pose(p=np.array([0.0, 0.0, 0.0]), q=np.array([1, 0, 0, 0]))
    )

    body_mesh = None
    if (
        mesh_file_body is not None
        and isinstance(mesh_file_body, str)
        and len(mesh_file_body) > 0
    ):
        body_builder = scene.create_actor_builder()
        body_builder.add_convex_collision_from_file(mesh_file_body)
        body_builder.add_visual_from_file(mesh_file_body)
        body_mesh = body_builder.build_static(name="body_mesh")
        body_mesh.set_pose(
            sapien.Pose(p=np.array([0.0, 0.0, 0.0]), q=np.array([1, 0, 0, 0]))
        )
        HAS_BODY = True
    else:
        body_mesh = None
        HAS_BODY = False

    # ---------------------------------------------------------------------- #
    # Add random lighting to the scene.
    # ---------------------------------------------------------------------- #
    # n_lights = np.random.randint(1, max_lights + 1)
    # add_random_lighting(scene, num_lights=n_lights)

    n_lights = np.random.randint(1, max_lights + 1)

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

    rgba_images = []
    depth_images = []
    fov_rads = []
    camera_poses = []
    bbox_all = []
    mask_all = []
    union_mask_all = []
    bg_crop_xy_all = []

    # fov_deg = np.random.uniform(45, 90)
    fov_deg = 40
    fov_rad = np.deg2rad(fov_deg)

    camera = scene.add_camera(
        name="camera",
        width=intrinsics["width"],
        height=intrinsics["height"],
        fovy=fov_rad,
        near=intrinsics["near"],
        far=intrinsics["far"],
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

        seg_labels = camera.get_picture(
            "Segmentation"
        )  # [H, W, 4]; channel 0 = instance/entity id
        label_img = seg_labels[..., 0].astype(np.int32)

        # --- masks by auto-assigned entity IDs ---
        if HAS_BODY:
            hand_mask = (label_img == 1).astype(np.uint8)
            body_mask = (label_img == 2).astype(np.uint8)
        else:
            # Fallback: if somehow id missing, treat any nonzero as hand (degenerate single-mesh case)
            hand_mask = (label_img > 0).astype(np.uint8)
            body_mask = np.zeros_like(hand_mask, dtype=np.uint8)
        union_mask = ((hand_mask | body_mask) > 0).astype(np.uint8)

        # print("hand_mask", np.sum(hand_mask), "body_mask", np.sum(body_mask), "union_mask", np.sum(union_mask))

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
            small = cv2.resize(
                rgba_img, (max(1, W // s), max(1, H // s)), interpolation=cv2.INTER_AREA
            )
            rgba_img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

        # if bg_img_array is not None:
        #     # Union mask (hand ∪ body), then border removal is done inside compositor
        #     composite_rgb = composite_on_random_bg(
        #         rgba_img=rgba_img,
        #         mask=union_mask,
        #         bg_img_array=bg_img_array,
        #         feather_radius_px = edge_soften_radius_px if edge_soften_flag else 0.0,
        #         feather_shrink_px = edge_soften_shrink_px if edge_soften_flag else 0,
        #     )
        #     rgba_img = composite_rgb.astype(np.uint8)
        if bg_img_array is not None:
            # Union mask (hand ∪ body); request crop offsets
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

        # rgba_img = np.transpose(rgba_img, (1, 0, 2))
        # depth_img = np.transpose(depth_img, (1, 0))

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


def get_random_background(bg_files, target_H, target_W):
    from PIL import Image

    if bg_files:
        bg_file = random.choice(bg_files)
        bg_img_pil = Image.open(bg_file).convert("RGB")
        scale_factor = random.uniform(1.2, 2)
        new_width = int(bg_img_pil.width * scale_factor)
        new_height = int(bg_img_pil.height * scale_factor)
        bg_img_pil = bg_img_pil.resize((new_width, new_height), Image.BICUBIC)
        bg_img_array = np.array(bg_img_pil)
        # Ensure the background is at least as large as the target camera resolution.
        if bg_img_array.shape[0] < target_H or bg_img_array.shape[1] < target_W:
            scale_up = max(
                target_H / bg_img_array.shape[0], target_W / bg_img_array.shape[1]
            )
            new_width = int(np.ceil(bg_img_array.shape[1] * scale_up))
            new_height = int(np.ceil(bg_img_array.shape[0] * scale_up))
            bg_img_array = np.array(
                Image.fromarray(bg_img_array).resize(
                    (new_width, new_height), Image.BICUBIC
                )
            )

    return bg_img_array


def shift_hue(np_img, hue_shift_deg: float):
    """
    Shift the hue of a numpy RGB image by hue_shift_deg degrees.
    """
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.float32) + hue_shift_deg / 2) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def bilinear_sample(env, map_x, map_y):
    H, W, C = env.shape

    # wrap x and clamp y to valid range
    map_x = np.mod(map_x, W).astype(np.float32)
    map_y = np.clip(map_y, 0, H - 1).astype(np.float32)

    # integer neighbours
    x0 = np.floor(map_x).astype(np.int32)
    x1 = (x0 + 1) % W
    y0 = np.floor(map_y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)

    # fractional part
    dx = map_x - x0
    dy = map_y - y0
    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    # gather – each returns (H, W, 3)
    Ia = env[y0, x0]
    Ib = env[y1, x0]
    Ic = env[y0, x1]
    Id = env[y1, x1]

    # weighted sum (broadcast channel-wise)
    out = (
        Ia * wa[..., None]
        + Ib * wb[..., None]
        + Ic * wc[..., None]
        + Id * wd[..., None]
    )

    return out.astype(np.float32)


# simple cache for loaded envmaps
_env_cache = {}


def render_envmap_to_ldr(
    env_map_path: str, width: int, height: int, gamma: float = 2.2
) -> np.ndarray:
    """
    Load the full EXR once (cached), then each call just picks
    a random W×H patch from it and tonemaps to 8-bit sRGB.

    Returns:
        uint8 array of shape (height, width, 3)
    """
    # ---------------------------------------------- 1. Random POV + FOV
    yaw = np.random.uniform(0, 2 * np.pi)
    pitch = np.random.uniform(-np.pi / 16, np.pi / 16)  # +/- 45° pitch

    # OpenGL-style forward direction: Z+ forward
    forward = np.array(
        [
            np.cos(pitch) * np.sin(yaw),  # x
            np.sin(pitch),  # y
            np.cos(pitch) * np.cos(yaw),  # z
        ],
        dtype=np.float32,
    )

    forward /= np.linalg.norm(forward)

    # Fixed world up (Y+ up)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Right vector (X+) via cross product
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    # Recomputed true up vector (Y+)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Assemble rotation matrix R: columns are right, up, forward (OpenGL convention)
    R = np.stack([right, up, forward], axis=1)  # shape (3, 3)

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, :3] = R

    fov_y_rad = np.deg2rad(np.random.uniform(45, 60))

    # ---------------------------------------------- 2. Load EXR via OpenEXR (cache)
    if env_map_path not in _env_cache:
        exr = OpenEXR.InputFile(env_map_path)
        dw = exr.header()["dataWindow"]
        W_env = dw.max.x - dw.min.x + 1
        H_env = dw.max.y - dw.min.y + 1

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(H_env, W_env)
        g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(H_env, W_env)
        b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(H_env, W_env)
        env = np.stack([r, g, b], axis=-1)  # (H_env, W_env, 3)

        _env_cache[env_map_path] = env
    else:
        env = _env_cache[env_map_path]

    H_env, W_env = env.shape[:2]

    # ---------------------------------------------- 3. Generate camera ray directions
    jj, ii = np.meshgrid(np.arange(width), np.arange(height))  # pixel grid
    cx, cy = width * 0.5, height * 0.5
    fy = (height * 0.5) / np.tan(fov_y_rad * 0.5)
    fx = fy * (width / height)

    dirs_cam = np.stack(
        [
            (jj - cx) / fx,  # +X right
            -(ii - cy) / fy,  # +Y down
            np.ones_like(jj),  # +Z forward
        ],
        axis=-1,
    )

    norm = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam /= np.clip(norm, 1e-8, None)

    # dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # ---------------------------------------------- 4. Rotate rays to world space
    R_wc = cam_pose[:3, :3].astype(np.float32)
    dirs_world = dirs_cam @ R_wc.T  # (H, W, 3)

    x, y, z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
    phi = np.arctan2(x, -z)  # longitude
    theta = np.arccos(np.clip(y, -1.0, 1.0))  # latitude

    u = (phi / (2 * np.pi) + 0.5) * W_env
    v = (theta / np.pi) * H_env
    v = np.clip(v, 0, H_env - 1.0)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    if np.isnan(map_x).any() or np.isnan(map_y).any():
        print("NaNs detected in sampling grid!")
        np.save("debug_env.npy", env)
        np.save("debug_mapx.npy", map_x)
        np.save("debug_mapy.npy", map_y)
        raise RuntimeError("Aborting due to NaNs in remap grid.")

    # ---------------------------------------------- 5. Sample from lat-long env

    try:
        hdr = cv2.remap(
            env,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        ).astype(np.float32)
    except cv2.error as e:
        print(f"[WARN] remap failed – falling back to NumPy: {e}")
        hdr = bilinear_sample(env, map_x, map_y)

    # ---------------------------------------------- 6. Tonemap to LDR
    p95 = np.percentile(hdr, 95)
    exposure = 0.95**gamma / max(p95, 1e-6)
    ldr = np.clip(hdr * exposure, 0.0, 1.0) ** (1.0 / gamma)

    return (ldr * 255 + 0.5).astype(np.uint8)


# --- ZIP-FACE HELPERS -------------------------------------------------
def _tm_load_as_trimesh(path: str, process: bool = False):
    import trimesh, numpy as np

    m = trimesh.load(path, process=process, maintain_order=True)
    if isinstance(m, trimesh.Trimesh):
        return m
    if isinstance(m, trimesh.Scene):
        parts = []
        if hasattr(m, "graph") and hasattr(m.graph, "nodes_geometry"):
            try:
                for node_name, geom_name in m.graph.nodes_geometry:
                    if geom_name not in m.geometry:
                        continue
                    g = m.geometry[geom_name].copy()
                    try:
                        T = m.graph.get(node_name)
                    except Exception:
                        T = np.eye(4)
                    if T is None:
                        T = np.eye(4)
                    g.apply_transform(T)
                    parts.append(g)
            except Exception:
                parts.extend([g.copy() for g in m.geometry.values()])
        else:
            parts.extend([g.copy() for g in m.geometry.values()])
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if isinstance(m, (list, tuple)):
        parts = []
        for it in m:
            if isinstance(it, trimesh.Trimesh):
                parts.append(it)
            elif isinstance(it, trimesh.Scene):
                # recurse: flatten a child scene
                parts.append(_tm_load_as_trimesh(it, process=process))
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    raise TypeError(f"Unsupported mesh type from loader for {path}: {type(m)}")


def apply_zip_faces_to_body(
    body_obj_path: str, hand_obj_path: str, zip_pkl_path: str, out_path: str = None
) -> str:
    """Read BODY and HAND OBJs, then:
      - copy the needed HAND vertices (from zip_pkl) into BODY vertex array
      - append the mixed faces to BODY

    NOTE: We intentionally avoid loading with trimesh here.
    Once BODY has UV seams (f v/vt), trimesh may duplicate vertices and break
    the vertex indexing expected by zip_arm.pkl.
    """

    def _read_obj_v_vt_f(path: str):
        V = []
        VT = []
        F = []
        FVT = []  # list[(vt0,vt1,vt2)] or None per face

        def _parse_face_tok(tok: str):
            sp = tok.split("/")
            v = int(sp[0]) - 1
            vt = None
            if len(sp) >= 2 and sp[1] != "":
                vt = int(sp[1]) - 1
            return v, vt

        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        V.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vt "):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        VT.append([float(parts[1]), float(parts[2])])
                elif line.startswith("f "):
                    toks = line.strip().split()[1:]
                    if len(toks) < 3:
                        continue
                    corners = [_parse_face_tok(tok) for tok in toks]
                    # fan triangulate if needed
                    for i in range(1, len(corners) - 1):
                        tri = [corners[0], corners[i], corners[i + 1]]
                        F.append([tri[0][0], tri[1][0], tri[2][0]])
                        if (
                            tri[0][1] is not None
                            and tri[1][1] is not None
                            and tri[2][1] is not None
                        ):
                            FVT.append([tri[0][1], tri[1][1], tri[2][1]])
                        else:
                            FVT.append(None)

        V = np.asarray(V, dtype=np.float32)
        VT = np.asarray(VT, dtype=np.float32) if len(VT) else None
        F = np.asarray(F, dtype=np.int64)

        has_uv = (
            (VT is not None) and all(x is not None for x in FVT) and len(FVT) == len(F)
        )
        FVT_arr = np.asarray(FVT, dtype=np.int64) if has_uv else None
        return V, VT, F, FVT_arr

    Vb, VTb, Fb, FVTb = _read_obj_v_vt_f(body_obj_path)
    Vh, _, _, _ = _read_obj_v_vt_f(hand_obj_path)

    with open(zip_pkl_path, "rb") as f:
        payload = pickle.load(f)

    faces_spec = payload.get("faces", [])
    body_only_faces = payload.get("body_only_faces", [])

    # map each used HAND vertex index -> appended BODY index
    needed_h = sorted(
        {int(idx) for face in faces_spec for (lab, idx) in face if lab == "H"}
    )
    h2b_new = {}

    Vb_list = [Vb]
    for hv in needed_h:
        h2b_new[hv] = Vb.shape[0] + len(h2b_new)
    if needed_h:
        Vb_list.append(Vh[np.array(needed_h, dtype=int)])

    Vb2 = np.vstack(Vb_list)

    # build appended faces
    new_faces = []
    new_faces_uv = []

    has_uv = (VTb is not None) and (FVTb is not None)
    v_default = None
    face_centroids = None
    face_uv_mean = None
    kdtree = None
    extra_vts = []

    if has_uv:
        # pick a stable vt per body vertex (used for newly created faces that don't exist in template)
        v_default = {}
        for tri_v, tri_vt in zip(Fb, FVTb):
            for vv, vt in zip(tri_v.tolist(), tri_vt.tolist()):
                if vv not in v_default:
                    v_default[vv] = vt

        # --- seam UV heuristic -------------------------------------------------
        # Zip faces are manually created (mixing BODY and appended HAND vertices),
        # so they don't have canonical UVs. To avoid them sampling the texture at
        # a dummy location (0,0), we assign each zip face a *constant* UV sampled
        # from the nearest original BODY face. This makes the zip triangles take
        # a plausible local skin color from whatever SMPLitex texture is bound.
        face_centroids = Vb[Fb].mean(axis=1)  # (M,3)
        face_uv_mean = VTb[FVTb].mean(axis=1)  # (M,2)
        try:
            from scipy.spatial import cKDTree

            kdtree = cKDTree(face_centroids)
        except Exception:
            kdtree = None

        # keep a dummy vt for safety (should not be used after the heuristic)
        VTb = np.vstack([VTb, np.array([[0.0, 0.0]], dtype=np.float32)])

    for face in faces_spec:
        tri = []
        for lab, idx in face:
            idx = int(idx)
            if lab == "B":
                tri.append(idx)
            else:  # 'H'
                tri.append(h2b_new[idx])
        new_faces.append(tri)
        if has_uv:
            c = Vb2[np.asarray(tri, dtype=np.int64)].mean(axis=0)
            if kdtree is not None:
                _, nn = kdtree.query(c, k=1)
                nn = int(nn)
            else:
                # brute force fallback
                nn = int(np.argmin(((face_centroids - c[None, :]) ** 2).sum(axis=1)))

            uv = face_uv_mean[nn].astype(np.float32)
            vt_idx = int(VTb.shape[0] + len(extra_vts))
            extra_vts.append(uv)
            # use the same UV at all 3 corners → approximately the mean face color
            new_faces_uv.append([vt_idx, vt_idx, vt_idx])

    if has_uv and len(extra_vts) > 0:
        VTb = np.vstack([VTb, np.asarray(extra_vts, dtype=np.float32)])

    # ---- add body-only faces (no new vertices needed) ----
    Fb_sorted = set(map(tuple, np.sort(Fb, axis=1)))

    body_only_tris = []
    body_only_tris_uv = []
    for face in body_only_faces:
        btri = [int(idx) for (lab, idx) in face]
        if any(lab != "B" for (lab, _) in face):
            continue
        if not all(0 <= v < Vb.shape[0] for v in btri):
            raise ValueError(
                f"Body-only face references out-of-range vertex: {btri} (Vb={Vb.shape[0]})"
            )
        key = tuple(sorted(btri))
        if key not in Fb_sorted:
            body_only_tris.append(btri)
            Fb_sorted.add(key)
            if has_uv:
                body_only_tris_uv.append([int(v_default.get(v, 0)) for v in btri])

    # stitch all faces together
    stacks = [Fb]
    if new_faces:
        stacks.append(np.asarray(new_faces, dtype=np.int64))
    if body_only_tris:
        stacks.append(np.asarray(body_only_tris, dtype=np.int64))
    Fb2 = np.vstack(stacks)

    FVTb2 = None
    if has_uv:
        uv_stacks = [FVTb]
        if new_faces_uv:
            uv_stacks.append(np.asarray(new_faces_uv, dtype=np.int64))
        if body_only_tris_uv:
            uv_stacks.append(np.asarray(body_only_tris_uv, dtype=np.int64))
        FVTb2 = np.vstack(uv_stacks)

    # write
    out_path = out_path or body_obj_path

    with open(out_path, "w") as obj:
        mtl_name = os.path.basename(os.path.splitext(out_path)[0] + "_body.mtl")
        obj.write("mtllib " + mtl_name + "\n")
        for v in Vb2:
            obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_uv:
            for uv in VTb:
                obj.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        obj.write("usemtl body\n")

        if has_uv:
            for f, ft in zip(Fb2, FVTb2):
                a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
                ua, ub, uc = int(ft[0]) + 1, int(ft[1]) + 1, int(ft[2]) + 1
                obj.write(f"f {a}/{ua} {b}/{ub} {c}/{uc}\n")
        else:
            for f in Fb2:
                a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
                obj.write(f"f {a} {b} {c}\n")

    return out_path


# -----------------------------------------------------
def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)
    return img


def _alpha_mask_from_bg(img_bgr: np.ndarray, thr: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thr).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _blend_masked(
    base_bgr: np.ndarray, changed_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    m = (mask / 255.0)[:, :, None]
    out = changed_bgr.astype(np.float32) * m + base_bgr.astype(np.float32) * (1 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------- augmentation building blocks --------------------
def lab_shift(
    img_bgr: np.ndarray,
    dL: float,
    da: float,
    db: float,
    sL: float = 1.0,
    gamma: float = 1.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    img_bgr = _ensure_uint8(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    L2 = np.clip(L * sL + dL, 0, 255)
    a2 = np.clip(a + da, 0, 255)
    b2 = np.clip(b + db, 0, 255)
    lab2 = cv2.merge([L2, a2, b2]).astype(np.uint8)
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if abs(gamma - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] = np.clip(255.0 * ((hsv[..., 2] / 255.0) ** (1.0 / gamma)), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if mask is not None:
        out = _blend_masked(img_bgr, out, mask)
    return out


def color_transfer_lab_mean_std(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip: bool = True,
) -> np.ndarray:
    """Match src LAB mean/std to reference LAB mean/std."""
    src = _ensure_uint8(src_bgr)
    ref = _ensure_uint8(ref_bgr)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Only compute stats over masked foreground to avoid black bg skew
    if mask is not None:
        m = mask.astype(bool)
        m3 = np.dstack([m, m, m])
        src_vals = src_lab[m3].reshape(-1, 3)
    else:
        src_vals = src_lab.reshape(-1, 3)
    ref_vals = ref_lab.reshape(-1, 3)

    src_mean, src_std = src_vals.mean(axis=0), src_vals.std(axis=0) + 1e-6
    ref_mean, ref_std = ref_vals.mean(axis=0), ref_vals.std(axis=0) + 1e-6

    out_lab = (src_lab - src_mean) / src_std * ref_std + ref_mean

    if clip:
        out_lab = np.clip(out_lab, 0, 255)
    out_bgr = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    if mask is not None:
        out_bgr = _blend_masked(src, out_bgr, mask)
    return out_bgr


def augment_skin_tone_random(
    img_bgr: np.ndarray,
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    params: Optional[dict] = None,
    mask: Optional[np.ndarray] = None,
    return_params: bool = False,
):
    img_bgr = _ensure_uint8(img_bgr)
    if rng is None:
        rng = np.random.default_rng()

    # sample params once (or reuse provided)
    if params is None:
        if rng.random() > p:
            out = img_bgr.copy()
            return (out, None) if return_params else out

        t = float(rng.uniform(-1.0, 1.0))
        params = dict(
            dL=25.0 * t,
            da=15.0 * t,
            db=12.0 * t,
            sL=float(np.clip(1.0 + 0.15 * t, 0.7, 1.3)),
            gamma=float(np.clip(1.0 + 0.10 * t, 0.7, 1.5)),
        )

    if mask is None:
        mask = _alpha_mask_from_bg(img_bgr)

    out = lab_shift(img_bgr, mask=mask, **params)
    return (out, params) if return_params else out


def alpha_mask_from_bg(img_bgr: np.ndarray, thr: int = 8) -> np.ndarray:
    return _alpha_mask_from_bg(img_bgr, thr=thr)


def skin_mask_ycrcb_rgb(
    img_rgb_u8: np.ndarray, y_min: int = 20, y_max: int = 245
) -> np.ndarray:
    img = np.asarray(img_rgb_u8, dtype=np.uint8)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    mask = (
        (Cr >= 133)
        & (Cr <= 173)
        & (Cb >= 77)
        & (Cb <= 127)
        & (Y >= y_min)
        & (Y <= y_max)
    )
    return mask.astype(np.uint8) * 255


def skin_fraction_ycrcb_rgb(samples_rgb_u8: np.ndarray) -> float:
    s = np.asarray(samples_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    if s.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(s.reshape(1, -1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    Y, Cr, Cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]
    m = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127) & (Y >= 20) & (Y <= 245)
    return float(m.mean())


def is_skin_like_rgb_samples(
    samples_rgb_u8: np.ndarray, frac_thr: float = 0.35
) -> bool:
    s = np.asarray(samples_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    if s.shape[0] < 8:
        return False
    mean = s.mean(axis=0)
    if mean.max() < 35 or mean.min() > 245:
        return False
    return skin_fraction_ycrcb_rgb(s) >= frac_thr


def resolve_obj_map_kd(obj_path: str) -> Optional[str]:
    import re

    if not obj_path or not os.path.exists(obj_path):
        return None
    obj_dir = os.path.dirname(obj_path)
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        obj_txt = f.read()

    mtllib_m = re.search(r"^\\s*mtllib\\s+([^\\r\\n]+)\\s*$", obj_txt, flags=re.M)
    if not mtllib_m:
        return None
    mtl_name = mtllib_m.group(1).strip()
    mtl_path = mtl_name if os.path.isabs(mtl_name) else os.path.join(obj_dir, mtl_name)
    if not os.path.exists(mtl_path):
        return None

    usemtl_m = re.search(r"^\\s*usemtl\\s+([^\\r\\n]+)\\s*$", obj_txt, flags=re.M)
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        mtl_txt = f.read()

    target = usemtl_m.group(1).strip() if usemtl_m else None
    if target:
        block = re.search(
            rf"^\\s*newmtl\\s+{re.escape(target)}\\s*$([\\s\\S]*?)(?=^\\s*newmtl\\s+|\\Z)",
            mtl_txt,
            flags=re.M,
        )
        scope = block.group(1) if block else mtl_txt
    else:
        scope = mtl_txt

    mk = re.search(r"^\\s*map_Kd\\s+([^\\r\\n]+)\\s*$", scope, flags=re.M)
    if not mk:
        return None
    tex = mk.group(1).strip()
    return tex if os.path.isabs(tex) else os.path.normpath(os.path.join(obj_dir, tex))


####################################################################################
def _to_xyxy(b: np.ndarray) -> np.ndarray:
    """
    Normalize various bbox formats to XYXY per-sample.
    Supports shapes:
      - (4,) as [x_min, y_min, x_max, y_max]
      - (4,2) as four corners [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    Returns (4,) float32 XYXY. If invalid (e.g., all -1), returns [-1,-1,-1,-1].
    """
    b = np.asarray(b)
    if b.shape == (4,):
        return b.astype(np.float32)
    if b.shape == (4, 2):
        xs = b[:, 0]
        ys = b[:, 1]
        if np.all(xs < 0) or np.all(ys < 0):
            return np.array([-1, -1, -1, -1], dtype=np.float32)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    raise ValueError(f"Unsupported bbox shape {b.shape}; expected (4,) or (4,2).")


def _infer_layout(images: np.ndarray) -> Tuple[str, Tuple[int, int, int, int]]:
    """
    Infer NHWC vs NCHW. Returns ('NHWC' or 'NCHW', shape).
    """
    if images.ndim != 4:
        raise ValueError(f"images must be 4D, got shape {images.shape}")
    B, A, C, D = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
    # Heuristic: if the last dim looks like channels, assume NHWC; else NCHW
    if images.shape[-1] in (3, 4) and images.shape[1] not in (3, 4):
        return "NHWC", images.shape
    if images.shape[1] in (3, 4) and images.shape[-1] not in (3, 4):
        return "NCHW", images.shape
    # Fallback: prefer NHWC if both could be channels
    return "NHWC", images.shape


def occlude_bbox_with_colored_rect_batch(
    images: np.ndarray,
    bboxes: Union[np.ndarray, list],
    p: float = 0.5,
    size_scale_range: Tuple[float, float] = (0.75, 1.25),
    max_cover_frac: float = 0.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Post-generation augmentation:
      For each image in the batch, with probability p, draw a filled rectangle
      of random color whose size is ~ [0.75, 1.25] × bbox. Position it so that
      the overlap with the original bbox covers a random fraction t ∈ [0, max_cover_frac]
      of the bbox area. Returns a *new* array with the same layout as input.

    Args:
      images: np.uint8 batch. NHWC (B,H,W,3|4) or NCHW (B,3|4,H,W).
      bboxes: batch-aligned bboxes, one per image. Either:
              - (B,4) XYXY, or
              - (B,4,2) four corners (top-left, top-right, bottom-right, bottom-left).
      p: per-image probability to apply augmentation (default 0.5).
      size_scale_range: rectangle size scale range relative to bbox (default (0.75,1.25)).
      max_cover_frac: max fraction of bbox area to be covered (default 0.5).
      rng: optional np.random.Generator for reproducibility.

    Returns:
      images_out: np.uint8 array with same shape/layout as input.
    """
    if images.dtype != np.uint8:
        raise ValueError(f"images must be uint8, got {images.dtype}")
    if rng is None:
        rng = np.random.default_rng()

    layout, shape = _infer_layout(images)
    if layout == "NCHW":
        # convert to NHWC for drawing
        imgs = np.transpose(images, (0, 2, 3, 1)).copy()
    else:
        imgs = images.copy()

    B = imgs.shape[0]
    H, W = imgs.shape[1], imgs.shape[2]

    # normalize bbox container to per-sample XYXY list
    bboxes = np.array(bboxes, dtype=np.float32)
    if bboxes.ndim == 2 and bboxes.shape[1] == 4:
        xyxy_list = [bboxes[i] for i in range(min(B, bboxes.shape[0]))]
    elif bboxes.ndim == 3 and bboxes.shape[1:] == (4, 2):
        xyxy_list = [_to_xyxy(bboxes[i]) for i in range(min(B, bboxes.shape[0]))]
    else:
        raise ValueError(
            f"Unsupported bboxes shape {bboxes.shape}; expected (B,4) or (B,4,2)."
        )

    for i in range(B):
        if rng.random() > p:
            continue

        xyxy = _to_xyxy(xyxy_list[i])
        x_min, y_min, x_max, y_max = [float(v) for v in xyxy]

        # skip invalid/empty bbox
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
            continue

        # Clamp bbox to image frame
        x_min_c = max(0, min(W - 1, int(round(x_min))))
        y_min_c = max(0, min(H - 1, int(round(y_min))))
        x_max_c = max(0, min(W, int(round(x_max))))
        y_max_c = max(0, min(H, int(round(y_max))))

        bw = max(1, x_max_c - x_min_c)
        bh = max(1, y_max_c - y_min_c)
        bbox_area = bw * bh

        # Random rectangle size relative to bbox
        s = float(rng.uniform(size_scale_range[0], size_scale_range[1]))
        rw = max(1, int(round(bw * s)))
        rh = max(1, int(round(bh * s)))
        # Keep rect within image bounds by clipping its final placement later

        # Target overlap area: fraction t in [0, max_cover_frac] of bbox area
        t = float(rng.uniform(0.0, max_cover_frac))
        target_overlap = t * bbox_area

        # Quick overlap area helper
        def _overlap_area(rx0, ry0, rw, rh):
            rx1 = rx0 + rw
            ry1 = ry0 + rh
            ix0 = max(rx0, x_min_c)
            iy0 = max(ry0, y_min_c)
            ix1 = min(rx1, x_max_c)
            iy1 = min(ry1, y_max_c)
            return max(0, ix1 - ix0) * max(0, iy1 - iy0)

        # Rectangle size ~ [0.75, 1.25] × bbox
        s = float(rng.uniform(size_scale_range[0], size_scale_range[1]))
        rw = max(1, int(round(bw * s)))
        rh = max(1, int(round(bh * s)))

        # # Pick P uniformly inside the bbox
        # px = int(rng.integers(x_min_c, x_max_c))
        # py = int(rng.integers(y_min_c, y_max_c))
        # --- pick P inside bbox, restricted to central band of the image ---
        # central band ranges (exclusive high for rng.integers)
        cx_low = int(np.ceil(0.25 * W))
        cx_high = int(np.floor(0.75 * W))
        cy_low = int(np.ceil(0.25 * H))
        cy_high = int(np.floor(0.75 * H))

        # intersection of bbox with central band
        px_low = max(x_min_c, cx_low)
        px_high = min(x_max_c, cx_high)  # exclusive in rng.integers
        py_low = max(y_min_c, cy_low)
        py_high = min(y_max_c, cy_high)  # exclusive in rng.integers

        # if the intersection is empty (e.g., bbox outside the central band), fall back to bbox-uniform
        if px_low >= px_high or py_low >= py_high:
            px = int(rng.integers(x_min_c, x_max_c))
            py = int(rng.integers(y_min_c, y_max_c))
        else:
            px = int(rng.integers(px_low, px_high))
            py = int(rng.integers(py_low, py_high))

        # Determine quadrant of P relative to bbox center
        cx = (x_min_c + x_max_c) * 0.5
        cy = (y_min_c + y_max_c) * 0.5
        right = px >= cx
        up = (
            py >= cy
        )  # y increases downward in image coords; here "up" means toward larger y (screen down).
        # If you prefer math-style "up" (toward smaller y), flip this: up = py <= cy

        # Available room from P to bbox edges along the chosen quadrant
        if right and up:  # Upper-right (screen coords)
            avail_w = max(0, x_max_c - px)
            avail_h = max(0, y_max_c - py)
            quad = "UR"
        elif (not right) and up:  # Upper-left
            avail_w = max(0, px - x_min_c)
            avail_h = max(0, y_max_c - py)
            quad = "UL"
        elif (not right) and (not up):  # Lower-left
            avail_w = max(0, px - x_min_c)
            avail_h = max(0, py - y_min_c)
            quad = "LL"
        else:  # Lower-right
            avail_w = max(0, x_max_c - px)
            avail_h = max(0, py - y_min_c)
            quad = "LR"

        # Max overlap dims cannot exceed avail room nor rectangle size
        max_iw = min(avail_w, rw)
        max_ih = min(avail_h, rh)

        if target_overlap < 1.0 or max_iw <= 0 or max_ih <= 0:
            # No enforced overlap: small jittered placement from P in quadrant direction
            jitter = int(0.25 * max(bw, bh))
            if quad == "UR":
                rx0 = np.clip(px - rw // 4, 0, max(0, W - rw))
                ry0 = np.clip(py - rh // 4, 0, max(0, H - rh))
            elif quad == "UL":
                rx0 = np.clip(px - rw + rw // 4, 0, max(0, W - rw))
                ry0 = np.clip(py - rh // 4, 0, max(0, H - rh))
            elif quad == "LL":
                rx0 = np.clip(px - rw + rw // 4, 0, max(0, W - rw))
                ry0 = np.clip(py - rh + rh // 4, 0, max(0, H - rh))
            else:  # LR
                rx0 = np.clip(px - rw // 4, 0, max(0, W - rw))
                ry0 = np.clip(py - rh + rh // 4, 0, max(0, H - rh))
        else:
            # Choose iw,ih so iw*ih ≈ target_overlap but bounded by max_iw,max_ih
            iw = int(round(np.sqrt(target_overlap)))
            iw = max(1, min(iw, int(max_iw)))
            ih = int(round(target_overlap / iw))
            ih = max(1, min(ih, int(max_ih)))
            iw = max(1, min(int(round(target_overlap / max(ih, 1))), int(max_iw)))

            # Compute the far corner of the overlap patch from P in the quadrant
            if quad == "UR":
                far_x, far_y = px + iw, py + ih  # this will be the rect's top-right
                rx0 = far_x - rw
                ry0 = far_y - rh
            elif quad == "UL":
                far_x, far_y = px - iw, py + ih  # rect's top-left
                rx0 = far_x
                ry0 = far_y - rh
            elif quad == "LL":
                far_x, far_y = px - iw, py - ih  # rect's bottom-left
                rx0 = far_x
                ry0 = far_y
            else:  # LR
                far_x, far_y = px + iw, py - ih  # rect's bottom-right
                rx0 = far_x - rw
                ry0 = far_y

            # Clamp into image
            rx0 = int(np.clip(rx0, 0, max(0, W - rw)))
            ry0 = int(np.clip(ry0, 0, max(0, H - rh)))

            # Safeguard: if clamping increases overlap above target, shrink and nudge
            actual = _overlap_area(rx0, ry0, rw, rh)
            if actual > target_overlap:
                shrink = np.sqrt((target_overlap + 1e-6) / (actual + 1e-6))
                rw = max(1, int(round(rw * shrink)))
                rh = max(1, int(round(rh * shrink)))
                rx0 = int(np.clip(rx0, 0, max(0, W - rw)))
                ry0 = int(np.clip(ry0, 0, max(0, H - rh)))
                # Nudge away from bbox a bit (opposite to quadrant direction)
                for _ in range(4):
                    actual = _overlap_area(rx0, ry0, rw, rh)
                    if actual <= target_overlap:
                        break
                    if quad in ("UR", "LR") and rx0 > 0:
                        rx0 -= 1
                    elif quad in ("UL", "LL") and rx0 < W - rw:
                        rx0 += 1
                    if quad in ("UR", "UL") and ry0 > 0:
                        ry0 -= 1
                    elif quad in ("LR", "LL") and ry0 < H - rh:
                        ry0 += 1

        rx1 = int(np.clip(rx0 + rw, 1, W))
        ry1 = int(np.clip(ry0 + rh, 1, H))

        # Random RGB color (uint8)
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.rectangle(imgs[i], (rx0, ry0), (rx1 - 1, ry1 - 1), color, thickness=-1)

    # Convert back to original layout
    if layout == "NCHW":
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def patch_obj_mtl(
    obj_path: str,
    Ka=(0.2, 0.18, 0.16),
    Kd=(1.0, 0.85, 0.78),
    Ks=(0.0, 0.0, 0.0),
    Ns=1,
    illum=1,
    keep_map_kd: bool = True,
) -> str:
    """
    Edit the MTL referenced by `obj_path`:
      • finds the mtllib + usemtl,
      • preserves map_Kd if requested,
      • overwrites the target material block with Ka/Kd/Ks/Ns/illum.
    Returns the mtl file path (for logging).
    """
    import os, re

    if not os.path.exists(obj_path):
        print(f"[mtl][warn] OBJ not found: {obj_path}")
        return ""

    with open(obj_path, "r") as f:
        obj_txt = f.read()

    # 1) find mtllib and material name used by the mesh
    mtllib_match = re.search(r"^mtllib\s+([^\r\n]+)", obj_txt, flags=re.MULTILINE)
    usemtl_match = re.search(r"^usemtl\s+([^\r\n]+)", obj_txt, flags=re.MULTILINE)
    if not mtllib_match:
        print(f"[mtl][warn] No mtllib in {obj_path}")
        return ""
    mtl_name = usemtl_match.group(1).strip() if usemtl_match else None

    obj_dir = os.path.dirname(obj_path)
    mtllib_name = mtllib_match.group(1).strip()
    mtl_path = os.path.join(obj_dir, mtllib_name)

    # 2) parse existing mtl (if any) to preserve map_Kd
    existing = ""
    if os.path.exists(mtl_path):
        with open(mtl_path, "r") as f:
            existing = f.read()

    # pick a material name if OBJ didn't have one; try first newmtl, else "hand"
    if not mtl_name:
        m = re.search(r"^newmtl\s+([^\r\n]+)", existing, flags=re.MULTILINE)
        mtl_name = m.group(1).strip() if m else "hand"

    # find map_Kd inside the target material block (if present)
    def extract_map_kd(txt: str, target: str):
        pat = re.compile(
            rf"^newmtl\s+{re.escape(target)}\s*$([\s\S]*?)(?=^newmtl\s+|\Z)",
            flags=re.MULTILINE,
        )
        b = pat.search(txt)
        if not b:
            return None
        tb = b.group(1)
        mk = re.search(r"^\s*map_Kd\s+([^\r\n]+)", tb, flags=re.MULTILINE)
        return mk.group(1).strip() if mk else None

    map_kd = extract_map_kd(existing, mtl_name) if keep_map_kd and existing else None

    # 3) build the new block
    def fmt3(v):
        return f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"

    block = []
    block.append(f"newmtl {mtl_name}\n")
    block.append(f"Ka {fmt3(Ka)}\n")
    block.append(f"Kd {fmt3(Kd)}\n")
    block.append(f"Ks {fmt3(Ks)}\n")
    block.append(f"Ns {int(Ns)}\n")
    block.append(f"illum {int(illum)}\n")
    if map_kd:
        block.append(f"map_Kd {map_kd}\n")
    new_block = "".join(block)

    # 4) replace or append in the mtl file
    pat_target = re.compile(
        rf"^newmtl\s+{re.escape(mtl_name)}\s*$([\s\S]*?)(?=^newmtl\s+|\Z)",
        flags=re.MULTILINE,
    )
    if existing and pat_target.search(existing):
        updated = pat_target.sub(new_block, existing)
    else:
        updated = (
            existing
            + ("\n" if existing and not existing.endswith("\n") else "")
            + new_block
        )

    os.makedirs(os.path.dirname(mtl_path), exist_ok=True)
    with open(mtl_path, "w") as f:
        f.write(updated)

    # print(f"[mtl] Patched {mtl_path} (material '{mtl_name}')")
    return mtl_path


# ── Arm texture helper ─────────────────────────────────────────────────────────
import re, glob, os, shutil, zipfile, zlib
import fnmatch
from pathlib import Path

# Cache zip listings so we don't re-scan the zip every sample
_ZIP_TEXTURE_CACHE = {}


def _parse_mtl_sections(mtl_text: str):
    secs, name, buf = {}, None, []
    for ln in mtl_text.splitlines():
        if ln.strip().startswith("newmtl "):
            if name is not None:
                secs[name] = "\n".join(buf).strip() + "\n"
            name = ln.strip().split(None, 1)[1]
            buf = [ln]
        else:
            buf.append(ln)
    if name is not None:
        secs[name] = "\n".join(buf).strip() + "\n"
    return secs


def _ensure_map_kd(sec_text: str, png_rel_path: str):
    png_rel_path = png_rel_path.replace("\\", "/")  # OBJ/MTL prefer forward slashes
    lines = sec_text.strip().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("map_kd"):
            lines[i] = f"map_Kd {png_rel_path}"
            return "\n".join(lines) + "\n"
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("kd "):
            lines.insert(i + 1, f"map_Kd {png_rel_path}")
            return "\n".join(lines) + "\n"
    lines.append(f"map_Kd {png_rel_path}")
    return "\n".join(lines) + "\n"


def _pick_random_texture(
    textures_dir: str, patterns=("arm_skin_v2_*.png", "arm_skin_*.png")
) -> Path:
    all_pngs = []
    for pat in patterns:
        all_pngs.extend(glob.glob(os.path.join(textures_dir, pat)))
    if not all_pngs:
        raise FileNotFoundError(f"No PNGs in {textures_dir} matching {patterns}")
    import random

    return Path(random.choice(all_pngs))


def _list_zip_pngs(zip_path: str, patterns=("*",)) -> list[str]:
    zp = os.path.abspath(zip_path)
    key = (zp, tuple(patterns))
    if key in _ZIP_TEXTURE_CACHE:
        return _ZIP_TEXTURE_CACHE[key]
    with zipfile.ZipFile(zp, "r") as zf:
        names = []
        for n in zf.namelist():
            if n.endswith("/") or (not n.lower().endswith(".png")):
                continue
            base = os.path.basename(n)
            if patterns and not any(
                fnmatch.fnmatch(base, p) or fnmatch.fnmatch(n, p) for p in patterns
            ):
                continue
            names.append(n)
    if not names:
        raise FileNotFoundError(f"No PNGs in zip {zp} matching {patterns}")
    _ZIP_TEXTURE_CACHE[key] = names
    return names


def _pick_random_texture_from_zip(zip_path: str, patterns=("*",)) -> tuple[str, str]:
    import random

    names = _list_zip_pngs(zip_path, patterns=patterns)
    return os.path.abspath(zip_path), random.choice(names)


def _extract_png_from_zip(zip_path: str, inner_name: str, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(inner_name).stem
    suf = Path(inner_name).suffix
    crc = zlib.crc32(inner_name.encode("utf-8")) & 0xFFFFFFFF
    out_name = f"{stem}_{crc:08x}{suf}"
    out_path = dst_dir / out_name
    if out_path.exists():
        return out_path
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_name, "r") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def assign_random_arm_texture(
    obj_path: str,
    textures_dir: str = None,
    textures_zip: str = None,
    copy_to_dir=None,
    patterns=None,
) -> str:
    """
    Pick a PNG from textures_dir OR textures_zip and ensure the BODY MTL uses it as map_Kd.
    If copy_to_dir is provided, copy the PNG into that directory (used by render temp),
    and reference it by filename only so the renderer resolves it locally.
    Returns absolute path of the PNG actually used.
    """
    obj_p = Path(obj_path)

    if textures_zip:
        print(f"[armtex] Picking random texture from zip {textures_zip}")
        if patterns is None:
            patterns = ("*.png",)
        zip_path, inner = _pick_random_texture_from_zip(
            str(textures_zip), patterns=patterns
        )
        dst_root = Path(copy_to_dir) if copy_to_dir is not None else obj_p.parent
        png_dst = _extract_png_from_zip(zip_path, inner, dst_root)
        if copy_to_dir is not None:
            map_kd_value = png_dst.name
        else:
            rel = os.path.relpath(png_dst, start=obj_p.parent)
            map_kd_value = rel.replace("\\", "/")
        used_png_abs = str(png_dst)
    else:
        if textures_dir is None:
            raise ValueError(
                "assign_random_arm_texture: provide textures_zip or textures_dir"
            )
        if patterns is None:
            patterns = ("arm_skin_v2_*.png", "arm_skin_*.png")
        png_src = _pick_random_texture(str(textures_dir), patterns=patterns)
        png_name = png_src.name
        if copy_to_dir is not None:
            dst_dir = Path(copy_to_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            png_dst = dst_dir / png_name
            if not png_dst.exists():
                shutil.copy2(png_src, png_dst)
            map_kd_value = png_dst.name
            used_png_abs = str(png_dst)
        else:
            rel = os.path.relpath(png_src, start=obj_p.parent)
            map_kd_value = rel.replace("\\", "/")
            used_png_abs = str(png_src)

    obj_txt = obj_p.read_text(encoding="utf-8", errors="ignore")
    mtllib_m = re.search(r"^\s*mtllib\s+([^\r\n]+)", obj_txt, flags=re.MULTILINE)
    if mtllib_m:
        mtl_name = mtllib_m.group(1).strip()
        mtl_p = obj_p.parent / mtl_name
        mtl_txt = (
            mtl_p.read_text(encoding="utf-8", errors="ignore") if mtl_p.exists() else ""
        )
    else:
        mtl_name = obj_p.stem + "_armtex.mtl"
        mtl_p = obj_p.parent / mtl_name
        mtl_txt = ""

    usemtls = re.findall(r"^\s*usemtl\s+([^\s]+)", obj_txt, flags=re.MULTILINE) or [
        "material0"
    ]

    secs = _parse_mtl_sections(mtl_txt) if mtl_txt.strip() else {}
    if not secs:
        for m in set(usemtls):
            base = f"""newmtl {m}
Ka 0.20 0.18 0.16
Kd 0.95 0.80 0.75
Ks 0.00 0.00 0.00
Ns 1
illum 1
"""
            secs[m] = _ensure_map_kd(base, map_kd_value)
    else:
        any_updated = False
        for m in set(usemtls):
            if m in secs:
                secs[m] = _ensure_map_kd(secs[m], map_kd_value)
                any_updated = True
        if not any_updated:
            for k in list(secs.keys()):
                secs[k] = _ensure_map_kd(secs[k], map_kd_value)

    mtl_p.write_text("\n\n".join(secs[k] for k in secs), encoding="utf-8")

    if not mtllib_m:
        obj_txt = f"mtllib {mtl_name}\n" + obj_txt
        obj_p.write_text(obj_txt, encoding="utf-8")

    return used_png_abs
