import hydra
from omegaconf import DictConfig
import numpy as np
import os

import trimesh


from convert_h5_to_tar import (
    bbox_to_center_scale,
    float16_to_uint16,
    project_joints_to_image,
)

from smplx_helper import (
    export_body_arm_aligned_to_mano_seam,
    _get_hand_uv_template,
    _load_smpl_uv_template,
    _sample_texture_colors_at_uvs,
)
from sync_data_utils import (
    alpha_mask_from_bg,
    is_skin_like_rgb_samples,
    lab_shift,
    resolve_obj_map_kd,
    skin_mask_ycrcb_rgb,
)


def generate_texture(cfg):
    """
    Generate textures using the provided sample_uv_batch function, splitting the output into
    multiple NPZ files if necessary. Each NPZ file will contain at most `max_npz_batch_size`
    textures.

    Note: This function assumes that sample_uv_batch is defined in sample_uv.py and has been
    imported properly.

    Docker run example:
      docker run --gpus all -it --rm -v ${PWD}:/workspace/handy -w /workspace/handy handy bash
      python create_data.py generate_texture.execute=true

    Args:
      cfg: A configuration object (e.g., DictConfig) with at least:
             generate_texture.n_textures (total textures),
             generate_texture.max_batch_size (GPU generation batch size),
             generate_texture.max_npz_batch_size (max textures saved per npz file),
             generate_texture.save_dir (save folder),
             generate_texture.prefix (NPZ filename prefix).
    """
    # Import the function for batch texture generation
    from sample_uv import sample_uv_batch
    from tqdm import tqdm

    N = cfg.generate_texture.n_textures
    MAX_SEED = 2**16 - 1
    max_gpu_batch_size = cfg.generate_texture.max_batch_size  # GPU batch size
    max_npz_batch_size = (
        cfg.generate_texture.max_npz_batch_size
    )  # e.g., 1024 per npz file
    save_dir = cfg.generate_texture.save_path
    prefix = cfg.generate_texture.file_prefix

    # Create the output directory if it doesn't exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    current_chunk = []  # Accumulate images for one NPZ file
    file_counter = 0  # To number the files
    total_generated = 0  # Global counter

    for start in tqdm(
        range(0, N, max_gpu_batch_size), desc="Generating textures", leave=False
    ):
        current_gpu_batch_size = min(max_gpu_batch_size, N - start)
        texture_seed_batch = np.random.randint(1, MAX_SEED, size=current_gpu_batch_size)
        batch_textures = sample_uv_batch(
            texture_seed_batch
        )  # Shape: (batch_size, H, W, 3)

        # Append each texture from the current batch to the chunk list.
        for texture in batch_textures:
            current_chunk.append(texture)
            total_generated += 1
            # If we reached the maximum number of images per npy file, save and reset.
            if len(current_chunk) >= max_npz_batch_size:
                file_path = os.path.join(save_dir, f"{prefix}_{file_counter}.npy")
                np.save(file_path, np.array(current_chunk))
                print(f"Saved {len(current_chunk)} textures to {file_path}")
                file_counter += 1
                current_chunk = []  # Reset for the next npy file.

    # Save any remaining textures in the last file.
    if current_chunk:
        file_path = os.path.join(save_dir, f"{prefix}_{file_counter}.npy")
        np.save(file_path, np.array(current_chunk))
        print(f"Saved {len(current_chunk)} textures to {file_path}")

    print(f"Total textures generated: {total_generated}")


def load_texture(cfg, image_index, visualize=False):
    """
    Retrieve a single texture image from the saved NPZ files given its global index.

    The files are expected to be saved with names:
        {save_dir}/{prefix}_0.npz, {save_dir}/{prefix}_1.npz, ...
    with each file containing at most `max_npz_batch_size` images.

    Args:
      image_index (int): Global index of the texture (e.g., 0 to N-1)
      cfg: A configuration object containing:
             generate_texture.save_dir (folder path),
             generate_texture.prefix (file prefix),
             generate_texture.max_npz_batch_size (max images per NPZ file).
      visualize: Visualize if True.

    Returns:
      A NumPy array representing the texture image (e.g., shape (1024, 1024, 3)).
    """
    ROOT_DIR = cfg.generate_mesh.root_dir
    save_dir = ROOT_DIR + "/" + cfg.generate_texture.save_path
    prefix = cfg.generate_texture.file_prefix
    max_npz_batch_size = cfg.generate_texture.max_npz_batch_size

    # Determine which file should contain the image.
    file_index = image_index // max_npz_batch_size
    index_in_file = image_index % max_npz_batch_size
    file_path = os.path.join(save_dir, f"{prefix}_{file_index}.npy")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use memory mapping to load the file, avoiding full file read into memory.
    textures = np.load(file_path, mmap_mode="r")

    # Check that the index is within bounds for the current file.
    if index_in_file < 0 or index_in_file >= textures.shape[0]:
        raise IndexError(
            f"Image index {image_index} is out-of-bounds in file {file_path}"
        )

    texture = textures[index_in_file]

    if visualize:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(texture)
        plt.axis("off")
        plt.title(f"Texture {image_index}")
        plt.show()

    return texture


def generate_pose_batch(
    root_dir: str,
    npz_path: str,
    N: int,
    extrap_gamma: float = 0.1,  # alpha ~ U(-gamma, +gamma)
    mix_noise_frac: float = 0.01,  # added noise = mix_noise_frac * per-dim std
    manolayer=None,
    bmc_threshold=None,
):
    """
    Returns:
        pose_batch: torch.FloatTensor of shape (N, 48)
        beta_batch: torch.FloatTensor of shape (N, 10)
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from bmc_loss.bmc_loss import BMCLoss, BMCLossConfig

    # Load data — support both freihand format (key "poses") and wilor format (key "hand_pose")
    data = np.load(npz_path)
    pose_key = "poses" if "poses" in data else "hand_pose"
    poses_45 = data[pose_key][:, 3:].astype(
        np.float32, copy=False
    )  # drop global (root) rotation
    betas_all = data["betas"].astype(np.float32, copy=False)
    print(
        f"Loaded {npz_path.split('/')[-1]}: poses={poses_45.shape}, betas={betas_all.shape}"
    )

    M, D_pose = poses_45.shape
    M_beta, D_beta = betas_all.shape
    assert D_pose == 45 and D_beta == 10

    # Std with floor
    pose_std = np.maximum(poses_45.std(axis=0, ddof=1), 1e-6).astype(np.float32)
    beta_std = np.maximum(betas_all.std(axis=0, ddof=1), 1e-6).astype(np.float32)

    # Init BMC
    use_bmc = (manolayer is not None) and (bmc_threshold is not None)
    if use_bmc:
        bmc_cfg = BMCLossConfig()
        bmc_cfg.path = root_dir + "/" + "data/BMC"
        bmc_cfg.lambda_bl = 1.0
        bmc_cfg.lambda_rb = 0.0
        bmc_cfg.lambda_a = 1.0
        bmc_cfg.scale = 1.0
        bmc = BMCLoss(bmc_cfg)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        @torch.no_grad()
        def bmc_ok(pose48_1xD: torch.Tensor) -> bool:
            pose48_1xD = pose48_1xD.to("cpu")
            out = manolayer.forward(pose48_1xD, torch.zeros(1, D_beta))
            joints = out.joints.to(device)
            loss = bmc.compute_loss(joints=joints)["bmc_loss"]
            return float(loss) <= float(bmc_threshold)

    rng = np.random.default_rng()
    poses_out, betas_out = [], []

    def sample_mixup_extrap() -> tuple[np.ndarray, np.ndarray]:
        i, j = rng.integers(0, M, size=2)
        A_pose, B_pose = poses_45[i], poses_45[j]
        # sample betas independently (pools may differ in size)
        bi, bj = rng.integers(0, M_beta, size=2)
        A_beta, B_beta = betas_all[bi], betas_all[bj]

        alpha = rng.uniform(-extrap_gamma, +extrap_gamma)

        pose = A_pose + alpha * (B_pose - A_pose)
        beta = A_beta + alpha * (B_beta - A_beta)

        pose += rng.normal(size=D_pose).astype(np.float32) * (mix_noise_frac * pose_std)
        beta += rng.normal(size=D_beta).astype(np.float32) * (mix_noise_frac * beta_std)

        return pose, beta

    # Generate
    for _ in tqdm(range(N), desc=f"Generating poses [mixup_extrap]", leave=False):
        attempt = 0
        while True:
            attempt += 1
            pose45, beta = sample_mixup_extrap()
            pose48 = np.concatenate([np.zeros(3, dtype=np.float32), pose45], axis=0)

            if not use_bmc:
                poses_out.append(torch.from_numpy(pose48))
                betas_out.append(torch.from_numpy(beta))
                break
            else:
                t = torch.from_numpy(pose48).unsqueeze(0).to(device)
                if bmc_ok(t) or attempt >= 10_000:
                    poses_out.append(t.squeeze(0).detach().cpu())
                    betas_out.append(torch.from_numpy(beta))
                    break

    return torch.stack(poses_out, dim=0), torch.stack(betas_out, dim=0)


def generate_data(cfg: DictConfig):
    from tqdm import tqdm
    import h5py
    import torch
    from manotorch.manolayer import ManoLayer
    from handylayer import HandyLayer
    from PIL import Image, ImageDraw
    import random
    import json
    import joblib

    # ------------------------------------------------------------------ #
    # SINGLE-BATCH MODE:
    # If you launch this script from a wrapper (see §3) we *want*
    # exactly one batch per process.  When num_generate > batch_size
    # we silently clamp it so the function exits after the first 800.
    # ------------------------------------------------------------------ #
    if cfg.generate_mesh.num_generate > cfg.generate_mesh.batch_size:
        cfg.generate_mesh.num_generate = cfg.generate_mesh.batch_size

    tqdm.monitor_interval = 0

    from sync_data_utils import (
        run_simulation_sapien,
        get_random_background,
        render_envmap_to_ldr,
        shift_hue,
        apply_zip_faces_to_body,
        augment_skin_tone_random,
        occlude_bbox_with_colored_rect_batch,
        patch_obj_mtl,
        assign_random_arm_texture,
    )
    from moge_utils import (
        moge_depth_from_rgb,
        adjust_bg_depth_layering,
        compose_hand_bg_depth,
        warmup_moge,
    )

    use_moge = getattr(cfg.simulate, "moge_bg_depth", None) and getattr(
        cfg.simulate.moge_bg_depth, "flag", False
    )
    if use_moge:
        warmup_moge(model_name=cfg.simulate.moge_bg_depth.model_name)

    # ---- Lazy singletons so we only load the big model once ----

    N = cfg.generate_mesh.num_generate
    M = cfg.generate_mesh.batch_size
    ROOT_DIR = cfg.generate_mesh.root_dir

    save_dir = cfg.generate_mesh.mesh_save_dir
    temp_dir = cfg.generate_mesh.mesh_temp_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    if cfg.generate_mesh.save_images_and_meta:
        img_dir = os.path.join(save_dir, "images")
        meta_dir = os.path.join(save_dir, "meta")
        mask_dir = os.path.join(save_dir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

    # Initialize MANO params
    beta_dim = cfg.generate_mesh.mano_betas_size
    beta_std = cfg.generate_mesh.beta_std
    pose_dim = cfg.generate_mesh.mano_poses_size
    pose_std = cfg.generate_mesh.pose_std

    # n_vertices = cfg.generate_mesh.n_vertices
    # n_joints = cfg.generate_mesh.n_joints
    RGB_width = cfg.generate_mesh.RGB_width
    RGB_height = cfg.generate_mesh.RGB_height

    n_images = cfg.simulate.n_images
    n_textures = cfg.generate_texture.n_textures

    file_prefix = cfg.generate_mesh.file_prefix

    background_dir = ROOT_DIR + "/" + cfg.simulate.background.background_dir
    env_map_dir = ROOT_DIR + "/" + cfg.simulate.background.env_map_dir

    mano_layer = ManoLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=pose_dim,
        mano_assets_root=ROOT_DIR + "/" + "manotorch/assets/mano",
    )
    handylayer = HandyLayer(
        use_pca=False,
        flat_hand_mean=True,
        ncomps=pose_dim,
        shape_ncomps=beta_dim,
        handy_assets_root=ROOT_DIR + "/" + "manotorch/assets/handy",
        mano_assets_root=ROOT_DIR + "/" + "manotorch/assets/mano",
    )
    # mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False, ncomps=pose_dim, mano_assets_root="./manotorch/assets/mano")
    # handylayer = HandyLayer(use_pca=False, flat_hand_mean=False, ncomps=pose_dim, shape_ncomps=beta_dim, handy_assets_root="./manotorch/assets/handy", mano_assets_root="./manotorch/assets/mano")

    n_batches = int(np.ceil(N / M))

    for batch_id in range(n_batches):
        start_idx = batch_id * M
        batch_size = min(M, N - start_idx)

        print(
            f"\n=== Batch {batch_id + 1}/{n_batches} "
            f"({start_idx} ... {start_idx + batch_size - 1}) ==="
        )

        # --------------------------------------------------------------
        # 2 · allocate **batch‑local** buffers
        # --------------------------------------------------------------

        beta_data = torch.rand(batch_size, beta_dim) * beta_std

        if cfg.generate_mesh.use_dposerx_sample_pose:
            import subprocess

            gen_pose_path = os.path.join(ROOT_DIR, "gen_poses.npy")
            # cmd = f"python -m {ROOT_DIR}.DposerX.sample_poses --config {ROOT_DIR}/DposerX/configs/hand/subvp/timefc.py --sample_num {batch_size}"
            cmd = (
                f"cd {ROOT_DIR} && "
                f"python -m DposerX.sample_poses "
                f"--bodymodel-path manotorch/assets/mano/models "
                f"--ckpt-path DposerX/pretrained_models/hand/BaseMLP/last.ckpt "
                f"--root_dir . "
                f"--output-path . "
                f"--sample_num {batch_size} "
                f"--device cuda"
                # f"--config DposerX/configs/hand/subvp/timefc.py "
            )
            print(f"[DposerX] Generating poses with command:\n{cmd}")

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    capture_output=True,  # <-- Capture stdout and stderr
                    text=True,  # <-- Decode output as string
                )
                print(result.stdout)  # Print standard output
            except subprocess.CalledProcessError as e:
                print("[DposerX] Pose sampling failed!")
                print(f"Exit code: {e.returncode}")
                print("--- STDOUT ---")
                print(e.stdout)
                print("--- STDERR ---")
                print(e.stderr)
                raise

            pose_45 = np.load(gen_pose_path)  # shape: (batch_size, 45)
            assert pose_45.shape == (batch_size, 45), (
                f"[Error]: shape is {pose_45.shape}"
            )

            # Also get FH-style extrap betas
            fh_pose_data, beta_data = generate_pose_batch(
                root_dir=ROOT_DIR,
                npz_path=ROOT_DIR + "/" + cfg.generate_mesh.real_pose_npz_path,
                N=batch_size,
                extrap_gamma=cfg.generate_mesh.real_pose_extrap_gamma,
                mix_noise_frac=cfg.generate_mesh.real_pose_mix_noise_frac,
                manolayer=None,
                bmc_threshold=None,
            )

            pose_data = torch.cat(
                [fh_pose_data[:, :3], torch.tensor(pose_45)], dim=1
            )  # shape: (N, 48)

        elif cfg.generate_mesh.use_real_pose:
            pose_data, beta_data = generate_pose_batch(
                root_dir=ROOT_DIR,
                npz_path=ROOT_DIR + "/" + cfg.generate_mesh.real_pose_npz_path,
                N=batch_size,
                extrap_gamma=cfg.generate_mesh.real_pose_extrap_gamma,
                mix_noise_frac=cfg.generate_mesh.real_pose_mix_noise_frac,
                manolayer=mano_layer,
                bmc_threshold=cfg.generate_mesh.real_pose_bmc_threshold,
            )
        elif cfg.generate_mesh.gmm_path != "":
            gmm = joblib.load(ROOT_DIR + "/" + cfg.generate_mesh.gmm_path)
            pose_data = generate_pose_batch(
                gmm=gmm,
                means=gmm.means_,
                N=batch_size,
                manolayer=mano_layer,
                scale=cfg.generate_mesh.gmm_scale,
                bmc_threshold=cfg.generate_mesh.bmc_threshold,
            )

        else:
            pose_data = torch.rand(batch_size, pose_dim) * pose_std
            pose_data = torch.cat([torch.zeros(batch_size, 3), pose_data], dim=1)

        tex_idx_data = np.random.choice(n_textures, batch_size, replace=True).astype(
            np.int32
        )

        image_rgba_data = np.zeros(
            (batch_size, n_images, RGB_width, RGB_height, 3), dtype=np.uint8
        )
        depth_data = np.zeros(
            (batch_size, n_images, RGB_width, RGB_height), dtype=np.float16
        )
        camera_poses_data = np.zeros((batch_size, n_images, 4, 4), dtype=np.float16)
        bbox_data = np.zeros((batch_size, n_images, 4, 2), dtype=np.float16)
        fov_rads_data = np.zeros((batch_size, n_images), dtype=np.float16)
        mask_data = np.zeros(
            (batch_size, n_images, RGB_width, RGB_height), dtype=np.float16
        )

        # --------------------------------------------------------------
        # 3 · forward pass once per batch
        # --------------------------------------------------------------
        with torch.no_grad():
            mano_output = mano_layer(pose_data, beta_data)
            if cfg.generate_mesh.use_handy_vertices:
                computed_vertices = handylayer.mano_to_handy(mano_output.verts)
            else:
                computed_vertices = mano_output.verts
            faces = mano_layer.th_faces.numpy()
            mano_joints = mano_output.joints
            if getattr(cfg.generate_mesh, "save_verts", False):
                verts_data = mano_output.verts.cpu().numpy()  # (batch_size, 778, 3)

        if cfg.simulate.background.env_map_dir != "":
            with open(os.path.join(env_map_dir, "configs.json")) as file:
                env_map_data = json.load(file)
                env_map_list = env_map_data["env_maps"]

            env_map_selected = random.choices(env_map_list, k=batch_size)

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

        for i in tqdm(range(batch_size), desc="Generatimg Scene", leave=False):
            tex_idx = tex_idx_data[i]

            # ---- build a textured mesh file on disk ----
            # Always keep the original RGB texture available (used later for hand↔arm sync)
            orig_texture = load_texture(cfg, tex_idx)  # RGB uint8

            if cfg.generate_mesh.texture_color_shift.flag:
                # augment_skin_tone_random is OpenCV-based (expects BGR). Convert RGB→BGR→RGB.
                aug_bgr = augment_skin_tone_random(
                    orig_texture[..., ::-1],
                    p=cfg.generate_mesh.texture_color_shift.percent,
                )
                aug_texture = aug_bgr[..., ::-1]
                texture_image = Image.fromarray(aug_texture)
            else:
                texture_image = Image.fromarray(orig_texture)

            hand_obj_path = os.path.join(temp_dir, f"{i}_hand.obj")
            handy_verts_i = computed_vertices[i].detach().cpu().numpy()  # Handy topo
            _ = handylayer.to_textured_obj(
                handy_verts=computed_vertices[i].detach().cpu().numpy(),
                texture_img=texture_image,
                save_path=hand_obj_path,
            )

            if (
                getattr(cfg.generate_mesh, "override_hand_mtl", None)
                and cfg.generate_mesh.override_hand_mtl.flag
            ):
                m = cfg.generate_mesh.override_hand_mtl
                patch_obj_mtl(
                    obj_path=hand_obj_path,
                    Ka=tuple(m.Ka),
                    Kd=tuple(m.Kd),
                    Ks=tuple(m.Ks),
                    Ns=m.Ns,
                    illum=m.illum,
                    keep_map_kd=getattr(m, "keep_map_kd", True),
                )

            # Optionally fuse SMPL-H body + higher-res Handy right hand (no alignment)
            if getattr(cfg.generate_mesh.add_arm, "flag", False):
                # === BODY+ARM (no hand) aligned to MANO seam center (function still needs mano verts) ===
                body_obj_path = os.path.join(temp_dir, f"{i}_body.obj")
                mano_raw_verts_i = mano_output.verts[i].detach().cpu().numpy()

                hand_origin = handy_verts_i.mean(axis=0).astype(
                    np.float32
                )  # NEW: hand mesh centroid

                # Write BODY, but ask the exporter to recenter final coordinates to 'hand_origin'
                uv_template_obj = getattr(
                    cfg.generate_mesh.add_arm, "uv_template_obj", None
                )
                if uv_template_obj and not os.path.isabs(uv_template_obj):
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
                        hand_beta=beta_data[i].detach().cpu().numpy(),
                        hand_pose=pose_data[i].detach().cpu().numpy(),
                        smplh_pkl_path=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.smplh_pkl_path,
                        mano_models_dir=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.mano_models_dir,
                        smpl2mano_right_path=ROOT_DIR
                        + "/"
                        + cfg.generate_mesh.add_arm.smpl2mano_right_path,
                        mano_vertices=mano_raw_verts_i,
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
                        handy_ref_verts=handy_verts_i,
                        remove_mask_pkl=ROOT_DIR + "/" + "keep_right_arm.pkl",
                        prefer_pkl=True,
                        uv_template_obj=uv_template_obj,
                        return_wrist_info=True,
                    )
                )

                # If a zip-face mapping exists, graft those faces onto BODY.
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

                # If enabled, stamp a realistic arm texture onto BODY MTL
                # Stamp a realistic arm texture; also copy the PNG into the temp dir so the renderer can see it.
                rar = getattr(cfg.generate_mesh.add_arm, "realistic_arm_texture", None)
                if rar and getattr(rar, "flag", False):
                    # tex_dir = rar.textures_dir
                    # if not os.path.isabs(tex_dir):
                    #     tex_dir = os.path.join(ROOT_DIR, tex_dir)

                    tex_dir = getattr(rar, "textures_dir", None)
                    tex_zip = getattr(rar, "textures_zip", None)

                    if tex_dir:
                        if not os.path.isabs(tex_dir):
                            tex_dir = os.path.join(ROOT_DIR, tex_dir)
                    if tex_zip:
                        if not os.path.isabs(tex_zip):
                            tex_zip = os.path.join(ROOT_DIR, tex_zip)

                    # print("tex_dir", tex_dir)
                    # print("tex_zip", tex_zip)

                    # Copy extracted/cached texture PNG next to the mesh OBJ/MTL that SAPIEN loads.
                    # IMPORTANT: mesh_temp_dir is a per-job local scratch path; do NOT prefix with ROOT_DIR.

                    copy_dir = None
                    try:
                        copy_dir = getattr(cfg.generate_mesh, "mesh_temp_dir", None)
                    except Exception:
                        copy_dir = None

                    try:
                        used_png = assign_random_arm_texture(
                            obj_path=body_obj_path,
                            textures_dir=tex_dir,
                            textures_zip=tex_zip,
                            copy_to_dir=copy_dir,
                        )
                        # print(f"[armtex] BODY map_Kd ← {used_png}")

                        # --- NEW: short-sleeve heuristic + skin-tone synchronization ---
                        try:
                            do_sync = True
                            sync_cfg = getattr(rar, "sync_hand_arm_skin", None)
                            if sync_cfg is not None:
                                do_sync = bool(getattr(sync_cfg, "flag", False))

                            if do_sync and used_png:
                                import cv2

                                hand_map_kd = resolve_obj_map_kd(hand_obj_path)
                                if (
                                    hand_map_kd
                                    and os.path.exists(hand_map_kd)
                                    and os.path.exists(used_png)
                                ):
                                    # base hand texture: ORIGINAL, so we align first then mutate jointly
                                    hand_rgb_base = np.asarray(
                                        orig_texture, dtype=np.uint8
                                    )
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
                                            idx = rng_local.choice(
                                                arm_skin.shape[0],
                                                size=max_samples,
                                                replace=False,
                                            )
                                            arm_samples_u8 = arm_skin[idx]
                                        else:
                                            arm_samples_u8 = arm_skin

                                    if short_sleeve and (arm_samples_u8 is not None):
                                        # masks: skin only (and avoid hand texture background)
                                        hand_bgr = hand_rgb_base[..., ::-1].copy()
                                        mask_hand = cv2.bitwise_and(
                                            alpha_mask_from_bg(hand_bgr),
                                            skin_mask_ycrcb_rgb(hand_rgb_base),
                                        )

                                        # align hand seam mean -> arm seam mean (LAB)
                                        hand_wrist_lab = (
                                            cv2.cvtColor(
                                                hand_wrist_u8.reshape(1, -1, 3)[
                                                    ..., ::-1
                                                ],
                                                cv2.COLOR_BGR2LAB,
                                            )
                                            .reshape(-1, 3)
                                            .astype(np.float32)
                                        )
                                        arm_wrist_lab = (
                                            cv2.cvtColor(
                                                arm_samples_u8.reshape(1, -1, 3)[
                                                    ..., ::-1
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
                                        if cfg.generate_mesh.texture_color_shift.flag:
                                            hand_bgr, params = augment_skin_tone_random(
                                                hand_bgr,
                                                p=cfg.generate_mesh.texture_color_shift.percent,
                                                return_params=True,
                                                mask=mask_hand,
                                            )
                                            if params is not None:
                                                arm_bgr = arm_rgb[..., ::-1].copy()
                                                # mask_arm = skin_mask_ycrcb_rgb(arm_rgb)
                                                arm_bgr = augment_skin_tone_random(
                                                    arm_bgr,
                                                    p=1.0,
                                                    params=params,
                                                    mask=mask_arm,
                                                )
                                                Image.fromarray(
                                                    arm_bgr[..., ::-1]
                                                ).save(used_png)

                                        # overwrite the hand texture referenced by the hand OBJ
                                        Image.fromarray(hand_bgr[..., ::-1]).save(
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

                # if getattr(cfg.generate_mesh, "override_hand_mtl", None) and cfg.generate_mesh.override_hand_mtl.flag:
                #     m = cfg.generate_mesh.override_hand_mtl
                #     patch_obj_mtl(
                #         obj_path=body_obj_path,          # ← point at the BODY OBJ
                #         Ka=tuple(m.Ka),
                #         Kd=tuple(m.Kd),
                #         Ks=tuple(m.Ks),
                #         Ns=m.Ns,
                #         illum=m.illum,
                #         keep_map_kd=getattr(m, "keep_map_kd", True),
                #     )

            bg_img_array = None
            if BG_FLAG == "env_map":
                env_map_path = (
                    env_map_selected[i].replace("./", ROOT_DIR + "/")
                    if env_map_selected[i].startswith("./")
                    else os.path.join(env_map_dir, env_map_selected[i])
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

            use_moge = getattr(cfg.simulate, "moge_bg_depth", None) and getattr(
                cfg.simulate.moge_bg_depth, "flag", False
            )
            bg_depth_scene = None
            if use_moge and (bg_img_array is not None):
                try:
                    # MoGe on the exact background patch we pass to SAPIEN
                    bg_depth_scene = moge_depth_from_rgb(
                        bg_img_array, model_name=cfg.simulate.moge_bg_depth.model_name
                    )
                    # (Optionally cache per bg image if you reuse backgrounds across scenes.)
                except Exception as e:
                    print(
                        f"[MoGe][warn] BG depth failed for scene {i}: {type(e).__name__}: {e}"
                    )
                    bg_depth_scene = None

            # ---- call SAPIEN simulation ----
            if getattr(cfg.generate_mesh.add_arm, "flag", False):
                # Provide BOTH meshes to the renderer: hand + body
                (
                    rgba_images,
                    depth_images,
                    camera_poses_all,
                    fov_rads_all,
                    bbox_all,
                    mask_all,
                    union_mask_all,
                    bg_crop_xy_all,
                ) = run_simulation_sapien(
                    mesh_file=hand_obj_path,  # e.g., f"{i}_hand.obj"
                    mesh_file_body=body_obj_path,  # e.g., f"{i}_body.obj"
                    max_lights=cfg.simulate.max_lights,
                    n_images=cfg.simulate.n_images,
                    cam_dist_mean=cfg.simulate.cam_dist_mean,
                    cam_dist_std=cfg.simulate.cam_dist_std,
                    cam_loc_dist=cfg.simulate.cam_loc_std,
                    intrinsics=cfg.simulate.intrinsics,
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
            else:
                (
                    rgba_images,
                    depth_images,
                    camera_poses_all,
                    fov_rads_all,
                    bbox_all,
                    mask_all,
                    union_mask_all,
                    bg_crop_xy_all,
                ) = run_simulation_sapien(
                    mesh_file=hand_obj_path,
                    mesh_file_body=None,
                    max_lights=cfg.simulate.max_lights,
                    n_images=cfg.simulate.n_images,
                    cam_dist_mean=cfg.simulate.cam_dist_mean,
                    cam_dist_std=cfg.simulate.cam_dist_std,
                    cam_loc_dist=cfg.simulate.cam_loc_std,
                    intrinsics=cfg.simulate.intrinsics,
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

            # ---- (B) per-image composite with patch-aligned BG depth ----
            if use_moge and (bg_depth_scene is not None):
                H, W = depth_images.shape[1], depth_images.shape[2]
                depth_full = np.zeros_like(depth_images, dtype=np.float32)
                for j in range(depth_images.shape[0]):
                    hand = depth_images[j].astype(np.float32) / 1000.0  # mm → m
                    m01 = (union_mask_all[j] > 0.5).astype(np.uint8)  # hand ∪ arm

                    # crop the MoGe BG depth at the same (y0, x0) used for RGB compositing
                    cy, cx = map(int, bg_crop_xy_all[j])
                    # safety clamp (shouldn't trigger, but avoids OOB if BG is tight)
                    cy = max(0, min(cy, bg_depth_scene.shape[0] - H))
                    cx = max(0, min(cx, bg_depth_scene.shape[1] - W))
                    bg_crop = bg_depth_scene[cy : cy + H, cx : cx + W]

                    # optional boundary align/de-align to control the hand–BG gap
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

                # store composites (float16 in H5)
                depth_data[i] = depth_full.astype(np.float16)

            else:
                # fallback: original SAPIEN depth
                depth_data[i] = depth_images

            if getattr(cfg.simulate.moge_bg_depth, "keep_hand_depth", False):
                # keep the original SAPIEN hand-only depth as a new dataset later (see §4)
                depth_hand_only = depth_images.astype(np.float32)

            # --- optional post-generation occlusion augmentation ---
            if getattr(cfg.generate_mesh.occlude_hand, "flag", False):
                rgba_images = occlude_bbox_with_colored_rect_batch(
                    images=rgba_images,
                    bboxes=bbox_all,
                    p=cfg.generate_mesh.occlude_hand.percent,
                    size_scale_range=cfg.generate_mesh.occlude_hand.size_scale_range,
                    max_cover_frac=cfg.generate_mesh.occlude_hand.max_cover_frac,
                )

            # ---- stash into batch arrays ----
            try:
                image_rgba_data[i] = rgba_images
            except ValueError:
                image_rgba_data[i] = rgba_images[..., :3]  # drop alpha if present
            # depth_data[i]       = depth_images
            camera_poses_data[i] = camera_poses_all
            fov_rads_data[i] = fov_rads_all
            bbox_data[i] = bbox_all
            mask_data[i] = union_mask_all

            # if i == 0:
            #     cam_pose_0 = camera_poses_all[0]
            #     image_0 = rgba_images[0]
            #     print("image_0 size:", rgba_images[0].shape)
            #     from convert_h5_to_tar import project_joints_to_image
            #     import cv2
            #     keypoints_2d = project_joints_to_image(
            #         joints_3d=mano_output.joints[i].detach().cpu().numpy(),
            #         cam_pose=cam_pose_0.astype(np.float32),
            #         fov_rad=fov_rads_all[0],
            #         width=cfg.simulate.intrinsics.width,
            #         height=cfg.simulate.intrinsics.height,
            #     )
            #     KP_COLOR = (0, 255, 0)
            #     for i, (x, y) in enumerate(keypoints_2d.astype(int)):
            #         cv2.circle(image_0, (x, y), 3, KP_COLOR, -1)
            #         cv2.putText(image_0, str(i), (x + 2, y - 2),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            #                     (255, 255, 255), 1, cv2.LINE_AA)

            #     cv2.imwrite("image_0.png", image_0)

        # out_path = os.path.join(save_dir, f"{file_prefix}_{batch_id:05d}.h5")
        existing = [
            f
            for f in os.listdir(save_dir)
            if f.startswith(file_prefix) and f.endswith(".h5")
        ]
        next_id = len(existing)  # ≤ last file index + 1
        out_path = os.path.join(save_dir, f"{file_prefix}_{next_id:05d}.h5")

        print("Saving H5 ...")
        with h5py.File(out_path, "w") as hf:
            hf.attrs["start_idx"] = start_idx
            hf.attrs["end_idx"] = start_idx + batch_size  # exclusive

            hf.create_dataset(
                "beta", data=beta_data, dtype="float32", compression="gzip"
            )
            hf.create_dataset(
                "pose", data=pose_data, dtype="float32", compression="gzip"
            )
            hf.create_dataset(
                "tex_idx", data=tex_idx_data, dtype="int32", compression="gzip"
            )
            hf.create_dataset(
                "image_rgba", data=image_rgba_data, dtype="uint8", compression="gzip"
            )
            hf.create_dataset(
                "depth", data=depth_data, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "mask", data=mask_data, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "camera_poses",
                data=camera_poses_data,
                dtype="float16",
                compression="gzip",
            )
            hf.create_dataset(
                "fov_rads", data=fov_rads_data, dtype="float16", compression="gzip"
            )
            hf.create_dataset(
                "bbox", data=bbox_data, dtype="float16", compression="gzip"
            )

            if getattr(cfg.generate_mesh, "save_verts", False):
                hf.create_dataset(
                    "verts", data=verts_data, dtype="float32", compression="gzip"
                )

            if use_moge and getattr(
                cfg.simulate.moge_bg_depth, "keep_hand_depth", False
            ):
                hf.create_dataset(
                    "depth_hand_only",
                    data=depth_hand_only.astype(np.float16),
                    dtype="float16",
                    compression="gzip",
                )

        print(f"✔ H5 saved {out_path}")

        if cfg.generate_mesh.save_images_and_meta:
            print("Saving images ...")
            n_images = cfg.simulate.n_images

            # ----------------------------------------------------------------
            #  Use *existing* PNG count as an absolute offset, so every
            #  create_sync_data.py run continues where the previous one ended.
            # ----------------------------------------------------------------
            existing_pngs = [f for f in os.listdir(img_dir) if f.endswith(".png")]
            img_offset = len(existing_pngs)  # e.g. 4096, 8192, …

            meta_dict = {}

            for scene_idx in range(batch_size):  # ← NOT range(M)!
                for img_idx in range(n_images):
                    # global_img_idx = global_scene_idx * n_images + img_idx

                    # # --------   RGBA → PNG --------------------------------
                    # img = Image.fromarray(image_rgba_data[scene_idx, img_idx])
                    # img.save(os.path.join(img_dir,
                    #                     f"{global_img_idx:09d}.png"))

                    global_img_idx = img_offset + scene_idx * n_images + img_idx

                    # --------  RGBA → PNG ----------------------------
                    img = Image.fromarray(image_rgba_data[scene_idx, img_idx])
                    img.save(os.path.join(img_dir, f"{global_img_idx:09d}.png"))

                    mask_arr = (mask_data[scene_idx, img_idx] * 255).astype(np.uint8)
                    mask_img = Image.fromarray(mask_arr, mode="L")
                    mask_img.save(os.path.join(mask_dir, f"{global_img_idx:09d}.png"))

                    # --------   metadata → dict ---------------------------
                    meta_dict[f"{global_img_idx:09d}"] = {
                        "beta": beta_data[scene_idx].tolist(),
                        "pose": pose_data[scene_idx].tolist(),
                        "tex_idx": int(tex_idx_data[scene_idx]),
                        "camera_pose": camera_poses_data[scene_idx, img_idx].tolist(),
                        "fov_rads": float(fov_rads_data[scene_idx, img_idx]),
                        "bbox": bbox_data[scene_idx, img_idx].tolist(),
                    }

            # dump *one* JSON file
            print("Saving meta ...")
            meta_path = os.path.join(
                meta_dir,
                f"{file_prefix}_{next_id:05d}.json",  # sync id with .h5
            )
            with open(meta_path, "w") as fp:
                json.dump(meta_dict, fp)
            print(f"    ↳ meta saved to {meta_path}")

        # --------------------------------------------------------------
        # 6 · free RAM *before* next batch
        # --------------------------------------------------------------
        del beta_data, pose_data, tex_idx_data
        del image_rgba_data, depth_data, camera_poses_data, fov_rads_data, bbox_data
        del mano_output, computed_vertices
        torch.cuda.empty_cache()  # if you were on GPU

        if not cfg.generate_mesh.leave_tmp_files:
            for f in os.listdir(temp_dir):  # purge temp meshes
                os.remove(os.path.join(temp_dir, f))

    print("Generation Completed!")


def plot_joints_on_image(
    image, joints_2d: np.ndarray, radius: int = 3, colors: list = None
):
    """
    Draw 2D joint markers on a PIL image, using different colors for each joint.

    Args:
        image:      PIL.Image (mode 'RGB' or 'RGBA').
        joints_2d:  numpy array of shape (N,2) with pixel coords [u,v].
        radius:     marker radius in pixels.
        colors:     Optional list of RGB tuples. If None, automatically assigns colors.

    Returns:
        A new PIL.Image with joints drawn.
    """
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    out = image.copy()
    draw = ImageDraw.Draw(out)
    H, W = out.height, out.width
    N = joints_2d.shape[0]

    if colors is None:
        # Generate N distinct colors using a matplotlib colormap
        cmap = plt.get_cmap("hsv")
        colors = [tuple(int(c * 255) for c in cmap(i / N)[:3]) for i in range(N)]

    for i, (u, v) in enumerate(joints_2d):
        x, y = int(round(u)), int(round(v))
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=colors[i])

    return out


def visualize_sample(
    mesh_save_dir: str,
    file_prefix: str,
    sample_idx: int,
    output_dir: str,
    vis_joints=False,
):
    """
    Loads the HDF5 file, extracts the RGB images for a given sample index,
    saves each of the n_images as PNG files to the designated folder, and returns
    a list of PIL Image objects.

    Parameters:
        mesh_save_dir (str): Path to the HDF5 files (e.g., cfg.generate_mesh.mesh_save_dir).
        sample_idx (int): Index of the sample to visualize.
        output_dir (str): Directory where the images will be saved.

    Returns:
        images (list): A list of PIL.Image objects corresponding to the RGB images.
                   Each image is assumed to be stored with 4 channels (RGBA).
    """
    from HandDataRouter import HandDataRouter

    import torch
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    router = HandDataRouter(save_dir=mesh_save_dir, M=72, file_prefix=file_prefix)
    sample = router.get(sample_idx)
    router.close()

    sample_images = sample["image_rgba"]

    beta = sample["beta"]
    beta = np.expand_dims(beta, axis=0)
    beta = torch.from_numpy(beta)
    pose = sample["pose"]
    pose = np.expand_dims(pose, axis=0)
    pose = torch.from_numpy(pose)

    camera_poses = sample["camera_poses"]
    fov_rads = sample["fov_rads"]

    if vis_joints:
        from manotorch.manolayer import ManoLayer
        from sync_data_utils import project_joints_to_image

        mano_layer = ManoLayer(
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            mano_assets_root="./manotorch/assets/mano",
        )

        # beta = beta.to(torch.float32)
        # pose = pose.to(torch.float32)

        mano_output = mano_layer(pose, beta)
        joints_loc_3d = mano_output.joints[0].cpu().numpy()

    n_images = sample_images.shape[0]
    im_width = sample_images.shape[1]
    im_height = sample_images.shape[2]

    print(n_images, im_width, im_height)

    for i in range(n_images):
        file_name = f"sample_{sample_idx}_image_{i}.png"
        file_path = os.path.join(output_dir, file_name)

        image = Image.fromarray(sample_images[i])

        if vis_joints:
            joints_loc_2d = project_joints_to_image(
                joints_loc_3d,
                cam_pose=camera_poses[i].astype(np.float32),
                fov_rad=fov_rads[i].astype(np.float32),
                width=im_width,
                height=im_height,
            )

            image = plot_joints_on_image(image, joints_loc_2d)

        image.save(file_path)

        print(f"Saved RGB image: {file_path}")


@hydra.main(version_base=None, config_path="configs", config_name="create_sync_data")
def main(cfg: DictConfig):
    if cfg.generate_texture.execute:
        generate_texture(cfg)
        load_texture(cfg, 0, True)

    elif cfg.generate_mesh.execute:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        generate_data(cfg)
        # visualize_sample(h5_file_path=os.path.join(cfg.generate_mesh.mesh_save_dir, "hand_data.h5"),
        #                  sample_idx=0,
        #                  output_dir="./data/mesh")

    elif cfg.visualize.execute:
        visualize_sample(
            mesh_save_dir=cfg.generate_mesh.mesh_save_dir,
            file_prefix=cfg.generate_mesh.file_prefix,
            sample_idx=cfg.visualize.index,
            output_dir=cfg.visualize.output_dir,
            vis_joints=True,
        )
    else:
        print("generate_mesh.execute flag is False. To run generate_mesh, execute:")
        print("    python create_data.py generate_mesh.execute=true")

    return 0


if __name__ == "__main__":
    main()
