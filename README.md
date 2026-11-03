# AnyHand Generation Pipeline

This repository hosts the synthetic data generation pipeline for [AnyHand](https://github.com/chen-si-cs/AnyHand). The pipeline generates photorealistic synthetic hand images with full ground-truth annotations (2D/3D joints, depth, segmentation masks, camera parameters) for training hand pose estimation models.

## Overview

The pipeline supports two generation modes:

1. **Hand-only generation** — Diverse hand poses with optional forearm, realistic textures, and composited backgrounds.
2. **GraspXL generation** — Hands grasping objects from the GraspXL/Objaverse datasets.

The pipeline combines MANO/Handy hand models, pose sampling, BMC filtering for biomechanical validity, and SAPIEN rendering.

For the released dataset and tools to read it, see the [AnyHand dataset](https://huggingface.co/datasets/chen-si-02/AnyHand-Dataset) and [AnyHand toolkit](https://github.com/chen-si-cs/anyhand-toolkit).

---

## Prerequisites

### Required External Data (not included)

Due to licensing, the following must be obtained separately and placed in the specified directories:

| Data | Directory | Source |
|------|-----------|--------|
| MANO hand models | `mano_v1_2/` | MANO website |
| DposerX pretrained model (`last.ckpt`) | `DposerX/pretrained_models/hand/BaseMLP/` | See DposerX repository |
| StyleGAN3 (for Handy texture generation) | `stylegan3/` | StyleGAN3 repo |
| SMPL-X body models (`SMPLH_male.pkl`) | `smplx_models/smplh/` | SMPL-X website |
| Handy assets | `manotorch/assets/` | See Handy / manotorch repo |
| Background images (MIT Indoor Scenes) | `data/MIT_Indoor_Scenes_Picked/` | MIT Indoor Scenes |
| Environment maps (optional) | `data/env_maps/` | HDR environment maps for lighting |
| SMPLitex arm textures (optional) | `data/SMPLitex-textures.zip` | SMPLitex |
| GraspXL data (for grasp mode) | `data/GraspXL/` | See GraspXL dataset |


---

## Installation

```bash
git clone https://github.com/chen-si-cs/AnyHand-Generation.git
cd AnyHand-Generation
```

### Environment

We recommend using a conda environment with Python 3.10+.

**Python dependencies:**

```bash
pip install cmake hydra-core pyexr joblib scikit-learn smplx tqdm h5py pillow opencv-python numpy scipy trimesh chumpy matplotlib OpenEXR
pip install sapien
pip install ./manotorch
pip install git+https://github.com/microsoft/MoGe.git
pip install ml_collections pytorch_lightning
```

---

Install CUDA-enabled PyTorch for your system before running the pipeline. Obtain the external `manotorch`, `DposerX`, and `stylegan3` codebases and their required assets separately, and place them at the paths used by the scripts. Run commands from the repository root.

### Configure external assets

The supplied configurations refer to external resources that are not included here:

- Real-pose samples under `real_poses/` (including `freihand_train_poses_betas.npz`) and optional GMM parameters under `gmm_params/`. Set the pose-source options and paths for your available assets.
- Handy texture weights at `manotorch/assets/handy/Handy_Models/hands_texture_model.pkl`.
- A MoGe checkpoint: `simulate.moge_bg_depth.model_name` currently points to `/moge-2-vitl-normal`; replace it with your checkpoint path or model identifier, or disable background depth with `simulate.moge_bg_depth.flag=false`. Background depth inferred by MoGe is an estimate; hand geometry and rendered hand depth come from the synthetic scene.

Review both YAML files under `configs/` before generation. This release contains the generation scripts and configuration examples; external models, pose samples, textures, and background datasets must be prepared separately.

## Usage

### 1. Generate Hand Textures (Optional)

If using Handy-based texture generation:

```bash
python create_sync_data.py generate_texture.execute=true generate_texture.n_textures=1024
```

### 2. Generate Hand-Only Data

Edit `configs/create_sync_data.yaml` to configure generation settings (resolution, number of samples, arm attachment, background type, etc.), then run:

```bash
python create_sync_data.py generate_mesh.execute=true generate_mesh.num_generate=1024
```

To save rendered images alongside mesh/annotation data:

```bash
python create_sync_data.py generate_mesh.execute=true generate_mesh.save_images_and_meta=true
```

### 3. Generate GraspXL Data (Hand + Object)

Edit `configs/create_sync_data_graspxl.yaml`, then run:

```bash
python create_sync_data_graspxl.py generate_mesh.execute=true generate_mesh.num_generate=1024
```
---

## Third-party assets and dependencies

External dependencies, model files, textures, and datasets remain subject to their respective licenses and terms. Obtain restricted assets from their original providers.
