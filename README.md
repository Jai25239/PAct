# PAct: Part-Decomposed Single-View Articulated Object Generation

> **Generate an articulated, simulation-ready 3D object from a single-view input.**
>
> *SIGGRAPH Asia 2026 · Conditionally Accepted*

**Authors:**
[Qingming Liu](https://github.com/Mobiuslqm)<sup>1,2</sup>,
[Xinyue Yao](https://scholar.google.com/citations?user=ZOf_esUAAAAJ&hl=en)<sup>1</sup>,
[Shuyuan Zhang](https://sanbingyouyong.github.io/)<sup>1</sup>,
[Yueci Deng](https://github.com/yuecideng)<sup>1,2</sup>,
[Guiliang Liu](https://guiliang.me)<sup>1</sup>,
[Zhen Liu](https://itszhen.com)<sup>1,†</sup>,
[Kui Jia](http://kuijia.site)<sup>1,2</sup>

<sup>1</sup>The Chinese University of Hong Kong, Shenzhen &nbsp;&nbsp; <sup>2</sup>DexForce Technology

<sup>†</sup>Corresponding author

<a href="https://pact-project.github.io/"><img src="https://img.shields.io/badge/Project-Page-1f6feb" alt="Project Page"></a>
<a href="https://arxiv.org/pdf/2602.14965"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Paper"></a>
<a href="https://arxiv.org/abs/2602.14965"><img src="https://img.shields.io/badge/arXiv-2602.14965-b31b1b" alt="arXiv"></a>
<a href="https://github.com/Mobiuslqm/PAct"><img src="https://img.shields.io/badge/Code-GitHub-181717" alt="Code"></a>
<a href="https://huggingface.co/spaces/PAct000/PAct"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet" alt="Demo"></a>
<a href="https://huggingface.co/PAct000/PAct"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Model"></a>
<a href="#8-citation"><img src="https://img.shields.io/badge/BibTeX-Cite-blue" alt="BibTeX"></a>

<!-- <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a> -->
![teaser](assets/doc/teaser.jpg)

Given a single image, **PAct** generates an articulated 3D object by predicting a part-decomposed structure, synthesizing high-fidelity part geometry and appearance, and estimating articulation parameters for physics-based simulation.
This repository provides the inference pipeline, both PAct training stages, and
the raw-data preprocessing workflow needed to construct their inputs.


## 1. Open-Source Timeline

| Stage | Deliverable | Target Date | Notes |
| --- | --- | --- | --- |
| Inference Release | Cleaned `infer_imgs.py`, pretrained checkpoints, sample configs. | 2026-02-07 | ✅ available now. |
| Dataset Preprocessing | Raw articulated assets to PAct latents and dataset index. | 2026-08-24 | ✅ available now. |
| Training Stage 1 | Part-based sparse-structure flow training. | 2026-08-24 | ✅ available now. |
| Training Stage 2 | Structured-latent flow and articulation-head training. | 2026-08-24 | ✅ available now. |


Dates reflect our best-effort plan; we will update this table and tag releases in the repo as milestones land.

Thanks to Codex and Kimi Coding Agent for helping us refactor the code and accelerate the open-sourcing process.

We also apologize for the repeated delays—sometimes chronic procrastination gets the better of us, lol.


## 2. Environment Setup

1. **Clone**
	```bash
	git clone https://github.com/PAct-project/PAct.git
	cd PAct
	```
2. **Conda environment (recommended; aligned with TRELLIS, SINGAPO, and OmniPart)**
	```bash
	conda env create -f PAct_env.yml
	conda activate PAct
    pip install git+https://github.com/facebookresearch/detectron2.git
	```

## 3. Launch Gradio Demo

```bash
python app.py
```
For convenience, we provide a Hugging Face demo that also allows downloading exported URDFs. The exported URDF files can be interactively viewed in VS-Code with [URDF Visualizer](https://marketplace.visualstudio.com/items?itemName=morningfrog.urdf-visualizer).
![teaser](assets/doc/urdf_vis_ext.png)
## 4. Inference via scritps


### 4.1 Running Inference

Call the batch inference script with your config and overrides:

```bash
python infer_imgs.py \
  --data_dir assets/real_world_examples \
  --outdir outputs/real_world \
  --batch_size 2 \
  --save_glb --export_arti_objects 
```

Results (videos, GLBs, Gaussian splats, logs) are written under the `--outdir` folder in subdirectories named with your sampling configuration and random seed.  Generation process of an object typically takes ~15s, comparable to TRELLIS; exporting a mesh is optional, but the subsequent textured-mesh step can be significantly more time-consuming.

### 4.2 Key Arguments

`infer_imgs.py` exposes every previously hard-coded hyperparameter as a CLI flag. Important options are summarized below (see [infer_imgs.py](infer_imgs.py) for the full list):

| Flag | Purpose | Default |
| --- | --- | --- |
| `--ss_steps`, `--slat_steps` | Sampler iterations for sparse structure / SLAT stages. | `25`, `25` |
| `--ss_cfg_strength`, `--slat_cfg_strength` | Guidance strength for each sampler. | `7.0`, `7.0` |
| `--explode_coords_ratio`, `--gaussian_explosion_scale` | Explosion used when visualizing voxels or Gaussians. | `0.5`, `0.3` |
| `--render_num_frames`, `--render_radius`, `--render_fov`, `--render_bg_color` | Camera sweep + appearance of rendered videos. | `60`, `2.3`, `60`, `(1,1,1)` |
| `--video_fps`, `--grid_size`, `--save_video_grid`, `--save_cond_vis_grid` | Control mosaic layout and playback speed when saving videos/images. | `20`, `4`, enabled |
| `--save_glb`, `--save_gs`, `--export_arti_objects` | Toggle mesh/splat export to SINGAPO/URDF formats. | disabled |
| `--mesh_simplify_ratio`, `--texture_size`, `--textured_mesh` | Mesh post-processing knobs when `--save_glb` is set. | `0.95`, `1024`, enabled |

Every argument can also be specified inside the JSON config; CLI values take precedence.

### 4.3 Output Layout

Each inference batch produces:

- `grid_vids_samples_videos_*`: Articulation and exploded-part video mosaics.
- `grids_cond_vis_*`: Conditioning image grids.
- `*_arti_animation*.mp4`, `*exploded_part*.mp4/png`: Per-object renders when mosaics are disabled.
- `run_command.txt`: Command provenance for reproducibility.
- `exported_arti_objects`: Optional GLB/Gaussian assets and articulation info if the corresponding flags are enabled.

### 4.4 Export Articulated Object to URDFs

Use the helper script to convert every `object.json` in an exported inference run into URDF files ( must set `--export_arti_objects` in Sec. 4.1):

```bash
python scripts/batch_json_to_urdf.py \
	--exported_art_objs_dir outputs/<run_name>/exported_arti_objects
```

Each generated URDF is placed next to its source metadata as `<object_name>_fromJson2urdf.urdf`.

## 5. Dataset preprocessing

The training release uses one pipeline and one on-disk representation:

```text
raw articulated object folders
  -> annotation and asset validation
  -> isolated staged dataset
  -> conditioning images and semantic masks
  -> normalized multi-view part geometry
  -> 64^3 voxels
  -> sparse-structure and structured latents
  -> dataset_index.jsonl / data_split.json
```

The raw root can contain object folders directly or nested under category and
dataset directories. Each object needs either `object.json` or an already
merged `object_merge_fixed.json`, with a flat `diffuse_tree`. For raw
`object.json`, the pipeline merges every non-root fixed node into its parent
before schema validation; it never writes the derived annotation into the raw
folder. Every remaining node needs `id`, `name`, OBJ paths in `objs`,
`aabb.center`, `aabb.size`, and `joint` fields for type, two-value range, axis
direction, and axis origin. The released model uses the labels `door`,
`drawer`, `base`, `handle`, `wheel`, `knob`, `shelf`, and `tray`, with at most
eight parts after merging. Geometry paths are relative to the object directory.
The annotations and source datasets are not redistributed by PAct.

Preprocessing additionally requires Blender on `PATH`, Open3D, `utils3d`,
DINOv2 access, and the public TRELLIS encoders. The pipeline never
installs system packages and never writes into `--input-root`.

The renderer and encoder algorithms were reconciled with the official
TRELLIS/SINGAPO sources instead of executing the research scripts unchanged.
The exact source commits, license, and PAct-specific differences are recorded
in [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md).

```bash
python scripts/preprocess_data.py \
  --config configs/preprocessing/pact.json \
  --input-root /path/to/raw_objects \
  --output-root /path/to/pact_dataset \
  --stages all
```

If Blender is not on the activated `PAct` environment's `PATH`, provide its
executable explicitly. Python orchestration and encoding still run in `PAct`:

```bash
conda activate PAct
python scripts/preprocess_data.py \
  --config configs/preprocessing/pact.json \
  --input-root /path/to/raw_objects \
  --output-root /path/to/pact_dataset \
  --stages all \
  --blender /path/to/blender
```

Geometry rendering follows the paper preprocessing path and uses
Cycles/OPTIX by default. Its output directory retains the historical
`render_merged_fixed_cycles` name used by released PAct data even though the
renderer is Cycles. Select the GPU(s) with `CUDA_VISIBLE_DEVICES` (the index is
evaluated before Blender starts), for example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_data.py \
  --config configs/preprocessing/pact.json \
  --input-root /path/to/raw_objects \
  --output-root /path/to/pact_dataset \
  --stages render_geometry,index \
  --blender /path/to/blender \
  --cycles-device-type OPTIX
```

Use `--cycles-device-type CUDA` on systems without OptiX. Existing geometry
outputs are skipped only when their metadata, mesh, and a nonempty-alpha
preview all exist, so interrupted or blank renders are regenerated safely.

For an archive with the same layout as the provided `pm.zip` sample:

```bash
mkdir -p /path/to/pm_raw
unzip /path/to/pm.zip -d /path/to/pm_raw
python scripts/preprocess_data.py \
  --config configs/preprocessing/pact.json \
  --input-root /path/to/pm_raw \
  --output-root /path/to/pm_processed \
  --stages validate,merge_fixed,stage,index \
  --skip-invalid
```

To exercise merge, both real Blender renderers, voxelization, and indexing on
a small extracted subset without producing training-quality data, add
`--smoke-test` and select those stages. Smoke output uses two 64px conditioning
views and one geometry view and must not be used for training.

Nested paths such as `Dishwasher/11622` become the collision-safe output ID
`Dishwasher__11622`, because the existing PAct dataset scans one object-folder
level below its root. `--skip-invalid` records rejected objects and reasons in
`preprocessing_manifest.json`; without it, preprocessing stops at the first
invalid object.

Stages can be resumed safely because completed outputs are skipped. To split
work across machines, pass a subset such as
`--stages validate,merge_fixed,stage,render_conditioning,render_geometry` first and then
`--stages voxelize,encode_ss,extract_features,encode_slat,index`. Logs from
external renderers are stored in `/path/to/pact_dataset/logs`.
`data_split.json` is generated deterministically from `validation_fraction` in
the preprocessing config and is consumed automatically by both training stages.

The resulting layout is:

```text
/path/to/pact_dataset/
  data_split.json
  dataset_index.jsonl
  <object_id>/
    object_merge_fixed.json
    imgs/00.png ... 19.png
    imgs/semantic_masks_merge_fixed/00.npz ... 19.npz
    trellis_part_preprocess/
      render_merged_fixed_cycles/{full,part_<id>_<name>}/
      voxels_merged_fixed/{full,part_<id>_<name>}/mesh.ply
      ss_latents/ss_enc_conv3d_16l8_fp16/*.npz
      features/dinov2_vitl14_reg/*.npz
      latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/*.npz
```

For a dependency-light validation/staging smoke test:

```bash
tmp_dir=$(mktemp -d)
python scripts/preprocessing/create_synthetic_raw.py --output-root "$tmp_dir/raw"
python scripts/preprocess_data.py \
  --config configs/preprocessing/pact.json \
  --input-root "$tmp_dir/raw" \
  --output-root "$tmp_dir/processed" \
  --stages validate,merge_fixed,stage,index
```

## 6. Training

The two released configs instantiate exactly the model classes loaded by
`PActPipeline`: Stage 1 trains `PartBasedSparseStructureFlowModel`; Stage 2
trains `SLatFlowModel` wrapped by `ArticulationRegressionHead`. Both use the
original TRELLIS `Trainer -> BasicTrainer -> FlowMatchingTrainer` lifecycle,
flow matching, frozen DINOv2 conditioning, classifier-free dropout, TRELLIS
inflated-FP16 training, gradient accumulation, adaptive clipping, EMA,
distributed training, and resumable samplers/checkpoints. The exact class MRO
is documented in
[`modules/pact/trainers/README.md`](modules/pact/trainers/README.md).

Stage 1 may start from random initialization, but paper reproduction should
pass the local TRELLIS/OmniPart sparse-structure state dict with
`--pretrained-denoiser`. Stage 2 uses LoRA and therefore requires the local
structured-latent base state dict. The argument accepts a plain PyTorch
state-dict `.pt`/`.ckpt`; it does not accept a pipeline directory.

Single-GPU training:

```bash
python train.py \
  --config configs/training/pact_stage1.json \
  --data-dir /path/to/pact_dataset \
  --split-info /path/to/pact_dataset/data_split.json \
  --pretrained-denoiser /path/to/ss_flow_base.ckpt \
  --output-dir outputs/pact_stage1 \
  --num-gpus 1

python train.py \
  --config configs/training/pact_stage2.json \
  --data-dir /path/to/pact_dataset \
  --split-info /path/to/pact_dataset/data_split.json \
  --pretrained-denoiser /path/to/slat_flow_base.ckpt \
  --output-dir outputs/pact_stage2 \
  --num-gpus 1
```

Multi-GPU training preserves TRELLIS' `torch.multiprocessing.spawn` entry:

```bash
python train.py \
  --config configs/training/pact_stage1.json \
  --data-dir /path/to/pact_dataset \
  --split-info /path/to/pact_dataset/data_split.json \
  --pretrained-denoiser /path/to/ss_flow_base.ckpt \
  --output-dir outputs/pact_stage1 \
  --num-gpus 4
```

Resume and validation use the same config and output directory:

```bash
python train.py \
  --config configs/training/pact_stage1.json \
  --data-dir /path/to/pact_dataset \
  --split-info /path/to/pact_dataset/data_split.json \
  --output-dir outputs/pact_stage1 \
  --load-dir outputs/pact_stage1 \
  --ckpt latest \
  --num-gpus 1

python train.py \
  --config configs/training/pact_stage2.json \
  --data-dir /path/to/pact_dataset \
  --split-info /path/to/pact_dataset/data_split.json \
  --pretrained-denoiser /path/to/slat_flow_base.ckpt \
  --output-dir outputs/pact_stage2 \
  --load-dir outputs/pact_stage2 \
  --ckpt latest \
  --validate-only \
  --num-gpus 1
```

Each experiment contains `config.json`, the exact `command.txt`, TensorBoard
events, `log.txt`, and TRELLIS split checkpoints under `ckpts/`:
`denoiser_stepXXXXXXX.pt`, `denoiser_ema<rate>_stepXXXXXXX.pt`, and
`misc_stepXXXXXXX.pt` (optimizer, scheduler/scaler when enabled, gradient
clipper, step, and resumable sampler). At normal completion and in the smoke
test, the entry also writes `ckpts/inference/model{.json,.safetensors}` for
Stage 1 or `model{_slat.json,_arti.json,.safetensors}` for Stage 2. LoRA is
merged before export, so these files load directly through the existing
inference model registry.

To assemble both trained stages with the public decoders and run the existing
inference entrypoint:

```bash
python scripts/export_inference_checkpoint.py \
  --stage1-artifact outputs/pact_stage1/ckpts/inference/model \
  --stage2-artifact outputs/pact_stage2/ckpts/inference/model \
  --output-dir outputs/pact_trained_pipeline \
  --base-model PAct000/PAct

python infer_imgs.py \
  --model outputs/pact_trained_pipeline \
  --data_dir /path/to/input_images \
  --outdir outputs/inference \
  --batch_size 2
```

The environment file includes the newly exercised runtime dependencies:
`safetensors`, `tensorboard`, `peft`, and `open3d`. Public
pretrained TRELLIS/OmniPart weights are downloaded through Hugging Face. Raw
articulated annotations, meshes, and their redistribution rights remain the
user's responsibility.
## 7. Contributing & Support

## 8. Citation

If you build upon this work, please cite the PAct paper:
```bibtex
@article{liu2026pact,
    title   = {PAct: Part-Decomposed Single-View Articulated Object Generation},
    author  = {Liu, Qingming and Yao, Xinyue and Zhang, Shuyuan and Deng, Yueci and Liu, Guiliang and Liu, Zhen and Jia, Kui},
    journal = {arXiv preprint arXiv:2602.14965},
    year    = {2026}
}
```


And we sincerely thank the authors of TRELLIS and OmniPart, whose codes were used in our work.

```
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}.
```

```bitex
@article{yang2025omnipart,
        title={Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion},
        author={Yang, Yunhan and Zhou, Yufan and Guo, Yuan-Chen and Zou, Zi-Xin and Huang, Yukun and Liu, Ying-Tian and Xu, Hao and Liang, Ding and Cao, Yan-Pei and Liu, Xihui},
        journal={arXiv preprint arXiv:2507.06165},
        year={2025}
}
```
