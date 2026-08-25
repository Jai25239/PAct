"""Reusable PAct preprocessing operations.

Voxelization, multi-view DINO projection, and latent encoding follow Microsoft
TRELLIS ``dataset_toolkits`` (MIT, TRELLIS@d7f8816). The directory traversal is
PAct-specific: every ``full``/``part_<id>_<name>`` result lives under
``trellis_part_preprocess``, exactly where the released datasets load it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PREPROCESS_DIR = "trellis_part_preprocess"
# Kept for compatibility with released PAct data. The research launcher used
# Cycles despite this historical ``_eevee`` directory name.
RENDER_DIR = "render_merged_fixed_cycles"
VOXEL_DIR = "voxels_merged_fixed"


def voxelize_mesh(mesh_path: Path, output_path: Path, resolution: int = 64) -> int:
    import open3d as o3d
    import utils3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {mesh_path}")
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    indices = np.asarray([voxel.grid_index for voxel in grid.get_voxels()])
    if indices.size == 0:
        raise ValueError(f"Voxelization produced no occupied cells: {mesh_path}")
    points = (indices + 0.5) / resolution - 0.5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utils3d.io.write_ply(str(output_path), points.astype(np.float32))
    return len(points)


class LatentEncoder:
    def __init__(
        self,
        *,
        device: str,
        feature_model: str,
        ss_encoder: str,
        slat_encoder: str,
        batch_size: int,
    ):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.feature_model_name = feature_model
        self.ss_encoder_source = ss_encoder
        self.slat_encoder_source = slat_encoder
        self.ss_encoder_name = ss_encoder.split("/")[-1]
        self.slat_encoder_name = f"{feature_model}_{slat_encoder.split('/')[-1]}"
        self._feature_model = None
        self._ss_encoder = None
        self._slat_encoder = None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
            3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    @property
    def feature_model(self):
        if self._feature_model is None:
            self._feature_model = (
                self.torch.hub.load(
                    "facebookresearch/dinov2",
                    self.feature_model_name,
                    pretrained=True,
                )
                .eval()
                .to(self.device)
            )
        return self._feature_model

    @property
    def ss_encoder(self):
        if self._ss_encoder is None:
            from modules.pact import models

            self._ss_encoder = (
                models.from_pretrained(self.ss_encoder_source).eval().to(self.device)
            )
        return self._ss_encoder

    @property
    def slat_encoder(self):
        if self._slat_encoder is None:
            from modules.pact import models

            self._slat_encoder = (
                models.from_pretrained(self.slat_encoder_source).eval().to(self.device)
            )
        return self._slat_encoder

    def release(self, component: str) -> None:
        attribute = f"_{component}"
        model = getattr(self, attribute)
        if model is not None:
            del model
            setattr(self, attribute, None)
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()

    def encode_ss(
        self, voxel_path: Path, destination: Path, resolution: int = 64
    ) -> None:
        import utils3d

        torch = self.torch
        points = utils3d.io.read_ply(str(voxel_path))[0]
        coords = ((torch.as_tensor(points) + 0.5) * resolution).long()
        coords.clamp_(0, resolution - 1)
        occupancy = torch.zeros(
            1, 1, resolution, resolution, resolution, device=self.device
        )
        occupancy[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        with torch.inference_mode():
            latent = self.ss_encoder(occupancy, sample_posterior=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, mean=latent[0].float().cpu().numpy())

    def extract_features(
        self, render_dir: Path, voxel_path: Path, destination: Path
    ) -> None:
        import torch.nn.functional as F
        import utils3d
        from PIL import Image

        torch = self.torch
        metadata = json.loads(
            (render_dir / "transforms.json").read_text(encoding="utf-8")
        )
        points_np = utils3d.io.read_ply(str(voxel_path))[0]
        points = torch.as_tensor(points_np, dtype=torch.float32, device=self.device)
        indices = ((points + 0.5) * 64).long().clamp(0, 63)
        sampled_views = []
        for start in range(0, len(metadata["frames"]), self.batch_size):
            frames = metadata["frames"][start : start + self.batch_size]
            images, extrinsics, intrinsics = [], [], []
            for frame in frames:
                image = Image.open(render_dir / frame["file_path"]).resize(
                    (518, 518), Image.Resampling.LANCZOS
                )
                rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
                rgb = rgba[..., :3] * rgba[..., 3:]
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
                images.append((tensor - self.mean) / self.std)
                camera_to_world = torch.tensor(
                    frame["transform_matrix"], dtype=torch.float32, device=self.device
                )
                camera_to_world[:3, 1:3] *= -1
                extrinsics.append(torch.inverse(camera_to_world))
                fov = torch.tensor(frame["camera_angle_x"], device=self.device)
                intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
            with torch.inference_mode():
                features = self.feature_model(torch.stack(images), is_training=True)[
                    "x_prenorm"
                ]
            patch_count = 518 // 14
            patchtokens = (
                features[:, self.feature_model.num_register_tokens + 1 :]
                .permute(0, 2, 1)
                .reshape(len(frames), -1, patch_count, patch_count)
            )
            uv = (
                utils3d.torch.project_cv(
                    points, torch.stack(extrinsics), torch.stack(intrinsics)
                )[0]
                * 2
                - 1
            )
            sampled = (
                F.grid_sample(
                    patchtokens, uv.unsqueeze(1), mode="bilinear", align_corners=False
                )
                .squeeze(2)
                .permute(0, 2, 1)
            )
            sampled_views.append(sampled.cpu())
        averaged = torch.cat(sampled_views).mean(dim=0).numpy().astype(np.float16)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            indices=indices.cpu().numpy().astype(np.uint8),
            patchtokens=averaged,
        )

    def encode_slat(self, feature_path: Path, destination: Path) -> None:
        from modules.pact.modules.sparse.basic import SparseTensor

        torch = self.torch
        features = np.load(feature_path)
        sparse = SparseTensor(
            feats=torch.from_numpy(features["patchtokens"]).float(),
            coords=torch.cat(
                [
                    torch.zeros(len(features["indices"]), 1, dtype=torch.int32),
                    torch.from_numpy(features["indices"]).int(),
                ],
                dim=1,
            ),
        ).to(self.device)
        with torch.inference_mode():
            latent = self.slat_encoder(sparse, sample_posterior=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            feats=latent.feats.float().cpu().numpy(),
            coords=latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
        )
