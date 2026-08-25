"""Dataset and dataloader construction for released PAct training stages."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset



class SyntheticSparseStructureDataset(Dataset):
    """Deterministic, in-memory data used by the documented smoke test."""

    is_test = False

    def __init__(self, model_args: dict[str, Any], length: int = 8):
        self.length = length
        self.loads = [4] * length
        self.channels = int(model_args["in_channels"])
        self.resolution = int(model_args["resolution"])
        self.cond_channels = int(model_args["cond_channels"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(index)
        num_parts = 2
        cond = torch.randn(
            num_parts, 8, self.cond_channels, generator=generator
        )
        return {
            "x_0": torch.randn(
                num_parts,
                self.channels,
                self.resolution,
                self.resolution,
                self.resolution,
                generator=generator,
            ),
            "num_parts": num_parts,
            "part_idx": torch.arange(num_parts),
            "cond": cond,
            "cond_features": cond,
            "ordered_mask_dino": torch.tensor(
                [[[0, 1], [2, 0]], [[0, 1], [2, 0]]], dtype=torch.long
            ),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], split_size: int | None = None):
        if split_size and split_size > 1:
            groups = [batch[index::split_size] for index in range(split_size)]
            return [
                SyntheticSparseStructureDataset.collate_fn(group)
                for group in groups
                if group
            ]
        return {
            "x_0": torch.cat([item["x_0"] for item in batch]),
            "num_parts": torch.tensor([item["num_parts"] for item in batch]),
            "part_idx": torch.cat([item["part_idx"] for item in batch]),
            "cond": torch.cat([item["cond"] for item in batch]),
            "cond_features": torch.cat([item["cond_features"] for item in batch]),
            "ordered_mask_dino": torch.cat(
                [item["ordered_mask_dino"] for item in batch]
            ),
        }


class SyntheticStructuredLatentDataset(Dataset):
    """Small sparse samples for optional Stage-2 environment checks."""

    is_test = False

    def __init__(self, model_args: dict[str, Any], length: int = 8):
        self.length = length
        self.loads = [4] * length
        self.channels = int(model_args["in_channels"])
        self.cond_channels = int(model_args["cond_channels"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(index)
        coords = torch.tensor(
            [[1, 1, 1], [1, 1, 2], [2, 2, 2], [2, 2, 3]], dtype=torch.int32
        )
        cond = torch.randn(1, 8, self.cond_channels, generator=generator)[0]
        return {
            "coords": coords,
            "feats": torch.randn(4, self.channels, generator=generator),
            "part_layouts": [slice(0, 2), slice(2, 4)],
            "articulation": torch.cat(
                [torch.linspace(-0.5, 0.5, 24).unsqueeze(0), torch.ones(1, 24)],
                dim=-1,
            ),
            "cond": cond,
            "cond_features": cond,
            "ordered_mask_dino": torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], split_size: int | None = None):
        if split_size and split_size > 1:
            groups = [batch[index::split_size] for index in range(split_size)]
            return [
                SyntheticStructuredLatentDataset.collate_fn(group)
                for group in groups
                if group
            ]
        from ..modules.sparse.basic import SparseTensor

        coords = []
        feats = []
        arti_coords = []
        arti_feats = []
        for batch_id, item in enumerate(batch):
            coords.append(
                torch.cat(
                    [
                        torch.full(
                            (len(item["coords"]), 1), batch_id, dtype=torch.int32
                        ),
                        item["coords"],
                    ],
                    dim=-1,
                )
            )
            feats.append(item["feats"])
            arti_coords.append(
                torch.full(
                    (len(item["articulation"]), 1), batch_id, dtype=torch.int32
                )
            )
            arti_feats.append(item["articulation"])
        x_0 = SparseTensor(coords=torch.cat(coords), feats=torch.cat(feats))
        x_0_arti = SparseTensor(
            coords=torch.cat(arti_coords), feats=torch.cat(arti_feats)
        )
        return {
            "x_0": x_0,
            "x_0_arti": x_0_arti,
            "part_layouts": [item["part_layouts"] for item in batch],
            "cond": torch.stack([item["cond"] for item in batch]),
            "cond_features": torch.stack([item["cond_features"] for item in batch]),
            "ordered_mask_dino": torch.stack(
                [item["ordered_mask_dino"] for item in batch]
            ),
        }
