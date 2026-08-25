"""Export helpers for TRELLIS checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_model_artifact(
    model: torch.nn.Module,
    destination: Path,
    stage: str,
    model_config: dict[str, Any],
    articulation_config: dict[str, Any] | None = None,
) -> Path:
    """Save a prefix consumable by the PAct model registry."""
    from safetensors.torch import save_file

    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in unwrap_model(model).state_dict().items()
    }
    save_file(state, str(destination.with_suffix(".safetensors")))
    if stage == "sparse_structure":
        destination.with_suffix(".json").write_text(
            json.dumps(model_config, indent=2) + "\n", encoding="utf-8"
        )
    else:
        if articulation_config is None:
            raise ValueError("articulation_config is required for structured_latent")
        Path(f"{destination}_slat.json").write_text(
            json.dumps(model_config, indent=2) + "\n", encoding="utf-8"
        )
        Path(f"{destination}_arti.json").write_text(
            json.dumps(articulation_config, indent=2) + "\n", encoding="utf-8"
        )
    return destination
