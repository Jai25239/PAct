#!/usr/bin/env python3
"""Assemble Stage-1 and Stage-2 training artifacts into a PActPipeline directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def resolve_file(base: str, relative: str, revision: str) -> Path:
    local = Path(base).expanduser()
    if local.is_dir():
        path = local / relative
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(base, relative, revision=revision))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_prefix(source: Path, destination: Path, suffixes: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        source_file = Path(f"{source}{suffix}")
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        shutil.copy2(source_file, Path(f"{destination}{suffix}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-artifact", required=True, type=Path)
    parser.add_argument("--stage2-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="PAct000/PAct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline_source = resolve_file(args.base_model, "pipeline.json", args.revision)
    pipeline = json.loads(pipeline_source.read_text(encoding="utf-8"))
    models = pipeline["args"]["models"]

    copy_prefix(
        args.stage1_artifact.expanduser().resolve(),
        output / models["sparse_structure_flow_model"],
        [".json", ".safetensors"],
    )
    copy_prefix(
        args.stage2_artifact.expanduser().resolve(),
        output / models["slat_arti_flow_model"],
        ["_arti.json", "_slat.json", ".safetensors"],
    )
    for key in ("sparse_structure_decoder", "slat_decoder_gs", "slat_decoder_mesh"):
        prefix = models[key]
        for suffix in (".json", ".safetensors"):
            relative = f"{prefix}{suffix}"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                resolve_file(args.base_model, relative, args.revision), destination
            )
    (output / "pipeline.json").write_text(
        json.dumps(pipeline, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
