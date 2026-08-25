#!/usr/bin/env python3
"""Prepare articulated object folders for PAct Stage-1 and Stage-2 training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, help="Preprocessing JSON config")
    result.add_argument(
        "--input-root", required=True, type=Path, help="Raw object root"
    )
    result.add_argument(
        "--output-root", required=True, type=Path, help="Processed dataset root"
    )
    result.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages, or all; stages always run in canonical order",
    )
    result.add_argument(
        "--skip-invalid", action="store_true", help="Record and skip invalid objects"
    )
    result.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 2 conditioning views, 1 geometry view, 64px, and 1 render sample",
    )
    result.add_argument("--device", help="Override encoder device from the config")
    result.add_argument(
        "--blender", help="Override the Blender executable from the config"
    )
    result.add_argument(
        "--cycles-device-type",
        choices=("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"),
        help="Override the Cycles compute backend",
    )
    result.add_argument(
        "--cycles-devices",
        help="Comma-separated Cycles device names; defaults to all GPUs",
    )
    result.add_argument(
        "--cycles-samples", type=int, help="Override Cycles samples per pixel"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from preprocessing.pipeline import ALL_STAGES, Preprocessor

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.smoke_test:
        config.update(
            {
                "smoke_test": True,
                "conditioning_views": 2,
                "geometry_views": 1,
                "image_resolution": 64,
                "conditioning_samples": 1,
                "cycles_samples": 1,
            }
        )
    if args.blender:
        config["blender"] = args.blender
    if args.cycles_device_type:
        config["cycles_device_type"] = args.cycles_device_type
    if args.cycles_devices:
        config["cycles_devices"] = args.cycles_devices
    if args.cycles_samples is not None:
        if args.cycles_samples < 1:
            raise ValueError("--cycles-samples must be positive")
        config["cycles_samples"] = args.cycles_samples
    stages = list(ALL_STAGES) if args.stages == "all" else args.stages.split(",")
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown preprocessing stages: {sorted(unknown)}")
    selected = [stage for stage in ALL_STAGES if stage in stages]
    Preprocessor(
        config,
        args.input_root,
        args.output_root,
        selected,
        args.skip_invalid,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
