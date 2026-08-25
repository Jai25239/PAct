"""Orchestration for the reproducible PAct preprocessing pipeline."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .annotation import merge_fixed_parts
from .ops import PREPROCESS_DIR, RENDER_DIR, VOXEL_DIR, LatentEncoder, voxelize_mesh

LOGGER = logging.getLogger("pact.preprocessing")
ALL_STAGES = (
    "validate",
    "merge_fixed",
    "stage",
    "render_conditioning",
    "render_geometry",
    "voxelize",
    "encode_ss",
    "extract_features",
    "encode_slat",
    "index",
)


SEMANTIC_LABELS = {"door", "drawer", "base", "handle", "wheel", "knob", "shelf", "tray"}
JOINT_TYPES = {"fixed", "revolute", "prismatic", "screw", "continuous"}


def _nodes(annotation: dict[str, Any]) -> Iterable[dict[str, Any]]:
    # PAct annotations store the flattened kinematic tree in diffuse_tree;
    # parent/child relationships are IDs, not nested node dictionaries.
    yield from annotation.get("diffuse_tree", [])


def discover_objects(input_root: Path, annotation_names: Iterable[str]) -> list[Path]:
    names = tuple(dict.fromkeys(annotation_names))
    if any((input_root / name).is_file() for name in names):
        return [input_root]
    objects = {
        annotation.parent
        for name in names
        for annotation in input_root.rglob(name)
        if PREPROCESS_DIR not in annotation.parts
    }
    return sorted(objects)


def validate_object(
    object_dir: Path,
    annotation: dict[str, Any],
    annotation_path: Path,
    max_parts: int = 8,
) -> dict[str, Any]:
    nodes = list(_nodes(annotation))
    if not nodes:
        raise ValueError(f"No diffuse_tree parts in {annotation_path}")
    if len(nodes) > max_parts:
        raise ValueError(
            f"{annotation_path} has {len(nodes)} parts; maximum is {max_parts}"
        )
    ids = [int(node["id"]) for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate part IDs in {annotation_path}")
    asset_count = 0
    for node in nodes:
        for required in ("id", "name", "objs", "joint", "aabb"):
            if required not in node:
                raise ValueError(f"Part {node.get('id')} is missing {required}")
        if node["name"] not in SEMANTIC_LABELS:
            raise ValueError(
                f"Unsupported semantic label {node['name']!r}; "
                f"expected one of {sorted(SEMANTIC_LABELS)}"
            )
        if not re.fullmatch(r"[a-z0-9_.-]+", node["name"]):
            raise ValueError(f"Unsafe part name: {node['name']!r}")
        aabb = node["aabb"]
        for key in ("center", "size"):
            if not isinstance(aabb.get(key), list) or len(aabb[key]) != 3:
                raise ValueError(f"Part {node['id']} aabb.{key} must have 3 values")
        joint = node["joint"]
        if joint.get("type") not in JOINT_TYPES:
            raise ValueError(f"Part {node['id']} has unsupported joint type")
        if not isinstance(joint.get("range"), list) or len(joint["range"]) != 2:
            raise ValueError(f"Part {node['id']} joint.range must have 2 values")
        axis = joint.get("axis", {})
        for key in ("direction", "origin"):
            if not isinstance(axis.get(key), list) or len(axis[key]) != 3:
                raise ValueError(
                    f"Part {node['id']} joint.axis.{key} must have 3 values"
                )
        if not node["objs"]:
            raise ValueError(f"Part {node['id']} has no geometry assets")
        for relative in node["objs"]:
            asset = (object_dir / relative).resolve()
            try:
                asset.relative_to(object_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Part asset escapes its object directory: {relative}"
                ) from exc
            if not asset.is_file():
                raise FileNotFoundError(f"Missing part asset: {asset}")
            if asset.suffix.lower() != ".obj":
                raise ValueError(f"Condition rendering currently requires OBJ: {asset}")
            asset_count += 1
    return {"parts": len(nodes), "assets": asset_count}


def _safe_link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Conflicting symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing staged asset: {destination}"
        )
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _object_id(input_root: Path, source: Path) -> str:
    relative = source.relative_to(input_root) if source != input_root else Path(source.name)
    components = relative.parts
    if not components or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in components):
        raise ValueError(f"Unsafe object path relative to input root: {relative}")
    return "__".join(components)


def stage_object(
    source: Path,
    output_root: Path,
    annotation: dict[str, Any],
    object_id: str,
) -> Path:
    destination = output_root / object_id
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.name == "object_merge_fixed.json":
            continue
        # These directories are pipeline outputs. Linking them would make a
        # renderer write through the symlink into the raw dataset.
        if child.name not in {PREPROCESS_DIR, "imgs"}:
            _safe_link(child, target)
    annotation_path = destination / "object_merge_fixed.json"
    serialized = json.dumps(annotation, indent=2, ensure_ascii=False) + "\n"
    if annotation_path.exists():
        if annotation_path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"Derived annotation changed; use a fresh output root: {annotation_path}"
            )
    else:
        annotation_path.write_text(serialized, encoding="utf-8")
    return destination


def _views(count: int, seed: int) -> list[dict[str, float]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    offset = rng.random(2)
    views = []
    golden = (1 + math.sqrt(5)) / 2
    for index in range(count):
        u = ((index + 0.5) / count + offset[0]) % 1.0
        v = (index / golden + offset[1]) % 1.0
        pitch = math.asin(2 * u - 1)
        yaw = 2 * math.pi * v
        views.append(
            {"yaw": yaw, "pitch": pitch, "radius": 2.0, "fov": math.radians(40)}
        )
    return views


def _has_visible_foreground(path: Path) -> bool:
    """Reject renderer outputs that are present but fully transparent/black."""
    if not path.is_file():
        return False
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A")
    return alpha.getbbox() is not None


class Preprocessor:
    def __init__(
        self,
        config: dict[str, Any],
        input_root: Path,
        output_root: Path,
        stages: list[str],
        skip_invalid: bool,
    ):
        self.config = config
        self.input_root = input_root.resolve()
        self.output_root = output_root.resolve()
        self.stages = stages
        self.skip_invalid = skip_invalid
        self.annotation_name = config.get("annotation_name", "object_merge_fixed.json")
        self.source_annotation_name = config.get("source_annotation_name", "object.json")
        self.merge_fixed = bool(config.get("merge_fixed_parts", True))
        self.records: list[dict[str, Any]] = []
        self.staged: list[Path] = []

    def run(self) -> None:
        if (
            self.input_root == self.output_root
            or self.output_root.is_relative_to(self.input_root)
            or self.input_root.is_relative_to(self.output_root)
        ):
            raise ValueError("Input and output roots must be disjoint directories")
        if (
            int(self.config.get("conditioning_views", 20)) < 20
            and not self.config.get("smoke_test", False)
        ):
            raise ValueError("PAct training requires at least 20 conditioning views")
        if int(self.config.get("voxel_resolution", 64)) != 64:
            raise ValueError(
                "The released PAct latent encoders require voxel_resolution=64"
            )
        objects = discover_objects(
            self.input_root, (self.annotation_name, self.source_annotation_name)
        )
        if not objects:
            raise ValueError(
                f"No {self.annotation_name} or {self.source_annotation_name} objects "
                f"found under {self.input_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        for source in objects:
            object_id = _object_id(self.input_root, source)
            record = {"id": object_id, "source": str(source), "valid": False}
            try:
                annotation_path = source / self.annotation_name
                if annotation_path.is_file():
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                    removed: list[int] = []
                    annotation_source = self.annotation_name
                else:
                    source_path = source / self.source_annotation_name
                    annotation = json.loads(source_path.read_text(encoding="utf-8"))
                    annotation_source = self.source_annotation_name
                    if not self.merge_fixed:
                        raise ValueError(
                            f"{source_path} requires merge_fixed_parts=true because "
                            f"{self.annotation_name} is absent"
                        )
                    annotation, removed = merge_fixed_parts(annotation)
                record.update(
                    validate_object(
                        source,
                        annotation,
                        source / annotation_source,
                        int(self.config.get("max_parts", 8)),
                    )
                )
                record["annotation_source"] = annotation_source
                record["merged_fixed_part_ids"] = removed
                record["valid"] = True
                self.staged.append(
                    stage_object(source, self.output_root, annotation, object_id)
                )
            except Exception as exc:
                record["error"] = str(exc)
                LOGGER.error("Invalid object %s: %s", source, exc)
                if not self.skip_invalid:
                    self.records.append(record)
                    self._write_manifest()
                    raise
            self.records.append(record)
        LOGGER.info("Validated %d object(s)", len(self.staged))

        for stage in self.stages:
            if stage in {"validate", "merge_fixed", "stage"}:
                continue
            LOGGER.info("Starting stage: %s", stage)
            getattr(self, f"run_{stage}")()
        self._write_manifest()

    def run_render_conditioning(self) -> None:
        worker = Path(__file__).parent / "blender" / "render_conditioning.py"
        for index, object_dir in enumerate(self.staged):
            marker = object_dir / "imgs" / "_SUCCESS"
            if marker.is_file():
                continue
            command = [
                self.config.get("blender", "blender"),
                "--background",
                "--python-exit-code",
                "2",
                "--python",
                str(worker),
                "--",
                "--data",
                str(object_dir),
                "--num-views",
                str(self.config.get("conditioning_views", 20)),
                "--resolution",
                str(self.config.get("image_resolution", 512)),
                "--seed",
                str(int(self.config.get("seed", 42)) + index),
                "--engine",
                self.config.get("conditioning_render_engine", "CYCLES"),
                "--samples",
                str(int(self.config.get("conditioning_samples", 64))),
            ]
            self._run(command, object_dir, "render_conditioning")

    def run_render_geometry(self) -> None:
        worker = Path(__file__).parent / "blender" / "render_geometry.py"
        views = json.dumps(
            _views(
                int(self.config.get("geometry_views", 150)),
                int(self.config.get("seed", 42)),
            )
        )
        for object_dir in self.staged:
            output = object_dir / PREPROCESS_DIR / RENDER_DIR
            annotation = json.loads(
                (object_dir / "object_merge_fixed.json").read_text(encoding="utf-8")
            )
            names = ["full"] + [
                f"part_{int(node['id'])}_{node['name']}" for node in _nodes(annotation)
            ]
            expected = [
                output / name / filename
                for name in names
                for filename in ("transforms.json", "mesh.ply")
            ]
            previews = [output / name / "000.png" for name in names]
            if all(path.is_file() for path in expected) and all(
                _has_visible_foreground(path) for path in previews
            ):
                continue
            if all(path.is_file() for path in expected):
                LOGGER.warning(
                    "Existing geometry render is blank or incomplete; rerendering %s",
                    object_dir,
                )
            command = [
                self.config.get("blender", "blender"),
                "--background",
                "--python-exit-code",
                "2",
                "--python",
                str(worker),
                "--",
                "--views",
                views,
                "--object",
                str(object_dir / "object_merge_fixed.json"),
                "--output_folder",
                str(output),
                "--resolution",
                str(self.config.get("image_resolution", 512)),
                "--engine",
                self.config.get("render_engine", "CYCLES"),
                "--cycles-samples",
                str(self.config.get("cycles_samples", 128)),
                "--save_mesh",
            ]
            if self.config.get("cycles_device_type"):
                command.extend(
                    ["--cycles-device-type", self.config["cycles_device_type"]]
                )
            if self.config.get("cycles_devices"):
                command.extend(["--cycles-devices", self.config["cycles_devices"]])
            self._run(command, object_dir, "render_geometry")
            invalid = [path for path in previews if not _has_visible_foreground(path)]
            if invalid:
                raise RuntimeError(f"Geometry renderer produced blank images: {invalid}")

    def run_voxelize(self) -> None:
        resolution = int(self.config.get("voxel_resolution", 64))
        for object_dir in self.staged:
            render_root = object_dir / PREPROCESS_DIR / RENDER_DIR
            meshes = sorted(render_root.glob("*/mesh.ply"))
            if not meshes:
                raise FileNotFoundError(f"No rendered meshes found under {render_root}")
            for mesh in meshes:
                destination = (
                    object_dir
                    / PREPROCESS_DIR
                    / VOXEL_DIR
                    / mesh.parent.name
                    / "mesh.ply"
                )
                if destination.is_file():
                    continue
                count = voxelize_mesh(mesh, destination, resolution)
                LOGGER.info("Voxelized %s (%d cells)", mesh, count)

    def _encoder(self) -> LatentEncoder:
        if not hasattr(self, "encoder"):
            self.encoder = LatentEncoder(
                device=self.config.get("device", "cuda"),
                feature_model=self.config.get("feature_model", "dinov2_vitl14_reg"),
                ss_encoder=self.config.get(
                    "ss_encoder",
                    "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
                ),
                slat_encoder=self.config.get(
                    "slat_encoder",
                    "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16",
                ),
                batch_size=int(self.config.get("encoder_batch_size", 16)),
            )
        return self.encoder

    def run_encode_ss(self) -> None:
        encoder = self._encoder()
        for object_dir in self.staged:
            voxel_root = object_dir / PREPROCESS_DIR / VOXEL_DIR
            voxels = sorted(voxel_root.glob("*/mesh.ply"))
            if not voxels:
                raise FileNotFoundError(f"No voxel meshes found under {voxel_root}")
            for voxel in voxels:
                destination = (
                    object_dir
                    / PREPROCESS_DIR
                    / "ss_latents"
                    / encoder.ss_encoder_name
                    / f"{voxel.parent.name}.npz"
                )
                if not destination.is_file():
                    encoder.encode_ss(voxel, destination)
        encoder.release("ss_encoder")

    def run_extract_features(self) -> None:
        encoder = self._encoder()
        for object_dir in self.staged:
            voxel_root = object_dir / PREPROCESS_DIR / VOXEL_DIR
            render_root = object_dir / PREPROCESS_DIR / RENDER_DIR
            voxels = sorted(voxel_root.glob("*/mesh.ply"))
            if not voxels:
                raise FileNotFoundError(f"No voxel meshes found under {voxel_root}")
            for voxel in voxels:
                destination = (
                    object_dir
                    / PREPROCESS_DIR
                    / "features"
                    / encoder.feature_model_name
                    / f"{voxel.parent.name}.npz"
                )
                if not destination.is_file():
                    encoder.extract_features(
                        render_root / voxel.parent.name, voxel, destination
                    )
        encoder.release("feature_model")

    def run_encode_slat(self) -> None:
        encoder = self._encoder()
        for object_dir in self.staged:
            feature_root = (
                object_dir / PREPROCESS_DIR / "features" / encoder.feature_model_name
            )
            features = sorted(feature_root.glob("*.npz"))
            if not features:
                raise FileNotFoundError(f"No DINO features found under {feature_root}")
            for feature in features:
                destination = (
                    object_dir
                    / PREPROCESS_DIR
                    / "latents"
                    / encoder.slat_encoder_name
                    / feature.name
                )
                if not destination.is_file():
                    encoder.encode_slat(feature, destination)
        encoder.release("slat_encoder")

    def run_index(self) -> None:
        entries = []
        for object_dir in self.staged:
            preprocess = object_dir / PREPROCESS_DIR
            entry = {
                "id": object_dir.name,
                "path": str(object_dir),
                "conditioning": (object_dir / "imgs" / "_SUCCESS").is_file(),
                "geometry": (
                    preprocess / RENDER_DIR / "full" / "transforms.json"
                ).is_file()
                and _has_visible_foreground(
                    preprocess / RENDER_DIR / "full" / "000.png"
                ),
                "voxels": (preprocess / VOXEL_DIR / "full" / "mesh.ply").is_file(),
            }
            entries.append(entry)
        index_path = self.output_root / "dataset_index.jsonl"
        index_path.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
            encoding="utf-8",
        )
        tags = [f"{self.output_root.name}/{entry['id']}" for entry in entries]
        random.Random(int(self.config.get("seed", 42))).shuffle(tags)
        validation_fraction = float(self.config.get("validation_fraction", 0.05))
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        validation_count = max(1, round(len(tags) * validation_fraction))
        validation_count = min(validation_count, max(0, len(tags) - 1))
        if len(tags) == 1:
            split = {"train": tags, "test": tags}
        else:
            split = {
                "train": tags[validation_count:],
                "test": tags[:validation_count],
            }
        (self.output_root / "data_split.json").write_text(
            json.dumps(split, indent=2) + "\n", encoding="utf-8"
        )

    def _run(self, command: list[str], object_dir: Path, stage: str) -> None:
        LOGGER.info("[%s] %s", stage, object_dir.name)
        log_dir = self.output_root / "logs"
        log_dir.mkdir(exist_ok=True)
        with (log_dir / f"{object_dir.name}.{stage}.log").open(
            "w", encoding="utf-8"
        ) as log:
            environment = os.environ.copy()
            if self.config.get("clear_blender_ld_library_path", True) and stage in {
                "render_conditioning",
                "render_geometry",
            }:
                # Conda's libstdc++ can shadow the one bundled by Blender and
                # make Embree fail during process startup.
                environment.pop("LD_LIBRARY_PATH", None)
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )

    def _write_manifest(self) -> None:
        (self.output_root / "preprocessing_manifest.json").write_text(
            json.dumps(
                {
                    "input_root": str(self.input_root),
                    "output_root": str(self.output_root),
                    "stages": self.stages,
                    "records": self.records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
