"""Render PAct conditioning images and semantic masks inside Blender.

This is the PAct-specific, dependency-light form of SINGAPO's Blender-native
``scripts/preprocess/render_script_4.py`` (MIT, SINGAPO@30c0ef8). It consumes
the merged annotation, preserves its numeric part IDs, and writes exactly the
RGBA/NPZ representation loaded by the released PAct datasets.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.materials, bpy.data.textures, bpy.data.images):
        for item in list(collection):
            collection.remove(item)


def configure_render(engine: str, resolution: int, samples: int) -> None:
    scene = bpy.context.scene
    if engine == "BLENDER_EEVEE" and bpy.app.version >= (4, 2, 0):
        engine = "BLENDER_EEVEE_NEXT"
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True


def import_obj(path: Path) -> list[bpy.types.Object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    return [obj for obj in set(bpy.data.objects) - before if obj.type == "MESH"]


def load_parts(annotation: dict, object_dir: Path) -> None:
    loaded = 0
    for node in annotation["diffuse_tree"]:
        semantic_id = int(node["id"])
        for relative in node["objs"]:
            for obj in import_obj((object_dir / relative).resolve()):
                # Part ID 0 is valid. The dataset loader distinguishes it from
                # background with the rendered RGBA alpha mask before adding 1.
                obj.pass_index = semantic_id
                obj["category_id"] = semantic_id
                loaded += 1
    if not loaded:
        raise ValueError("No mesh objects were loaded")
    for material in bpy.data.materials:
        material.use_backface_culling = False


def setup_camera_and_lighting():
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.lens_unit = "FOV"
    camera.data.angle = math.radians(40)
    target = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(target)
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    constraint.target = target

    area_data = bpy.data.lights.new("PActArea", type="AREA")
    area_data.energy = 30000
    area = bpy.data.objects.new("PActArea", area_data)
    bpy.context.scene.collection.objects.link(area)
    area.location = (0, 0, 1.3)
    area.scale = (100, 100, 100)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = 0.2
    return camera


def setup_index_output(mask_dir: Path):
    scene = bpy.context.scene
    scene.use_nodes = True
    bpy.context.view_layer.use_pass_object_index = True
    bpy.context.view_layer.update()
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(mask_dir)
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    index_socket = next(
        (socket for socket in layers.outputs if socket.name == "IndexOB"), None
    )
    if index_socket is None or index_socket.is_unavailable:
        available = [socket.name for socket in layers.outputs]
        raise RuntimeError(f"Blender object-index pass is unavailable: {available}")
    scene.node_tree.links.new(index_socket, output.inputs[0])
    return output


def camera_views(count: int, seed: int):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    phi_segments = np.linspace(math.pi / 3, math.pi / 2, count + 1)
    theta_segments = np.linspace(-5 * math.pi / 6, -math.pi / 6, count + 1)
    phis = [rng.uniform(phi_segments[i], phi_segments[i + 1]) for i in range(count)]
    thetas = [
        rng.uniform(theta_segments[i], theta_segments[i + 1]) for i in range(count)
    ]
    random.shuffle(phis)
    random.shuffle(thetas)
    for phi, theta in zip(phis, thetas):
        radius = rng.uniform(3.0, 3.5)
        yield (
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi),
        )


def read_index_pass(path: Path, resolution: int) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        if (width, height) != (resolution, resolution):
            raise ValueError(f"Unexpected semantic-mask size: {(width, height)}")
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        return np.rint(np.flipud(pixels[..., 0])).astype(np.int32)
    finally:
        bpy.data.images.remove(image)


def render(
    object_dir: Path,
    num_views: int,
    resolution: int,
    seed: int,
    engine: str,
    samples: int,
) -> None:
    annotation_path = object_dir / "object_merge_fixed.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_dir = object_dir / "imgs"
    mask_dir = image_dir / "semantic_masks_merge_fixed"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    configure_render(engine, resolution, samples)
    load_parts(annotation, object_dir)
    camera = setup_camera_and_lighting()
    index_output = setup_index_output(mask_dir)
    for index, location in enumerate(camera_views(num_views, seed)):
        camera.location = location
        bpy.context.view_layer.update()
        stem = f"{index:02d}"
        bpy.context.scene.render.filepath = str(image_dir / f"{stem}.png")
        index_output.file_slots[0].path = f"{stem}_index_"
        bpy.ops.render.render(write_still=True)
        matches = glob.glob(str(mask_dir / f"{stem}_index_*.exr"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one semantic EXR for view {stem}: {matches}")
        intermediate = Path(matches[0])
        semantic_mask = read_index_pass(intermediate, resolution)
        intermediate.unlink()
        np.savez_compressed(mask_dir / f"{stem}.npz", semantic_mask=semantic_mask)
    (image_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--num-views", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--samples", type=int, default=64)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    render(
        args.data.resolve(),
        args.num_views,
        args.resolution,
        args.seed,
        args.engine,
        args.samples,
    )


if __name__ == "__main__":
    main()
