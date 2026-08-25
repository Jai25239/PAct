"""Render normalized multi-view geometry used by PAct latent preprocessing.

This worker runs inside Blender. It renders the complete object and each part
with one shared normalization, then exports the normalized mesh and cameras.
Its camera/normalization/export conventions follow Microsoft TRELLIS
``dataset_toolkits/blender_script/render.py`` (MIT, TRELLIS@d7f8816), with the
PAct per-part traversal from the research code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.materials, bpy.data.textures, bpy.data.images):
        for item in list(collection):
            collection.remove(item)


def configure_render(
    engine: str,
    resolution: int,
    cycles_samples: int,
    cycles_device_type: str | None,
    cycles_devices: str | None,
) -> None:
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
    if engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 64
    elif engine == "CYCLES":
        cycles = scene.cycles
        cycles.samples = cycles_samples
        cycles.filter_type = "BOX"
        cycles.filter_width = 1
        cycles.diffuse_bounces = 1
        cycles.glossy_bounces = 1
        cycles.transparent_max_bounces = 3
        cycles.transmission_bounces = 3
        cycles.use_denoising = True
        selected_names = (
            {name.strip() for name in cycles_devices.split(",") if name.strip()}
            if cycles_devices
            else None
        )
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
        except KeyError:
            bpy.ops.preferences.addon_enable(module="cycles")
            preferences = bpy.context.preferences.addons["cycles"].preferences
        if cycles_device_type:
            try:
                preferences.compute_device_type = cycles_device_type
            except TypeError as exc:
                print(
                    f"[WARN] Cycles backend {cycles_device_type} is unavailable: {exc}",
                    file=sys.stderr,
                )
        preferences.get_devices()
        gpu_types = {"CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"}
        enabled = False
        for device in preferences.devices:
            use = device.type in gpu_types and (
                selected_names is None or device.name in selected_names
            )
            device.use = use
            enabled = enabled or use
        cycles.device = "GPU" if enabled else "CPU"
        if not enabled:
            print("[WARN] No Cycles GPU enabled; using CPU.", file=sys.stderr)


def load_meshes(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.lower() != ".obj":
            raise ValueError(f"PAct conditioning preprocessing requires OBJ: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    if not any(obj.type == "MESH" for obj in bpy.context.scene.objects):
        raise ValueError("No mesh objects were imported")


def root_object():
    roots = [obj for obj in bpy.context.scene.objects if not obj.parent]
    if len(roots) == 1:
        return roots[0]
    parent = bpy.data.objects.new("PActRoot", None)
    bpy.context.scene.collection.objects.link(parent)
    for obj in roots:
        obj.parent = parent
    return parent


def scene_bounds() -> tuple[Vector, Vector]:
    lower = Vector((math.inf, math.inf, math.inf))
    upper = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            lower = Vector(tuple(min(a, b) for a, b in zip(lower, world)))
            upper = Vector(tuple(max(a, b) for a, b in zip(upper, world)))
    if not found:
        raise ValueError("Cannot normalize a scene without meshes")
    return lower, upper


def normalize_scene(fixed: tuple[float, list[float]] | None = None):
    root = root_object()
    if fixed is None:
        lower, upper = scene_bounds()
        extent = max(upper - lower)
        if extent <= 0:
            raise ValueError("Object bounding box has zero extent")
        scale = 1.0 / extent
        root.scale *= scale
        bpy.context.view_layer.update()
        lower, upper = scene_bounds()
        offset = -(lower + upper) / 2
    else:
        scale, raw_offset = fixed
        offset = Vector(raw_offset)
        root.scale *= scale
        bpy.context.view_layer.update()
    root.matrix_world.translation += offset
    bpy.context.view_layer.update()
    return float(scale), [float(value) for value in offset]


def add_camera_and_lights():
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.sensor_width = 32
    camera.data.sensor_height = 32
    target = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(target)
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    constraint.target = target
    for kind, energy, location, scale in (
        ("POINT", 1000, (4, 1, 6), (1, 1, 1)),
        ("AREA", 10000, (0, 0, 10), (100, 100, 100)),
        ("AREA", 1000, (0, 0, -10), (1, 1, 1)),
    ):
        data = bpy.data.lights.new(f"PActLight{len(bpy.data.lights)}", type=kind)
        data.energy = energy
        light = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(light)
        light.location = location
        light.scale = scale
    return camera


def camera_matrix(camera) -> list[list[float]]:
    position, rotation, _ = camera.matrix_world.decompose()
    matrix = rotation.to_matrix()
    return [
        [matrix[row][0], matrix[row][1], matrix[row][2], position[row]]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def export_mesh(destination: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    if hasattr(bpy.ops.wm, "ply_export"):
        bpy.ops.wm.ply_export(
            filepath=str(destination), export_selected_objects=True
        )
    else:
        bpy.ops.export_mesh.ply(filepath=str(destination), use_selection=True)


def render_meshes(
    paths: list[Path],
    output: Path,
    views: list[dict[str, float]],
    engine: str,
    resolution: int,
    cycles_samples: int,
    cycles_device_type: str | None,
    cycles_devices: str | None,
    fixed_normalization: tuple[float, list[float]] | None = None,
) -> tuple[float, list[float]]:
    reset_scene()
    configure_render(
        engine, resolution, cycles_samples, cycles_device_type, cycles_devices
    )
    load_meshes(paths)
    scale, offset = normalize_scene(fixed_normalization)
    camera = add_camera_and_lights()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": offset,
        "frames": [],
    }
    for index, view in enumerate(views):
        camera.location = (
            view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["pitch"]),
        )
        camera.data.lens = 16 / np.tan(view["fov"] / 2)
        bpy.context.view_layer.update()
        filename = f"{index:03d}.png"
        bpy.context.scene.render.filepath = str(output / filename)
        bpy.ops.render.render(write_still=True)
        metadata["frames"].append(
            {
                "file_path": filename,
                "camera_angle_x": view["fov"],
                "transform_matrix": camera_matrix(camera),
            }
        )
    (output / "transforms.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    export_mesh(output / "mesh.ply")
    return scale, offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True, help="JSON camera-view list")
    parser.add_argument("--object", required=True, type=Path, help="PAct annotation")
    parser.add_argument("--output_folder", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--cycles-samples", type=int, default=128)
    parser.add_argument("--cycles-device-type")
    parser.add_argument("--cycles-devices")
    parser.add_argument("--save_mesh", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.save_mesh:
        raise ValueError("PAct preprocessing requires --save_mesh")
    annotation_path = args.object.resolve()
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    base = annotation_path.parent
    parts = annotation["diffuse_tree"]
    views = json.loads(args.views)
    all_paths = [base / relative for part in parts for relative in part["objs"]]
    normalization = render_meshes(
        all_paths,
        args.output_folder / "full",
        views,
        args.engine,
        args.resolution,
        args.cycles_samples,
        args.cycles_device_type,
        args.cycles_devices,
    )
    for part in parts:
        name = f"part_{int(part['id'])}_{part['name']}"
        render_meshes(
            [base / relative for relative in part["objs"]],
            args.output_folder / name,
            views,
            args.engine,
            args.resolution,
            args.cycles_samples,
            args.cycles_device_type,
            args.cycles_devices,
            normalization,
        )


if __name__ == "__main__":
    main()
