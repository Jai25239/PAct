"""PAct articulation-annotation preparation.

The fixed-part merge follows the project research utility
``PartArt-Gen/scripts/merge_fixed_parts.py``.  It is kept dependency-free here
so raw ``object.json`` annotations can be prepared before Blender or CUDA
dependencies are loaded.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Iterable


def _unique(values: Iterable[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _aabb_bounds(aabb: dict[str, Any]) -> tuple[list[float], list[float]]:
    center = [float(value) for value in aabb["center"]]
    size = [float(value) for value in aabb["size"]]
    return (
        [center[index] - size[index] / 2 for index in range(3)],
        [center[index] + size[index] / 2 for index in range(3)],
    )


def _union_aabb(left: dict[str, Any], right: dict[str, Any]) -> dict[str, list[float]]:
    left_min, left_max = _aabb_bounds(left)
    right_min, right_max = _aabb_bounds(right)
    lower = [min(left_min[index], right_min[index]) for index in range(3)]
    upper = [max(left_max[index], right_max[index]) for index in range(3)]
    return {
        "center": [(lower[index] + upper[index]) / 2 for index in range(3)],
        "size": [upper[index] - lower[index] for index in range(3)],
    }


def _depths(nodes: dict[int, dict[str, Any]]) -> dict[int, int]:
    depths: dict[int, int] = {}
    queue: deque[int] = deque()
    for node_id, node in nodes.items():
        if int(node.get("parent", -1)) == -1:
            depths[node_id] = 0
            queue.append(node_id)
    while queue:
        node_id = queue.popleft()
        for child_id in nodes[node_id].get("children", []) or []:
            child_id = int(child_id)
            if child_id not in nodes:
                raise ValueError(f"Part {node_id} references missing child {child_id}")
            if child_id in depths:
                raise ValueError("diffuse_tree contains a cycle or multiple parents")
            depths[child_id] = depths[node_id] + 1
            queue.append(child_id)
    if len(depths) != len(nodes):
        missing = sorted(set(nodes) - set(depths))
        raise ValueError(f"diffuse_tree contains unreachable parts: {missing}")
    return depths


def merge_fixed_parts(annotation: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    """Merge every non-root fixed node into its parent, deepest nodes first."""

    source_nodes = annotation.get("diffuse_tree", [])
    if not source_nodes:
        raise ValueError("Annotation has no diffuse_tree parts")
    nodes = {int(node["id"]): deepcopy(node) for node in source_nodes}
    if len(nodes) != len(source_nodes):
        raise ValueError("diffuse_tree contains duplicate part IDs")
    depths = _depths(nodes)
    removed: list[int] = []

    for node_id in sorted(nodes, key=lambda value: depths[value], reverse=True):
        if node_id in removed:
            continue
        # Read the live node: a deeper fixed child may already have been merged
        # into it earlier in this pass.
        node = nodes[node_id]
        parent_id = int(node.get("parent", -1))
        if (node.get("joint") or {}).get("type") != "fixed" or parent_id == -1:
            continue
        if parent_id not in nodes or parent_id in removed:
            raise ValueError(f"Cannot merge part {node_id}: missing parent {parent_id}")
        parent = nodes[parent_id]
        parent["objs"] = _unique(
            (parent.get("objs") or []) + (node.get("objs") or [])
        )
        parent["plys"] = _unique(
            (parent.get("plys") or []) + (node.get("plys") or [])
        )
        children = [
            int(child)
            for child in (parent.get("children") or [])
            if int(child) != node_id
        ]
        node_children = [int(child) for child in (node.get("children") or [])]
        parent["children"] = _unique(children + node_children)
        for child_id in node_children:
            nodes[child_id]["parent"] = parent_id
        if parent.get("aabb") and node.get("aabb"):
            parent["aabb"] = _union_aabb(parent["aabb"], node["aabb"])
        removed.append(node_id)

    merged = deepcopy(annotation)
    merged["diffuse_tree"] = [
        nodes[node_id] for node_id in sorted(nodes) if node_id not in removed
    ]
    if isinstance(merged.get("meta"), dict):
        merged["meta"]["n_diff_parts"] = len(merged["diffuse_tree"])
    return merged, sorted(removed)
