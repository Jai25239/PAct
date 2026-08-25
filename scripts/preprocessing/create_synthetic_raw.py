#!/usr/bin/env python3
"""Create a tiny articulated raw-data fixture for preprocessing smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OBJ_TEMPLATE = """v {x0} 0 0
v {x1} 0 0
v {x0} 0.2 0
v {x0} 0 0.2
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    object_dir = args.output_root.expanduser().resolve() / "synthetic_object"
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "base.obj").write_text(
        OBJ_TEMPLATE.format(x0=-0.3, x1=-0.1), encoding="utf-8"
    )
    (object_dir / "door.obj").write_text(
        OBJ_TEMPLATE.format(x0=0.1, x1=0.3), encoding="utf-8"
    )
    annotation = {
        "diffuse_tree": [
            {
                "id": 0,
                "parent": -1,
                "name": "base",
                "objs": ["base.obj"],
                "aabb": {"center": [-0.2, 0.1, 0.1], "size": [0.2, 0.2, 0.2]},
                "joint": {
                    "type": "fixed",
                    "range": [0.0, 0.0],
                    "axis": {"direction": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0]},
                },
            },
            {
                "id": 1,
                "parent": 0,
                "name": "door",
                "objs": ["door.obj"],
                "aabb": {"center": [0.2, 0.1, 0.1], "size": [0.2, 0.2, 0.2]},
                "joint": {
                    "type": "revolute",
                    "range": [0.0, 90.0],
                    "axis": {"direction": [0.0, 0.0, 1.0], "origin": [0.1, 0.0, 0.0]},
                },
            },
        ]
    }
    (object_dir / "object_merge_fixed.json").write_text(
        json.dumps(annotation, indent=2) + "\n", encoding="utf-8"
    )
    print(object_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
