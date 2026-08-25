# PAct preprocessing provenance and format

This directory does not execute the research scripts verbatim. Those scripts
were PAct experiments layered on two upstream codebases and still contained
different directory names, machine paths, installer commands, and one-off
logging. The public pipeline was reconciled against:

- Microsoft TRELLIS commit `d7f8816f70fb7866abe6415aad74569169f93ade`
  (`dataset_toolkits/render.py`, `voxelize.py`, `extract_feature.py`,
  `encode_ss_latent.py`, and `encode_latent.py`), MIT License.
- SINGAPO commit `30c0ef8958e54257984f55e6dcd8dc770e2d0590`
  (`scripts/preprocess/0_render_input_img.py` and `render_script.py`), MIT
  License.
- The PAct adaptations in `PartArt-Gen@siggrah_asia_revision`, especially its
  per-part rendering, merged-fixed annotation, semantic-mask, and per-part
  latent layout changes.

The resulting mapping is:

| Public module | Upstream responsibility | PAct adaptation |
| --- | --- | --- |
| `annotation.py` | PAct research `merge_fixed_parts.py` | dependency-free bottom-up merge without modifying raw data |
| `blender/render_conditioning.py` | SINGAPO Blender-native camera, RGBA, and object-index render path | merged-fixed tree and PAct semantic-mask files |
| `blender/render_geometry.py` | TRELLIS normalization, cameras, mesh export | shared full-object normalization for every part |
| `ops.py::voxelize_mesh` | TRELLIS 64³ surface voxelization | per-part folder traversal |
| `ops.py::extract_features` | TRELLIS multi-view DINO projection | full and part feature files |
| `ops.py::encode_ss` | TRELLIS sparse-structure encoder | PAct Stage-1 per-part latents |
| `ops.py::encode_slat` | TRELLIS structured-latent encoder | PAct Stage-2 per-part latents |

The canonical folder is `trellis_part_preprocess` because that is the path
consumed by the released PAct datasets. The older research scripts sometimes
wrote `trellis_preprocess`; using them directly therefore does not produce a
complete training dataset for this repository.

Run the pipeline through `scripts/preprocess_data.py`. It recursively discovers
raw `object.json` files, merges non-root fixed nodes into their parents, writes
the derived `object_merge_fixed.json` only in the output tree, validates the
result, invokes the two renderer workers, creates both latent representations,
and writes `dataset_index.jsonl` and `data_split.json`. Raw folders may be
nested by category or dataset. Their relative path components are joined with
`__` to produce unique one-level object IDs for the existing PAct dataloader.
Completed outputs are detected and skipped on reruns.

The formal PAct preprocessing launcher overrides the legacy wrapper defaults
with TRELLIS' `CYCLES` engine. The public config therefore also defaults to
Cycles with the OPTIX backend; `CUDA_VISIBLE_DEVICES` limits which physical
GPUs Blender can enumerate, while `--cycles-device-type` and
`--cycles-devices` provide explicit backend and device-name overrides. The
geometry stage validates rendered alpha content in addition to file presence,
so a failed render cannot be recorded as complete merely because it emitted a
PNG and `transforms.json`.

For compatibility, the result remains named `render_merged_fixed_cycles`: the
research launcher appended `_eevee` unconditionally even when its command line
selected Cycles. This historical folder name describes the released dataset
layout, not the active render engine.

If a raw object already has `object_merge_fixed.json`, that annotation is used
as-is. The merge operation moves `objs`/`plys` to the parent, reparents children,
unions AABBs, removes the fixed child, and updates `meta.n_diff_parts`. It is the
cleaned public form of the merge implementation supplied with the PAct research
repository; it is not a SINGAPO or TRELLIS operation.

All Python commands are intended to run in the `PAct` conda environment.
Blender is an external executable and need not be installed into a second
Python environment. `--blender` accepts its executable path. Conda's
`LD_LIBRARY_PATH` is removed only for Blender subprocesses to avoid shadowing
Blender's bundled C++ libraries. The public worker uses SINGAPO's tracked
Blender-native renderer rather than depending on BlenderProc's separately
downloaded Blender runtime.
