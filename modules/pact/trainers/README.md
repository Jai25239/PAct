# PAct trainer lineage

The public training path intentionally preserves the TRELLIS trainer lifecycle
and inheritance structure. PAct adds only the two paper-stage trainers at the
end of that hierarchy:

```text
Trainer
└── BasicTrainer
    └── FlowMatchingTrainer
        ├── FlowMatchingCFGTrainer
        │   └── PartBasedImageConditionedFlowMatchingCFGTrainer (Stage 1)
        └── SparseFlowMatchingTrainer
            └── SparseFlowMatchingCFGTrainer
                └── ImageConditionedSparseFlowMatchingCFGTrainer_Articulation (Stage 2)
```

The base classes, optimizer/EMA/checkpoint lifecycle, resumable sampler,
mixed-precision behavior, CFG mixin, and image-conditioning mixin are adapted
from Microsoft TRELLIS at commit
`d7f8816f70fb7866abe6415aad74569169f93ade` (MIT License). The two leaf trainers
come from the formal PAct training path in
`PartArt-Gen@siggrah_asia_revision`, with package names changed from
`modules.part_synthesis` to `modules.pact`.

The experimental `GRPOFinetuningTrainer` and the versioned `_ditHead` trainer
are deliberately not registered or copied because neither is referenced by the
released Stage-1/Stage-2 configs.
