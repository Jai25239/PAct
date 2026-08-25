"""TRELLIS-compatible trainers used by the released PAct stages."""

import importlib

__attributes = {
    "BasicTrainer": "basic",
    "FlowMatchingTrainer": "flow_matching.flow_matching",
    "FlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "ImageConditionedFlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "PartBasedImageConditionedFlowMatchingCFGTrainer": "flow_matching.flow_matching",
    "SparseFlowMatchingTrainer": "flow_matching.sparse_flow_matching",
    "SparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
    "ImageConditionedSparseFlowMatchingCFGTrainer": "flow_matching.sparse_flow_matching",
    "ImageConditionedSparseFlowMatchingCFGTrainer_Articulation": "flow_matching.sparse_flow_matching",
}

__all__ = list(__attributes)


def __getattr__(name):
    if name not in __attributes:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(f".{__attributes[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
