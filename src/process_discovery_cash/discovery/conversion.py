from __future__ import annotations

from typing import Any


def to_petri_net(model: Any, model_type: str) -> tuple[Any | None, str | None]:
    """Best-effort conversion to a Petri net tuple for metric computation."""
    if model is None:
        return None, "No discovered model available"
    if isinstance(model, tuple) and len(model) == 3:
        return model, None

    if model_type in {"process_tree", "process_tree_or_petri_net"}:
        try:
            from pm4py.objects.conversion.process_tree import converter as pt_converter

            return pt_converter.apply(model), None
        except Exception as exc:
            return None, f"Could not convert process tree to Petri net: {exc}"

    if model_type == "bpmn":
        try:
            from pm4py.objects.conversion.bpmn import converter as bpmn_converter

            return bpmn_converter.apply(model), None
        except Exception as exc:
            return None, f"Could not convert BPMN model to Petri net: {exc}"

    return None, f"Unsupported model type for Petri net conversion: {model_type}"
