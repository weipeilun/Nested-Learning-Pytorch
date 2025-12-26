from __future__ import annotations

from typing import Any, Dict, Callable


def build_inner_optimizer(block_type: str, optimizer_configs: Dict[str, dict], params: Dict[str, Any] | None = None) -> Callable | tuple[Callable, Callable]:
    if "Titans" in block_type:
        from .builder import DeepMomentumGradientDesentBuilder
        return DeepMomentumGradientDesentBuilder(**params, **optimizer_configs["DeepMomentumGradientDesent"])
    elif "FFN" in block_type:
        from .builder import DeepMomentumGradientDesentBuilder
        return DeepMomentumGradientDesentBuilder(**params, **optimizer_configs["DeepMomentumGradientDesent"])
    else:
        raise ValueError(f"Unsupported block type {block_type}")


def build_outer_optimizer(block_type: str, optimizer_configs: Dict[str, dict], params: Dict[str, Any] | None = None) -> Any:
    from .builder import AdamWBuilder
    return AdamWBuilder(**optimizer_configs["AdamW"])
