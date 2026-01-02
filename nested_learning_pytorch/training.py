from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .memory.assoc_memory import AssocMemSpec
from .model import HOPEModel, ModelConfig


LOSS_REGISTRY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "mae": nn.L1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "nll": nn.NLLLoss,
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")
    return LOSS_REGISTRY[name](**kwargs)

@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device


def unwrap_config(cfg: DictConfig) -> DictConfig:
    """Hydra can wrap grouped configs (e.g., hope/pilot) under the group name."""
    if "model" in cfg:
        return cfg
    if "hope" in cfg:
        return cfg.hope
    if "ablations" in cfg:
        return cfg.ablations
    return cfg


def create_assoc_mem_spec(config_dict: dict, inner_loss_fn: nn.Module, outer_loss_fn: nn.Module) -> AssocMemSpec:
    """Create an AssocMemSpec from a config dict, putting unknown fields in extra_params.
    
    Args:
        config_dict: Configuration dictionary from YAML
        
    Returns:
        AssocMemSpec with standard fields and extra_params populated
    """
    # Define the standard fields that AssocMemSpec accepts
    standard_fields = {
        'name', 'type', 'update_period', 'warmup_steps', 'jitter',
        'inner_lr', 'outer_lr', 'lr_multiple', 'hidden_multiplier', 'children_blocks'
    }
    
    # Separate standard fields from extra fields
    standard_params = {}
    extra_params = {}
    
    for key, value in config_dict.items():
        if key in standard_fields:
            # Handle nested children_blocks recursively
            if key == 'children_blocks' and value:
                standard_params[key] = [create_assoc_mem_spec(child, inner_loss_fn, outer_loss_fn) for child in value]
            else:
                standard_params[key] = value
        else:
            extra_params[key] = value
    
    # add inner_loss_fn and outer_loss_fn to standard_params
    standard_params['inner_loss_fn'] = inner_loss_fn
    standard_params['outer_loss_fn'] = outer_loss_fn
    
    # Special handling for FullyAdaptiveTitans: provide default update_period
    block_type = config_dict.get('type', '')
    if block_type == 'FullyAdaptiveTitans' and 'update_period' not in standard_params:
        # Use the minimum of the two periods, or 0 if neither exists
        periods = []
        if 'update_period_titans' in extra_params:
            periods.append(extra_params['update_period_titans'])
        if 'update_period_adaptive' in extra_params:
            periods.append(extra_params['update_period_adaptive'])
        standard_params['update_period'] = min(periods) if periods else 0
    
    # Add extra_params if there are any
    if extra_params:
        standard_params['extra_params'] = extra_params
    
    return AssocMemSpec(**standard_params)


def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module:
    optimizer_cfg = {}
    if "optimizers" in model_cfg:
        optimizer_cfg = OmegaConf.to_container(model_cfg.optimizers, resolve=True) # type: ignore[arg-type]
    inner_loss_fn = get_loss(model_cfg.get("inner_loss_fn", "mse"))
    outer_loss_fn = get_loss(model_cfg.get("outer_loss_fn", "mse"))
    blocks = [create_assoc_mem_spec(entry, inner_loss_fn, outer_loss_fn) for entry in model_cfg.blocks]
    hope_cfg = ModelConfig(
        num_tokens=model_cfg.num_tokens,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        blocks=blocks,
        optimizers=optimizer_cfg,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        is_training=model_cfg.get("is_training", True),
    )
    return HOPEModel(hope_cfg, inner_loss_fn, outer_loss_fn)
