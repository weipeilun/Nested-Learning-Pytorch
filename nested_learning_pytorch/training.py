from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf

from .memory.assoc_memory import AssocMemSpec
from .model import HOPEModel, ModelConfig


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


def create_assoc_mem_spec(config_dict: dict) -> AssocMemSpec:
    """Create an AssocMemSpec from a config dict, putting unknown fields in extra_params.
    
    Args:
        config_dict: Configuration dictionary from YAML
        
    Returns:
        AssocMemSpec with standard fields and extra_params populated
    """
    # Define the standard fields that AssocMemSpec accepts
    standard_fields = {
        'name', 'type', 'update_period', 'warmup_steps', 'jitter',
        'inner_lr', 'outer_lr', 'hidden_multiplier', 'children_blocks'
    }
    
    # Separate standard fields from extra fields
    standard_params = {}
    extra_params = {}
    
    for key, value in config_dict.items():
        if key in standard_fields:
            # Handle nested children_blocks recursively
            if key == 'children_blocks' and value:
                standard_params[key] = [create_assoc_mem_spec(child) for child in value]
            else:
                standard_params[key] = value
        else:
            extra_params[key] = value
    
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
    blocks = [create_assoc_mem_spec(entry) for entry in model_cfg.blocks]
    hope_cfg = ModelConfig(
        num_tokens=model_cfg.num_tokens,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        blocks=blocks,
        optimizers=optimizer_cfg,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        surprise_threshold=model_cfg.get("surprise_threshold", None),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )
    return HOPEModel(hope_cfg)
