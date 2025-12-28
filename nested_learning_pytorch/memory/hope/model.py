from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..assoc_memory import AssocMemSpec, AssocMemState, AssocMemory

@dataclass
class HOPEBlockConfig:
    dim: int
    blocks: Sequence[AssocMemSpec]
    inner_loss_fn: nn.Module
    outer_loss_fn: nn.Module
    activation: str = "gelu"
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


class HOPEBlock(AssocMemory):
    def __init__(self, block_name: str, config: HOPEBlockConfig):
        super().__init__(
            block_name=block_name,
            chunk_size=None,
            inner_optimizer=None,
            outer_optimizer=None,
            dim=None,
            inner_lr=None,
            outer_lr=None,
            inner_loss_fn=config.inner_loss_fn,
            outer_loss_fn=config.outer_loss_fn,
            )
        
        # Initialize blocks using factory pattern based on AssocMemSpec.type
        
        self.init_children_blocks(config.dim, config.blocks, config.optimizer_configs)
        
        self.config = config

        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x, state: dict[str, AssocMemState] | None = None) -> tuple[torch.Tensor, dict[str, AssocMemState]]:
        logits = x
        child_state = state
        for child_block in self.children_blocks:
            logits, child_state = child_block(x=logits, state=child_state)
        return logits, child_state
