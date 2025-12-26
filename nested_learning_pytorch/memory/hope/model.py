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

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_allowed_blocks(self, allowed: Set[str] | None) -> None:
        self.allowed_blocks = allowed.copy() if allowed is not None else None

    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        block_name = self.config.titan_block.name
        if not self._is_block_allowed("titan"):
            return
        if not self.block_manager.should_update(block_name):
            return
        if not self._passes_surprise(surprise_value):
            self._record_gate(block_name, hit=False)
            return
        # Use full sequence for granular updates (Critique P1)
        # Note: We intentionally do not pool over dim=1 (sequence) here.
        # teach_signal is (B, T, D), attn_out is (B, T, D)
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        # context_vec is still pooled for the optimizer interface which expects a vector/low-rank hint
        context_vec = attn_out.detach().mean(dim=(0, 1))
        
        with torch.enable_grad():
            query = attn_out.detach().requires_grad_(True)
            target = (teach_signal.detach() + modifier).detach() # modifier is now (B, T, D)
            prediction = self.titan_memory(query)
            
            # Granular Masking (Critique P1)
            loss = F.mse_loss(prediction, target, reduction='none')
            if self.surprise_threshold is not None:
                # Mask out tokens where surprise < threshold
                with torch.no_grad():
                     norms = teach_signal.norm(dim=-1, keepdim=True)
                     mask = (norms >= self.surprise_threshold).float()
                loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                loss = loss.mean()
        
        magnitude = self.block_manager.optimize(block_name, self.titan_memory, loss, context=context_vec)
        extra_metrics = self.block_manager.pop_last_metrics(block_name)
        stats = {"grad_norm": magnitude, "gate_hit": 1.0}
        if surprise_value is not None:
            stats["surprise_value"] = surprise_value
        stats.update(extra_metrics)
        self.last_update_stats[f"titan.{block_name}"] = stats

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        delta = teach_signal.detach().mean(dim=1, keepdim=True)
        error_signals = {spec.name: delta for spec in self.config.cms_blocks}
        chunk_map = self.cms.accumulate_chunks(
            inputs=cms_inputs,
            outputs=cms_outputs,
            error_signals=error_signals,
        )
        for spec in self.config.cms_blocks:
            block_name = spec.name
            if not self._is_block_allowed(block_name):
                continue
            if not self.block_manager.should_update(block_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(block_name, hit=False)
                continue
            chunk = chunk_map.get(block_name)
            if chunk is None:
                continue
            chunk_inputs, chunk_targets, chunk_mask = chunk
            with torch.enable_grad():
                chunk_inp = chunk_inputs.detach().requires_grad_(True)
                prediction = self.cms.blocks[block_name](chunk_inp)
                diff = prediction - chunk_targets
                diff_sq = diff.pow(2)
                mask_expanded = chunk_mask.unsqueeze(-1).expand_as(diff_sq)
                loss = (diff_sq * mask_expanded).sum() / mask_expanded.sum().clamp(min=1.0)
            context_vec = chunk_inputs.mean(dim=(0, 1))
            magnitude = self.block_manager.optimize(
                block_name,
                self.cms.blocks[block_name],
                loss,
                context=context_vec,
            )
            extra_metrics = self.block_manager.pop_last_metrics(block_name)
            stats_payload = {
                "grad_norm": magnitude,
                "chunk_samples": float(chunk_inputs.shape[0]),
                "gate_hit": 1.0,
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            stats_payload.update(extra_metrics)
            self.last_update_stats[f"cms.{block_name}"] = stats_payload
            self.cms.consume_chunk(block_name)

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _is_block_allowed(self, block_name: str) -> bool:
        if self.allowed_blocks is None:
            return True
        return block_name in self.allowed_blocks or (
            block_name.startswith("titan") and "titan" in self.allowed_blocks
        )

    def _record_gate(self, block_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{block_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0
