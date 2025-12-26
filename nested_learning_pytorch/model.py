from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

from .memory.hope.model import HOPEBlock, HOPEBlockConfig
from .memory.assoc_memory import AssocMemState, AssocMemSpec
from .manager import FrequencyManager

from .utils import pack_one_with_inverse, safe_cat


@dataclass
class ModelConfig:
    num_tokens: int
    dim: int
    num_layers: int
    blocks: Sequence[AssocMemSpec]
    optimizers: Dict[str, dict] | None = None
    gradient_checkpointing: bool = False
    surprise_threshold: float | None = None
    freeze_backbone: bool = False
    is_training: bool = True


class HOPEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self._surprise_threshold = config.surprise_threshold
        self._allowed_update_blocks: set[str] | None = None
        block_config = HOPEBlockConfig(
            dim=config.dim,
            blocks=config.blocks,
            optimizer_configs=config.optimizers or {},
        )
        self.blocks = nn.ModuleList([HOPEBlock(f'hope_{i}', block_config) for i in range(config.num_layers)])
            
        # update frequency manager for hierarchical blocks
        self.freq_manager = FrequencyManager(self.blocks)
        
        # classes input
        self.to_inner = nn.Linear(config.num_tokens, config.dim)

        # classes output
        self.norm = nn.LayerNorm(config.dim)
        self.to_logits = nn.Linear(config.dim, config.num_tokens)
            
        # whether to update the fast_weights of the blocks
        # this is actually the switch to obtain information from the environment in realtime, making it a "streaming information system" or a traditional AI model
        self.inner_training = config.is_training

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self._surprise_threshold = threshold
        for block in self.blocks:
            block.set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_allowed_update_blocks(self, blocks: set[str] | None) -> None:
        self._allowed_update_blocks = blocks.copy() if blocks is not None else None
        for block in self.blocks:
            block.set_allowed_blocks(self._allowed_update_blocks)

    def get_allowed_update_blocks(self) -> set[str] | None:
        return None if self._allowed_update_blocks is None else self._allowed_update_blocks.copy()
    
    def check_identical_step(self, state: dict[str, AssocMemState]) -> bool:
        """Check that all step values in all batch items are identical across all states.
        
        Args:
            state: Dictionary mapping block names to their states
            
        Returns:
            True if all steps are identical
            
        Raises:
            ValueError: If step values are not identical across batch or blocks
        """
        if not state:
            return True
        
        # Collect all step tensors from the state dictionary
        step_tensors = []
        last_update_step_tensors = []
        block_names = []
        
        for block_name, block_state in state.items():
            if block_state.step is not None and isinstance(block_state.step, torch.Tensor):
                step_tensors.append(block_state.step)
                last_update_step_tensors.append(block_state.last_update_step)
                block_names.append(block_name)
        
        if not step_tensors:
            return True   
             
        # Check that all last_update_step within each tensor are identical (across batch dimension)
        for block_name, last_update_step_tensor in zip(block_names, last_update_step_tensors):
            if last_update_step_tensor.numel() > 1:
                first_last_update_step = last_update_step_tensor[0]
                if not torch.all(last_update_step_tensor == first_last_update_step):
                    raise ValueError(
                        f"Block '{block_name}': last_update_step values are not identical across batch. "
                        f"Found values: {last_update_step_tensor.tolist()}"
                    )
        
        # Check that all blocks and batchs have the same step value
        step = step_tensors[0][0].item()
        for block_name, step_tensor in zip(block_names, step_tensors):
            if not torch.all(step_tensor == step):
                raise ValueError(
                    f"Step mismatch: block '{block_names[0]}' has step {step}, "
                    f"but block '{block_name}' has step {step_tensor.tolist()}"
                )
        
        return True
        
    def forward_inner_batch(self, x: torch.Tensor, state: dict[str, AssocMemState], start_step: int, y: torch.Tensor | None = None, pbar: tqdm | None = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        # Create progress bar on first call
        close_pbar = False
        if pbar is None and start_step == 0:
            pbar = tqdm(total=seq_len, desc="Processing sequence", unit="step")
            close_pbar = True
        
        if start_step >= seq_len:
            if close_pbar and pbar is not None:
                pbar.close()
            return None
        
        next_update_step = self.freq_manager.next_update_step(state)
        
        if next_update_step > seq_len:
            seq_end_step = seq_len
        else:
            seq_end_step = next_update_step
        
        seq = x[:, start_step:seq_end_step, :]
        for block in self.blocks:
            seq, state = block(x=seq, state=state)
            
        seq = self.norm(seq)
        logits = self.to_logits(seq)
        
        # calculate gradients for each block
        for block in self.blocks:
            if self.inner_training:
                block.cal_inner_grads(logits=logits, state=state, y=y[:, start_step:seq_end_step, :])
            else:
                block.cal_inner_grads(logits=logits, state=state, y=logits)
        
        # update fast_weights if needed
        for block in self.blocks:
            block.inner_update(state=state)
        
        # Update progress bar
        if pbar is not None:
            pbar.update(seq_end_step - start_step)
        
        result = safe_cat([logits, self.forward_inner_batch(x, state, seq_end_step, y=y, pbar=pbar)], dim=1)
        
        # Close progress bar after final recursion
        if close_pbar and pbar is not None:
            pbar.close()
        
        return result

    def forward(
        self,
        x: torch.Tensor,
        state: dict[str, AssocMemState] | None = None,
        *,
        y: torch.Tensor | None = None,
    ) -> dict[str, AssocMemState]:
        # pack into batch dimension
        
        x, inverse_pack = pack_one_with_inverse(x, '* l d')
        
        if self.inner_training:
            assert y is not None, "y is required when inner_training is True"
            y, _ = pack_one_with_inverse(y, '* l d')
        
        # handle previous state init

        if state is None:
            for block in self.blocks:
                state = block.init_state(state=state, batch_size=x.shape[0])
        
        self.check_identical_step(state)
        
        x = self.to_inner(x)
            
        logits = self.forward_inner_batch(x, state, 0, y=y)
        
        logits = inverse_pack(logits)
        
        return logits, state
    
    def update(self, logits: torch.Tensor, state: dict[str, AssocMemState], y: torch.Tensor) -> None:
        for block in self.blocks:
            block.cal_outer_grads(logits=logits, state=state, y=y)
            
        for block in self.blocks:
            block.optimize()

    def _gather_block_stats(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "pop_update_stats"):
                stats = block.pop_update_stats()
                for block_name, payload in stats.items():
                    prefix = f"layer{idx}.{block_name}"
                    for key, value in payload.items():
                        metrics[f"{prefix}.{key}"] = value
        return metrics

    def pop_update_metrics(self) -> Dict[str, float]:
        metrics = self._latest_update_metrics
        self._latest_update_metrics = {}
        return metrics

    def freeze_backbone(self) -> None:
        """
        Freeze the shared transformer spine (embeddings, attention blocks, norm, LM head).
        HOPE/TITAN/CMS memories remain trainable for adapter-style finetuning.
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for block in self.blocks:
            if hasattr(block, "attn"):
                for p in block.attn.parameters():
                    p.requires_grad = False
