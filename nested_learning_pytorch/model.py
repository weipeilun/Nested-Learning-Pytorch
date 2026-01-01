from __future__ import annotations

from dataclasses import dataclass
from turtle import write_docstringdict
import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm
from .memory.hope.model import HOPEBlock, HOPEBlockConfig
from .memory.assoc_memory import AssocMemState, AssocMemSpec, AssocMemory
from .manager import FrequencyManager, rebuild_state

from .utils import pack_one_with_inverse, safe_cat
from torch.func import vmap, grad
from tensordict import TensorDict

@dataclass
class ModelConfig:
    num_tokens: int
    dim: int
    num_layers: int
    blocks: Sequence[AssocMemSpec]
    optimizers: Dict[str, dict] | None = None
    gradient_checkpointing: bool = False
    freeze_backbone: bool = False
    is_training: bool = True
    inner_loss_fn: str = 'mse'
    outer_loss_fn: str = 'mse'


class HOPEModel(nn.Module):
    
    def __init__(self, config: ModelConfig, inner_loss_fn: nn.Module, outer_loss_fn: nn.Module):
        super().__init__()
        self.config = config
        self.inner_loss_fn = inner_loss_fn
        self.outer_loss_fn = outer_loss_fn
        self.gradient_checkpointing = config.gradient_checkpointing
        block_config = HOPEBlockConfig(
            dim=config.dim,
            blocks=config.blocks,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            optimizer_configs=config.optimizers or {},
        )
        self.blocks = nn.ModuleList([HOPEBlock(f'hope_{i}', block_config) for i in range(config.num_layers)])
            
        # update frequency manager for hierarchical blocks
        self.freq_manager = FrequencyManager(self.blocks)
        
        # classes input
        self.to_inner = nn.Linear(config.num_tokens, config.dim)

        # classes output
        # self.norm = nn.LayerNorm(config.dim)
        self.to_logits = nn.Linear(config.dim, config.num_tokens)
            
        # whether to update the fast_weights of the blocks
        # this is actually the switch to obtain information from the environment in realtime, making it a "streaming information system" or a traditional AI model
        self.inner_training = config.is_training
        self.inner_training = config.is_training
        
        # Inner loop vmap.
        
        self.per_task_forward_cal_grads = grad(self.forward_inner_chunk, has_aux = True)
        
        # Outer loop vmap.
        
        outer_grad_fn = grad(self.cal_outer_loss, has_aux = True)
        
        self.per_task_update_cal_grads = vmap(outer_grad_fn, in_dims = (0, 0, 0, 0, None, None, None, None))
    
    def forward_inner_chunk(self,
                            grad_weight_values_list: list[torch.Tensor],
                            non_grad_weight_values_list: list[torch.Tensor],
                            x: torch.Tensor,
                            y: torch.Tensor,
                            grad_weight_keys_list: list[str],
                            non_grad_weight_keys_list: list[str],
                            step_need_update_dict: dict[str, bool]
                            ) -> torch.Tensor:
        forward_chunk_state = rebuild_state(grad_weight_values_list, non_grad_weight_values_list, grad_weight_keys_list, non_grad_weight_keys_list, weights_keys=['fast_weights'])
            
        # Update fast_weights if needed.
        # Since we are in a vmap, in order to make the grad() grasp all the gradients it needed, step t - 1's update must be put to step t's start.
        for block in self.blocks:
            block.maybe_inner_update(state=forward_chunk_state, step_need_update_dict=step_need_update_dict)
        
        seq = x
        for block in self.blocks:
            seq, forward_chunk_state = block(x=seq, state=forward_chunk_state)
        
        logits = self.to_logits(seq)
        
        loss = self.inner_loss_fn(logits, y)
        return loss, (logits, forward_chunk_state)
    
    def cal_outer_loss(self, 
                       grad_weight_values_list: list[torch.Tensor],
                       non_grad_weight_values_list: list[torch.Tensor],
                       x: torch.Tensor,
                       y: torch.Tensor,
                       grad_weight_keys_list: list[str],
                       non_grad_weight_keys_list: list[str],
                       chunk_sizes: list[int],
                       step_need_update_dict_list: list[dict[str, bool]]
                       ) -> torch.Tensor:
        state = rebuild_state(grad_weight_values_list, non_grad_weight_values_list, grad_weight_keys_list, non_grad_weight_keys_list, weights_keys=['weights', 'fast_weights'])
        
        x = self.to_inner(x)
        
        logits_list = []
        current_step = 0
        
        # This is the inner loop of meta learning.
        # In section 8.1 (https://abehrouz.github.io/files/NL.pdf): "Note that, again, the initial states of all memories, i.e., M□,0
        # for any □ ∈ {𝒌, 𝒗, 𝒒, 𝜂, 𝛼, memory} are meta-learned across all sequences/contexts, and so are optimized in the higher
        # levels (or outer-loop)."
        pbar = tqdm(
            zip(chunk_sizes, step_need_update_dict_list, strict=True),
            total=len(chunk_sizes),
            leave=False
        )
        for chunk_size, step_need_update_dict in pbar:
            seq_start_step = current_step
            seq_end_step = seq_start_step + chunk_size
            seq = x[seq_start_step:seq_end_step, :]
            target = y[seq_start_step:seq_end_step, :]
            
            pbar.set_description(f"seq_len {seq_start_step}")
            
            grad_weight_keys_list: list[str] = []
            grad_weight_values_list: list[torch.Tensor] = []
            for block in self.blocks:
                grad_weight_keys, grad_weight_values = block.get_calcuable_weights(state=state, parameter_weight_key='fast_weights')
                grad_weight_keys_list.extend(grad_weight_keys)
                grad_weight_values_list.extend(grad_weight_values)
            
            non_grad_weight_keys_list: list[str] = []
            non_grad_weight_values_list: list[torch.Tensor] = []
            for block in self.blocks:
                non_grad_weight_keys, non_grad_weight_values = block.get_inner_non_calcuable_fast_weights(state=state)
                non_grad_weight_keys_list.extend(non_grad_weight_keys)
                non_grad_weight_values_list.extend(non_grad_weight_values)
            
            # A higher order gradient is needed in inner loop. grad() can handle it.
            grads, (logits, updated_state) = self.per_task_forward_cal_grads(grad_weight_values_list, non_grad_weight_values_list, seq, target, grad_weight_keys_list, non_grad_weight_keys_list, step_need_update_dict)
            
            state = self.update_state(state, updated_state)
            
            # Collect the fast weights gradients
            block_grad_dict = {}
            for grad_weight_keys, grad_weight_value in zip(grad_weight_keys_list, grads, strict=True):
                block_name, grad_key = grad_weight_keys.split(AssocMemory.DEFAULT_GRADIENT_KEY_SPLITTER)
                if block_name not in block_grad_dict:
                    block_grad_dict[block_name] = {}
                block_grad_dict[block_name][grad_key] = grad_weight_value
                
            # Cache the inner gradients in case blocks' inner update frequency is different
            for block in self.blocks:
                block.cache_inner_grads(state=state, block_grads_dict=block_grad_dict)
                
            # Accumulate results
            logits_list.append(logits)
        
        logits_tensor = safe_cat(logits_list, dim=0)
        loss = self.outer_loss_fn(logits_tensor, y)
        
        return loss, state
    
    def forward_inner_loop(self, x: torch.Tensor, y: torch.Tensor, state: dict[str, AssocMemState] | None = None) -> torch.Tensor:
        
        # pack into batch dimension
        
        x, inverse_pack = pack_one_with_inverse(x, '* l d')
        
        batch_size, sequence_len = x.shape[:2]
        
        if self.inner_training:
            assert y is not None, "y is required when inner_training is True"
            y, _ = pack_one_with_inverse(y, '* l d')
        
        # handle previous state init

        if state is None:
            for block in self.blocks:
                state = block.init_state(state=state, batch_size=batch_size)
        
        self.check_identical_step(state)
        
        chunk_sizes = self.freq_manager.get_all_chuck_sizes(state, sequence_len)
        
        step_need_update_dict_list = self.if_step_need_update(state=state, chunk_sizes=chunk_sizes)
        
        # Use vmap to forward each task in batch dimension.
        
        grad_weight_keys_list: list[str] = []
        grad_weight_values_list: list[torch.Tensor] = []
        for block in self.blocks:
            grad_weight_keys, grad_weight_values = block.get_calcuable_weights(state=state, parameter_weight_key='weights')
            grad_weight_keys_list.extend(grad_weight_keys)
            grad_weight_values_list.extend(grad_weight_values)
        
        non_grad_weight_keys_list: list[str] = []
        non_grad_weight_values_list: list[torch.Tensor] = []
        for block in self.blocks:
            non_grad_weight_keys, non_grad_weight_values = block.get_outer_non_calcuable_fast_weights(state=state)
            non_grad_weight_keys_list.extend(non_grad_weight_keys)
            non_grad_weight_values_list.extend(non_grad_weight_values)
        
        # vmap is used to compute gradients for each task in batch dimension. It's curical to keep the gradient computation right and efficient.
        outer_grads, state = self.per_task_update_cal_grads(
            grad_weight_values_list,
            non_grad_weight_values_list,
            x,
            y,
            grad_weight_keys_list,
            non_grad_weight_keys_list,
            chunk_sizes,
            step_need_update_dict_list
        )
        
        outer_grad_dict = {grad_weight_key: outer_grad for grad_weight_key, outer_grad in zip(grad_weight_keys_list, outer_grads, strict=True)}
        
        return outer_grad_dict, state
    
    def outer_update(self, grads_dict: dict[str, torch.Tensor]) -> None:
        for block in self.blocks:
            block.outer_update(grads_dict=grads_dict)
    
    def check_identical_step(self, state: TensorDict[str, TensorDict]) -> bool:
        """Check that all step values in all batch items are identical across all states.
        
        Args:
            state: Dictionary mapping block names to their states
            
        Returns:
            True if all steps are identical
            
        Raises:
            ValueError: If step values are not identical across batch or blocks
        """
        if len(state) == 0:
            return True
        
        # Collect all step tensors from the state dictionary
        step_tensors = []
        last_update_step_tensors = []
        block_names = []
        
        for block_name, block_state in state.items():
            if 'step' in block_state and isinstance(block_state['step'], torch.Tensor):
                step_tensors.append(block_state['step'])
                last_update_step_tensors.append(block_state['last_update_step'])
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
    
    def update_state(self, state: dict[str, torch.Tensor | dict], updated_state: dict[str, torch.Tensor | dict]) -> None:
        """Recursively update state with values from updated_state.
        
        Args:
            state: The state dictionary to be updated (modified in-place)
            updated_state: The dictionary containing new values to update
        """
        new_state = dict()
        
        # First pass: process all keys from updated_state
        for key, updated_value in updated_state.items():
            if isinstance(updated_value, dict):
                # For dict, recursively create the merged state
                if key in state and isinstance(state[key], dict):
                    # If state also has this key as dict, merge them
                    value_new_state = self.update_state(state[key], updated_value)
                    new_state[key] = value_new_state
                else:
                    # If state doesn't have it or it's not a dict, use updated_value
                    new_state[key] = updated_value
            else:
                # For Tensor or other types, directly add from updated_state
                new_state[key] = updated_value
        
        # Second pass: add keys from state that are not in updated_state
        for key, value in state.items():
            if key not in new_state:
                new_state[key] = value
        
        return new_state

    def if_step_need_update(self, state: dict[str, AssocMemState], chunk_sizes: list[int]) -> dict[str, bool]:
        # The update of step t is placed in step t + 1's start, so we need to generate the step_need_update_dict for step t - 1 in advance.
        step_need_update_dict_list = [{}]
        start_step = 0
        for i in range(len(chunk_sizes) - 1):
            chunk_size = chunk_sizes[i]
            end_step = start_step + chunk_size
            step_need_update_dict = {}
            for block_name in state.keys():
                if '_optimizer' not in block_name:
                    target_update_step = math.ceil(end_step / self.freq_manager.frequency_map[block_name]) * self.freq_manager.frequency_map[block_name]
                    if start_step < target_update_step <= end_step:
                        step_need_update_dict[block_name] = True
                    else:
                        step_need_update_dict[block_name] = False
            step_need_update_dict_list.append(step_need_update_dict)
            start_step = end_step
        return step_need_update_dict_list
