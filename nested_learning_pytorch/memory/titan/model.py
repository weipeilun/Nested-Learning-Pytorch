from __future__ import annotations
from math import e
from typing import Callable
import warnings
from dataclasses import dataclass
from functools import partial
from torch.func import vmap, grad

from tensordict import TensorDict
import torch
from torch import nn

from ..assoc_memory import AssocMemory, AssocMemState

from ..ffn.model import FFNBlock
from ...manager import rebuild_state_for_titans

from ...utils import pack_one_with_inverse, default, safe_cat, exists, repeat_dict_values

"""
ein notation:
b - batch
h - heads
bh - batch and heads
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
o - momentum orders
u - key / value updates - allowing a token to emit multiple key / values
"""

# Fully Additive Titans
# Section 8.1 of the paper Nested Learning
# This corresponds to formulas 79, 80, 81, and 82


class FullyAdditiveTitansBlock(AssocMemory):
    """Fully Adaptive Titans Block implementation with automatic registration."""
    
    # Register this class with type name "Memory"
    _type_name = "FullyAdaptiveTitans"
    
    # @dataclass
    # class TitansState(AssocMemState):
    #     k: torch.Tensor | None = None
    #     v: torch.Tensor | None = None
    #     eta: torch.Tensor | None = None
    #     alpha: torch.Tensor | None = None
    #     last_alpha_fast_weight: torch.Tensor | None = None
    #     last_eta_fast_weight: torch.Tensor | None = None
    #     last_q_fast_weight: torch.Tensor | None = None
    #     last_k_fast_weight: torch.Tensor | None = None
    #     last_v_fast_weight: torch.Tensor | None = None
    #     last_step_updated: bool = False
        
    def __init__(
        self,
        block_name: str,
        chunk_size_titans: int,
        chunk_size_adaptive: int,
        dim: int,
        inner_optimizer: Callable | tuple[Callable, Callable],
        outer_optimizer: Callable | tuple[Callable, Callable],
        inner_lr: float,
        outer_lr: float,
        inner_loss_fn: nn.Module,
        outer_loss_fn: nn.Module,
        pre_rmsnorm = True,
        max_grad_norm: float | None = None,
        spectral_norm_surprises = False,
        mem_model_norm_add_residual = True, # by default, layernorm output and add residual as proposed in TTT paper, but could be removed
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        )
    ):
        super().__init__(
            block_name=block_name,
            chunk_size=None,
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            dim=dim,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
        )
        
        # Store chunk sizes for the chunk_sizes property
        self.chunk_size_titans = chunk_size_titans
        self.chunk_size_adaptive = chunk_size_adaptive
        
        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        
        titans_optimizer = inner_optimizer.clone()
        titans_optimizer.kwargs['chunk_size'] = titans_optimizer.kwargs.pop('chunk_size_titans')
        titans_optimizer.kwargs.pop('chunk_size_adaptive')
        titans_block_name = f'{self.block_name}_titans'
        titans_optimizer.kwargs['block_name'] = f'{titans_block_name}_inner_optimizer'
        self.titans_memory = FFNBlock(
            block_name=titans_block_name,
            chunk_size=chunk_size_titans,
            dim=dim,
            inner_optimizer=titans_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            normalization=None,
            default_model_kwargs=default_model_kwargs,
        )
        
        q_optimizer = inner_optimizer.clone()
        q_optimizer.kwargs['chunk_size'] = q_optimizer.kwargs.pop('chunk_size_adaptive')
        q_optimizer.kwargs.pop('chunk_size_titans')
        q_block_name = f'{self.block_name}_q'
        q_optimizer.kwargs['block_name'] = f'{q_block_name}_inner_optimizer'
        q_kwargs = default_model_kwargs.copy()
        q_kwargs['depth'] = 1
        self.q_memory = FFNBlock(
            block_name=q_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=q_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            normalization=None,
            default_model_kwargs=q_kwargs,
        )
        
        k_optimizer = inner_optimizer.clone()
        k_optimizer.kwargs['chunk_size'] = k_optimizer.kwargs.pop('chunk_size_adaptive')
        k_optimizer.kwargs.pop('chunk_size_titans')
        k_block_name = f'{self.block_name}_k'
        k_optimizer.kwargs['block_name'] = f'{k_block_name}_inner_optimizer'
        k_kwargs = default_model_kwargs.copy()
        k_kwargs['depth'] = 1
        self.k_memory = FFNBlock(
            block_name=k_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=k_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            normalization=None,
            default_model_kwargs=k_kwargs,
        )
        
        v_optimizer = inner_optimizer.clone()
        v_optimizer.kwargs['chunk_size'] = v_optimizer.kwargs.pop('chunk_size_adaptive')
        v_optimizer.kwargs.pop('chunk_size_titans')
        v_block_name = f'{self.block_name}_v'
        v_optimizer.kwargs['block_name'] = f'{v_block_name}_inner_optimizer'
        v_kwargs = default_model_kwargs.copy()
        v_kwargs['depth'] = 1
        self.v_memory = FFNBlock(
            block_name=v_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=v_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            normalization=None,
            default_model_kwargs=v_kwargs,
        )
        
        eta_optimizer = inner_optimizer.clone()
        eta_optimizer.kwargs['chunk_size'] = eta_optimizer.kwargs.pop('chunk_size_adaptive')
        eta_optimizer.kwargs.pop('chunk_size_titans')
        eta_block_name = f'{self.block_name}_eta'
        eta_optimizer.kwargs['block_name'] = f'{eta_block_name}_inner_optimizer'
        eta_kwargs = default_model_kwargs.copy()
        eta_kwargs['out_dim'] = 1
        self.eta_memory = FFNBlock(
            block_name=eta_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=eta_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            is_multi_head=False,
            with_bias=True,
            normalization=None,
            default_model_kwargs=eta_kwargs,
        )
        
        alpha_optimizer = inner_optimizer.clone()
        alpha_optimizer.kwargs['chunk_size'] = alpha_optimizer.kwargs.pop('chunk_size_adaptive')
        alpha_optimizer.kwargs.pop('chunk_size_titans')
        alpha_block_name = f'{self.block_name}_alpha'
        alpha_optimizer.kwargs['block_name'] = f'{alpha_block_name}_inner_optimizer'
        alpha_kwargs = default_model_kwargs.copy()
        alpha_kwargs['out_dim'] = 1
        self.alpha_memory = FFNBlock(
            block_name=alpha_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=alpha_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            is_multi_head=False,
            with_bias=True,
            normalization=None,
            default_model_kwargs=alpha_kwargs,
        )
        
        # Initialize eta_alpha weights to produce alpha ≈ 0 and eta ≈ 0 initially
        self._initialize_eta_alpha_weights()

        self.model = None # will disable the normal update stages
        object.__setattr__(self, 'inner_optimizer', None)
        object.__setattr__(self, 'outer_optimizer', None)
        
        def titans_memory_loss_fn(titans_fast_weight_values, k, v, eta, titans_fast_weight_keys, titans_memory_block_name):
            titans_memory_state = rebuild_state_for_titans(titans_fast_weight_values, titans_fast_weight_keys, titans_memory_block_name)
            titans_logits, titans_memory_state = self.titans_memory(x=k, state=titans_memory_state, auto_update_step=False, pattern='s d, d o -> s o')
            weighted_titans_logits = titans_logits * eta.unsqueeze(-1)
            return self.inner_loss_fn(weighted_titans_logits, v)
        
        self._titans_grad_fn = grad(titans_memory_loss_fn)
        
    
    def _initialize_eta_alpha_weights(self):
        """
        Initialize eta and alpha model biases to produce desired initial outputs.
        
        - eta bias: negative values so sigmoid(logits) ≈ 0 initially
        - alpha bias: positive values so sigmoid(logits) ≈ 0.95 initially
        
        This ensures eta starts near zero (slow learning) while alpha starts
        near 1 (high retention), providing stable initialization.
        """
        with torch.no_grad():
            # Get the last layer bias of the memory model
            # The model may be wrapped in ResidualNorm, so check for that
            if hasattr(self.eta_memory.model, 'model'):
                # If wrapped in ResidualNorm
                last_eta_bias = self.eta_memory.model.model.biases[-1]
            else:
                # If not wrapped
                last_eta_bias = self.eta_memory.model.biases[-1]
            
            if hasattr(self.alpha_memory.model, 'model'):
                # If wrapped in ResidualNorm
                last_alpha_bias = self.alpha_memory.model.model.biases[-1]
            else:
                # If not wrapped
                last_alpha_bias = self.alpha_memory.model.biases[-1]
            
            # Initialize alpha bias to produce large positive outputs
            # Example: sigmoid(3) ≈ 0.95, sigmoid(5) ≈ 0.993
            # Using uniform distribution between -6.0 and -4.0 to get eta close to 0
            nn.init.uniform_(last_eta_bias, -6.0, -4.0)
            
            # Using uniform distribution between 4.0 and 6.0 to get alpha close to 1
            nn.init.uniform_(last_alpha_bias, 4.0, 6.0)
    
    @property
    def chunk_sizes(self) -> dict[str, int]:
        return {
            self.eta_memory.block_name: self.chunk_size_adaptive,
            self.alpha_memory.block_name: self.chunk_size_adaptive,
            self.q_memory.block_name: self.chunk_size_adaptive,
            self.k_memory.block_name: self.chunk_size_adaptive,
            self.v_memory.block_name: self.chunk_size_adaptive,
            self.titans_memory.block_name: self.chunk_size_titans,
        }
        
    def forward(
        self,
        x,
        state: dict[str, AssocMemState] | None = None,
    ):
        # scheduler handles memory updates, this only handles retrieve
        
        if self.children_blocks is not None:
            if len(self.children_blocks) > 1:
                raise ValueError(f"FFNBlock {self.block_name} has multiple children blocks, but only one is supported")
            elif len(self.children_blocks) > 0:
                x, state = self.children_blocks[0](x=x, state=state)
        
        # handle previous state init

        if state is None:
            state = self.init_state(state, batch_size=x.shape[0])
            
        x_normed = self.retrieve_norm(x)
        
        # Step 1: calculate all titans memories parameters.
            
        q, state = self.q_memory(x=x_normed, state=state)
        k, state = self.k_memory(x=x_normed, state=state)
        v, state = self.v_memory(x=x_normed, state=state)
        
        eta_logits, state = self.eta_memory(x=x_normed, state=state)
        eta = eta_logits.sigmoid().squeeze(dim=-1)
        
        alpha_logits, state = self.alpha_memory(x=x_normed, state=state)
        alpha = alpha_logits.sigmoid().squeeze(dim=-1).mean()
        
        # Step 2: calculate titans memory's loss and gradients.
        
        titans_fast_weight_keys = []
        titans_fast_weight_values = []
        for titans_fast_weight_name, titans_fast_weight_value in state[self.titans_memory.block_name]['fast_weights'].items():
            titans_fast_weight_keys.append(titans_fast_weight_name)
            titans_fast_weight_values.append(titans_fast_weight_value)
        
        # As you can see, this should be the fastest level in main gradient stream.
        titans_grads = self._titans_grad_fn(titans_fast_weight_values, k, v, eta, titans_fast_weight_keys, self.titans_memory.block_name)
        
        # Step 3: update titans memory's gradients using the optimizer.
        
        titans_grads_dict = {key: grad for key, grad in zip(titans_fast_weight_keys, titans_grads, strict=True)}
        
        # Here we introduce DGD with weight decay, which projects the momentums to the orthogonal space, hoping to keep the optimizer memorizing through a long context
        # Update the optimizer first then use it to optimize the Titans memory.
        # It's important to update the optimizer first, otherwise will lead to a suboptimal result - both instinctly and according to equation 50 in NL paper.
        
        # Step 3.1: apply gradients and one-step update preconditioner.
        # The goal is to eliminate the most obvious non-orthogonal components in the current batch/task.
        # We need t's preconditioner gradients rather than step t - 1's at step t + 1, so we must apply inner_optimizer's gradients before calling inner_optimizer.forward().
        updated = self.titans_memory.inner_optimizer.update_preconditioners(grads_dict=titans_grads_dict, state=state)
        assert updated, "Preconditioner update failed"
        
        # Step 3.2: update the momentums (fast weights) using inner optimizer.
        
        titans_grads_dict_optimized, state = self.titans_memory.inner_optimizer(x=titans_grads_dict, state=state, alpha=alpha)
        
        # Fix step update failure because titans memory is not called yet.
        
        state[self.titans_memory.inner_optimizer.block_name]['step'] = state[self.titans_memory.inner_optimizer.block_name]['step'] + x.shape[0]
        
        # Step 4: update titans memory's fast weights using the optimized gradients.
        # Equation 50 in NL paper.
        new_fast_weights = {}
        for fast_weight_key, fast_weight_value in state[self.titans_memory.block_name]['fast_weights'].items():
            momentum_key = f"{self.titans_memory.inner_optimizer.param_name_mapping[fast_weight_key]}.momentum"
            
            assert momentum_key in titans_grads_dict_optimized, f"Momentum key {momentum_key} not found in titans_grads_dict_optimized"
            
            new_fast_weights[fast_weight_key] = fast_weight_value + titans_grads_dict_optimized[momentum_key]
            
        state[self.titans_memory.block_name]['fast_weights'] = new_fast_weights
        
        # Step 5: query titans memory.
        
        titans_logits, state = self.titans_memory(x=q, state=state, pattern='s d, d o -> s o')

        return titans_logits + x, state
          
    def cache_inner_grads(self, state: dict[str, AssocMemState], block_grads_dict: dict[str, torch.Tensor]) -> None:
        self.q_memory.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
        self.k_memory.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
        self.v_memory.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
        self.eta_memory.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
        self.alpha_memory.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
        # Titans memory fast weights' gradients are not calculated in main gradients stream.
        self.titans_memory.inner_optimizer.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
    
    def maybe_inner_update(self, state: dict[str, AssocMemState], step_need_update_dict: dict[str, bool]) -> None:
        self.q_memory.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)
        self.k_memory.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)
        self.v_memory.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)
        self.alpha_memory.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)
        updated = self.eta_memory.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)
                
        return updated
    
    def outer_update(self, grads_dict: dict[str, torch.Tensor]) -> None:
        self.titans_memory.outer_update(grads_dict=grads_dict)
        self.q_memory.outer_update(grads_dict=grads_dict)
        self.k_memory.outer_update(grads_dict=grads_dict)
        self.v_memory.outer_update(grads_dict=grads_dict)
        self.eta_memory.outer_update(grads_dict=grads_dict)
        self.alpha_memory.outer_update(grads_dict=grads_dict)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.outer_update(grads_dict=grads_dict)
    
    def init_state(self, state: dict[str, AssocMemState] | None = None, batch_size: int | None = None) -> dict[str, AssocMemState]:
        if state is None:
            state = {}
            
        if self.q_memory.block_name not in state:
            self.q_memory.init_state(state=state, batch_size=batch_size)
            
        if self.k_memory.block_name not in state:
            self.k_memory.init_state(state=state, batch_size=batch_size)
            
        if self.v_memory.block_name not in state:
            self.v_memory.init_state(state=state, batch_size=batch_size)
            
        if self.eta_memory.block_name not in state:
            self.eta_memory.init_state(state=state, batch_size=batch_size)
            
        if self.alpha_memory.block_name not in state:
            self.alpha_memory.init_state(state=state, batch_size=batch_size)
            
        if self.titans_memory.block_name not in state:
            self.titans_memory.init_state(state=state, batch_size=batch_size)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.init_state(state=state, batch_size=batch_size)

        return state
    
    def get_calcuable_weights(self, state: dict[str, AssocMemState], parameter_weight_key: str) -> dict[str, torch.Tensor]:
        weight_keys: list[str] = []
        weight_values: list[torch.Tensor] = []
        
        # Add all memories' fast weights.
        memory_keys, memory_values = self.q_memory.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
        weight_keys.extend(memory_keys)
        weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.k_memory.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
        weight_keys.extend(memory_keys)
        weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.v_memory.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
        weight_keys.extend(memory_keys)
        weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.alpha_memory.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
        weight_keys.extend(memory_keys)
        weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.eta_memory.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
        weight_keys.extend(memory_keys)
        weight_values.extend(memory_values)
        
        if parameter_weight_key == 'weights':
            memory_keys, memory_values = self.titans_memory.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
            weight_keys.extend(memory_keys)
            weight_values.extend(memory_values)
        elif parameter_weight_key == 'fast_weights':
            # All titans memory parameters are updated in a deeper level, so ignore the meta learning's inner gradient flow.
            memory_keys, memory_values = self.titans_memory.inner_optimizer.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
            weight_keys.extend(memory_keys)
            weight_values.extend(memory_values)
        else:
            raise ValueError(f"Unsupported parameter weight key: {parameter_weight_key}")
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
                weight_keys.extend(child_block_keys)
                weight_values.extend(child_block_values)
        
        return weight_keys, weight_values
    
    def get_inner_non_calcuable_fast_weights(self, state: dict[str, AssocMemState]) -> dict[str, torch.Tensor]:
        fast_weight_keys: list[str] = []
        fast_weight_values: list[torch.Tensor] = []
        
        memory_keys, memory_values = self.q_memory.get_inner_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.k_memory.get_inner_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.v_memory.get_inner_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.alpha_memory.get_inner_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.eta_memory.get_inner_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.titans_memory.inner_optimizer.get_inner_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        # Add all titans memory parameters here, since they are updated in a deeper level.
        if self.titans_memory.block_name in state:
            self.add_non_calcuable_weights(state[self.titans_memory.block_name]['fast_weights'], self.titans_memory.block_name, 'fast_weights', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.titans_memory.block_name]['step'], self.titans_memory.block_name, 'step', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.titans_memory.block_name]['last_update_step'], self.titans_memory.block_name, 'last_update_step', fast_weight_keys, fast_weight_values)
            if 'updates' in state[self.titans_memory.block_name]:
                self.add_non_calcuable_weights(state[self.titans_memory.block_name]['updates'], self.titans_memory.block_name, 'updates', fast_weight_keys, fast_weight_values)
            if 'n_updates' in state[self.titans_memory.block_name]:
                self.add_non_calcuable_weights(state[self.titans_memory.block_name]['n_updates'], self.titans_memory.block_name, 'n_updates', fast_weight_keys, fast_weight_values)

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_inner_non_calcuable_fast_weights(state=state)
                fast_weight_keys.extend(child_block_keys)
                fast_weight_values.extend(child_block_values)
        
        return fast_weight_keys, fast_weight_values
    
    def get_outer_non_calcuable_fast_weights(self, state: dict[str, AssocMemState]) -> dict[str, torch.Tensor]:
        fast_weight_keys: list[str] = []
        fast_weight_values: list[torch.Tensor] = []
        
        memory_keys, memory_values = self.q_memory.get_outer_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.k_memory.get_outer_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.v_memory.get_outer_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.alpha_memory.get_outer_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        memory_keys, memory_values = self.eta_memory.get_outer_non_calcuable_fast_weights(state=state)
        fast_weight_keys.extend(memory_keys)
        fast_weight_values.extend(memory_values)
        
        # Add all titans memory parameters here, since they are updated by a deeper level.
        if self.titans_memory.block_name in state:
            self.add_non_calcuable_weights(state[self.titans_memory.block_name]['step'], self.titans_memory.block_name, 'step', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.titans_memory.block_name]['last_update_step'], self.titans_memory.block_name, 'last_update_step', fast_weight_keys, fast_weight_values)
            if 'updates' in state[self.titans_memory.block_name]:
                self.add_non_calcuable_weights(state[self.titans_memory.block_name]['updates'], self.titans_memory.block_name, 'updates', fast_weight_keys, fast_weight_values)
            if 'n_updates' in state[self.titans_memory.block_name]:
                self.add_non_calcuable_weights(state[self.titans_memory.block_name]['n_updates'], self.titans_memory.block_name, 'n_updates', fast_weight_keys, fast_weight_values)
        
        if self.titans_memory.inner_optimizer.block_name in state:
            self.add_non_calcuable_weights(state[self.titans_memory.inner_optimizer.block_name]['step'], self.titans_memory.inner_optimizer.block_name, 'step', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.titans_memory.inner_optimizer.block_name]['last_update_step'], self.titans_memory.inner_optimizer.block_name, 'last_update_step', fast_weight_keys, fast_weight_values)
            if 'updates' in state[self.titans_memory.inner_optimizer.block_name]:
                self.add_non_calcuable_weights(state[self.titans_memory.inner_optimizer.block_name]['updates'], self.titans_memory.inner_optimizer.block_name, 'updates', fast_weight_keys, fast_weight_values)
            if 'n_updates' in state[self.titans_memory.inner_optimizer.block_name]:
                self.add_non_calcuable_weights(state[self.titans_memory.inner_optimizer.block_name]['n_updates'], self.titans_memory.inner_optimizer.block_name, 'n_updates', fast_weight_keys, fast_weight_values)

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_outer_non_calcuable_fast_weights(state=state)
                fast_weight_keys.extend(child_block_keys)
                fast_weight_values.extend(child_block_values)
        
        return fast_weight_keys, fast_weight_values
