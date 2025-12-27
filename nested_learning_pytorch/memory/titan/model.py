from __future__ import annotations
from math import e
from typing import Callable
import warnings
from dataclasses import dataclass
from functools import partial

from tensordict import TensorDict
import torch
from torch import nn
from torch.nn import Linear, Module, Parameter
from torch.func import functional_call

from ..assoc_memory import AssocMemory, AssocMemState

from ..ffn.model import FFNBlock

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


def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)


class FullyAdditiveTitansBlock(AssocMemory):
    """Fully Adaptive Titans Block implementation with automatic registration."""
    
    # Register this class with type name "Memory"
    _type_name = "FullyAdaptiveTitans"
    
    @dataclass
    class TitansState(AssocMemState):
        k: torch.Tensor | None = None
        v: torch.Tensor | None = None
        eta: torch.Tensor | None = None
        alpha: torch.Tensor | None = None
        last_alpha_fast_weight: torch.Tensor | None = None
        last_step_updated: bool = False
        
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
        inner_loss_fn: Callable = default_loss_fn,
        outer_loss_fn: Callable = default_loss_fn,
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
            memory_state_clz=self.TitansState,
            default_model_kwargs=default_model_kwargs,
        )
        
        k_q_v_optimizer = inner_optimizer.clone()
        k_q_v_optimizer.kwargs['chunk_size'] = k_q_v_optimizer.kwargs.pop('chunk_size_adaptive')
        k_q_v_optimizer.kwargs.pop('chunk_size_titans')
        k_q_v_block_name = f'{self.block_name}_k_q_v'
        k_q_v_optimizer.kwargs['block_name'] = f'{k_q_v_block_name}_inner_optimizer'
        k_q_v_kwargs = default_model_kwargs.copy()
        k_q_v_kwargs['depth'] = 1
        self.k_q_v_memories = FFNBlock(
            block_name=k_q_v_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=k_q_v_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            is_multi_head=True,
            heads=3,
            default_model_kwargs=k_q_v_kwargs,
        )
        
        eta_optimizer = inner_optimizer.clone()
        eta_optimizer.kwargs['chunk_size'] = eta_optimizer.kwargs.pop('chunk_size_adaptive')
        eta_optimizer.kwargs.pop('chunk_size_titans')
        eta_block_name = f'{self.block_name}_eta'
        eta_optimizer.kwargs['block_name'] = f'{eta_block_name}_inner_optimizer'
        eta_kwargs = default_model_kwargs.copy()
        eta_kwargs['depth'] = 1
        eta_kwargs['out_dim'] = 1
        self.eta_memory = FFNBlock(
            block_name=eta_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=eta_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            is_multi_head=False,
            with_bias=True,
            normalization='pre_norm_only',
            default_model_kwargs=eta_kwargs,
        )
        
        alpha_optimizer = inner_optimizer.clone()
        alpha_optimizer.kwargs['chunk_size'] = alpha_optimizer.kwargs.pop('chunk_size_adaptive')
        alpha_optimizer.kwargs.pop('chunk_size_titans')
        alpha_block_name = f'{self.block_name}_alpha'
        alpha_optimizer.kwargs['block_name'] = f'{alpha_block_name}_inner_optimizer'
        alpha_kwargs = default_model_kwargs.copy()
        alpha_kwargs['depth'] = 1
        alpha_kwargs['out_dim'] = 1
        self.alpha_memory = FFNBlock(
            block_name=alpha_block_name,
            chunk_size=chunk_size_adaptive,
            dim=dim,
            inner_optimizer=alpha_optimizer,
            outer_optimizer=outer_optimizer,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            is_multi_head=False,
            with_bias=True,
            normalization='pre_norm_only',
            default_model_kwargs=alpha_kwargs,
        )
        
        # Initialize eta_alpha weights to produce alpha ≈ 0 and eta ≈ 0 initially
        self._initialize_eta_alpha_weights()

        self.model = None # will disable the normal update stages
        object.__setattr__(self, 'inner_optimizer', None)
        object.__setattr__(self, 'outer_optimizer', None)
    
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
            self.k_q_v_memories.block_name: self.chunk_size_adaptive,
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
        
        # pack into batch dimension
        
        x, inverse_pack = pack_one_with_inverse(x, '* l d')
        
        # handle previous state init

        if state is None:
            state = self.init_state(state, batch_size=x.shape[0])
            
        k_q_v_logits, state = self.k_q_v_memories(x=x, state=state)
        
        k, v, q = k_q_v_logits.unbind(dim=1)
        
        eta_logits, state = self.eta_memory(x=x, state=state)
        eta = eta_logits.sigmoid().squeeze(dim=-1)
        
        alpha_logits, state = self.alpha_memory(x=x, state=state)
        alpha = alpha_logits.sigmoid().squeeze(dim=-1)
        
        titans_logits, state = self.titans_memory(x=q, state=state)
        
        # record k, v, eta, alpha for update
        titans_state = state[self.titans_memory.block_name]
        titans_state.k = safe_cat([titans_state.k, k], dim=1)
        titans_state.v = safe_cat([titans_state.v, v], dim=1)
        titans_state.eta = safe_cat([titans_state.eta, eta], dim=1)
        titans_state.alpha = safe_cat([titans_state.alpha, alpha], dim=1)
        
        titans_logits = inverse_pack(titans_logits, '* l d')

        return titans_logits, state
    
    def init_state(self, state: dict[str, AssocMemState] | None = None, batch_size: int | None = None) -> dict[str, AssocMemState]:
        if state is None:
            state = {}
        
        batch_size = default(batch_size, 1)
        
        if self.k_q_v_memories.block_name not in state:
            self.k_q_v_memories.init_state(state=state, batch_size=batch_size)
            
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
    
    # Note: Titans memory must be the fastest weight; gradients can only be passed to adaptive memories after Titans memory is updated.
    # So we update fast weights directly in this function, no need to cache the gradients into `updates`.
    def cal_inner_grads(self, logits: torch.Tensor, state: dict[str, AssocMemState], y: torch.Tensor, losses: torch.Tensor | None = None) -> None:
        titans_state = state[self.titans_memory.block_name]
        k = titans_state.k
        v = titans_state.v
        eta = titans_state.eta      # learning rate
        alpha = titans_state.alpha.mean(dim=-1)  # retention gate (mean over sequence)
        titans_fast_weights = titans_state.fast_weights
        
        with torch.enable_grad():
            titans_logits, state = self.titans_memory(x=k, state=state, auto_update_step=False)
            titans_losses = self.inner_loss_fn(titans_logits, v)
            
            # The subscript of eta is t. It's the learning rate of each token.
            titans_weighted_losses = titans_losses * eta
            
            titans_fast_weight_keys = []
            titans_fast_weight_values = []
            for titans_fast_weight_name, titans_fast_weight_value in titans_fast_weights.items():
                titans_fast_weight_keys.append(titans_fast_weight_name)
                titans_fast_weight_values.append(titans_fast_weight_value)
            
            # Compute gradients with respect to fast_weights
            # We need to compute the second order gradients in higher levels, so we need to set retain_graph and create_graph to True
            # In section 8.1 (https://abehrouz.github.io/files/NL.pdf): "Note that, again, the initial states of all memories, i.e., M□,0
            # for any □ ∈ {𝒌, 𝒗, 𝒒, 𝜂, 𝛼, memory} are meta-learned across all sequences/contexts, and so are optimized in the higher
            # levels (or outer-loop)."
            # We sum the batch dimension of the loss to avoid the division by the number of tasks, which is not expected in meta learning.
            titans_grads = torch.autograd.grad(
                outputs=titans_weighted_losses.mean(dim=1).sum(dim=0),
                inputs=titans_fast_weight_values,
                retain_graph=True,
                create_graph=True,
            )
            
            # print(f"----------------{self.titans_memory.block_name}----------------")
            # for weight_key, weight_grad in zip(titans_fast_weight_keys.keys(), titans_grads, strict=True):
            #     print(f"Weight {weight_key} gradient: {weight_grad.norm().item()}")
            
            # Here we introduce DGD with weight decay, which projects the momentums to the orthogonal space, hoping to keep the optimizer memorizing through a long context
            titans_grads_dict = TensorDict({key: grad for key, grad in zip(titans_fast_weight_keys, titans_grads, strict=True)})
            
            # Update the optimizer first then use it to optimize the Titans memory.
            # It's important to update the optimizer first, otherwise will lead to a suboptimal result.
            # Both instinctly and according to equation 50 in NL paper.
            self.titans_memory.inner_optimizer.cal_inner_grads_update(grads_dict=titans_grads_dict, state=state, alpha=alpha)

            titans_grads_dict_optimized, state = self.titans_memory.inner_optimizer(x=titans_grads_dict, state=state)
            
            # print(f"----------------{self.titans_memory.block_name}----------------")
            # for weight_key, weight_grad in titans_grads_dict_optimized.items():
            #     print(f"Weight {weight_key} gradient: {weight_grad.norm().item()}")
            
            new_titans_fast_weights = {}
            for (titans_fast_weight_key, titans_grad_optimized), titans_fast_weight_value in zip(titans_grads_dict_optimized.items(), titans_fast_weight_values, strict=True):
                new_titans_fast_weights[titans_fast_weight_key] = titans_fast_weight_value + titans_grad_optimized * (-self.inner_lr)
            
            titans_state.fast_weights = new_titans_fast_weights
            
            titans_state.k = None
            titans_state.v = None
            titans_state.eta = None
            titans_state.alpha = None
            titans_state.last_update_step = titans_state.step.clone()
            state[self.titans_memory.inner_optimizer.block_name].last_update_step = titans_state.step.clone()
        
            self.k_q_v_memories.cal_inner_grads(logits=titans_logits, state=state, y=v, losses=titans_weighted_losses)
            self.eta_memory.cal_inner_grads(logits=titans_logits, state=state, y=v, losses=titans_weighted_losses)
            # The alpha gradient can only be computed using the fast weights at step t-1.
            if titans_state.last_alpha_fast_weight is not None:
                # We do a little hack here.
                current_step_alpha_fast_weight = state[self.alpha_memory.block_name].fast_weights
                state[self.alpha_memory.block_name].fast_weights = titans_state.last_alpha_fast_weight
                self.alpha_memory.cal_inner_grads(logits=titans_logits, state=state, y=v, losses=titans_weighted_losses)
                state[self.alpha_memory.block_name].fast_weights = current_step_alpha_fast_weight
                if titans_state.last_step_updated:
                    titans_state.last_alpha_fast_weight = current_step_alpha_fast_weight
            else:
                titans_state.last_alpha_fast_weight = state[self.alpha_memory.block_name].fast_weights
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.cal_inner_grads(logits=logits, state=state, y=y)
                
    def inner_update(self, state: dict[str, AssocMemState]) -> None:
        self.k_q_v_memories.inner_update(state=state)
        self.eta_memory.inner_update(state=state)
        is_updated = self.alpha_memory.inner_update(state=state)
        state[self.titans_memory.block_name].last_step_updated = is_updated

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.inner_update(state=state)
                
    def cal_outer_grads(self, logits: torch.Tensor, state: dict[str, AssocMemState], y: torch.Tensor) -> None:
        self.titans_memory.cal_outer_grads(logits=logits, state=state, y=y)
        
        # adaptive memories' gradients are only available after the first update of the titans memory
        titans_chunk_size = self.titans_memory.chunk_size
        adaptive_logits = logits[:, titans_chunk_size:, :]
        adaptive_y = y[:, titans_chunk_size:, :]
        self.k_q_v_memories.cal_outer_grads(logits=adaptive_logits, state=state, y=adaptive_y)
        self.eta_memory.cal_outer_grads(logits=adaptive_logits, state=state, y=adaptive_y)
        self.alpha_memory.cal_outer_grads(logits=adaptive_logits, state=state, y=adaptive_y)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.cal_outer_grads(logits=logits, state=state, y=y)
    
    def optimize(self) -> None:
        self.titans_memory.optimize()
        self.k_q_v_memories.optimize()
        self.eta_memory.optimize()
        self.alpha_memory.optimize()
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.optimize()
