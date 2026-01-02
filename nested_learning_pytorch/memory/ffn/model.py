from __future__ import annotations
from typing import Callable
import warnings

import torch
from torch import nn
from torch.nn import Module
from torch.func import functional_call

from ..memory_models import MemoryMLP, NormalizationBuilder
from ..assoc_memory import AssocMemory, AssocMemState

from ...utils import exists, default

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

# main memory
# section 5.1 of the paper Nested Learning: The Illusion of Deep Learning Architecture:
# we observed that both optimization process of neural networks as well as neural architectures can be formulated as a set
# of nested and/or parallel optimization problems, in which the memory structure is a feedforward layer (e.g., either Deep
# MLPs, linear layers, etc.) and the objective is optimized with gradient descent or Newton's methods.


class FFNBlock(AssocMemory):
    """FFN Block implementation with automatic registration."""
    
    # Register this class with type name "Memory"
    _type_name = "FFN"
    
    def __init__(
        self,
        block_name: str,
        chunk_size: int,
        dim: int,
        inner_optimizer: Callable | tuple[Callable, Callable],
        outer_optimizer: Callable | tuple[Callable, Callable],
        inner_lr: float,
        outer_lr: float,
        lr_multiple: float,
        inner_loss_fn: nn.Module,
        outer_loss_fn: nn.Module,
        model: Module | None = None,
        is_multi_head = False,
        heads: int = 1,
        with_bias: bool = False,
        normalization: str = 'residual_pre_norm',
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        )
    ):
        super().__init__(
            block_name=block_name,
            chunk_size=chunk_size,
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            dim=dim,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            lr_multiple=lr_multiple,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
        )

        # memory model

        if not exists(model):
            model = MemoryMLP(dim, is_multi_head = is_multi_head, n_heads = heads, with_bias = with_bias, **default_model_kwargs)

        # the memory is the weights of the model

        if normalization:
            model = NormalizationBuilder(normalization)(dim = dim, is_multi_head = is_multi_head, model = model, **default_model_kwargs)

        self.model = model
        
        self.is_multi_head = is_multi_head
        depth = default_model_kwargs['depth']
        if is_multi_head:
            self.default_pattern = ['s d, h d o -> h s o']
            for _ in range(depth - 1):
                self.default_pattern.append('h s d, h d o -> h s o')
        else:
            self.default_pattern = ['s d, d o -> s o'] * depth

        # validate memory model

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_shape = (2, dim)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                test_weights = self.memory_model_parameter_dict

                mem_model_output = functional_call(self.model, dict(test_weights), (test_input,), {'pattern': self.default_pattern})
            except Exception as e:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}: {e}')

            out_dim = default_model_kwargs.get('out_dim', test_shape[-1])
            if is_multi_head:
                assert mem_model_output.shape == (heads, test_shape[0], out_dim), f'{self.block_name} output of memory model needs to be same shape as input'
            else:
                assert mem_model_output.shape == (test_shape[0], out_dim), f'{self.block_name} output of memory model needs to be same shape as input'

        # override optimizers - use object.__setattr__ to bypass PyTorch's module registration
        inner_opt = self.inner_optimizer(model = self.model)
        object.__setattr__(self, 'inner_optimizer', inner_opt)
        outer_opt = self.outer_optimizer(model = self.model, lr=outer_lr)
        object.__setattr__(self, 'outer_optimizer', outer_opt)
    
    def _apply(self, fn):
        """Override _apply to also handle optimizer states when moving devices."""
        # First apply to all module parameters and buffers
        super()._apply(fn)
        
        # Apply the same function to optimizer
        for optimizer in [self.inner_optimizer, self.outer_optimizer]:
            if isinstance(optimizer, AssocMemory):
                optimizer._apply(fn)
        
        return self
    
    def forward(
        self,
        x,
        state: dict[str, AssocMemState] | None = None,
        pattern: str | list[str] | None = None,
    ):
        # scheduler handles memory updates, this only handles retrieve
        
        if self.children_blocks is not None:
            if len(self.children_blocks) > 1:
                raise ValueError(f"FFNBlock {self.block_name} has multiple children blocks, but only one is supported")
            elif len(self.children_blocks) > 0:
                x, state = self.children_blocks[0](x=x, state=state)
        
        fast_weights = state[self.block_name]['fast_weights']

        logits = functional_call(self.model, fast_weights, (x,), {'pattern': default(pattern, self.default_pattern)})

        return logits, state
