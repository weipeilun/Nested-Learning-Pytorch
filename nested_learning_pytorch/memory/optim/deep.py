from __future__ import annotations
from typing import Any, Callable
import warnings

from functools import partial

import torch
from torch import nn
from torch.nn import Linear, Module, Parameter
import torch.nn.functional as F
from torch.func import functional_call
from tensordict import TensorDict
from dataclasses import dataclass
from einops import rearrange
from torch.func import grad, vmap

from ..assoc_memory import AssocMemory, AssocMemState
from ...utils import default, repeat_dict_values, pack_one_with_inverse
from ..memory_models import MemoryMLP, NormalizationBuilder, MomentumModule


class DeepMomentumGradientDesent(AssocMemory):
    """Deep Gradient Descent implementation (formula 43, 49, 50 in NL paper)."""
    
    # Register this class with type name "Memory"
    _type_name = "DeepMomentumGradientDesent"
    
    def __init__(
        self,
        block_name: str,
        chunk_size: int,
        dim: int,
        inner_optimizer: Callable | tuple[Callable, Callable],
        outer_optimizer: Callable | tuple[Callable, Callable],
        inner_lr: float,
        outer_lr: float,
        params: dict[str, torch.Tensor],
        inner_loss_fn: nn.Module,
        outer_loss_fn: nn.Module,
        alpha: float = 0.9,
        eta: float = 0.1,
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
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
        )
        
        params_dict = {}
        self.param_name_mapping = {}
        for name, param in params:
            safe_name = name.replace('.', '_')
            
            is_multi_head = False
            n_heads = 1
            if param.ndim == 3:
                is_multi_head = True
                n_heads = param.shape[0]
                
            # Param for momentum (wrapped in a module for ModuleDict compatibility)
            momentum = MomentumModule(torch.zeros_like(param))
            params_dict[safe_name] = momentum
            
            # There is no geometry in bias, so we ignore the preconditioner to prevent over fitting.
            if param.ndim > 1:
                # Equation 51, we can use a higher-order feature map to enhance the capacity of a memory.
                # This can be configured via hope_xxx.yaml file.
                # It's a gradient, all imformation of grads are important - scale, angle, curvature, correlation, energy etc. 
                # Any normalization / residual / non-linearity will distroy its geometry.
                preconditioner = MemoryMLP(dim=param.shape[-1], is_multi_head=is_multi_head, n_heads=n_heads, activation_fn=nn.Identity(), **default_model_kwargs)
                
                params_dict[f'{safe_name}_preconditioner'] = preconditioner

            self.param_name_mapping[name] = safe_name
        
        self.model = nn.ModuleDict(params_dict)
        self.default_pattern = '... d, d o -> ... o'
        self.default_inverse = nn.Identity()
        
        # Define eta and alpha as non-learnable constants.
        self.register_buffer('alpha', torch.tensor([alpha], dtype=torch.float32))
        self.register_buffer('eta', torch.tensor([eta], dtype=torch.float32))
        self.register_buffer('eta_ones', torch.ones([1], dtype=torch.float32))
        self.optimizer_key_idx_map = {name: idx for idx, name in enumerate(self.get_optimizer_params().keys())}
        
        # The inner optimizer gradient vmap, to calculate preconditioner gradient within each task (in batch dimension)
        self.preconditioner_grad_fn = grad(self.cal_preconditioner_loss_vmap)
        
        # override optimizers - use object.__setattr__ to bypass PyTorch's module registration
        object.__setattr__(self, 'inner_optimizer', None)
        outer_opt = self.outer_optimizer(model = self.model, lr=outer_lr)
        object.__setattr__(self, 'outer_optimizer', outer_opt)
        
    def retrive_params_by_prefix_lstrip(self, prefix: str, fast_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name.replace(f'{prefix}.', ''): fast_weights[name] for name in fast_weights.keys() if name.startswith(f'{prefix}.')}
        
    def cal_preconditioner_loss_vmap(self, preconditioner_fast_weight_values: list[torch.Tensor], grad: torch.Tensor, preconditioner_fast_weight_keys: list[torch.Tensor], preconditioner_safe_name: str, eps: float = 1e-6) -> dict[str, torch.Tensor]:
        # Using the loss function in equation 43: ||O.T @ O - I||_F^2 to update the preconditioner.
        # A vmap loss calculator. No batch dimension.
        # Gram matrix: [d, d]
        preconditioner_fast_weight = dict(zip(preconditioner_fast_weight_keys, preconditioner_fast_weight_values, strict=True))
        
        O = functional_call(self.model[preconditioner_safe_name], preconditioner_fast_weight, (grad,), {'pattern': self.default_pattern})        # (d1, d2)

        assert O.ndim == 2, "Unsupported O shape"

        # ---- per-task normalization ----
        O_norm = F.normalize(O, dim=-1, eps=eps)  # (d1, d2)

        # ---- Gram matrix per task ----
        # G_i = O_i O_i^T  (rank-1)
        G = torch.matmul(
            O_norm,                             # (d1, d2)
            O_norm.transpose(-2, -1)            # (d2, d1)
        )                                       # (d1, d1)

        # Eliminate off-diagonal elements. This is the orthogonal space.
        # equation 43 (Delta rule) 
        # Unrestricting the length of vectors of size O_i means that: 
        # 1. each vector can have a different scale, which is a good thing in a learned optimizer; 
        # 2. gradient is insensitive to scale, which is more conducive to maintaining numerical stability.
        # preconditioner_loss = ((G - torch.eye(G.shape[-1], device=G.device)) ** 2).mean()
        preconditioner_loss = ((G - torch.diag_embed(G.diagonal(dim1=-2, dim2=-1))) ** 2).mean()
        
        return preconditioner_loss
    
    def update_preconditioners(self, grads_dict: TensorDict, state: dict[str, AssocMemState]) -> TensorDict:
        if hasattr(self, 'model') and self.model is not None:
            fast_weights = state[self.block_name]['fast_weights']
            
            # Step 1: apply gradients to the preconditioner.
            if 'updates' in state[self.block_name] and 'n_updates' in state[self.block_name]:
                updates = state[self.block_name]['updates']
                n_updates = state[self.block_name]['n_updates']
                
                # Updates are added through multiple chunks, we need to divide by the number of chunks to get the average gradient.
                averaged_updates = self.average_updates(updates=updates, n_updates=n_updates)
                
                # Equation 41 in NL paper.
                applied_grads_fast_weights = {}
                for fast_weight_key, fast_weight_value in fast_weights.items():
                    if '_preconditioner' in fast_weight_key:
                        if fast_weight_key in averaged_updates:
                            applied_grads_fast_weights[fast_weight_key] = fast_weight_value + averaged_updates[fast_weight_key] * (-self.inner_lr)
                        else:
                            applied_grads_fast_weights[fast_weight_key] = fast_weight_value
            else:
                # This should only happen in the first step of each inner loop.
                applied_grads_fast_weights = fast_weights
            
            # Step 2: optimize the preconditioner to orthogonal space.
            new_fast_weights = {}
            for grad_name, grad_value in grads_dict.items():
                need_pack = False
                
                if grad_value.dim() == 3:
                    g_eff = grad_value
                    need_pack = True
                elif grad_value.dim() == 2:
                    # g: (d1, d2) -> (d2)
                    g_eff = grad_value
                elif grad_value.dim() == 1:
                    # There is no geometry in bias, so we ignore the preconditioner to prevent over fitting.
                    continue
                else:
                    raise ValueError("Unsupported g shape")
                
                preconditioner_safe_name = f'{self.param_name_mapping[grad_name]}_preconditioner'
                preconditioner_fast_weight_dict = self.retrive_params_by_prefix_lstrip(preconditioner_safe_name, applied_grads_fast_weights) # (d2, d2)
                if need_pack:
                    preconditioner_fast_weight_dict = repeat_dict_values(preconditioner_fast_weight_dict, 'b h i j -> (b h) i j')
                    
                preconditioner_fast_weight_keys = []
                preconditioner_fast_weight_values = []
                for preconditioner_fast_weight_key, preconditioner_fast_weight_value in preconditioner_fast_weight_dict.items():
                    preconditioner_fast_weight_keys.append(preconditioner_fast_weight_key)
                    preconditioner_fast_weight_values.append(preconditioner_fast_weight_value)
                
                # We update the preconditioner with one-step inner loop first, and then use the updated preconditioner to update the momentum.
                # The goal is to eliminate the most obvious non-orthogonal components in the current batch/task, which is a geometric correction of the orthogonal vector space, rather than a numerical optimization in SGD.
                # torch.no_grad() to cut the computation graph of the preconditioner, because we don't want higher order derivatives.
                with torch.no_grad():
                    preconditioner_grads = self.preconditioner_grad_fn(preconditioner_fast_weight_values, g_eff, preconditioner_fast_weight_keys, preconditioner_safe_name)
                
                # Update the preconditioner with one-step inner loop.
                # Equation 44 in NL paper.
                for preconditioner_fast_weight_key, preconditioner_grad, preconditioner_fast_weight_value in zip(preconditioner_fast_weight_keys, preconditioner_grads, preconditioner_fast_weight_values, strict=True):
                    new_fast_weights[f'{preconditioner_safe_name}.{preconditioner_fast_weight_key}'] = preconditioner_fast_weight_value + preconditioner_grad * (-self.inner_lr)
            
            for fast_weight_key, fast_weight_value in fast_weights.items():
                if fast_weight_key not in new_fast_weights:
                    new_fast_weights[fast_weight_key] = fast_weight_value
            
            # Step 3: check if only preconditioner is updated.
            assert len(new_fast_weights) == len(fast_weights), "New fast weights should have the same length as old fast weights"
            # for fast_weight_key, fast_weight_value in new_fast_weights.items():
            #     if '_preconditioner' not in fast_weight_key:
            #         assert (new_fast_weights[fast_weight_key] == fast_weights[fast_weight_key]).all(), f"Momentum {fast_weight_key} should not be updated"
            #     else:
            #         assert (new_fast_weights[fast_weight_key] != fast_weights[fast_weight_key]).any(), f"Preconditioner {fast_weight_key} should be updated"
            
            state[self.block_name]['fast_weights'] = new_fast_weights
            if 'updates' in state[self.block_name]:
                state[self.block_name]['updates'] = {update_key: torch.zeros_like(update_value) for update_key, update_value in state[self.block_name]['updates'].items()}
            if 'n_updates' in state[self.block_name]:
                state[self.block_name]['n_updates'] = torch.zeros_like(state[self.block_name]['n_updates'])

            return True
        return False
    
    def forward(self, x: TensorDict, state: dict[str, AssocMemState], alpha: torch.Tensor | None = None) -> TensorDict:
        fast_weights = state[self.block_name]['fast_weights']
        
        if alpha is None:
            alpha = self.alpha
            eta = self.eta
        else:
            # We assume the learning rate (eta) is dealt outside when calculating loss.
            alpha = alpha
            eta = self.eta_ones
            
        # Optimize the momentum and the fast weights for each optimizer parameter.
        new_momentum_dict = dict()
        for grad_name, grad_value in x.items():
            need_pack = False
            
            if grad_value.dim() == 3:
                # g: (h, d1, d2)
                g_eff = grad_value
                need_pack = True
            elif grad_value.dim() == 2:
                # g: (d1, d2)
                g_eff = grad_value
            elif grad_value.dim() == 1:
                # g: (d2)
                g_eff = grad_value
            else:
                raise ValueError("Unsupported g shape")
            
            # Step 1: map the gradient to orthogonalized space.
            if grad_value.dim() > 1:
                preconditioner_safe_name = f'{self.param_name_mapping[grad_name]}_preconditioner'
                preconditioner_fast_weight_dict = self.retrive_params_by_prefix_lstrip(preconditioner_safe_name, fast_weights) # (d2, d2)                if need_pack:
                if need_pack:
                    preconditioner_fast_weight_dict = repeat_dict_values(preconditioner_fast_weight_dict, 'b h i j -> (b h) i j')
                
                grad_in_orthogonalized_space = functional_call(self.model[preconditioner_safe_name], preconditioner_fast_weight_dict, (g_eff,), {'pattern': self.default_pattern})
            elif grad_value.dim() == 1:
                # There is no geometry in bias, so we ignore the preconditioner to prevent over fitting.
                grad_in_orthogonalized_space = g_eff
            else:
                raise ValueError("Unsupported g shape")
            
            # Step 2: update the momentum (fast weights) of the optimizer.
            # Equation 47 (Hebbian-rule) / 49 (delta rule), depending on self.inner_loss_fn, calculated at the end of vmap.
            momentum_fast_weights_dict = self.retrive_params_by_prefix_lstrip(self.param_name_mapping[grad_name], fast_weights)
            if need_pack:
                momentum_fast_weights_dict = repeat_dict_values(momentum_fast_weights_dict, 'b h i j -> (b h) i j')

            assert len(momentum_fast_weights_dict) == 1, f"Momentum should have only one fast weight: {self.block_name}"
            
            for momentum_fast_weights_key, momentum_fast_weights_value in momentum_fast_weights_dict.items():
                alpha_unsqueezed = alpha
                while alpha_unsqueezed.ndim != momentum_fast_weights_value.ndim:
                    alpha_unsqueezed = alpha_unsqueezed.unsqueeze(-1)
                eta_unsqueezed = eta
                while eta_unsqueezed.ndim != momentum_fast_weights_value.ndim:
                    eta_unsqueezed = eta_unsqueezed.unsqueeze(-1)
                    
                new_momentum_dict[f'{self.param_name_mapping[grad_name]}.{momentum_fast_weights_key}'] = momentum_fast_weights_value * alpha_unsqueezed - grad_in_orthogonalized_space * eta_unsqueezed
            
        new_momentum_fast_weights_dict = new_momentum_dict.copy()
        for fast_weight_key, fast_weight_value in fast_weights.items():
            if fast_weight_key not in new_momentum_fast_weights_dict:
                new_momentum_fast_weights_dict[fast_weight_key] = fast_weight_value
                
        state[self.block_name]['fast_weights'] = new_momentum_fast_weights_dict
        return new_momentum_dict, state
    
    def get_calcuable_weights(self, state: dict[str, AssocMemState], parameter_weight_key: str) -> dict[str, torch.Tensor]:
        weight_keys: list[str] = []
        weight_values: list[torch.Tensor] = []
        
        if self.block_name in state:
            weights = state[self.block_name][parameter_weight_key]
            for key, value in weights.items():
                # preconditioner update is not in vmap, it's need to be added to non_calcuable_fast_weights
                if parameter_weight_key == 'weights':
                    weight_keys.append(f"{self.block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}{key}")
                    weight_values.append(value)
                elif '_preconditioner' in key:
                    weight_keys.append(f"{self.block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}{key}")
                    weight_values.append(value)
            
        return weight_keys, weight_values
    
    def get_inner_non_calcuable_fast_weights(self, state: dict[str, AssocMemState]) -> dict[str, torch.Tensor]:
        fast_weight_keys: list[str] = []
        fast_weight_values: list[torch.Tensor] = []
        
        if self.block_name in state:
            fast_weights = state[self.block_name]['fast_weights']
            preconditioner_fast_weights_dict = dict()
            for key, value in fast_weights.items():
                # preconditioner update is not in vmap, it's need to be added to non_calcuable_fast_weights
                if '_preconditioner' not in key:
                    preconditioner_fast_weights_dict[key] = value
            fast_weight_keys.append(f"{self.block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}fast_weights")
            fast_weight_values.append(preconditioner_fast_weights_dict)
            
            self.add_non_calcuable_weights(state[self.block_name]['step'], self.block_name, 'step', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.block_name]['last_update_step'], self.block_name, 'last_update_step', fast_weight_keys, fast_weight_values)
            if 'updates' in state[self.block_name]:
                self.add_non_calcuable_weights(state[self.block_name]['updates'], self.block_name, 'updates', fast_weight_keys, fast_weight_values)
            if 'n_updates' in state[self.block_name]:
                self.add_non_calcuable_weights(state[self.block_name]['n_updates'], self.block_name, 'n_updates', fast_weight_keys, fast_weight_values)
            
        return fast_weight_keys, fast_weight_values
    
    def get_optimizer_params(self) -> TensorDict:
        return TensorDict({name: param for name, param in self.model.named_parameters() if '_preconditioner' not in name})
    
    def get_preconditioner_params(self) -> TensorDict:
        return TensorDict({name: param for name, param in self.model.named_parameters() if '_preconditioner' in name})
    
    # def init_state(self, state: dict[str, AssocMemState] | None = None, batch_size: int | None = None) -> dict[str, AssocMemState]:
    #     state = super().init_state(state=state, batch_size=batch_size)
        
    #     batch_size = default(batch_size, 1)
    #     device = next(self.parameters()).device
        
    #     weights = self.get_optimizer_params()
    #     # Convert non-leaf tensors to leaf tensors for optimization
    #     # Use clone() to ensure separate memory storage for each tensor
    #     weights = {k: torch.stack([v.clone().detach().zero_().requires_grad_() for _ in range(batch_size)]) for k, v in weights.items()}
        
    #     state[self.block_name]['momentum'] = weights
        
    #     return state
