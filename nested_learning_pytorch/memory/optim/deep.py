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

from ..assoc_memory import AssocMemory, AssocMemState
from ...utils import default, repeat_dict_values, pack_one_with_inverse

from ..memory_models import MemoryMLP, ResidualNorm


def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)


class DeepGradientDesent(AssocMemory):
    """Deep Gradient Descent implementation (formula 43, 49, 50 in NL paper)."""
    
    # Register this class with type name "Memory"
    _type_name = "DeepGradientDesent"
    
    @dataclass
    class DeepGradientDesentState(AssocMemState):
        momentum: TensorDict | dict[str, torch.Tensor] | None = None
    
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
        inner_loss_fn: Callable = default_loss_fn,
        outer_loss_fn: Callable = default_loss_fn,
        memory_state_clz: type | None = DeepGradientDesentState,
        model: Module | None = None,
        alpha: float = 0.1,
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
            chunk_size=chunk_size,
            inner_optimizer=inner_optimizer,
            outer_optimizer=outer_optimizer,
            dim=dim,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            inner_loss_fn=inner_loss_fn,
            outer_loss_fn=outer_loss_fn,
            memory_state_clz=memory_state_clz,
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
                
            # todo: reduce momentum hidden dimension here to reduce memory usage
            default_model_kwargs['expansion_factor'] = 2.
            model = MemoryMLP(param.shape[-1], is_multi_head=is_multi_head, n_heads=n_heads, **default_model_kwargs)
            # Use small random initialization to break symmetry while staying close to identity
            for weight in model.weights:
                nn.init.normal_(weight, mean=0.0, std=0.01)
            params_dict[safe_name] = model
            
            # Here is the tricky part: the NL paper doesn't define the preconditioner clearly, but we could just use a linear layer
            preconditioner = MemoryMLP(dim=param.shape[-1], depth=1, is_multi_head=is_multi_head, n_heads=n_heads)
            params_dict[f'{safe_name}_preconditioner'] = preconditioner

            self.param_name_mapping[name] = safe_name
        
        self.model = nn.ModuleDict(params_dict)
        self.default_pattern = 'b ... d, b d o -> b ... o'
        self.default_inverse = nn.Identity()
        
        # Define eta and alpha as non-learnable constants.
        self.register_buffer('alpha', torch.tensor([alpha], dtype=torch.float32))
        self.optimizer_key_idx_map = {name: idx for idx, name in enumerate(self.get_optimizer_params().keys())}
        
        # override optimizers - use object.__setattr__ to bypass PyTorch's module registration
        object.__setattr__(self, 'inner_optimizer', None)
        object.__setattr__(self, 'outer_optimizer', None)
    
    def retrive_params_by_prefix_lstrip(self, prefix: str, fast_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return TensorDict({name.replace(f'{prefix}.', ''): fast_weights[name] for name in fast_weights.keys() if name.startswith(f'{prefix}.')})
        
    def forward(
        self,
        grads_dict: TensorDict,
        state: dict[str, AssocMemState] | None = None,
    ):
        # DeepGradientDesent should not have children blocks
        
        fast_weights = state[self.block_name].fast_weights
        
        new_grads_dict = TensorDict({})
        for name, grad in grads_dict.items():
            b = grad.shape[0]
            need_pack = False
            if grad.dim() == 4:
                grad_eff = rearrange(grad, 'b h i j -> (b h) i j')
                grad_eff = grad_eff.mean(dim=1)
                need_pack = True
            elif grad.dim() == 3:
                # matrix weight with batch
                grad_eff = grad.mean(dim=1) # (b, d2)
            elif grad.dim() == 2:
                grad_eff = grad # (b, d2)
            else:
                raise ValueError("Unsupported grad shape")
            
            safe_name = self.param_name_mapping[name]
            fast_weight = self.retrive_params_by_prefix_lstrip(safe_name, fast_weights)
            if need_pack:
                fast_weight = repeat_dict_values(fast_weight, 'b h i j -> (b h) i j')
            v = functional_call(self.model[safe_name], dict(fast_weight), (grad_eff,), {'pattern': self.default_pattern})
            
            # Normalize the orthogonal space, because we only care the direction in the orthogonal space, and keep the gradient's magnitude.
            v = torch.nn.functional.normalize(v, dim=1)  # (b, d2)
            
            # The rank-1 update operator, equals to v.unsqueeze(2) @ grad_eff.unsqueeze(2).transpose(1, 2)
            P = v.unsqueeze(2) * grad_eff.unsqueeze(1) # (b, d2, d2)
            
            # Apply to the gradient
            if grad.dim() == 4:
                P = rearrange(P, '(b h) i j -> b h i j', b = b)
                grad_delta = grad @ P # (b, h, d1, d2)
            elif grad.dim() == 3:
                grad_delta = grad @ P # (b, d1, d2)
            elif grad.dim() == 2:
                grad_delta = (grad.unsqueeze(1) @ P).squeeze(1) # (b, d2)
            else:
                raise ValueError("Unsupported grad shape")
            
            new_grads_dict[name] = grad_delta
            
        return new_grads_dict, state
    
    def update_preconditioner(self, preconditioner_fast_weight: dict[str, torch.Tensor], grad: torch.Tensor, preconditioner_safe_name: str, eps: float = 1e-6) -> dict[str, torch.Tensor]:
        # Using the loss function in equation 43: ||O.T @ O - I||_F^2 to update the preconditioner.
        # Gram matrix: [d, d]
        O = functional_call(self.model[preconditioner_safe_name], dict(preconditioner_fast_weight), (grad,), {'pattern': self.default_pattern})

        # ---- flatten non-batch dims ----
        O_flat = O.view(O.shape[0], -1)        # (b, D)

        # ---- per-task normalization ----
        O_norm = F.normalize(O_flat, dim=1, eps=eps)  # (b, D)

        # ---- Gram matrix per task ----
        # G_i = O_i O_i^T  (rank-1)
        G = torch.bmm(
            O_norm.unsqueeze(2),         # (b, D, 1)
            O_norm.unsqueeze(1)          # (b, 1, D)
        )                                # (b, D, D)

        # enforce unit energy
        # We sum the batch dimension of the loss to avoid the division by the number of tasks.
        preconditioner_loss = ((G.diagonal(dim1=1, dim2=2) - 1) ** 2).sum()
        
        preconditioner_fast_weight_keys = []
        preconditioner_fast_weight_values = []
        for preconditioner_fast_weight_name, preconditioner_fast_weight_value in preconditioner_fast_weight.items():
            preconditioner_fast_weight_keys.append(preconditioner_fast_weight_name)
            preconditioner_fast_weight_values.append(preconditioner_fast_weight_value)
        
        preconditioner_grads = torch.autograd.grad(
            outputs=preconditioner_loss,
            inputs=preconditioner_fast_weight_values,
            retain_graph=True,
            create_graph=True,
        )
        
        # Update the preconditioner with one-step inner loop.
        preconditioner_fast_weight_update = dict()
        for preconditioner_fast_weight_key, preconditioner_grad, preconditioner_fast_weight_value in zip(preconditioner_fast_weight_keys, preconditioner_grads, preconditioner_fast_weight_values, strict=True):
            preconditioner_fast_weight_update[preconditioner_fast_weight_key] = preconditioner_fast_weight_value + preconditioner_grad * (-1)
        return TensorDict(preconditioner_fast_weight_update)
    
    def cal_inner_grads_update(self, grads_dict: torch.Tensor, state: dict[str, AssocMemState], alpha: torch.Tensor | None = None, eps: float = 1e-6) -> None:
        updated = False
        if hasattr(self, 'model') and self.model is not None:
            fast_weights = state[self.block_name].fast_weights
            momentum = state[self.block_name].momentum
            
            alpha = default(alpha, self.alpha)
            
            with torch.enable_grad():
                fast_weights_updated = {}
                momentum_updated = {}
                for grad_name, grad in grads_dict.items():
                    b = grad.shape[0]
                    need_pack = False
                    
                    if grad.dim() == 4:
                        g_eff = rearrange(grad, 'b h i j -> (b h) i j')
                        g_eff = g_eff.mean(dim=1)
                        need_pack = True
                    elif grad.dim() == 3:
                        # g: (b, d1, d2) -> (b, d2)
                        g_eff = grad.mean(dim=1)
                    elif grad.dim() == 2:
                        # g: (b, d2)
                        g_eff = grad
                    else:
                        raise ValueError("Unsupported g shape")
                    
                    preconditioner_safe_name = f'{self.param_name_mapping[grad_name]}_preconditioner'
                    preconditioner_fast_weight_dict = self.retrive_params_by_prefix_lstrip(preconditioner_safe_name, fast_weights)
                    if need_pack:
                        preconditioner_fast_weight_dict = repeat_dict_values(preconditioner_fast_weight_dict, 'b h i j -> (b h) i j')
                    
                    # We update the preconditioner with one-step inner loop first, and then use the updated preconditioner to update the momentum.
                    # The goal is to eliminate the most obvious non-orthogonal components in the current batch/task, which is a geometric correction of the orthogonal vector space, rather than a numerical optimization in SGD.
                    preconditioner_fast_weight_dict = self.update_preconditioner(preconditioner_fast_weight_dict, g_eff, preconditioner_safe_name)
                    preconditioner_fast_weight = next(iter(preconditioner_fast_weight_dict.values()))
                    
                    # Then we use the updated preconditioner to update the optimizer and momentum.
                    optimizer_fast_weights = self.retrive_params_by_prefix_lstrip(self.param_name_mapping[grad_name], fast_weights)
                    if need_pack:
                        optimizer_fast_weights = repeat_dict_values(optimizer_fast_weights, 'b h i j -> (b h) i j')
                    
                    # Loss function in equation 49
                    grad_col = g_eff.unsqueeze(-1)  # (b, d2, 1)
                    
                    h = grad_col
                    optimizer_fast_weight_keys = []
                    optimizer_fast_weight_values = []
                    for optimizer_fast_weight_key, optimizer_fast_weight_value in optimizer_fast_weights.items():
                        optimizer_fast_weight_keys.append(optimizer_fast_weight_key)
                        optimizer_fast_weight_values.append(optimizer_fast_weight_value)
                        
                        h = torch.bmm(optimizer_fast_weight_value.transpose(1, 2), h)    # (b, dn, 1)
                        
                    m_g = torch.bmm(h, grad_col.transpose(1, 2))    # (b, d2, d2)
                    
                    # To orthogonalized / normalized space
                    m_g_flat = rearrange(m_g, 'b i j -> b (i j)')
                    P_flat = rearrange(preconditioner_fast_weight, 'b i j -> b (i j)')
                    # Normalize m_g_flat here because we don't want to optimize the magnitude of the orthogonalized space, only the direction.
                    m_g_hat = F.normalize(m_g_flat, dim=1, eps=eps)
                    P_hat = F.normalize(P_flat, dim=1, eps=eps)
                    
                    # L2 loss in orthogonalized space (delta rule)
                    # We sum the batch dimension of the loss to avoid the division by the number of tasks, which is not expected in meta learning.
                    loss = ((m_g_hat - P_hat) ** 2).sum(dim=1).sum()
                    
                    optimizer_fast_weight_grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=optimizer_fast_weight_values,
                        retain_graph=True,
                        create_graph=True,
                    )
                    if need_pack:
                        optimizer_fast_weight_grads = [rearrange(optimizer_fast_weight_grad, '(b h) i j -> b h i j', b = b) for optimizer_fast_weight_grad in optimizer_fast_weight_grads]
                    
                    # update the fast weights directly
                    for optimizer_fast_weight_key, optimizer_fast_weight_grad in zip(optimizer_fast_weight_keys, optimizer_fast_weight_grads, strict=True):
                        tmp_weight_key = f'{self.param_name_mapping[grad_name]}.{optimizer_fast_weight_key}'
                        
                        # When a multi-head mlp meets a multi-weighted alpha, it could be issue with the weight dimension. But since FullyAdditiveTitansBlock.titans_memory is not multi-headed, we will be fine.
                        if alpha.ndim == 2:
                            alpha_unsqueezed = alpha[:, self.optimizer_key_idx_map[tmp_weight_key]]
                        else:
                            alpha_unsqueezed = alpha
                        while alpha_unsqueezed.ndim != optimizer_fast_weight_grad.ndim:
                            alpha_unsqueezed = alpha_unsqueezed.unsqueeze(-1)
                            
                        # Equation 50
                        # alpha is initialized near 1, and optimizer_fast_weight_grad is already weighted by eta
                        momentum_updated[tmp_weight_key] = momentum[tmp_weight_key] * alpha_unsqueezed - optimizer_fast_weight_grad
                        fast_weights_updated[tmp_weight_key] = fast_weights[tmp_weight_key] + momentum_updated[tmp_weight_key]
                    
                    if need_pack:
                        preconditioner_fast_weight_dict = repeat_dict_values(preconditioner_fast_weight_dict, '(b h) i j -> b h i j', b = b)
                    for preconditioner_fast_weight_key, preconditioner_fast_weight_value in preconditioner_fast_weight_dict.items():
                        fast_weights_updated[f'{self.param_name_mapping[grad_name]}_preconditioner.{preconditioner_fast_weight_key}'] = preconditioner_fast_weight_value
                
                state[self.block_name].fast_weights = fast_weights_updated
                state[self.block_name].momentum = momentum_updated
                updated = True

        # There supposedly should be no children blocks for DeepGradientDesent
        # if self.children_blocks is not None:
        #     for child_block in self.children_blocks:
        #         child_block.cal_inner_grads(logits=logits, state=state, y=y)
        return updated
    
    def get_optimizer_params(self) -> TensorDict:
        return TensorDict({name: param for name, param in self.model.named_parameters() if '_preconditioner' not in name})
    
    def get_preconditioner_params(self) -> TensorDict:
        return TensorDict({name: param for name, param in self.model.named_parameters() if '_preconditioner' in name})
    
    def init_state(self, state: dict[str, AssocMemState] | None = None, batch_size: int | None = None) -> dict[str, AssocMemState]:
        state = super().init_state(state=state, batch_size=batch_size)
        
        batch_size = default(batch_size, 1)
        device = next(self.parameters()).device
        
        weights = self.get_optimizer_params()
        weights = repeat_dict_values(weights, '... -> b ...', b = batch_size).to(device)
        # Convert non-leaf tensors to leaf tensors for optimization
        # Use clone() to ensure separate memory storage for each tensor
        weights = {k: v.clone().detach().zero_().requires_grad_() for k, v in weights.items()}
        
        state[self.block_name].momentum = weights
        
        return state
