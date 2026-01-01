from __future__ import annotations

from typing import Protocol, Sequence, Dict, List
from dataclasses import dataclass, field, replace
from functools import wraps

from typing import Callable, Any
import warnings
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map
from tensordict import TensorDict

from .factory import _ASSOC_MEMORY_REGISTRY
from .optim.factory import build_inner_optimizer, build_outer_optimizer

from ..utils import default


DEFAULT_INNER_OPTIMIZER_SUF = '_inner_optimizer'
DEFAULT_OUTER_OPTIMIZER_SUF = '_outer_optimizer'


def update_step(forward_fn):
    """Decorator that automatically updates the step counter after forward returns."""
    @wraps(forward_fn)
    def wrapper(self, x, state: dict[str, AssocMemState] | None = None, auto_update_step: bool = True, *args, **kwargs):
        # Call the original forward method
        logits, state = forward_fn(self, x, state, *args, **kwargs)
        
        if auto_update_step:
            if self.block_name in state and state[self.block_name]['last_update_step'] is not None and isinstance(state[self.block_name]['last_update_step'], torch.Tensor):
                if self.block_name.endswith('_inner_optimizer'):
                    # We can't observe sequence length from optimizers because its inputs and outputs are gradients, so we use it's parent block's sequence length
                    parent_block_name = self.block_name.replace('_inner_optimizer', '')
                    state[self.block_name]['step'] = state[parent_block_name]['step'].clone()
                else:
                    if x.ndim == 2:
                        seq_len = x.shape[0]
                    elif x.ndim == 3:
                        seq_len = x.shape[1]
                    else:
                        raise ValueError(f"Invalid input shape: {x.shape}")
                    state[self.block_name]['step'] += seq_len
                
        return logits, state
    
    return wrapper


def is_optimizer_block(block_name: str) -> bool:
    return is_inner_optimizer_block(block_name) or is_outer_optimizer_block(block_name)

def is_inner_optimizer_block(block_name: str) -> bool:
    return block_name.endswith(DEFAULT_INNER_OPTIMIZER_SUF)

def is_outer_optimizer_block(block_name: str) -> bool:
    return block_name.endswith(DEFAULT_OUTER_OPTIMIZER_SUF)


class AssocMemory(nn.Module):
    """Base class for associative memories with explicit update hooks."""
    
    # Class attribute to enable automatic registration
    _type_name: str | None = None
    
    DEFAULT_GRADIENT_KEY_SPLITTER = '::'
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses if they have a _type_name attribute.
        Also automatically wraps the forward method with update_step decorator.
        """
        super().__init_subclass__(**kwargs)
        
        # Register class by type name
        if hasattr(cls, '_type_name') and cls._type_name is not None:
            if cls._type_name in _ASSOC_MEMORY_REGISTRY:
                # Allow re-registration in case of reloading
                pass
            _ASSOC_MEMORY_REGISTRY[cls._type_name] = cls
        
        # Automatically wrap forward method with update_step decorator
        # Only wrap if the subclass defines its own forward method
        if 'forward' in cls.__dict__:
            original_forward = cls.__dict__['forward']
            # Check if it's not already wrapped (avoid double-wrapping)
            if not hasattr(original_forward, '__wrapped__'):
                cls.forward = update_step(original_forward)
    
    def __init__(self,
                 block_name: str,
                 chunk_size: int | None,
                 inner_optimizer: Callable | tuple[Callable, Callable],
                 outer_optimizer: Callable | tuple[Callable, Callable],
                 dim: int | None,
                 inner_lr: float,
                 outer_lr: float,
                 inner_loss_fn: nn.Module,
                 outer_loss_fn: nn.Module,
                 ): 
        super().__init__()
        
        self.block_name = block_name
        
        # the chunk size within the paper where adaptive step, momentum, weight decay are shared
        
        self.chunk_size = chunk_size
        
        # since optimization is a critical part for meta learning, we need to pass the optimizers here
        # https://abehrouz.github.io/files/NL.pdf, sector 6
        object.__setattr__(self, 'inner_optimizer', inner_optimizer)
        object.__setattr__(self, 'outer_optimizer', outer_optimizer)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_loss_fn = inner_loss_fn
        self.outer_loss_fn = outer_loss_fn
        
        self.dim = dim
        
        self.children_blocks = None
        
    def init_children_blocks(self, dim: int, block_specs: Sequence[AssocMemSpec], optimizer_configs: Dict[str, dict]) -> None:
        children_blocks = []
        if block_specs is not None:
            for spec in block_specs:
                children_blocks.append(_build_block(
                    replace(spec, name=f"{self.block_name}.{spec.name}"),
                    optimizer_configs,
                    dim
                ))
        if children_blocks:
            self.children_blocks = nn.ModuleList(children_blocks)

    @update_step
    def forward(self, x, state: dict[str, AssocMemState] | None = None, *args, **kwargs) -> tuple[torch.Tensor, dict[str, AssocMemState]]:  # type: ignore[override]
        raise NotImplementedError
                
    # cache the gradients of sequence dimension for each block and sum to state.updates
    def cache_inner_grads(self, state: dict[str, AssocMemState], block_grads_dict: dict[str, torch.Tensor]) -> None:
        if hasattr(self, 'model') and self.model is not None:
            updates = state[self.block_name].get('updates', None)
            n_updates = state[self.block_name].get('n_updates', torch.zeros_like(state[self.block_name]['step'], dtype=torch.float32))
            
            assert self.block_name in block_grads_dict, f"Gradients for {self.block_name} not found in block_grads_dict, please check the gradient flow."
            
            grads_dict = block_grads_dict[self.block_name]
                
            new_updates = {}
            for fast_weight_key, fast_weight_grad in grads_dict.items(): 
                # add to historic gradients
                if updates is not None:
                    new_updates[fast_weight_key] = updates[fast_weight_key] + fast_weight_grad
                else:
                    new_updates[fast_weight_key] = fast_weight_grad
        
            if new_updates:
                state[self.block_name]['updates'] = new_updates
                state[self.block_name]['n_updates'] = n_updates + 1
            
            # cache the updates for the inner optimizer
            if '_inner_optimizer' not in self.block_name:
                self.inner_optimizer.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)
                
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.cache_inner_grads(state=state, block_grads_dict=block_grads_dict)

    def maybe_inner_update(self, state: dict[str, AssocMemState], step_need_update_dict: dict[str, bool]) -> bool:
        updated = False
        if self.block_name in step_need_update_dict and step_need_update_dict[self.block_name]:
            fast_weights = state[self.block_name]['fast_weights']
            updates = state[self.block_name]['updates']
            n_updates = state[self.block_name]['n_updates']
            
            # Updates are added through multiple chunks, we need to divide by the number of chunks to get the average gradient.
            averaged_updates = self.average_updates(updates=updates, n_updates=n_updates)
            
            # Here we introduce DGD with weight decay, which projects the momentums to the orthogonal space, hoping to keep the optimizer memorizing through a long context
            # Update the optimizer first then use it to optimize the Titans memory.
            # It's important to update the optimizer first, otherwise will lead to a suboptimal result - both instinctly and according to equation 50 in NL paper.
            
            # Step 1: apply gradients to preconditioner and one-step update preconditioner.
            # The goal of one-step update is to eliminate the most obvious non-orthogonal components in the current batch/task.
            # We need t's preconditioner gradients rather than t - 1's at step t + 1. So inner_optimizer's gradients must be applied before calling inner_optimizer.forward().
            updated = self.inner_optimizer.update_preconditioners(grads_dict=averaged_updates, state=state)
            assert updated, "Preconditioner update failed"
            
            # Step 2: get the momentums using inner optimizer.
            momentums, state = self.inner_optimizer(x=averaged_updates, state=state)
            
            # Step 3: update the fast weights using momentums.
            # Equation 50 in NL paper.
            new_fast_weights = {}
            for fast_weight_key, fast_weight_value in fast_weights.items():
                momentum_key = f"{self.inner_optimizer.param_name_mapping[fast_weight_key]}.momentum"
                
                assert momentum_key in momentums, f"Momentum key {momentum_key} not found in momentum_deltas"
                
                new_fast_weights[fast_weight_key] = fast_weight_value + momentums[momentum_key]
            
            # Update fast_weights with the new values.
            state[self.block_name]['fast_weights'] = new_fast_weights
                
            # handle state after update
            state[self.block_name]['updates'] = {update_key: torch.zeros_like(update_value) for update_key, update_value in updates.items()}
            state[self.block_name]['n_updates'] = torch.zeros_like(n_updates)
            state[self.block_name]['last_update_step'] = state[self.block_name]['step'].clone()
            state[self.inner_optimizer.block_name]['last_update_step'] = state[self.block_name]['step'].clone()
            updated = True

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.maybe_inner_update(state=state, step_need_update_dict=step_need_update_dict)
                
        return updated
    
    def outer_update(self, grads_dict: dict[str, torch.Tensor]) -> None:
        if hasattr(self, 'model') and self.model is not None:
            # Set gradients for model parameters
            for name, param in self.model.named_parameters():
                grad_key = f'{self.block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}{name}'
                if grad_key in grads_dict:
                    param.grad = grads_dict[grad_key].mean(dim = 0)
                else:
                    raise ValueError(f"Gradient key {grad_key} not found in grads_dict, please check the gradient flow.")
            
            # print(f"------------------{self.block_name}------------------")
            # for name, param in self.model.named_parameters():
            #     print(torch.norm(param).item())
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad(set_to_none=True)
            # for name, param in self.model.named_parameters():
            #     print(torch.norm(param).item())
            
            # Set all parameter gradients to None (more memory efficient than zero_grad)
            for param in self.model.parameters():
                param.grad = None
                
            if '_inner_optimizer' not in self.block_name:
                self.inner_optimizer.outer_update(grads_dict=grads_dict)
            
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.outer_update(grads_dict=grads_dict)
    
    @property
    def memory_model_parameter_dict(self) -> TensorDict | None:
        if hasattr(self, 'model') and self.model is not None:
            mem_model_params = dict(self.model.named_parameters())

            return TensorDict(dict(zip([*mem_model_params.keys()], [*mem_model_params.values()])))
        else:
            return None
    
    @property
    def chunk_sizes(self) -> dict[str, int]:
        if self.chunk_size is not None:
            return {self.block_name: self.chunk_size}
        else:
            return None
    
    def average_updates(self, updates: TensorDict, n_updates: torch.Tensor) -> TensorDict:
        return {update_key: update_value / n_updates for update_key, update_value in updates.items()}
    
    def init_state(self, state: dict[str, AssocMemState] | None = None, batch_size: int | None = None) -> dict[str, AssocMemState]:
        if state is None:
            state = dict()
        
        batch_size = default(batch_size, 1)
        
        if self.block_name not in state:
            weights = self.memory_model_parameter_dict
            last_update_step = None
            step = None
            
            if weights is not None:
                # Create batch_size dimensional tensors for step tracking
                device = next(self.parameters()).device
                last_update_step = torch.zeros(batch_size, dtype=torch.int32, device=device, requires_grad=False)
                step = torch.zeros(batch_size, dtype=torch.int32, device=device, requires_grad=False)
                
                # Convert non-leaf tensors to leaf tensors for optimization
                # Use clone() to ensure separate memory storage for each tensor
                weights = {k: torch.stack([v.clone().detach().requires_grad_().to(device) for _ in range(batch_size)]) for k, v in weights.items()}            
                
                state[self.block_name] = {
                    'last_update_step': last_update_step,
                    'step': step,
                    'weights': weights,
                    'fast_weights': weights,
                }        
        if self.inner_optimizer is not None and isinstance(self.inner_optimizer, AssocMemory):
            self.inner_optimizer.init_state(state=state, batch_size=batch_size)
        if self.outer_optimizer is not None and isinstance(self.outer_optimizer, AssocMemory):
            self.outer_optimizer.init_state(state=state, batch_size=batch_size)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.init_state(state=state, batch_size=batch_size)

        return state
    
    def add_calcuable_weights(
        self, 
        weight_dict: dict[str, torch.Tensor] | None, 
        memory_block_name: str,
        weight_keys: list[str],
        weight_values: list[torch.Tensor]
    ) -> None:
        """Helper method to add fast weights from a memory block to the accumulator lists."""
        if weight_dict is not None:
            for key, value in weight_dict.items():
                weight_keys.append(f"{memory_block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}{key}")
                weight_values.append(value)
    
    def add_non_calcuable_weights(
        self, 
        value_tensor: torch.Tensor | None, 
        memory_block_name: str,
        key_name: str,
        fast_weight_keys: list[str],
        fast_weight_values: list[torch.Tensor]
    ) -> None:
        """Helper method to add fast weights from a memory block to the accumulator lists."""
        if value_tensor is not None:
            fast_weight_keys.append(f"{memory_block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}{key_name}")
            fast_weight_values.append(value_tensor)
    
    def get_calcuable_weights(self, state: dict[str, AssocMemState], parameter_weight_key: str) -> dict[str, torch.Tensor]:
        weight_keys: list[str] = []
        weight_values: list[torch.Tensor] = []
        
        if self.block_name in state:
            try:
                self.add_calcuable_weights(state[self.block_name][parameter_weight_key], self.block_name, weight_keys, weight_values)
            except Exception as e:
                print(f"Error in get_calcuable_weights: {e}")
                print(f"state: {state}")
                print(f"parameter_weight_key: {parameter_weight_key}")
                raise e
        
            optimizer_keys, optimizer_values = self.inner_optimizer.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
            weight_keys.extend(optimizer_keys)
            weight_values.extend(optimizer_values)
            
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_calcuable_weights(state=state, parameter_weight_key=parameter_weight_key)
                weight_keys.extend(child_block_keys)
                weight_values.extend(child_block_values)
        
        return weight_keys, weight_values
    
    def get_inner_non_calcuable_fast_weights(self, state: dict[str, AssocMemState]) -> dict[str, torch.Tensor]:
        fast_weight_keys: list[str] = []
        fast_weight_values: list[torch.Tensor] = []
        
        if self.block_name in state:
            self.add_non_calcuable_weights(state[self.block_name]['step'], self.block_name, 'step', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.block_name]['last_update_step'], self.block_name, 'last_update_step', fast_weight_keys, fast_weight_values)
            if 'updates' in state[self.block_name]:
                self.add_non_calcuable_weights(state[self.block_name]['updates'], self.block_name, 'updates', fast_weight_keys, fast_weight_values)
            if 'n_updates' in state[self.block_name]:
                self.add_non_calcuable_weights(state[self.block_name]['n_updates'], self.block_name, 'n_updates', fast_weight_keys, fast_weight_values)
            
            optimizer_keys, optimizer_values = self.inner_optimizer.get_inner_non_calcuable_fast_weights(state=state)
            fast_weight_keys.extend(optimizer_keys)
            fast_weight_values.extend(optimizer_values)

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_inner_non_calcuable_fast_weights(state=state)
                fast_weight_keys.extend(child_block_keys)
                fast_weight_values.extend(child_block_values)
        
        return fast_weight_keys, fast_weight_values
    
    def get_outer_non_calcuable_fast_weights(self, state: dict[str, AssocMemState]) -> dict[str, torch.Tensor]:
        fast_weight_keys: list[str] = []
        fast_weight_values: list[torch.Tensor] = []
        
        if self.block_name in state:
            self.add_non_calcuable_weights(state[self.block_name]['step'], self.block_name, 'step', fast_weight_keys, fast_weight_values)
            self.add_non_calcuable_weights(state[self.block_name]['last_update_step'], self.block_name, 'last_update_step', fast_weight_keys, fast_weight_values)
            if 'updates' in state[self.block_name]:
                self.add_non_calcuable_weights(state[self.block_name]['updates'], self.block_name, 'updates', fast_weight_keys, fast_weight_values)
            if 'n_updates' in state[self.block_name]:
                self.add_non_calcuable_weights(state[self.block_name]['n_updates'], self.block_name, 'n_updates', fast_weight_keys, fast_weight_values)
            
            if '_inner_optimizer' not in self.block_name:
                optimizer_keys, optimizer_values = self.inner_optimizer.get_outer_non_calcuable_fast_weights(state=state)
                fast_weight_keys.extend(optimizer_keys)
                fast_weight_values.extend(optimizer_values)

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_outer_non_calcuable_fast_weights(state=state)
                fast_weight_keys.extend(child_block_keys)
                fast_weight_values.extend(child_block_values)
        
        return fast_weight_keys, fast_weight_values
    
def _build_block(spec: AssocMemSpec, optimizer_configs: Dict[str, dict], dim: int, **kwargs) -> AssocMemory:
    """Factory function to create a block instance based on AssocMemSpec.type.
    
    This function automatically instantiates the correct AssocMemory subclass
    based on the registered type name.
    
    Args:
        spec: Block specification containing type and other parameters
        dim: Feature dimension for the layer
        **kwargs: Additional keyword arguments to pass to the memory constructor
        
    Returns:
        An initialized AssocMemory instance
        
    Raises:
        ValueError: If the specified type is not registered
    """
    # Ensure all subclasses are imported and registered
    _ensure_subclasses_imported()
    
    memory_type = spec.type
    
    if memory_type not in _ASSOC_MEMORY_REGISTRY:
        available_types = ', '.join(_ASSOC_MEMORY_REGISTRY.keys())
        raise ValueError(
            f"Unknown AssocMemory type '{memory_type}'. "
            f"Available types: {available_types or 'none'}"
        )
    
    memory_cls = _ASSOC_MEMORY_REGISTRY[memory_type]
    
    # Prepare extra params with potential mappings for specific memory types
    extra_params = dict(spec.extra_params)
    
    # Special handling for FullyAdaptiveTitans: map update_period_* to chunk_size_*
    if memory_type == "FullyAdaptiveTitans":
        if 'update_period_titans' in extra_params:
            extra_params['chunk_size_titans'] = extra_params.pop('update_period_titans')
        if 'update_period_adaptive' in extra_params:
            extra_params['chunk_size_adaptive'] = extra_params.pop('update_period_adaptive')
    
    # Build parameters for the memory class
    # FullyAdaptiveTitans doesn't use the standard chunk_size parameter
    params_base = {
        'dim': dim,
        'block_name': spec.name,
        'inner_optimizer': None,
        'outer_optimizer': None,
        'inner_lr': spec.inner_lr,
        'outer_lr': spec.outer_lr,
        'inner_loss_fn': spec.inner_loss_fn,
        'outer_loss_fn': spec.outer_loss_fn,
        **extra_params,  # Add all extra custom parameters
        **kwargs
    }
    
    # Add chunk_size for types that need it (not FullyAdaptiveTitans)
    if memory_type != "FullyAdaptiveTitans":
        params_base['chunk_size'] = spec.update_period
        
    params = params_base.copy()
    params_inner_optimizer = params_base.copy()
    params_outer_optimizer = params_base.copy()
    
    outer_optimizer_for_inner_optimizer = build_outer_optimizer(memory_type, optimizer_configs=optimizer_configs, params=params_outer_optimizer)
    params_inner_optimizer['outer_optimizer'] = outer_optimizer_for_inner_optimizer
    
    params_inner_optimizer['block_name'] = f"{spec.name}{DEFAULT_INNER_OPTIMIZER_SUF}"
    params_outer_optimizer['block_name'] = f"{spec.name}{DEFAULT_OUTER_OPTIMIZER_SUF}"
    inner_optimizer = build_inner_optimizer(memory_type, optimizer_configs=optimizer_configs, params=params_inner_optimizer)
    outer_optimizer = build_outer_optimizer(memory_type, optimizer_configs=optimizer_configs, params=params_outer_optimizer)
    
    params['inner_optimizer'] = inner_optimizer
    params['outer_optimizer'] = outer_optimizer
    
    # Filter params to only include those accepted by the class
    # This allows for flexible parameter passing
    try:
        block = memory_cls(**params)
    except TypeError as e:
        # If there's a parameter mismatch, provide helpful error message
        raise TypeError(
            f"Error instantiating {memory_type}: {str(e)}. "
            f"Parameters provided: {list(params.keys())}"
        ) from e
    
    # Initialize children blocks
    # children_blocks are already AssocMemSpec objects from create_assoc_mem_spec
    block.init_children_blocks(dim, spec.children_blocks, optimizer_configs)
    
    return block
        

def _ensure_subclasses_imported():
    """Lazy import all AssocMemory subclasses to ensure they are registered.
    
    This function is called by build_block to ensure all subclasses are loaded
    before attempting to instantiate them.
    """
    # Import all known AssocMemory subclasses here
    try:
        from .ffn.model import FFNBlock  # noqa: F401
        from .hope.model import HOPEBlock  # noqa: F401
        from .titan.model import FullyAdditiveTitansBlock  # noqa: F401
    except ImportError as e:
        raise ImportError(f"AssocMemory subclasses not found. Please ensure all AssocMemory subclasses are imported. Error: {e}")


@dataclass
class AssocMemState:
    last_update_step: torch.Tensor | int = -1  # shape: (batch_size,) or scalar
    step: torch.Tensor | int = 0  # shape: (batch_size,) or scalar
    weights: TensorDict | dict[str, torch.Tensor] | None = None,  # TensorDict with batch_size dimension
    fast_weights: TensorDict | dict[str, torch.Tensor] | None = None,  # TensorDict with batch_size dimension
    # updates: TensorDict[str, torch.Tensor] | None = None, # updates (gradients) for each parameter

def block_state_detach(
    state: AssocMemState
) -> AssocMemState:
    assert isinstance(state, AssocMemState)
    state = tree_map(lambda t: t.detach() if torch.is_tensor(t) else t, tuple(state))
    return AssocMemState(*state)


@dataclass(frozen=True)
class AssocMemSpec:
    """Configuration for a nested-learning associative memory."""

    name: str
    type: str
    update_period: int
    inner_loss_fn: nn.Module
    outer_loss_fn: nn.Module
    warmup_steps: int = 0
    jitter: int = 0
    inner_lr: float = 5.0e-3
    outer_lr: float = 5.0e-4
    hidden_multiplier: int = 4
    children_blocks: List[AssocMemSpec] = field(default_factory=list)
    extra_params: Dict = field(default_factory=dict)  # Store all custom parameters here

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            msg = f"update_period for block {self.name} must be positive"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = f"warmup_steps for block {self.name} must be non-negative"
            raise ValueError(msg)
        if self.jitter < 0:
            msg = f"jitter for block {self.name} must be non-negative"
            raise ValueError(msg)
