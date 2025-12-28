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

from ..utils import repeat_dict_values, default


DEFAULT_INNER_OPTIMIZER_SUF = '_inner_optimizer'
DEFAULT_OUTER_OPTIMIZER_SUF = '_outer_optimizer'


def update_step(forward_fn):
    """Decorator that automatically updates the step counter after forward returns."""
    @wraps(forward_fn)
    def wrapper(self, x, state: dict[str, AssocMemState] | None = None, auto_update_step: bool = True):
        # Call the original forward method
        logits, state = forward_fn(self, x, state)
        
        if auto_update_step:
            if self.block_name in state and state[self.block_name].last_update_step is not None and isinstance(state[self.block_name].last_update_step, torch.Tensor):
                if self.block_name.endswith('_inner_optimizer'):
                    # We can't observe sequence length from optimizers, so we use it's parent block's sequence length
                    parent_block_name = self.block_name.replace('_inner_optimizer', '')
                    state[self.block_name].step = state[parent_block_name].step.clone()
                else:
                    if x.ndim == 2:
                        seq_len = x.shape[0]
                    elif x.ndim == 3:
                        seq_len = x.shape[1]
                    else:
                        raise ValueError(f"Invalid input shape: {x.shape}")
                    state[self.block_name].step += seq_len
                
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
                 memory_state_clz: type | None = None
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
        self.memory_state_clz = memory_state_clz
        
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
    
    def call_children_blocks(self, x, state: dict[str, AssocMemState] | None = None) -> dict[str, tuple[torch.Tensor, dict[str, AssocMemState]]]:
        if self.children_blocks is not None:
            return {child_block.block_name: child_block(x=x, state=state) for child_block in self.children_blocks}
        else:
            return {}

    @update_step
    def forward(self, x, state: dict[str, AssocMemState] | None = None) -> tuple[torch.Tensor, dict[str, AssocMemState]]:  # type: ignore[override]
        raise NotImplementedError
    
    def add_calcuable_fast_weights(
        self, 
        fast_weight_dict: dict[str, torch.Tensor] | None, 
        memory_block_name: str,
        fast_weight_keys: list[str],
        fast_weight_values: list[torch.Tensor]
    ) -> None:
        """Helper method to add fast weights from a memory block to the accumulator lists."""
        if fast_weight_dict is not None:
            for key, value in fast_weight_dict.items():
                fast_weight_keys.append(f"{memory_block_name}{self.DEFAULT_GRADIENT_KEY_SPLITTER}{key}")
                fast_weight_values.append(value)
    
    def get_calcuable_fast_weights(self, state: dict[str, AssocMemState]) -> dict[str, torch.Tensor]:
        fast_weight_keys: list[str] = []
        fast_weight_values: list[torch.Tensor] = []
        
        self.add_calcuable_fast_weights(state[self.block_name].fast_weights, self.block_name, fast_weight_keys, fast_weight_values)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block_keys, child_block_values = child_block.get_calcuable_fast_weights(state=state)
                fast_weight_keys.extend(child_block_keys)
                fast_weight_values.extend(child_block_values)
        
        return fast_weight_keys, fast_weight_values
                
    # calculate the sum of gradients across sequence dimension for each block and sum to state.updates
    def cal_inner_grads(self, logits: torch.Tensor, state: dict[str, AssocMemState], y: torch.Tensor, block_grads_dict: dict[str, torch.Tensor]) -> None:
        if hasattr(self, 'model') and self.model is not None:
            fast_weights = state[self.block_name].fast_weights
            updates = state[self.block_name].updates
            
            if self.block_name in block_grads_dict:
                grads_dict = block_grads_dict[self.block_name]
                    
                key_grads_iter = grads_dict.items()
            else:
                warnings.warn(f"Autometically calculating gradients for {self.block_name} failed, please check the gradient flow.")
                with torch.enable_grad():
                    losses = self.inner_loss_fn(logits, y)
                    
                    fast_weight_keys = []
                    fast_weight_values = []
                    for fast_weight_name, fast_weight_value in fast_weights.items():
                        fast_weight_keys.append(fast_weight_name)
                        fast_weight_values.append(fast_weight_value)
                    
                    # Compute gradients with respect to fast_weights
                    # We need to compute the second order gradients in higher levels, so we need to set retain_graph and create_graph to True
                    # In section 8.1 (https://abehrouz.github.io/files/NL.pdf): "Note that, again, the initial states of all memories, i.e., M□,0
                    # for any □ ∈ {𝒌, 𝒗, 𝒒, 𝜂, 𝛼, memory} are meta-learned across all sequences/contexts, and so are optimized in the higher
                    # levels (or outer-loop)."
                    # We sum the batch dimension of the loss to avoid the division by the number of tasks, which is not expected in meta learning.
                    grads = torch.autograd.grad(
                        outputs=losses.sum(dim=1).mean(dim=0),
                        inputs=fast_weight_values,
                        retain_graph=True,
                        create_graph=True,
                    )
                    
                    key_grads_iter = zip(fast_weight_keys, grads, strict=True)
                    
            new_updates = {}
            for fast_weight_key, grad in key_grads_iter: 
                # add to historic gradients
                if grad is not None:
                    if updates is not None:
                        new_updates[fast_weight_key] = updates[fast_weight_key] + grad
                    else:
                        new_updates[fast_weight_key] = grad
        
            if new_updates:
                state[self.block_name].updates = new_updates
                
                # # check gradients towards weights
                # weights = state[self.block_name].weights
                # weights_list = list(weights.values())
                # weights_list_grads = torch.autograd.grad(
                #     outputs=losses.sum(dim=1).mean(dim=0),
                #     inputs=weights_list,
                #     retain_graph=True,
                # )
                # print(f"----------------{self.block_name}----------------")
                # for weight_key, weight_grad in zip(weights.keys(), weights_list_grads, strict=True):
                #     print(f"Weight {weight_key} gradient: {weight_grad.norm().item()}")
                
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.cal_inner_grads(logits=logits, state=state, y=y, block_grads_dict=block_grads_dict)

    def inner_update(self, state: dict[str, AssocMemState]) -> bool:
        updated = False
        if state[self.block_name].step is not None and (state[self.block_name].step - state[self.block_name].last_update_step >= self.chunk_size).all():
            fast_weights = state[self.block_name].fast_weights
            updates = state[self.block_name].updates
            
            if updates is not None:
                # Here we introduce DGD with weight decay, which projects the momentums to the orthogonal space, hoping to keep the optimizer memorizing through a long context
                # Update the optimizer first then use it to optimize the Titans memory.
                # It's important to update the optimizer first, otherwise will lead to a suboptimal result.
                # Both instinctly and according to equation 50 in NL paper.
                self.inner_optimizer.cal_inner_grads_update(grads_dict=updates, state=state)

                updates_optimized, state = self.inner_optimizer(x=updates, state=state)
                
                new_fast_weights = {}
                for update_key, grad_optimized in updates_optimized.items():
                    # fast weights need to be updated
                    weight_value = fast_weights[update_key]
                    
                    new_fast_weights[update_key] = weight_value + grad_optimized * (-self.inner_lr)
                
                # Update fast_weights with the new values
                state[self.block_name].fast_weights = new_fast_weights
                
            # handle state after update
            state[self.block_name].updates = None
            state[self.block_name].last_update_step = state[self.block_name].step.clone()
            state[self.inner_optimizer.block_name].last_update_step = state[self.block_name].step.clone()
            updated = True

        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.inner_update(state=state)
                
        return updated
    
    def cal_outer_grads(self, logits: torch.Tensor, state: dict[str, AssocMemState], y: torch.Tensor) -> None:
        if hasattr(self, 'model') and self.model is not None:
            # todo: test
            weights = state[self.block_name].weights
            
            with torch.enable_grad():
                losses = self.outer_loss_fn(logits, y)
                
                weight_keys = []
                weight_values = []
                for weight_name, weight_value in weights.items():
                    weight_keys.append(weight_name)
                    weight_values.append(weight_value)
                
                grads = torch.autograd.grad(
                    outputs=losses.mean(),
                    inputs=weight_values,
                    retain_graph=True,
                )
                # print(f"cal_outer_grads:----------------{self.block_name}----------------")
                # for weight_key, weight_grad in zip(weight_keys, grads, strict=True):
                #     print(f"Weight {weight_key} gradient: {weight_grad.norm().item()}")
                
                # Apply optimizer to gradients
                outer_grads = {}
                for weight_key, grad in zip(weight_keys, grads, strict=True):
                    outer_grads[weight_key] = grad
                
                self.outer_grads = outer_grads
                    
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.cal_outer_grads(logits=logits, state=state, y=y)
    
    def optimize(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            # Set gradients for model parameters
            for name, param in self.model.named_parameters():
                if name in self.outer_grads:
                    # mean over batch when performing outer optimization
                    param.grad = self.outer_grads[name].mean(dim = 0)
            
            # print(f"------------------{self.block_name}------------------")
            # for name, param in self.model.named_parameters():
            #     print(torch.norm(param).item())
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad(set_to_none=True)
            # for name, param in self.model.named_parameters():
            #     print(torch.norm(param).item())
                
            # Clear all gradients, retained graphs and GPU memory
            del self.outer_grads
            
            # Set all parameter gradients to None (more memory efficient than zero_grad)
            for param in self.model.parameters():
                param.grad = None
            
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.optimize()
    
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
    
    def init_state(self, state: dict[str, AssocMemState] | None = None, batch_size: int | None = None) -> dict[str, AssocMemState]:
        if state is None:
            state = {}
        
        batch_size = default(batch_size, 1)
        
        if self.block_name not in state:
            weights = self.memory_model_parameter_dict
            updates = None
            last_update_step = None
            step = None
            
            if weights is not None:
                # Create batch_size dimensional tensors for step tracking
                device = next(self.parameters()).device
                last_update_step = torch.zeros(batch_size, dtype=torch.int32, device=device, requires_grad=False)
                step = torch.zeros(batch_size, dtype=torch.int32, device=device, requires_grad=False)
                
                weights = repeat_dict_values(weights, '... -> b ...', b = batch_size).to(device)
                
                # Convert non-leaf tensors to leaf tensors for optimization
                # Use clone() to ensure separate memory storage for each tensor
                weights = {k: v.clone().detach().requires_grad_() for k, v in weights.items()}
            
            state_clz = self.memory_state_clz if self.memory_state_clz is not None else AssocMemState
            state[self.block_name] = state_clz(
                last_update_step=last_update_step,
                step=step,
                weights=weights,
                fast_weights=weights,
                updates=updates,
            )
        
        if self.inner_optimizer is not None and isinstance(self.inner_optimizer, AssocMemory):
            self.inner_optimizer.init_state(state=state, batch_size=batch_size)
        if self.outer_optimizer is not None and isinstance(self.outer_optimizer, AssocMemory):
            self.outer_optimizer.init_state(state=state, batch_size=batch_size)
        
        if self.children_blocks is not None:
            for child_block in self.children_blocks:
                child_block.init_state(state=state, batch_size=batch_size)

        return state
    
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
    updates: dict[str, torch.Tensor] | None = None, # updates (gradients) for each parameter

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
    inner_lr: float = 1.0e-4
    outer_lr: float = 1.0e-4
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
