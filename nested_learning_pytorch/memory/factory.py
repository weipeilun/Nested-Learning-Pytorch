from typing import Type, Dict

import torch.nn as nn


# Registry for all AssocMemory subclasses
_ASSOC_MEMORY_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_assoc_memory(type_name: str):
    """Decorator to register an AssocMemory subclass.
    
    Args:
        type_name: The string identifier for this memory type
        
    Example:
        @register_assoc_memory("Memory")
        class Memory(AssocMemory):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if type_name in _ASSOC_MEMORY_REGISTRY:
            raise ValueError(f"AssocMemory type '{type_name}' is already registered")
        _ASSOC_MEMORY_REGISTRY[type_name] = cls
        return cls
    return decorator


def get_registered_memory_types() -> list[str]:
    """Get a list of all registered AssocMemory types.
    
    Returns:
        List of registered type names
    """
    return list(_ASSOC_MEMORY_REGISTRY.keys())