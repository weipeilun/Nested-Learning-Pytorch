from __future__ import annotations

from typing import Dict, Tuple, List

import math
from tensordict import TensorDict
import torch

from .memory.assoc_memory import AssocMemory, AssocMemSpec


class FrequencyManager:
    """Manages update frequencies for hierarchical AssocMemory layers.
    
    This manager analyzes a list of AssocMemory modules and generates
    appropriate update frequencies based on their chunk_size and nesting
    structure. Child layers (in sub_blocks) are updated more frequently
    than their parent layers.
    """
    
    def __init__(self, memories: List[AssocMemory], base_period: int = 1):
        """Initialize the frequency manager.
        
        Args:
            memories: List of top-block AssocMemory modules to manage
            base_period: Base update period for the fastest updating layer (default: 1)
        """
        self.memories = memories
        self.base_period = base_period
        
        # Mappings from block_name to various properties
        self.frequency_map: Dict[str, int] = {}
        self.block_specs: Dict[str, AssocMemSpec] = {}
        self.hierarchy: Dict[str, List[str]] = {}  # parent -> children mapping
        
        # Build the frequency map and hierarchy
        self._analyze_hierarchy()
        
        self.min_frequencies = []
        self._compute_min_frequencies()
    
    def _compute_min_frequencies(self) -> None:
        """Compute the minimal base frequencies from all values in frequency_map.
        
        Examples:
        - {'key1': 3, 'key2': 6, 'key3': 9} -> [3]
        - {'key1': 3, 'key2': 6, 'key3': 9, 'key4': 4} -> [3, 4]
        
        Algorithm: Find all minimal numbers that cannot be divided by other numbers in the set.
        """
        if not self.frequency_map:
            self.min_frequencies = []
            return
        
        # Get all frequency values and remove duplicates
        frequencies = sorted(set(self.frequency_map.values()))
        
        if not frequencies:
            self.min_frequencies = []
            return
        
        # Find all numbers that cannot be divided by other numbers (i.e., base frequencies)
        min_freqs = []
        
        for freq in frequencies:
            # Check if freq can be divided by any already found base frequency
            is_multiple = False
            for base in min_freqs:
                if freq % base == 0:
                    is_multiple = True
                    break
            
            # If freq is not a multiple of existing bases, it might be a new base
            if not is_multiple:
                # Remove all existing bases that can be divided by freq (freq is smaller)
                min_freqs = [base for base in min_freqs if base % freq != 0]
                min_freqs.append(freq)
        
        self.min_frequencies = sorted(min_freqs)
    
    def _analyze_hierarchy(self) -> None:
        """Recursively analyze the memory hierarchy and assign update frequencies."""
        visited = set()
        
        # First pass: collect all layers and build hierarchy
        all_layers: List[Tuple[AssocMemory, int]] = []  # (memory, depth)
        
        def collect_layers(memory: AssocMemory, depth: int = 0, parent_name: str | None = None) -> None:
            """Recursively collect all layers with their depth.
            
            Skip memories with None chunk_size (they don't update), but still process their children.
            For Titans-type memories with chunk_sizes (plural), add entries for each sub-memory.
            """
            if memory.block_name in visited:
                return
            
            # Check if this memory has chunk_sizes (plural) property - for Titans-type memories
            has_chunk_sizes = hasattr(memory, 'chunk_sizes') and memory.chunk_sizes is not None
            
            if has_chunk_sizes:
                # Titans-type memory with multiple chunk_sizes
                chunk_sizes_dict = memory.chunk_sizes
                
                # Add each sub-memory to all_layers
                for sub_name, sub_chunk_size in chunk_sizes_dict.items():
                    if sub_name not in visited:
                        visited.add(sub_name)
                        # Create a synthetic memory object for each sub-memory
                        # We'll store the chunk_size and block_name for later processing
                        all_layers.append((memory, depth, sub_name, sub_chunk_size))
                        
                        # Track parent-child relationships
                        if parent_name is not None:
                            if parent_name not in self.hierarchy:
                                self.hierarchy[parent_name] = []
                            self.hierarchy[parent_name].append(sub_name)
            
            # Always recursively process sub-blocks (even if parent has None chunk_size)
            if hasattr(memory, 'children_blocks') and memory.children_blocks:
                for child_block in memory.children_blocks:
                    if isinstance(child_block, AssocMemory):
                        # All memories (including those with None chunk_size) can be parents
                        collect_layers(child_block, depth + 1, memory.block_name)
        
        # Collect all layers from top-block memories
        for memory in self.memories:
            collect_layers(memory)
        
        # Second pass: assign update frequencies based on depth and chunk_size
        # Deeper layers (higher depth) update more frequently
        # Larger chunk_size means less frequent updates
        
        # Group layers by depth
        depth_map: Dict[int, List[Tuple]] = {}
        for layer_info in all_layers:
            depth = layer_info[1]
            if depth not in depth_map:
                depth_map[depth] = []
            depth_map[depth].append(layer_info)
        
        # Assign frequencies: deeper blocks get lower periods (more frequent updates)
        # Base formula: period = base_period * chunk_size * (multiplier ^ depth)
        max_depth = max(depth_map.keys()) if depth_map else 0
        
        for layer_info in all_layers:
            # Check if this is a Titans-type sub-memory (4-tuple) or standard memory (2-tuple)
            if len(layer_info) == 4:
                # Titans-type sub-memory: (memory, depth, sub_name, sub_chunk_size)
                memory, depth, sub_name, sub_chunk_size = layer_info
                
                # Calculate update period based on chunk_size
                update_period = self.base_period * sub_chunk_size
                
                self.frequency_map[sub_name] = update_period
                
                # Create AssocMemSpec for this sub-memory
                self.block_specs[sub_name] = AssocMemSpec(
                    name=sub_name,
                    type=type(memory).__name__ + "_SubMemory",
                    update_period=update_period,
                    warmup_steps=0,
                    jitter=0,
                    inner_loss_fn=None,
                    outer_loss_fn=None,
                )
    
    def get_frequency(self, block_name: str) -> int:
        """Get the update frequency (period) for a given block.
        
        Args:
            block_name: Name of the block to query
            
        Returns:
            Update period (in steps) for the block
            
        Raises:
            KeyError: If block_name is not found
        """
        if block_name not in self.frequency_map:
            raise KeyError(f"Block '{block_name}' not found in frequency map")
        return self.frequency_map[block_name]
    
    def get_block_spec(self, block_name: str) -> AssocMemSpec:
        """Get the AssocMemSpec for a given block.
        
        Args:
            block_name: Name of the block to query
            
        Returns:
            AssocMemSpec for the block
            
        Raises:
            KeyError: If block_name is not found
        """
        if block_name not in self.block_specs:
            raise KeyError(f"Block '{block_name}' not found in block specs")
        return self.block_specs[block_name]
    
    def get_all_block_specs(self) -> List[AssocMemSpec]:
        """Get all BlockSpecs ordered by update frequency (fastest first).
        
        Returns:
            List of BlockSpecs sorted by update_period (ascending)
        """
        return sorted(self.block_specs.values(), key=lambda spec: spec.update_period)
    
    def get_children(self, block_name: str) -> List[str]:
        """Get the child layers of a given block.
        
        Args:
            block_name: Name of the parent block
            
        Returns:
            List of child block names (empty if no children)
        """
        return self.hierarchy.get(block_name, [])
    
    def get_all_chuck_sizes(self, state: TensorDict[str, TensorDict], sequence_len: int) -> List[int]:
        all_chunk_sizes = []
        current_start_step = 0
        while current_start_step < sequence_len:
            next_update_step = self.next_update_step(state, current_start_step)
            if next_update_step > sequence_len:
                seq_end_step = sequence_len
            elif next_update_step == current_start_step:
                break
            else:
                seq_end_step = next_update_step
            all_chunk_sizes.append(seq_end_step - current_start_step)
            current_start_step = seq_end_step
        return all_chunk_sizes
        
    def next_update_step(self, state: TensorDict[str, TensorDict], current_start_step: int) -> int:
        all_next_update_steps = []
        for block_name in state.keys():
            if block_name in self.frequency_map:
                block_update_frequency = self.frequency_map[block_name]
                block_next_update_step = math.ceil((current_start_step + 1) / block_update_frequency) * block_update_frequency
                all_next_update_steps.append(block_next_update_step)
        return min(all_next_update_steps) if all_next_update_steps else current_start_step
    

def rebuild_state(grad_weight_values_list: list[torch.Tensor],
                  non_grad_weight_values_list: list[torch.Tensor],
                  grad_weight_keys_list: list[str],
                  non_grad_weight_keys_list: list[str],
                  weights_keys: str) -> dict[str, dict]:
    new_state = {}
    for grad_weight_value, grad_weight_key in zip(grad_weight_values_list, grad_weight_keys_list, strict=True):
        block_name, grad_key = grad_weight_key.split(AssocMemory.DEFAULT_GRADIENT_KEY_SPLITTER)
        if block_name not in new_state:
            new_state[block_name] = {}
        for weights_key in weights_keys:
            if weights_key not in new_state[block_name]:
                new_state[block_name][weights_key] = {}
            new_state[block_name][weights_key][grad_key] = grad_weight_value
    for non_grad_weight_value, non_grad_weight_key in zip(non_grad_weight_values_list, non_grad_weight_keys_list, strict=True):
        block_name, grad_key = non_grad_weight_key.split(AssocMemory.DEFAULT_GRADIENT_KEY_SPLITTER)
        if block_name not in new_state:
            new_state[block_name] = {}
        if grad_key not in new_state[block_name]:
            new_state[block_name][grad_key] = non_grad_weight_value
        else:
            for weights_key in weights_keys:
                if grad_key == weights_key and isinstance(non_grad_weight_value, dict):
                    new_state[block_name][grad_key].update(non_grad_weight_value)
                else:
                    raise ValueError(f"Unsupported non-grad weight value type: {type(non_grad_weight_value)}")
    return new_state

def rebuild_state_for_titans(grad_weight_values_list: list[torch.Tensor],
                             grad_weight_keys_list: list[str],
                             block_name: str) -> dict[str, dict]:
    new_state = {}
    for grad_weight_value, grad_weight_key in zip(grad_weight_values_list, grad_weight_keys_list, strict=True):
        if block_name not in new_state:
            new_state[block_name] = {'fast_weights': {}}
        new_state[block_name]['fast_weights'][grad_weight_key] = grad_weight_value
    return new_state
