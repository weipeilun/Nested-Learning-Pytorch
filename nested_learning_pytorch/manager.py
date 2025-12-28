from __future__ import annotations

from typing import Dict, Tuple, List

import torch

from .memory.assoc_memory import AssocMemory, AssocMemSpec, AssocMemState


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
        
    def next_update_step(self, state: dict[str, AssocMemState]) -> int:
        all_next_update_steps = []
        for block_name, block_state in state.items():
            if block_state.last_update_step is not None and isinstance(block_state.last_update_step, torch.Tensor) and block_name in self.frequency_map:
                block_update_frequency = self.frequency_map[block_name]
                all_next_update_steps.append(block_state.last_update_step + block_update_frequency)
        return min(next_update_steps.min() for next_update_steps in all_next_update_steps).item()
