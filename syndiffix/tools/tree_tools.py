from __future__ import annotations

import ast
import itertools
from typing import Any, Iterator

import pandas as pd
import numpy as np
import random

from ..interval import Interval
from ..tree import Node, Leaf, Branch, tree_walker, dump_tree
from ..synthesizer import Synthesizer
from ..microdata import generate_value
from ..common import get_items_combination_list
from ..interval import Interval


def tree_to_df(node: Node) -> pd.DataFrame:
    """
    Convert a syndiffix tree to a DataFrame representation.
    
    Args:
        node: The root node of the tree to serialize
        suppress: If True, remove low-count nodes
        
    Returns:
        DataFrame with columns: node_id, node_type, ranges, count, children
    """

    rows = []
    node_id = 0
    node_to_id = {}
    
    # First pass: assign IDs to all nodes
    for current_node, _, _ in tree_walker(node):
        node_to_id[id(current_node)] = node_id
        node_id += 1
    
    # Second pass: serialize all nodes with correct child references
    node_id = 0
    for current_node, _, _ in tree_walker(node):
        
        suppress = False
        if isinstance(current_node, Leaf):
            low_threshold = current_node.context.anonymization_context.anonymization_params.low_count_params.low_threshold
            if not current_node.is_over_threshold(low_threshold):
                suppress = True

        # Create ranges_data based on conditional logic
        ranges_data = []
        for i in range(len(current_node.snapped_intervals)):
            actual_interval = current_node.actual_intervals[i]
            snapped_interval = current_node.snapped_intervals[i]
            
            # Use actual_intervals if min == max, otherwise use snapped_intervals
            if actual_interval.min == actual_interval.max:
                selected_interval = actual_interval
            else:
                selected_interval = snapped_interval
            
            ranges_data.append((round(float(selected_interval.min), 15), round(float(selected_interval.max), 15)))

        # Determine node type
        node_type = "Leaf" if isinstance(current_node, Leaf) else "Branch"
        
        # For Branch nodes, serialize children mapping with keys in sorted order
        # Only include children that are in node_to_id (i.e., not suppressed)
        children_data = None
        if isinstance(current_node, Branch):
            children_data = {child_index: node_to_id[id(child)] 
                           for child_index, child in sorted(current_node.children.items())
                           if id(child) in node_to_id}

        # Get exact count
        if isinstance(current_node, Leaf):
            true_count = len(current_node.rows)
        else:  # Branch
            true_count = len(list(current_node._matching_rows()))
        
        row_data = {
            'node_id': node_id,
            'node_type': node_type,
            'ranges': str(ranges_data),
            'count': current_node.noisy_count(),
            'true_count': true_count,
            'suppress': suppress,
            'children': str(children_data) if children_data else None
        }
        
        rows.append(row_data)
        node_id += 1
    
    return pd.DataFrame(rows)


def df_to_tree(df: pd.DataFrame) -> Any:
    """
    Reconstruct a tree from a DataFrame created by tree_to_df().
    
    Args:
        df: DataFrame containing serialized tree data
        
    Returns:
        The root node of the reconstructed tree
        
    Note: 
        This function cannot fully reconstruct nodes because it lacks context, subnodes, 
        and other essential data. It creates placeholder nodes with the serialized properties.
    """
    
    if df.empty:
        raise ValueError("Cannot reconstruct tree from empty DataFrame")
    
    # Create placeholder nodes (this is incomplete without context data)
    nodes = {}
    
    # First pass: create all nodes with basic properties
    for _, row in df.iterrows():
        node_id = row['node_id']
        node_type = row['node_type']
        
        # Parse intervals from string representation
        ranges_data = ast.literal_eval(row['ranges'])
        
        # Convert to Interval objects (values are already Python floats from serialization)
        ranges = tuple(Interval(min_val, max_val) for min_val, max_val in ranges_data)
        
        # Note: We cannot create real nodes without Context, so this creates incomplete placeholders
        if node_type == "Leaf":
            # Create a placeholder Leaf (this would normally require context and initial_row)
            placeholder_node = type('PlaceholderLeaf', (), {
                'node_id': node_id,
                'node_type': node_type,
                'ranges': ranges,
                'count': row['count'],
                'true_count': row['true_count'],
                'suppress': row['suppress'],
                'children': None,
            })()
        else:  # Branch
            placeholder_node = type('PlaceholderBranch', (), {
                'node_id': node_id,
                'node_type': node_type,
                'ranges': ranges,
                'count': row['count'],
                'true_count': row['true_count'],
                'suppress': row['suppress'],
                'children': {},
            })()
        
        nodes[node_id] = placeholder_node
    
    # Second pass: reconstruct parent-child relationships for Branch nodes
    for _, row in df.iterrows():
        node_id = row['node_id']
        if row['children'] and row['children'] != 'None':
            children_data = ast.literal_eval(row['children'])
            if hasattr(nodes[node_id], 'children'):
                nodes[node_id].children = {child_index: nodes[child_id] 
                                         for child_index, child_id in children_data.items() 
                                         if child_id is not None}
    
    # Return the first node (assumed to be root)
    return nodes[0]


def row_to_node(df_row: pd.Series) -> dict:
    """
    Convert a single row from a tree DataFrame to a dictionary.
    
    Args:
        df_row: A single row from a DataFrame created by tree_to_df()
        
    Returns:
        A dictionary containing all the node properties
    """
    
    node_type = df_row['node_type']
    
    # Parse intervals from string representation
    ranges_data = ast.literal_eval(df_row['ranges'])
    
    # Convert intervals to dictionaries with min/max keys
    ranges = [{'min': min_val, 'max': max_val} for min_val, max_val in ranges_data]
    
    # Create dictionary directly instead of placeholder object
    node_dict = {
        'node_id': df_row['node_id'],
        'ranges': ranges,
        'count': df_row['count'],
        'node_type': node_type,
        'true_count': df_row['true_count'],
        'suppress': df_row['suppress'],
    }
    
    # Add children data
    node_dict['children'] = {}
    if node_type == "Branch" and df_row['children'] and df_row['children'] != 'None':
        children_data = ast.literal_eval(df_row['children'])
        node_dict['children'] = children_data
    
    return node_dict


def dump_placeholder_tree(placeholder_node: Any, indent: int = 0) -> None:
    """Display the placeholder tree structure with directory-like indentation."""
    indent_str = "  " * indent

    # Format snapped_interval as [(min, max), (min, max), ...]
    intervals_str = ", ".join(f"({interval.min}, {interval.max})" for interval in placeholder_node.ranges)

    noisy_count = placeholder_node.count

    # Print this node's info
    print(f"{indent_str}[{intervals_str}] noisy count: {noisy_count}")

    # Recursively print children if this is a placeholder Branch
    if hasattr(placeholder_node, 'children') and placeholder_node.children:
        for child_index in sorted(placeholder_node.children.keys()):
            child = placeholder_node.children[child_index]
            dump_placeholder_tree(child, indent + 1)

class TestNode:
    """
    A class to convert and store tree DataFrame nodes.
    
    Converts each row in the DataFrame to a dictionary with the same format
    as df_to_complete_leafs() output: 'ranges' (from snapped_intervals), 'count' 
    (from _noisy_count_cache), and 'initial' (always True for existing nodes).

    The _in suffix refers to structures taken from the inner represention of syndiffix. The _ex suffix refers to the external representations.
    """
    
    def __init__(self, syn: Synthesizer, comb: tuple, leafs_mode: str = "none", range_extend_fraction: float = 0.25) -> None:
        self.leafs_mode = leafs_mode
        self.range_extend_fraction = range_extend_fraction
        self.nodes_in = {}
        self.nodes_supp_in = {}
        self.leafs_in = {}
        self.points_in = None
        self.histogram_in = None
        self.assigned_values_in = None
        self.assigned_values_ex = None
        self.convertors = get_items_combination_list(comb, syn.column_convertors)
        self.unsafe_rng = random.Random()
        self.null_mappings = get_items_combination_list(comb, syn.forest.null_mappings)
        self.root = syn.forest._tree_cache[comb]
        # Note that tree_to_df determines suppress status. tree_to_df in this form won't
        # be needed in the long run
        self.df_tree_in = tree_to_df(self.root)
        # set self.max_node_id to be the maximum node_id in self.df_tree_in
        self.max_node_id = self.df_tree_in['node_id'].max() if not self.df_tree_in.empty else -1
        
        for _, row in self.df_tree_in.iterrows():
            # Use row_to_node to convert the row to a dictionary
            node_dict = row_to_node(row)
            
            # Add the 'initial' field to match expected format
            node_dict['initial'] = True  # All nodes from DataFrame are original/initial nodes

            self.nodes_in[node_dict['node_id']] = node_dict

        if len(comb) == 1:
            self.is_1d = True
            self.is_2d = False
        elif len(comb) == 2:
            self.is_2d = True
            self.is_1d = False
        else:
            raise ValueError(f"Nodes have ranges with length {len(self.nodes_in[self.max_node_id]['ranges'])}, only 1 or 2 are supported")

        self.suppress()
        if self.leafs_mode != "none" and (self.is_1d or self.is_2d):
            self.add_leafs_to_nodes_supp()
        # Must be after add_leafs_to_nodes_supp
        self.top_down_count_adjust()
        # The following creates self.leafs_in, which contains only leaf nodes
        self.make_only_leafs()
        if self.is_1d:
            # The following creates self.points_in and self.histogram_in
            self.prepare_1d_leafs()
            col_name = syn.forest.orig_data.columns[comb[0]]
            self.assigned_values_in, self.assigned_values_ex = self.assign_1d_values(col_name = col_name)

        self.nodes_in_stats = self.compute_statistics(self.nodes_in)
        self.nodes_supp_in_stats = self.compute_statistics(self.nodes_supp_in)
        self.leafs_in_stats = self.compute_statistics(self.leafs_in)

    def dump_tree_from_root(self) -> None:
        dump_tree(self.root)

    def integrity_checks(self) -> tuple[bool, dict]:
        """
        Run all integrity checks and return a summary of errors found.
        
        Returns:
            dict: Dictionary with integrity check results
        """
        results = {}
        
        results['nodes_integrity'] = self._check_nodes_integrity(self.nodes_in)
        results['nodes_leafs_integrity'] = self._check_leafs_integrity(self.nodes_in)
        results['nodes_supp_integrity'] = self._check_nodes_integrity(self.nodes_supp_in)
        results['nodes_supp_leafs_integrity'] = self._check_leafs_integrity(self.nodes_supp_in)
        results['leafs_integrity'] = self._check_leafs_integrity(self.leafs_in)
        problems_found = False
        for key, value in results.items():
            if value:  # If the list is not empty, there are problems
                problems_found = True
        
        return problems_found, results

    def suppress(self):
        """
        Create a suppressed version of nodes by filtering out suppressed nodes
        and cleaning up child references.
        """
        # Initialize the suppressed nodes dictionary
        self.nodes_supp_in = {}
        
        # Add all elements where suppress==False
        for node_id, node_dict in self.nodes_in.items():
            if not node_dict['suppress']:
                self.nodes_supp_in[node_id] = node_dict.copy()
        
        # Loop through because a deletion in one round can result in
        # more invalid child references
        # Walk through nodes_supp and remove invalid child references
        for node_id, node_dict in self.nodes_supp_in.items():
            if 'children' in node_dict and node_dict['children']:
                # Filter children to only include those that exist in nodes_supp
                valid_children = {child_index: child_id
                                  for child_index, child_id in node_dict['children'].items()
                                  if child_id in self.nodes_supp_in}
                node_dict['children'] = valid_children
                # If no valid children remain, convert to a Leaf
                if not valid_children:
                    node_dict['node_type'] = 'Leaf'
                    node_dict['children'] = {}
        # copy self.nodes_supp_in to self.nodes_supp_in_temp   (TODO remove)
        self.nodes_supp_in_temp = self.nodes_supp_in.copy()

    def _check_nodes_integrity(self, nodes) -> list[str]:
        """
        Check the integrity of the nodes structure.
        
        Validates two conditions:
        1. For all items except node_id==0, there is an item with that node_id among its children
        2. For all node_id in children, there is a valid item with key==node_id
        
        Returns:
            list[str]: List of error messages, empty if all checks pass
        """
        errors = []
        
        # Check condition 2: All child node_ids must exist in nodes
        for node_id, node_dict in nodes.items():
            if 'children' in node_dict and node_dict['children']:
                for child_index, child_id in node_dict['children'].items():
                    if child_id not in nodes:
                        errors.append(f"Node {node_id} references child {child_id} which doesn't exist in nodes")

        # Check condition 1: All nodes except root (node_id==0) must be referenced as children
        referenced_nodes = set()
        for node_id, node_dict in nodes.items():
            if 'children' in node_dict and node_dict['children']:
                referenced_nodes.update(node_dict['children'].values())
        
        for node_id in nodes.keys():
            if node_id != 0 and node_id not in referenced_nodes:
                errors.append(f"Node {node_id} is not referenced as a child by any other node")
        
        # Ensure that all Branch nodes have at least one child
        for node_id, node_dict in nodes.items():
            if node_dict['node_type'] == 'Branch':
                if 'children' not in node_dict or not node_dict['children']:
                    errors.append(f"Branch node {node_id} has no children")
        
        return errors

    def _check_leafs_integrity(self, nodes) -> list[str]:
        """
        Check the integrity of leaf nodes by ensuring no ranges overlap.
        
        Validates that all leaf node ranges do not overlap with each other.
        Works for both 1D and 2D ranges.
        
        Returns:
            list[str]: List of error messages, empty if all checks pass
        """
        errors = []
        leaf_items = [(node_id, node_dict) for node_id, node_dict in nodes.items() if node_dict['node_type'] == 'Leaf']
        
        for i, (leaf_id1, leaf1) in enumerate(leaf_items):
            for j, (leaf_id2, leaf2) in enumerate(leaf_items[i + 1:], start=i + 1):
                ranges1 = leaf1['ranges']
                ranges2 = leaf2['ranges']
                
                # Check if ranges have same dimensionality
                if len(ranges1) != len(ranges2):
                    continue  # Skip comparison if different dimensions
                
                # Check for overlap in all dimensions
                # For boxes to overlap, they must overlap in ALL dimensions
                overlap = True
                for dim in range(len(ranges1)):
                    range1 = ranges1[dim]
                    range2 = ranges2[dim]
                    
                    # Two ranges don't overlap if one ends before the other starts
                    if range1['max'] <= range2['min'] or range2['max'] <= range1['min']:
                        overlap = False
                        break
                
                if overlap:
                    # Calculate widths for error message
                    widths1 = [r['max'] - r['min'] for r in ranges1]
                    widths2 = [r['max'] - r['min'] for r in ranges2]
                    err_message = f"Leaf {leaf_id1} and leaf {leaf_id2} have overlapping ranges: "\
                                  f"{ranges1} (widths: {widths1}) overlaps with {ranges2} (widths: {widths2})"
                    errors.append(err_message)

        return errors

    def add_leafs_to_nodes_supp(self) -> None:
        """
        Generate nodes for missing children in self.nodes_supp_in.

        self.leafs_mode can be 'simple' or 'leaf_only': 
        - 'simple': Add missing children for branches based only on count differences.
        - 'leaf_only': Additionally only if they would be adjacent to existing leaf nodes
       
        Raises:
            ValueError: If any node has ranges with length > 2
        """
        # Initialize leafs
        leaf_id = self.max_node_id + 1  # Start new node IDs after existing max
        
        leaf_nodes_list = []
        if self.leafs_mode == 'leaf_only':
            # Get all leaf nodes from the tree
            leaf_nodes_list = [node for node in self.nodes_supp_in.values() 
                                    if node['node_type'] == 'Leaf']
        
        # Walk through all nodes in nodes_supp_in
        leafs_to_add = []
        for node_id, node_dict in self.nodes_supp_in.items():
            # Check ranges length constraint
            if len(node_dict['ranges']) > 2:
                raise ValueError(f"Node {node_id} has ranges with length {len(node_dict['ranges'])}, maximum allowed is 2")
            
            if node_dict['node_type'] == 'Branch':
                # Check for missing children and potentially create new nodes
                children = node_dict.get('children', {})
                expected_children = 2 if self.is_1d else 4
                
                if len(children) < expected_children:
                    branch_count = node_dict['count']
                    child_sum = sum(self.nodes_supp_in[child_id]['count'] for child_id in children.values())
                    
                    missing_child_indices = set()
                    if branch_count > child_sum:
                        # Find missing children
                        if self.leafs_mode == 'simple':
                            existing_child_indices = set(children.keys())
                            all_child_indices = set(range(expected_children))
                            missing_child_indices = all_child_indices - existing_child_indices
                        elif self.leafs_mode == 'leaf_only':
                            missing_child_indices = self._get_leaf_only_missing_children(node_dict, leaf_nodes_list)
                        else:
                            raise ValueError(f"add_leafs_to_nodes_supp: Unknown mode: {self.leafs_mode}")

                    num_missing = len(missing_child_indices)
                    if num_missing > 0:
                        # Distribute remaining count as evenly as possible
                        remaining_count = branch_count - child_sum
                        child_counts = self._distribute_count_evenly(remaining_count, num_missing)
                        
                        # Create new nodes for missing children with non-zero counts
                        for i, child_index in enumerate(sorted(missing_child_indices)):
                            new_node_count = child_counts[i]
                            if new_node_count > 0:  # Only create nodes with positive count
                                node_dict['children'][child_index] = int(leaf_id)
                                new_ranges = self._calculate_child_ranges(node_dict['ranges'], child_index)
                                
                                new_node = {
                                    'node_id': int(leaf_id),
                                    'node_type': 'Leaf',
                                    'initial': False,
                                    'count': new_node_count,
                                    'ranges': new_ranges,
                                    'true_count': None,
                                    'suppress': False,
                                    'children': {}
                                }
                                leafs_to_add.append(new_node)
                                leaf_id += 1
        for new_node in leafs_to_add:
            self.nodes_supp_in[new_node['node_id']] = new_node

    def _distribute_count_evenly(self, total_count: float, num_recipients: int) -> list[int]:
        """
        Distribute a total count as evenly as possible among recipients.
        
        Args:
            total_count: Total count to distribute (can be float)
            num_recipients: Number of recipients to distribute to
            
        Returns:
            List of integer counts that sum to round(total_count)
        """
        if num_recipients == 0:
            return []
        
        # Round total to nearest integer
        total_int = int(round(total_count))
        
        # Base count for each recipient
        base_count = total_int // num_recipients
        remainder = total_int % num_recipients
        
        # Distribute base count to all, then add 1 to first 'remainder' recipients
        counts = [base_count] * num_recipients
        for i in range(remainder):
            counts[i] += 1
            
        return counts

    def _get_leaf_only_missing_children(self, node_dict: dict, leaf_nodes_list: list[dict]) -> set[int]:
        missing_child_indices = set()
        
        print(f"Checking node {node_dict} for leaf-only missing children")
        # Get all possible missing children
        children = node_dict.get('children', {})
        expected_children = 2 if self.is_1d else 4
        existing_child_indices = set(children.keys())
        all_child_indices = set(range(expected_children))
        potential_missing = all_child_indices - existing_child_indices
        
        # Check each potential missing child
        for child_index in potential_missing:
            # Calculate what the missing child's range would be
            child_ranges = self._calculate_child_ranges(node_dict['ranges'], child_index)
            print(f"  Potential missing child index {child_index} with ranges {child_ranges}")
            
            # Check if this missing child would be adjacent to any existing leaf
            for leaf_node in leaf_nodes_list:
                if self._ranges_are_adjacent(child_ranges, leaf_node['ranges']):
                    print(f"    Found adjacent leaf {leaf_node['node_id']} with ranges {leaf_node['ranges']}")
                    missing_child_indices.add(child_index)
                    break  # Found one adjacent leaf, that's enough
        
        return missing_child_indices

    def _ranges_are_adjacent(self, ranges1, ranges2):
        """Check if two multi-dimensional ranges are adjacent (share a border)"""
        if len(ranges1) != len(ranges2):
            return False
        
        # For ranges to be adjacent, they must:
        # 1. Touch at boundaries in exactly one dimension
        # 2. Overlap or touch in all other dimensions
        
        touching_dimensions = 0
        overlapping_dimensions = 0
        
        for i in range(len(ranges1)):
            r1_min, r1_max = ranges1[i]['min'], ranges1[i]['max']
            r2_min, r2_max = ranges2[i]['min'], ranges2[i]['max']
            
            # Check if ranges touch at boundaries in this dimension
            if r1_max == r2_min or r1_min == r2_max:
                touching_dimensions += 1
            # Check if ranges overlap in this dimension
            elif not (r1_max <= r2_min or r2_max <= r1_min):
                overlapping_dimensions += 1
            # If ranges are separate in this dimension, they can't be adjacent
            else:
                return False
        
        # Adjacent if they touch in exactly one dimension and overlap in all others
        return touching_dimensions == 1 and overlapping_dimensions == (len(ranges1) - 1)

    def _calculate_child_ranges(self, parent_ranges, child_index):
        """
        Calculate the ranges for a missing child based on parent ranges and child index.
        
        Args:
            parent_ranges: List of parent range dictionaries
            child_index: Index of the missing child (0-3)
            
        Returns:
            List of range dictionaries for the child
        """
        if self.is_1d:
            parent_range = parent_ranges[0]
            mid_point = round((parent_range['min'] + parent_range['max']) / 2, 15)
            
            if child_index == 0:
                # Lower half
                return [{'min': parent_range['min'], 'max': mid_point}]
            else:  # child_index == 1
                # Upper half
                return [{'min': mid_point, 'max': parent_range['max']}]
        
        else:  # is_2d
            range0 = parent_ranges[0]
            range1 = parent_ranges[1]
            mid_point0 = round((range0['min'] + range0['max']) / 2, 15)
            mid_point1 = round((range1['min'] + range1['max']) / 2, 15)
            
            if child_index == 0:
                # Lower half of range[0], lower half of range[1]
                return [
                    {'min': range0['min'], 'max': mid_point0},
                    {'min': range1['min'], 'max': mid_point1}
                ]
            elif child_index == 1:
                # Lower half of range[0], upper half of range[1]
                return [
                    {'min': range0['min'], 'max': mid_point0},
                    {'min': mid_point1, 'max': range1['max']}
                ]
            elif child_index == 2:
                # Upper half of range[0], lower half of range[1]
                return [
                    {'min': mid_point0, 'max': range0['max']},
                    {'min': range1['min'], 'max': mid_point1}
                ]
            else:  # child_index == 3
                # Upper half of range[0], upper half of range[1]
                return [
                    {'min': mid_point0, 'max': range0['max']},
                    {'min': mid_point1, 'max': range1['max']}
                ]
    
    def make_only_leafs(self):
        """
        Create self.leafs_in dict by collecting only leaf nodes from nodes_supp.
        """
        self.leafs_in = {}
        
        for node_id, node_dict in self.nodes_supp_in.items():
            if node_dict['node_type'] == 'Leaf':
                self.leafs_in[node_id] = node_dict.copy()
    
    def compute_statistics(self, nodes_dict: dict) -> dict:
        """
        Compute comprehensive statistics about a nodes dictionary.
        
        Args:
            nodes_dict: Dictionary of nodes to analyze
            
        Returns:
            Dictionary containing various statistics about the nodes
        """
        import numpy as np
        from collections import defaultdict
        
        if not nodes_dict:
            return {"error": "Empty nodes dictionary"}
        
        stats = {}
        
        # Basic counts
        stats['total_nodes'] = len(nodes_dict)
        
        # Node type counts
        node_types = defaultdict(int)
        for node in nodes_dict.values():
            node_types[node['node_type']] += 1
        stats['node_type_counts'] = dict(node_types)
        
        # Initial vs created nodes
        initial_counts = {'initial_true': 0, 'initial_false': 0}
        for node in nodes_dict.values():
            if node.get('initial', True):
                initial_counts['initial_true'] += 1
            else:
                initial_counts['initial_false'] += 1
        stats['initial_counts'] = initial_counts
        
        # Suppress counts
        suppress_counts = {'suppress_true': 0, 'suppress_false': 0}
        for node in nodes_dict.values():
            if node.get('suppress', False):
                suppress_counts['suppress_true'] += 1
            else:
                suppress_counts['suppress_false'] += 1
        stats['suppress_counts'] = suppress_counts
        
        # Range width statistics
        width_stats = defaultdict(lambda: {'counts': [], 'node_count': 0})
        dimensionality_counts = defaultdict(int)
        
        for node in nodes_dict.values():
            ranges = node['ranges']
            dimensionality_counts[len(ranges)] += 1
            
            # Calculate width (use minimum width for multi-dimensional ranges)
            if ranges:
                widths = [r['max'] - r['min'] for r in ranges]
                min_width = min(widths)
                width_key = round(min_width, 15)
                
                width_stats[width_key]['counts'].append(node['count'])
                width_stats[width_key]['node_count'] += 1
        
        # Convert width stats to final format
        range_width_stats = {}
        for width, data in width_stats.items():
            counts = np.array(data['counts'])
            range_width_stats[width] = {
                'node_count': data['node_count'],
                'count_avg': float(np.mean(counts)),
                'count_min': float(np.min(counts)),
                'count_max': float(np.max(counts)),
                'count_std': float(np.std(counts)),
                'count_median': float(np.median(counts))
            }
        
        stats['range_width_stats'] = range_width_stats
        stats['dimensionality_counts'] = dict(dimensionality_counts)
        
        # Count vs true_count statistics (only for nodes with true_count)
        count_diffs = []
        top_down_count_diffs = []
        count_vs_top_down_diffs = []
        nodes_with_true_count = 0
        nodes_without_true_count = 0
        nodes_with_top_down_and_true_count = 0
        nodes_with_count_and_top_down = 0
        
        for node in nodes_dict.values():
            if node.get('true_count') is not None:
                nodes_with_true_count += 1
                diff = node['count'] - node['true_count']
                count_diffs.append(diff)
                
                # Also check for top_down_count comparison
                if 'top_down_count' in node:
                    nodes_with_top_down_and_true_count += 1
                    top_down_diff = node['top_down_count'] - node['true_count']
                    top_down_count_diffs.append(top_down_diff)
            else:
                nodes_without_true_count += 1
            
            # Check for count vs top_down_count comparison
            if 'top_down_count' in node:
                nodes_with_count_and_top_down += 1
                count_top_down_diff = node['count'] - node['top_down_count']
                count_vs_top_down_diffs.append(count_top_down_diff)
        
        if count_diffs:
            count_diffs = np.array(count_diffs)
            stats['count_vs_true_count'] = {
                'nodes_with_true_count': nodes_with_true_count,
                'nodes_without_true_count': nodes_without_true_count,
                'diff_min': float(np.min(count_diffs)),
                'diff_max': float(np.max(count_diffs)),
                'diff_avg': float(np.mean(count_diffs)),
                'diff_std': float(np.std(count_diffs)),
                'diff_median': float(np.median(count_diffs))
            }
        else:
            stats['count_vs_true_count'] = {
                'nodes_with_true_count': 0,
                'nodes_without_true_count': nodes_without_true_count,
                'note': 'No nodes with true_count available'
            }
        
        # Top_down_count vs true_count statistics
        if top_down_count_diffs:
            top_down_count_diffs = np.array(top_down_count_diffs)
            stats['top_down_count_vs_true_count'] = {
                'nodes_with_top_down_and_true_count': nodes_with_top_down_and_true_count,
                'diff_min': float(np.min(top_down_count_diffs)),
                'diff_max': float(np.max(top_down_count_diffs)),
                'diff_avg': float(np.mean(top_down_count_diffs)),
                'diff_std': float(np.std(top_down_count_diffs)),
                'diff_median': float(np.median(top_down_count_diffs))
            }
        else:
            stats['top_down_count_vs_true_count'] = {
                'nodes_with_top_down_and_true_count': 0,
                'note': 'No nodes with both top_down_count and true_count available'
            }
        
        # Count vs top_down_count statistics
        if count_vs_top_down_diffs:
            count_vs_top_down_diffs = np.array(count_vs_top_down_diffs)
            stats['count_vs_top_down_count'] = {
                'nodes_with_count_and_top_down': nodes_with_count_and_top_down,
                'diff_min': float(np.min(count_vs_top_down_diffs)),
                'diff_max': float(np.max(count_vs_top_down_diffs)),
                'diff_avg': float(np.mean(count_vs_top_down_diffs)),
                'diff_std': float(np.std(count_vs_top_down_diffs)),
                'diff_median': float(np.median(count_vs_top_down_diffs))
            }
        else:
            stats['count_vs_top_down_count'] = {
                'nodes_with_count_and_top_down': 0,
                'note': 'No nodes with both count and top_down_count available'
            }

        # Count statistics
        all_counts = [node['count'] for node in nodes_dict.values()]
        if all_counts:
            all_counts = np.array(all_counts)
            stats['count_stats'] = {
                'min': float(np.min(all_counts)),
                'max': float(np.max(all_counts)),
                'avg': float(np.mean(all_counts)),
                'std': float(np.std(all_counts)),
                'median': float(np.median(all_counts)),
                'sum': float(np.sum(all_counts))
            }
        
        # Top-down count statistics (if available)
        top_down_counts = [node['top_down_count'] for node in nodes_dict.values() if 'top_down_count' in node]
        if top_down_counts:
            top_down_counts = np.array(top_down_counts)
            stats['top_down_count_stats'] = {
                'min': float(np.min(top_down_counts)),
                'max': float(np.max(top_down_counts)),
                'avg': float(np.mean(top_down_counts)),
                'std': float(np.std(top_down_counts)),
                'median': float(np.median(top_down_counts)),
                'sum': float(np.sum(top_down_counts)),
                'nodes_with_top_down_count': len(top_down_counts)
            }
        
        # Tree depth statistics (for nodes with children)
        branch_child_counts = []
        for node in nodes_dict.values():
            if node['node_type'] == 'Branch' and 'children' in node:
                branch_child_counts.append(len(node['children']))
        
        if branch_child_counts:
            branch_child_counts = np.array(branch_child_counts)
            stats['branch_children_stats'] = {
                'avg_children': float(np.mean(branch_child_counts)),
                'min_children': int(np.min(branch_child_counts)),
                'max_children': int(np.max(branch_child_counts)),
                'std_children': float(np.std(branch_child_counts))
            }
        
        # Range coverage statistics
        if nodes_dict:
            sample_node = next(iter(nodes_dict.values()))
            if sample_node['ranges']:
                num_dimensions = len(sample_node['ranges'])
                coverage_stats = {}
                
                for dim in range(num_dimensions):
                    dim_mins = [node['ranges'][dim]['min'] for node in nodes_dict.values() 
                               if len(node['ranges']) > dim]
                    dim_maxs = [node['ranges'][dim]['max'] for node in nodes_dict.values() 
                               if len(node['ranges']) > dim]
                    
                    if dim_mins and dim_maxs:
                        coverage_stats[f'dimension_{dim}'] = {
                            'overall_min': min(dim_mins),
                            'overall_max': max(dim_maxs),
                            'overall_range': max(dim_maxs) - min(dim_mins)
                        }
                
                stats['coverage_stats'] = coverage_stats
        
        return stats
    
    def top_down_count_adjust(self):
        """
        Create top_down_count adjustments working from root to leaves.
        
        Starting with the root node (node_id==0), propagates count adjustments
        down the tree so that parent top_down_count equals sum of children top_down_count.
        """
        # Initialize root node's top_down_count
        if 0 in self.nodes_supp_in:
            self.nodes_supp_in[0]['top_down_count'] = self.nodes_supp_in[0]['count']
        else:
            return  # No root node found
        
        # Process nodes level by level (breadth-first traversal)
        nodes_to_process = [0]  # Start with root
        
        while nodes_to_process:
            current_node_id = nodes_to_process.pop(0)
            current_node = self.nodes_supp_in[current_node_id]
            
            # Only process Branch nodes that have children
            if (current_node['node_type'] == 'Branch' and 
                'children' in current_node and 
                current_node['children']):
                
                # Get parent's top_down_count
                parent_top_down_count = current_node['top_down_count']
                
                # Calculate total count of children
                child_ids = list(current_node['children'].values())
                child_count_sum = sum(self.nodes_supp_in[child_id]['count'] for child_id in child_ids)

                if child_count_sum == 0:
                    raise ValueError("top_down_count_adjust: child_count_sum is zero, cannot adjust top_down_count proportionally")
                
                # Calculate proportional adjustment for each child
                for child_id in child_ids:
                    child_node = self.nodes_supp_in[child_id]
                    child_original_count = child_node['count']
                    
                    # Calculate proportional top_down_count
                    proportion = child_original_count / child_count_sum
                    child_node['top_down_count'] = parent_top_down_count * proportion
                    
                    # Add child to processing queue if it's a Branch
                    if child_node['node_type'] == 'Branch':
                        nodes_to_process.append(child_id)

    def prepare_1d_leafs(self):
        """
        Prepare 1D leafs data by separating into points and histogram, filling gaps.
        
        Creates self.points_in and self.histogram_in from self.leafs_in for 1D data.
        Points are exact values (min==max), histogram entries are ranges (min!=max).
        Gaps in histogram are filled with zero counts, ensuring coverage from 0 to 1.0.
        """
        # Check dimensionality
        if not self.leafs_in:
            self.points_in = []
            self.histogram_in = []
            return

        self.round_leafs_1d_in()

        # Initialize lists
        self.points_in = []
        histogram_intervals = []
        
        # Process each leaf
        for leaf in self.leafs_in.values():
            range_info = leaf['ranges'][0]
            min_val = range_info['min']
            max_val = range_info['max']
            count = leaf['rounded_count']
            
            if min_val == max_val:
                # Point data
                self.points_in.append((min_val, count))
            else:
                # Histogram interval
                histogram_intervals.append(((min_val, max_val), count))
        
        # Sort points by ascending P
        self.points_in.sort(key=lambda x: x[0])
        
        # Sort histogram intervals by ascending min
        histogram_intervals.sort(key=lambda x: x[0][0])
        
        # Fill gaps in histogram and ensure coverage from 0 to 1.0
        self.histogram_in = []
        current_pos = 0.0
        
        for (min_val, max_val), count in histogram_intervals:
            # Fill gap before this interval if needed
            if current_pos < min_val:
                self.histogram_in.append(((current_pos, min_val), 0))
            
            # Add the actual interval
            self.histogram_in.append(((min_val, max_val), count))
            current_pos = max_val
        
        # Fill gap at the end if needed
        if current_pos < 1.0:
            self.histogram_in.append(((current_pos, 1.0), 0))
        
        # Handle the case where there were no intervals at all
        if not histogram_intervals:
            self.histogram_in = [((0.0, 1.0), 0)]

    def check_rounding_1d(self):
        total_top_down = sum(leaf['top_down_count'] for leaf in self.leafs_in.values())
        total_rounded = sum(leaf.get('rounded_count', 0) for leaf in self.leafs_in.values())
        if round(total_top_down) != total_rounded:
            raise ValueError(f"check_rounding_1d: total rounded {total_rounded} does not match rounded total top_down {round(total_top_down)}")

    def round_leafs_1d_in(self) -> None:
        total_rounded = 0
        total_unrounded = 0
        up_rounded_diffs = []
        down_rounded_diffs = []
        for node_id, leaf in self.leafs_in.items():
            total_unrounded += leaf['top_down_count']
            leaf['rounded_count'] = int(round(leaf['top_down_count']))
            total_rounded += leaf['rounded_count']
            diff = leaf['rounded_count'] - leaf['top_down_count']
            if diff > 0:
                up_rounded_diffs.append([node_id, diff])
            else:
                down_rounded_diffs.append([node_id, -diff])
        needed_adjustment = round(total_unrounded) - total_rounded
        print(f"round_leafs_1d_in: total_unrounded={total_unrounded}, total_rounded={total_rounded}, needed_adjustment={needed_adjustment}")
        # having rounded all the counts, we may be off from the total we need
        # fix this by adjusting some of the rounded counts up or down by 1, working
        # from those with the largest rounding diffs
        if needed_adjustment > 0:
            # We need to increase some counts, so we work with those that we
            # rounded down the most
            sorted_diffs = sorted(down_rounded_diffs, key=lambda x: x[1], reverse=True)
            sorted_diffs += sorted(up_rounded_diffs, key=lambda x: x[1])
        else:
            # We need to decrease some counts, so we work with those that we
            # rounded up the most
            sorted_diffs = sorted(up_rounded_diffs, key=lambda x: x[1], reverse=True)
            sorted_diffs += sorted(down_rounded_diffs, key=lambda x: x[1])
        if len(sorted_diffs) < abs(needed_adjustment):
            raise ValueError("round_leafs_1d_in: not enough leafs to adjust to reach needed total")
        for i in range(abs(needed_adjustment)):
            node_id = sorted_diffs[i][0]
            if needed_adjustment > 0:
                self.leafs_in[node_id]['rounded_count'] += 1
                print(f"round_leafs_1d_in: increasing node_id={node_id} to {self.leafs_in[node_id]['rounded_count']}")
            else:
                if self.leafs_in[node_id]['rounded_count'] <= 0:
                    continue
                self.leafs_in[node_id]['rounded_count'] -= 1
                print(f"round_leafs_1d_in: decreasing node_id={node_id} to {self.leafs_in[node_id]['rounded_count']}")
        self.check_rounding_1d()

    def assign_1d_values(self, col_name: str = 'value') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create two 1-column DataFrames by assigning values based on 1D leafs and their top_down_counts. The first dataframe contains pre-converted values, the second contains converted values.
        
        For each leaf:
        - If min == max: assign that exact value
        - If min != max and both are ints: assign random integers inclusive of min, exclusive of max
        - If min != max and both are floats: assign random floats inclusive of both min and max
        
        Returns:
            pd.DataFrame with one column containing the assigned values
        """
        if not self.is_1d:
            raise ValueError("assign_1d_values only works with 1D data")
        
        if self.histogram_in is None or self.points_in is None:
            raise ValueError("assign_1d_values requires that prepare_1d_leafs be run first")
        
        all_preconverted_values = []
        all_converted_values = []

        for value, count in self.points_in:
            all_preconverted_values.extend([value] * count)
            # This is kindof heavyweight, but for now...
            interval = Interval(value, value)
            converted_value, _ = generate_value(interval=interval, convertor=self.convertors[0], null_mapping=self.null_mappings[0], rng=self.unsafe_rng)
            all_converted_values.extend([converted_value] * count)

        # Assign random values from self.histogram_in
        for i, cur_bin in enumerate(self.histogram_in):
            left_bin = self.histogram_in[i - 1] if i > 0 else None
            right_bin = self.histogram_in[i + 1] if i < len(self.histogram_in) - 1 else None
            
            cur_min, cur_max = cur_bin[0]
            cur_count = cur_bin[1]
            if cur_count == 0:
                continue

            rules = []
            cur_range = cur_max - cur_min
            l_edge = 0.0
            rules = []
            if left_bin is not None:
                left_min, left_max = left_bin[0]
                left_range = left_max - left_min
                min_range = min(cur_range, left_range)
                seg_range = min_range * self.range_extend_fraction
                seg_prob = (seg_range / cur_range) * 0.5
                rules.append([[cur_min - seg_range, cur_min], seg_prob])
                rules.append([[cur_min, cur_min+seg_range], seg_prob*2])
                l_edge = cur_min + seg_range
            if right_bin is not None:
                right_min, right_max = right_bin[0]
                right_range = right_max - right_min
                min_range = min(cur_range, right_range)
                seg_range = min_range * self.range_extend_fraction
                seg_prob = (seg_range / cur_range) * 0.5
                rules.append([[l_edge, cur_max-seg_range], 1.0-(seg_prob*2)])
                rules.append([[cur_max-seg_range, cur_max], 1.0-seg_prob])
                rules.append([[cur_max, cur_max+seg_range], 1.0])
            else:
                rules.append([[l_edge, cur_max], 1.0])
            #print(f"Current bin {cur_bin}")
            #print(rules)
            new_values = self._assign_values_from_rules(rules, cur_count)
            all_preconverted_values.extend(new_values)
            for value in new_values:
                interval = Interval(value, value)
                converted_value, _ = generate_value(interval=interval, convertor=self.convertors[0], null_mapping=self.null_mappings[0], rng=self.unsafe_rng)
                all_converted_values.append(converted_value)

        return pd.DataFrame({col_name: all_preconverted_values}), pd.DataFrame({col_name: all_converted_values})

    def _assign_values_from_rules(self, rules, count):
        new_values = []
        for _ in range(count):
            rand_prob = random.random()
            for rule in rules:
                if rand_prob <= rule[1]:
                    range_min, range_max = rule[0]
                    if isinstance(range_min, int) and isinstance(range_max, int):
                        value = random.randint(range_min, range_max - 1)
                    else:
                        value = random.uniform(range_min, range_max)
                    new_values.append(value)
                    break
        return new_values

class TestNodeForest:
    def __init__(self, syn: Synthesizer, leafs_mode: str = 'none', range_extend_fraction: float = 0.25):
        """
        Manages the complete set of TestNode objects for a given Synthesizer.
        """
        self.test_nodes = {}
        self.range_extend_fraction = range_extend_fraction
        for r in range(1, len(syn.column_convertors) + 1):
            for comb in itertools.combinations(range(len(syn.column_convertors)), r):
                self.test_nodes[comb] = TestNode(syn, comb, leafs_mode=leafs_mode, range_extend_fraction=range_extend_fraction)