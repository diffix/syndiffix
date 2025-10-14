from __future__ import annotations

import ast
from typing import Any, Iterator

import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np

from ..interval import Interval
from ..tree import Node, Leaf, Branch, tree_walker


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


class TestNodes:
    """
    A class to convert and store tree DataFrame nodes.
    
    Converts each row in the DataFrame to a dictionary with the same format
    as df_to_complete_leafs() output: 'ranges' (from snapped_intervals), 'count' 
    (from _noisy_count_cache), and 'initial' (always True for existing nodes).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize TestNodes with a DataFrame created by tree_to_df().
        
        Args:
            df: DataFrame created by tree_to_df()
        """
        self.nodes = {}
        self.nodes_supp = {}
        self.leafs = {}
        # set self.max_node_id to be the maximum node_id in df
        self.max_node_id = df['node_id'].max() if not df.empty else -1
        if df.empty:
            return
        
        for _, row in df.iterrows():
            # Use row_to_node to convert the row to a dictionary
            node_dict = row_to_node(row)
            
            # Add the 'initial' field to match expected format
            node_dict['initial'] = True  # All nodes from DataFrame are original/initial nodes

            self.nodes[node_dict['node_id']] = node_dict

        self.suppress()
        self.add_leafs_to_nodes_supp()
        # Must be after add_leafs_to_nodes_supp
        self.top_down_count_adjust()
        self.make_only_leafs()

        self.nodes_stats = self.compute_statistics(self.nodes)
        self.nodes_supp_stats = self.compute_statistics(self.nodes_supp)
        self.leafs_stats = self.compute_statistics(self.leafs)

    def integrity_checks(self) -> tuple[bool, dict]:
        """
        Run all integrity checks and return a summary of errors found.
        
        Returns:
            dict: Dictionary with integrity check results
        """
        results = {}
        
        results['nodes_integrity'] = self._check_nodes_integrity(self.nodes)
        results['nodes_leafs_integrity'] = self._check_leafs_integrity(self.nodes)
        results['nodes_supp_integrity'] = self._check_nodes_integrity(self.nodes_supp)
        results['nodes_supp_leafs_integrity'] = self._check_leafs_integrity(self.nodes_supp)
        results['leafs_integrity'] = self._check_leafs_integrity(self.leafs)
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
        self.nodes_supp = {}
        
        # Add all elements where suppress==False
        for node_id, node_dict in self.nodes.items():
            if not node_dict['suppress']:
                self.nodes_supp[node_id] = node_dict.copy()
        
        # Loop through because a deletion in one round can result in
        # more invalid child references
        # Walk through nodes_supp and remove invalid child references
        for node_id, node_dict in self.nodes_supp.items():
            if 'children' in node_dict and node_dict['children']:
                # Filter children to only include those that exist in nodes_supp
                valid_children = {child_index: child_id 
                                for child_index, child_id in node_dict['children'].items() 
                                if child_id in self.nodes_supp}
                node_dict['children'] = valid_children
                # If no valid children remain, convert to a Leaf
                if not valid_children:
                    node_dict['node_type'] = 'Leaf'
                    node_dict['children'] = {}

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

    def add_leafs_to_nodes_supp(self):
        """
        Create self.leafs list by collecting leaf nodes and generating nodes for missing children.
        
        Raises:
            ValueError: If any node has ranges with length > 2
        """
        # Initialize leafs
        leaf_id = self.max_node_id + 1  # Start new node IDs after existing max
        
        # Walk through all nodes in nodes_supp
        leafs_to_add = []
        for node_id, node_dict in self.nodes_supp.items():
            # Check ranges length constraint
            if len(node_dict['ranges']) > 2:
                raise ValueError(f"Node {node_id} has ranges with length {len(node_dict['ranges'])}, maximum allowed is 2")
            
            # Determine if node is 1d or 2d
            is_1d = len(node_dict['ranges']) == 1
            is_2d = len(node_dict['ranges']) == 2
            
            if node_dict['node_type'] == 'Branch':
                # Check for missing children and potentially create new nodes
                children = node_dict.get('children', {})
                expected_children = 2 if is_1d else 4
                
                if len(children) < expected_children:
                    branch_count = node_dict['count']
                    child_sum = sum(self.nodes_supp[child_id]['count'] for child_id in children.values())
                    
                    if branch_count > child_sum:
                        # Find missing children
                        existing_child_indices = set(children.keys())
                        all_child_indices = set(range(expected_children))
                        missing_child_indices = all_child_indices - existing_child_indices
                        
                        # Calculate count for each new node
                        num_missing = len(missing_child_indices)
                        new_node_count = (branch_count - child_sum) / num_missing
                        
                        # Create new nodes for missing children
                        for child_index in missing_child_indices:
                            node_dict['children'][child_index] = int(leaf_id)
                            new_ranges = self._calculate_child_ranges(node_dict['ranges'], child_index, is_1d)
                            
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
            self.nodes_supp[new_node['node_id']] = new_node
    
    def _calculate_child_ranges(self, parent_ranges, child_index, is_1d):
        """
        Calculate the ranges for a missing child based on parent ranges and child index.
        
        Args:
            parent_ranges: List of parent range dictionaries
            child_index: Index of the missing child (0-3)
            is_1d: True if 1D, False if 2D
            
        Returns:
            List of range dictionaries for the child
        """
        if is_1d:
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
        Create self.leafs dict by collecting only leaf nodes from nodes_supp.
        """
        self.leafs = {}
        
        for node_id, node_dict in self.nodes_supp.items():
            if node_dict['node_type'] == 'Leaf':
                self.leafs[node_id] = node_dict.copy()
    
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
        if 0 in self.nodes_supp:
            self.nodes_supp[0]['top_down_count'] = self.nodes_supp[0]['count']
        else:
            return  # No root node found
        
        # Process nodes level by level (breadth-first traversal)
        nodes_to_process = [0]  # Start with root
        
        while nodes_to_process:
            current_node_id = nodes_to_process.pop(0)
            current_node = self.nodes_supp[current_node_id]
            
            # Only process Branch nodes that have children
            if (current_node['node_type'] == 'Branch' and 
                'children' in current_node and 
                current_node['children']):
                
                # Get parent's top_down_count
                parent_top_down_count = current_node['top_down_count']
                
                # Calculate total count of children
                child_ids = list(current_node['children'].values())
                child_count_sum = sum(self.nodes_supp[child_id]['count'] for child_id in child_ids)
                
                # Avoid division by zero
                if child_count_sum > 0:
                    # Calculate proportional adjustment for each child
                    for child_id in child_ids:
                        child_node = self.nodes_supp[child_id]
                        child_original_count = child_node['count']
                        
                        # Calculate proportional top_down_count
                        proportion = child_original_count / child_count_sum
                        child_node['top_down_count'] = parent_top_down_count * proportion
                        
                        # Add child to processing queue if it's a Branch
                        if child_node['node_type'] == 'Branch':
                            nodes_to_process.append(child_id)
                else:
                    # If child_count_sum is 0, distribute equally among children
                    equal_share = parent_top_down_count / len(child_ids)
                    for child_id in child_ids:
                        child_node = self.nodes_supp[child_id]
                        child_node['top_down_count'] = equal_share
                        
                        # Add child to processing queue if it's a Branch
                        if child_node['node_type'] == 'Branch':
                            nodes_to_process.append(child_id)

    def make_gaussian_kde(self, bw_method: str = "scott", bw_scale: float = 1.0) -> gaussian_kde:
        """
        Create a Gaussian KDE from the histogram defined by leafs ranges and top_down_counts.
        
        Args:
            bw_method: Bandwidth method for KDE (default: "scott")
            bw_scale: Scale factor for bandwidth (default: 1.0)
            
        Returns:
            scipy.stats.gaussian_kde object
        """
        
        if not self.leafs:
            raise ValueError("No leafs available for KDE creation")
        
        # Check dimensionality
        sample_leaf = next(iter(self.leafs.values()))
        num_dims = len(sample_leaf['ranges'])
        
        if num_dims > 2:
            raise ValueError(f"KDE only supports 1D and 2D data, got {num_dims}D")
        
        # Collect data points by sampling within each bin
        data_points = []
        
        for leaf in self.leafs.values():
            # Skip leafs without top_down_count
            if 'top_down_count' not in leaf:
                continue
                
            weight = leaf['top_down_count']
            if weight <= 0:
                continue
            
            # Number of points to sample in this bin
            num_samples = round(weight)
            if num_samples == 0:
                continue
            
            ranges = leaf['ranges']
            
            if num_dims == 1:
                # For 1D, sample uniformly within the range
                min_val = ranges[0]['min']
                max_val = ranges[0]['max']
                
                if min_val == max_val:
                    # Point range - all samples at the same location
                    samples = np.full(num_samples, min_val)
                else:
                    # Uniform sampling within the range
                    samples = np.random.uniform(min_val, max_val, num_samples)
                
                data_points.extend(samples)
                
            elif num_dims == 2:
                # For 2D, sample uniformly within the rectangle
                min_x, max_x = ranges[0]['min'], ranges[0]['max']
                min_y, max_y = ranges[1]['min'], ranges[1]['max']
                
                if min_x == max_x and min_y == max_y:
                    # Point range - all samples at the same location
                    samples = np.full((num_samples, 2), [min_x, min_y])
                else:
                    # Uniform sampling within the rectangle
                    x_samples = np.random.uniform(min_x, max_x, num_samples)
                    y_samples = np.random.uniform(min_y, max_y, num_samples)
                    samples = np.column_stack([x_samples, y_samples])
                
                data_points.extend(samples)
        
        if not data_points:
            raise ValueError("No valid data points with top_down_count > 0 found")
        
        # Convert to numpy array
        data_points = np.array(data_points)
        
        # Transpose for scipy.stats.gaussian_kde (expects shape (n_dims, n_samples))
        if num_dims == 1:
            data_points = data_points.reshape(1, -1)
        else:
            data_points = data_points.T
        
        # Create KDE (no weights needed since we're sampling the right number of points)
        kde = gaussian_kde(data_points, bw_method=bw_method)
        
        # Apply bandwidth scaling if specified
        if bw_scale != 1.0:
            # Store original covariance_factor method
            original_covariance_factor = kde.covariance_factor
            
            # Create a new covariance_factor method that scales the bandwidth
            def scaled_covariance_factor():
                return original_covariance_factor() * bw_scale
            
            # Replace the method and recalculate bandwidth
            kde.covariance_factor = scaled_covariance_factor
            kde._compute_covariance()
        
        return kde