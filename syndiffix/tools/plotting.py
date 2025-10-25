from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_1d_nodes_bars(nodes: dict | list, column_index: int, file_name: str = "") -> plt:
    """
    Plot 1D bars from a list of nodes showing the distribution structure.
    
    Creates a visualization of tree nodes as colored bars on a 1D axis.
    The x-axis represents the range values for the specified column, and the y-axis
    represents the count values. Initial nodes are colored blue, while created
    (missing) nodes are colored yellow. Overlapping bars are visible due to alpha transparency.
    
    Args:
        nodes: Dict or list of node dictionaries returned by df_to_all_nodes() or df_to_complete_leafs()
               Each dict should have 'ranges', 'count', and 'initial' keys
        column_index: Integer specifying which dimension to plot from the ranges
        file_name: String to display as part of the plot title
        
    Returns:
        matplotlib.pyplot object ready for display or further customization
        
    Raises:
        ValueError: If column_index is not an integer
        IndexError: If column_index refers to non-existent dimension
        
    Example:
        >>> # Assuming you have a tree and have converted it to nodes
        >>> df_tree = tree_to_df(root_node)
        >>> nodes = df_to_complete_leafs(df_tree)
        >>> plt_obj = plot_1d_nodes_bars(nodes, 0, "my_data.csv")  # Plot first dimension
        >>> plt_obj.show()  # Display the plot
        >>> 
        >>> # Or save to file
        >>> plt_obj.savefig('tree_1d_visualization.png')
        
    Note:
        Requires matplotlib to be installed. Blue bars represent original tree nodes,
        yellow bars represent created missing nodes to fill gaps in coverage.
    """
    
    if not isinstance(column_index, int):
        raise ValueError("column_index must be an integer")
    
    # Convert dict to list if needed
    if isinstance(nodes, dict):
        nodes_list = list(nodes.values())
    else:
        nodes_list = nodes
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    # Process each node in the list
    for node in nodes_list:
        # Extract ranges from the node dictionary
        ranges = node['ranges']
        count = node['count']
        initial = node['initial']
        
        # Check if we have enough dimensions
        if len(ranges) <= column_index:
            raise IndexError(f"Not enough dimensions in ranges. "
                           f"Has {len(ranges)}, but column_index {column_index} requires at least {column_index + 1}")
        
        # Extract the specified dimension
        range_data = ranges[column_index]
        
        # Calculate bar coordinates
        x_min, x_max = range_data['min'], range_data['max']
        width = x_max - x_min
        height = count
        
        # Choose color based on whether node is initial or created
        color = 'blue' if initial else 'yellow'
            
        if width > 0:
            edgecolor='white'
        else:
            edgecolor=color

        rect = patches.Rectangle(
            (x_min, 0), width, height,
            linewidth=1, edgecolor=edgecolor, facecolor=color, alpha=0.3
        )
            
        # Add rectangle to plot
        ax.add_patch(rect)
    
    # Set up the plot
    ax.set_xlim(0, 1)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Set y-axis limit with some padding above the maximum count
    if nodes_list:
        max_count = max(node['count'] for node in nodes_list)
        min_count = min(node['count'] for node in nodes_list if node['count'] > 0)
        if max_count > 0:
            ax.set_ylim(min_count * 0.5 if min_count > 0 else 0.1, max_count * 2)
        else:
            ax.set_ylim(0.1, 1)
    else:
        ax.set_ylim(0.1, 1)
    
    ax.set_xlabel(f'Dimension {column_index}')
    ax.set_ylabel('Count')
    ax.set_title(f'1D Tree Structure Visualization\n(Dimension {column_index})\n{file_name}')
    
    # Create legend for initial vs created nodes
    legend_elements = [
        patches.Patch(facecolor='blue', edgecolor='white', alpha=0.3, label='Initial nodes'),
        patches.Patch(facecolor='yellow', edgecolor='white', alpha=0.3, label='Created nodes')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt


def plot_2d_nodes_boxes(nodes: dict | list, column_indexes: list[int], file_name: str = "", display_counts: bool = False):
    """
    Plot 2D boxes from a list of nodes showing the spatial structure.
    
    Creates a visualization of tree nodes as colored rectangles on a 1.0x1.0 square.
    Each box is colored based on its width - boxes with the same width get the same color.
    Boxes with width 1.0 are ignored. Larger boxes get thicker borders.
    Boxes with count 0 are filled with light grey and have no border.
    This is useful for visualizing how the tree partitions the space at different levels.
    
    Args:
        nodes: Dict or list of node dictionaries returned by df_to_all_nodes() or df_to_complete_leafs()
               Each dict should have 'ranges', 'count', and 'initial' keys
        column_indexes: List of two integers specifying which dimensions to plot
                       from the ranges (e.g., [0, 1] for first two dimensions)
        file_name: String to display as part of the plot title
        display_counts: If True, display the count value in the center of each box (default: False)
        
    Returns:
        matplotlib.pyplot object ready for display or further customization
        
    Raises:
        ValueError: If column_indexes doesn't contain exactly 2 integers
        IndexError: If column_indexes refer to non-existent dimensions
        
    Example:
        >>> # Assuming you have a tree and have converted it to nodes
        >>> df_tree = tree_to_df(root_node)
        >>> nodes = df_to_all_nodes(df_tree)
        >>> plt_obj = plot_2d_boxes(nodes, [0, 1], "my_data.csv")  # Plot first two dimensions
        >>> plt_obj.show()  # Display the plot
        >>> 
        >>> # Or save to file with counts displayed
        >>> plt_obj = plot_2d_boxes(nodes, [0, 1], "my_data.csv", display_counts=True)
        >>> plt_obj.savefig('tree_visualization.png')
        
    Note:
        Requires matplotlib to be installed. Each distinct box width gets a different
        color, making it easy to see the hierarchical structure of the tree.
        Boxes with width 1.0 are ignored as they represent the full domain.
        Boxes with count 0 are shown as solid light grey rectangles.
    """
    
    if len(column_indexes) != 2:
        raise ValueError(f"column_indexes must contain exactly 2 integers, got {len(column_indexes)}")
    
    if not all(isinstance(idx, int) for idx in column_indexes):
        raise ValueError("column_indexes must contain integers")
    
    # Convert dict to list if needed
    if isinstance(nodes, dict):
        nodes_list = list(nodes.values())
    else:
        nodes_list = nodes
    
    # First pass: collect all unique widths
    unique_widths = set()
    for node in nodes_list:
        ranges = node['ranges']
        count = node['count']
        
        if count > 0:  # Only process non-zero count boxes
            if len(ranges) <= max(column_indexes):
                continue
            
            x_range = ranges[column_indexes[0]]
            y_range = ranges[column_indexes[1]]
            
            width = x_range['max'] - x_range['min']
            height = y_range['max'] - y_range['min']
            
            box_width = round(min(width, height), 10)
            unique_widths.add(box_width)
    
    # Assign colors to unique widths
    width_to_color = {}
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, width in enumerate(sorted(unique_widths)):
        width_to_color[width] = colors[i % len(colors)]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    
    # Process each node in the list
    for node in nodes_list:
        # Extract ranges from the node dictionary
        ranges = node['ranges']
        count = node['count']
        
        # Check if we have enough dimensions
        if len(ranges) <= max(column_indexes):
            raise IndexError(f"Not enough dimensions in ranges. "
                           f"Has {len(ranges)}, but column_indexes {column_indexes} requires at least {max(column_indexes) + 1}")
        
        # Extract the two specified dimensions
        x_range = ranges[column_indexes[0]]
        y_range = ranges[column_indexes[1]]
        
        # Calculate box coordinates
        x_min, x_max = x_range['min'], x_range['max']
        y_min, y_max = y_range['min'], y_range['max']
        
        width = x_max - x_min
        height = y_max - y_min
        
        # For color assignment, use the minimum of width and height (assuming square boxes for trees)
        # Round to avoid floating point precision issues
        box_width = round(min(width, height), 10)
        
        # Handle zero-count boxes specially
        if count == 0:
            # Create solid light grey rectangle with no border
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=0, edgecolor='none', facecolor='lightgrey', alpha=0.7
            )
        else:
            color = width_to_color[box_width]
            
            # Calculate line width based on box width (larger boxes get thicker lines)
            # Scale linewidth from 1 to 4 based on box width
            line_width = 1 + (box_width * 3)
            
            # Check if this is a leaf node
            node_type = node.get('node_type', '')
            if node_type == 'Leaf':
                # Create rectangle with colored border and light grey fill for leaf nodes
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=line_width, edgecolor=color, facecolor='lightgrey', alpha=0.4
                )
            else:
                # Create rectangle patch with colored border, no fill, and alpha transparency
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=line_width, edgecolor=color, facecolor='none', alpha=0.7
                )
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add count text in the center of the box if requested (no background box)
        # Skip displaying zero counts
        if display_counts and count > 0:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Add text without background box
            ax.text(center_x, center_y, str(round(count)), 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=6, fontweight='normal', color='black')
    
    # Set up the plot
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Dimension {column_indexes[0]}')
    ax.set_ylabel(f'Dimension {column_indexes[1]}')
    title = '2D Tree Structure Visualization'
    if display_counts:
        title += '\nNumbers show box counts'
    title += f'\n{file_name}'
    ax.set_title(title)
    
    # Create legend for width-to-color mapping (excluding width 1.0)
    legend_elements = []
    for width, color in sorted(width_to_color.items()):
        if abs(width - 1.0) >= 1e-10:  # Only include non-full-width boxes in legend
            line_width = 1 + (width * 3)
            # Format width without trailing zeros
            width_str = f"{width:.8f}".rstrip('0').rstrip('.')
            legend_elements.append(patches.Patch(edgecolor=color, facecolor='none', 
                                               linewidth=line_width, label=f'Width: {width_str}'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return plt


def plot_kde(kde: gaussian_kde, file_name: str = "", grid_points: int = 100, **kwargs):
    """
    Plot a Gaussian KDE object with appropriate visualization for 1D or 2D data.
    
    For 1D KDE: Creates a line plot showing the probability density function
    For 2D KDE: Creates a contour plot showing probability density contours
    
    Args:
        kde: scipy.stats.gaussian_kde object to plot
        file_name: String to display as part of the plot title
        grid_points: Number of grid points per dimension for evaluation (default: 100)
        **kwargs: Additional keyword arguments passed to matplotlib plotting functions
                 For 1D: passed to plt.plot()
                 For 2D: passed to plt.contour() or plt.contourf()
        
    Returns:
        matplotlib.pyplot object ready for display or further customization
        
    Raises:
        ValueError: If KDE has more than 2 dimensions
        
    Example:
        >>> # Assuming you have a TestNodes object with leafs
        >>> test_nodes = TestNodes(df)
        >>> kde = test_nodes.make_gaussian_kde()
        >>> plt_obj = plot_kde(kde, "my_data.csv")
        >>> plt_obj.show()
        >>> 
        >>> # For 2D with filled contours
        >>> plt_obj = plot_kde(kde, "my_data.csv", levels=10, filled=True)
        >>> plt_obj.savefig('kde_plot.png')
        
    Note:
        Requires matplotlib and scipy to be installed. The grid resolution can be
        adjusted with the grid_points parameter for smoother or faster plots.
    """
    
    # Get dimensionality from KDE
    if hasattr(kde, 'd'):
        num_dims = kde.d
    else:
        # Fallback: get dimensionality from dataset shape
        num_dims = kde.dataset.shape[0]
    
    if num_dims > 2:
        raise ValueError(f"Cannot plot KDE with {num_dims} dimensions. Only 1D and 2D are supported.")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    if num_dims == 1:
        # 1D KDE plotting
        # Create evaluation grid from 0 to 1
        x = np.linspace(0, 1, grid_points)
        
        # Evaluate KDE
        density = kde(x)
        
        # Plot with default or user-specified parameters
        plot_kwargs = {'linewidth': 2, 'color': 'blue'}
        plot_kwargs.update(kwargs)
        
        ax.plot(x, density, **plot_kwargs)
        ax.fill_between(x, density, alpha=0.3, color=plot_kwargs.get('color', 'blue'))
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'1D Gaussian KDE\n{file_name}')
        ax.grid(True, alpha=0.3)
        
    else:
        # 2D KDE plotting
        # Create evaluation grid from 0 to 1 in both dimensions
        x = np.linspace(0, 1, grid_points)
        y = np.linspace(0, 1, grid_points)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate KDE on grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = kde(positions).reshape(X.shape)
        
        # Check if user wants filled contours
        filled = kwargs.pop('filled', False)
        levels = kwargs.pop('levels', 10)
        
        if filled:
            # Filled contour plot
            contour_kwargs = {'levels': levels, 'cmap': 'viridis', 'alpha': 0.7}
            contour_kwargs.update(kwargs)
            cs = ax.contourf(X, Y, density, **contour_kwargs)
            plt.colorbar(cs, ax=ax, label='Density')
        else:
            # Line contour plot
            contour_kwargs = {'levels': levels, 'colors': 'blue', 'linewidths': 1}
            contour_kwargs.update(kwargs)
            cs = ax.contour(X, Y, density, **contour_kwargs)
            ax.clabel(cs, inline=True, fontsize=8)
        
        # Plot original data points
        x_data = kde.dataset[0, :]
        y_data = kde.dataset[1, :]
        ax.scatter(x_data, y_data, c='red', s=20, alpha=0.6, label='Data points')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_title(f'2D Gaussian KDE\n{file_name}')
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return plt


def plot_1d_orig_anon_cdf(df_orig: pd.DataFrame, df_sdx: pd.DataFrame = None, df_test: pd.DataFrame = None, file_name: str = "") -> plt:
    """
    Plot CDF comparison between original and anonymized data.
    
    Creates a visualization comparing cumulative distribution functions of original data
    against one or two anonymized versions. Each dataset is sorted and plotted with
    index on x-axis and values on y-axis.
    
    Args:
        df_orig: 1-column DataFrame with original data values
        df_sdx: Optional 1-column DataFrame with Syndiffix anonymized values
        df_test: Optional 1-column DataFrame with test/alternative anonymized values
        file_name: String to display as part of the plot title
        
    Returns:
        matplotlib.pyplot object ready for display or further customization
        
    Raises:
        ValueError: If both df_sdx and df_test are None
        
    Example:
        >>> # Compare original with Syndiffix data
        >>> plt_obj = plot_1d_orig_anon_cdf(orig_df, sdx_df, file_name="comparison")
        >>> plt_obj.show()
        >>> 
        >>> # Compare original with both Syndiffix and test data
        >>> plt_obj = plot_1d_orig_anon_cdf(orig_df, sdx_df, test_df, "comparison")
        >>> plt_obj.savefig('cdf_comparison.png')
        
    Note:
        Requires matplotlib and pandas to be installed. At least one of df_sdx or df_test
        must be provided. Original data is plotted in light_green, Syndiffix in blue, and test in red.
    """
    
    if df_sdx is None and df_test is None:
        raise ValueError("At least one of df_sdx or df_test must be provided (not None)")
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    # Get column name (assuming single column)
    orig_col = df_orig.columns[0]
    
    # Sort original data and plot
    orig_sorted = df_orig[orig_col].sort_values().values
    orig_indices = np.arange(len(orig_sorted))
    ax.scatter(orig_indices, orig_sorted, c='palegreen', s=10, alpha=1.0, label='Original')
    
    # Plot Syndiffix data if provided
    if df_sdx is not None:
        sdx_col = df_sdx.columns[0]
        sdx_sorted = df_sdx[sdx_col].sort_values().values
        sdx_indices = np.arange(len(sdx_sorted))
        ax.scatter(sdx_indices, sdx_sorted, c='blue', s=1, alpha=0.7, label='Syndiffix')
    
    # Plot test data if provided
    if df_test is not None:
        test_col = df_test.columns[0]
        test_sorted = df_test[test_col].sort_values().values
        test_indices = np.arange(len(test_sorted))
        ax.scatter(test_indices, test_sorted, c='red', s=1, alpha=0.7, label='Test')
    
    # Set up the plot
    ax.set_xlabel('Index (sorted order)')
    ax.set_ylabel('Value')
    ax.set_title(f'1D CDF Comparison\n{file_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt
