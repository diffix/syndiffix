from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _convert_to_numeric(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert non-numeric values in two single-column dataframes to integers,
    ensuring the same values across both dataframes get the same integer mapping.
    
    Args:
        df1: First dataframe with exactly one column
        df2: Second dataframe with exactly one column
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Both dataframes with values converted to integers
    """
    # Extract the single columns as Series
    series1 = df1.iloc[:, 0]
    series2 = df2.iloc[:, 0]
    
    # Get all unique values from both series (excluding NaN)
    unique_values1 = set(series1.dropna().unique())
    unique_values2 = set(series2.dropna().unique())
    all_unique_values = sorted(unique_values1.union(unique_values2))
    
    # Create mapping from unique values to integers
    value_to_int = {value: i for i, value in enumerate(all_unique_values)}
    
    # Apply mapping to both series
    series1_numeric = series1.map(value_to_int)
    series2_numeric = series2.map(value_to_int)
    
    # Create new dataframes with the numeric data
    df1_numeric = pd.DataFrame(series1_numeric, columns=df1.columns)
    df2_numeric = pd.DataFrame(series2_numeric, columns=df2.columns)
    
    return df1_numeric, df2_numeric


def ks_measure(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic between two single-column dataframes.
    
    The KS statistic measures the maximum difference between the empirical cumulative
    distribution functions of two samples. Values range from 0 to 1, where 0 indicates
    identical distributions and 1 indicates completely different distributions.
    
    Args:
        df1: First dataframe with exactly one numeric column
        df2: Second dataframe with exactly one numeric column
        
    Returns:
        tuple[float, float]: A tuple containing:
            - KS statistic (between 0 and 1)
            - p-value for the hypothesis test
        
    Raises:
        ValueError: If either dataframe doesn't have exactly one column
        ValueError: If either dataframe is empty
        
    Example:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> df2 = pd.DataFrame({'values': [1.1, 2.1, 3.1, 4.1, 5.1]})
        >>> ks_stat, p_val = ks_measure(df1, df2)
        >>> print(f"KS statistic: {ks_stat:.4f}, p-value: {p_val:.4f}")
    """
    
    # Validate inputs
    if df1.shape[1] != 1:
        raise ValueError(f"ks_measure: df1 must have exactly 1 column, got {df1.shape[1]}")
    
    if df2.shape[1] != 1:
        raise ValueError(f"ks_measure: df2 must have exactly 1 column, got {df2.shape[1]}")
    
    if len(df1) == 0:
        raise ValueError("ks_measure: df1 cannot be empty")
    
    if len(df2) == 0:
        raise ValueError("ks_measure: df2 cannot be empty")
    
    # Extract the single columns as Series
    series1 = df1.iloc[:, 0]
    series2 = df2.iloc[:, 0]
    
    # Convert to numeric if needed
    if not pd.api.types.is_numeric_dtype(series1) or not pd.api.types.is_numeric_dtype(series2):
        df1, df2 = _convert_to_numeric(df1, df2)
        series1 = df1.iloc[:, 0]
        series2 = df2.iloc[:, 0]
    
    # Remove any NaN values
    series1_clean = series1.dropna()
    series2_clean = series2.dropna()
    
    if len(series1_clean) == 0:
        raise ValueError("ks_measure: df1 contains no valid numeric values after removing NaN")
    
    if len(series2_clean) == 0:
        raise ValueError("ks_measure: df2 contains no valid numeric values after removing NaN")
    
    # Calculate KS statistic using scipy
    ks_statistic, pvalue = stats.ks_2samp(series1_clean, series2_clean)
    
    return ks_statistic, pvalue


def energy_distance_2d(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Calculate the energy distance between two 2-column dataframes.
    
    The energy distance is a statistical distance between the distributions of two samples
    in multidimensional space. It ranges from 0 (identical distributions) to positive values
    (different distributions). Non-numeric columns are automatically converted to numeric values.
    If the dataframes have different numbers of rows, df2 is resampled (with replacement if smaller,
    or randomly reduced if larger) to match the size of df1.
    
    Args:
        df1: First dataframe with exactly two columns (size unchanged)
        df2: Second dataframe with exactly two columns (will be resampled to match df1's size)
        
    Returns:
        float: Energy distance between the two datasets
        
    Raises:
        ValueError: If either dataframe doesn't have exactly two columns
        ValueError: If either dataframe is empty
        ValueError: If column names don't match between dataframes
        
    Example:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> df2 = pd.DataFrame({'x': [1.1, 2.1, 3.1], 'y': [4.1, 5.1, 6.1]})
        >>> distance = energy_distance_2d(df1, df2)
        >>> print(f"Energy distance: {distance:.4f}")
    """
    
    # Validate inputs
    if df1.shape[1] != 2:
        raise ValueError(f"energy_distance_2d: df1 must have exactly 2 columns, got {df1.shape[1]}")
    
    if df2.shape[1] != 2:
        raise ValueError(f"energy_distance_2d: df2 must have exactly 2 columns, got {df2.shape[1]}")
    
    if len(df1) == 0:
        raise ValueError("energy_distance_2d: df1 cannot be empty")
    
    if len(df2) == 0:
        raise ValueError("energy_distance_2d: df2 cannot be empty")
    
    # Check that column names match
    if list(df1.columns) != list(df2.columns):
        raise ValueError(f"energy_distance_2d: column names must match. df1: {list(df1.columns)}, df2: {list(df2.columns)}")
    
    # Make copies to avoid modifying original dataframes
    df1_work = df1.copy()
    df2_work = df2.copy()
    
    # Convert non-numeric columns to numeric for each column
    for col in df1_work.columns:
        series1 = df1_work[col]
        series2 = df2_work[col]
        
        # Check if conversion is needed
        if not pd.api.types.is_numeric_dtype(series1) or not pd.api.types.is_numeric_dtype(series2):
            # Create temporary single-column dataframes for conversion
            temp_df1 = pd.DataFrame({col: series1})
            temp_df2 = pd.DataFrame({col: series2})
            
            # Convert to numeric
            temp_df1_numeric, temp_df2_numeric = _convert_to_numeric(temp_df1, temp_df2)
            
            # Update the working dataframes
            df1_work[col] = temp_df1_numeric.iloc[:, 0]
            df2_work[col] = temp_df2_numeric.iloc[:, 0]
    
    # Remove rows with any NaN values
    df1_clean = df1_work.dropna()
    df2_clean = df2_work.dropna()
    
    if len(df1_clean) == 0:
        raise ValueError("energy_distance_2d: df1 contains no valid rows after removing NaN values")
    
    if len(df2_clean) == 0:
        raise ValueError("energy_distance_2d: df2 contains no valid rows after removing NaN values")
    
    # Special case: if both dataframes are identical, return 0
    if df1_clean.equals(df2_clean):
        return 0.0
    
    # Adjust df2 to match df1's row count
    n1, n2 = len(df1_clean), len(df2_clean)
    if n1 != n2:
        if n2 < n1:
            # df2 is smaller: resample with replacement to match df1 size
            df2_clean = df2_clean.sample(n=n1, replace=True, random_state=42).reset_index(drop=True)
        else:
            # df2 is larger: randomly remove rows to match df1 size
            df2_clean = df2_clean.sample(n=n1, replace=False, random_state=42).reset_index(drop=True)
    
    # Manual array construction to avoid any pandas/numpy conversion issues
    # Extract values column by column and build arrays manually
    col1_name, col2_name = df1_clean.columns
    
    array1_list = []
    for idx in df1_clean.index:
        row = [float(df1_clean.loc[idx, col1_name]), float(df1_clean.loc[idx, col2_name])]
        array1_list.append(row)
    
    array2_list = []
    for idx in df2_clean.index:
        row = [float(df2_clean.loc[idx, col1_name]), float(df2_clean.loc[idx, col2_name])]
        array2_list.append(row)
    
    # Convert to numpy arrays with explicit construction
    array1 = np.array(array1_list, dtype=np.float64)
    array2 = np.array(array2_list, dtype=np.float64)
    
    # Ensure arrays are properly shaped
    if array1.shape[1] != 2 or array2.shape[1] != 2:
        raise ValueError(f"energy_distance_2d: arrays must have exactly 2 columns. Got shapes: {array1.shape}, {array2.shape}")
    
    # Check for infinite or NaN values
    if not np.all(np.isfinite(array1)):
        raise ValueError("energy_distance_2d: df1 contains infinite or NaN values")
    
    if not np.all(np.isfinite(array2)):
        raise ValueError("energy_distance_2d: df2 contains infinite or NaN values")
    
    # Manual energy distance calculation to avoid scipy issues
    try:
        # Calculate energy distance manually using the definition
        n, m = len(array1), len(array2)
        
        # Energy distance formula: 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        # where E[||X-X'||] and E[||Y-Y'||] include all pairs (including i=j where distance=0)
        
        # Distance between X and Y samples (all pairs)
        xy_sum = 0.0
        for i in range(n):
            for j in range(m):
                dist = np.sqrt(np.sum((array1[i] - array2[j]) ** 2))
                xy_sum += dist
        mean_xy = xy_sum / (n * m)
        
        # Distance between X and X samples (all pairs, including i=j)
        xx_sum = 0.0
        for i in range(n):
            for j in range(n):
                dist = np.sqrt(np.sum((array1[i] - array1[j]) ** 2))
                xx_sum += dist
        mean_xx = xx_sum / (n * n)
        
        # Distance between Y and Y samples (all pairs, including i=j)
        yy_sum = 0.0
        for i in range(m):
            for j in range(m):
                dist = np.sqrt(np.sum((array2[i] - array2[j]) ** 2))
                yy_sum += dist
        mean_yy = yy_sum / (m * m)
        
        # Calculate energy distance
        energy_dist = 2 * mean_xy - mean_xx - mean_yy
        
        return float(max(0.0, energy_dist))
        
    except Exception as e:
        raise ValueError(f"energy_distance_2d: Failed to calculate energy distance manually. Error: {e}")
        