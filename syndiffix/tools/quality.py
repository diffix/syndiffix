from __future__ import annotations

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
