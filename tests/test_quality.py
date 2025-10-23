import numpy as np
import pandas as pd
from pytest import approx
from scipy.stats import chi2_contingency, ks_2samp, pearsonr
from sklearn.metrics import mutual_info_score

from syndiffix import Synthesizer

from .conftest import *


def test_quality_float_str() -> None:
    """Test quality of synthetic data for float-string columns with dependency."""
    np.random.seed(42)

    num_rows = 1000
    # Create float column with normal distribution
    float_col = np.random.normal(50, 15, num_rows)

    # Create categorical string column with dependency on float values
    # Higher float values more likely to be in categories 6-9
    str_categories = [f"cat_{i}" for i in range(10)]

    # Create dependency: map float values to categories with some noise
    normalized_float = (float_col - float_col.min()) / (float_col.max() - float_col.min())
    category_indices = np.clip((normalized_float * 8 + np.random.normal(0, 1, num_rows)).astype(int), 0, 9)
    str_col = [str_categories[i] for i in category_indices]

    df = pd.DataFrame({"float_col": float_col, "str_col": str_col})
    # Ensure at least 20 instances of every category in str_col
    min_count = 20
    for cat in str_categories:
        count = (df["str_col"] == cat).sum()
        if count < min_count:
            # Find all rows with this category
            rows = df[df["str_col"] == cat]
            # If there are any, replicate random rows to reach min_count
            if not rows.empty:
                n_to_add = min_count - count
                replicated = rows.sample(n=n_to_add, replace=True, random_state=42)
                df = pd.concat([df, replicated], ignore_index=True)
            else:
                # If no rows exist, create new rows with random float_col
                new_rows = pd.DataFrame(
                    {"float_col": np.random.normal(50, 15, min_count), "str_col": [cat] * min_count}
                )
                df = pd.concat([df, new_rows], ignore_index=True)

    # Generate synthetic data
    df_syn = Synthesizer(df, anonymization_params=NOISELESS_PARAMS).sample()

    # 1. Single-column similarity tests

    # Float column: test distribution similarity using KS test
    ks_stat, ks_pvalue = ks_2samp(df["float_col"], df_syn["float_col"])
    assert ks_stat < 0.15, f"Float column distributions too different (KS statistic: {ks_stat})"

    # Float column: test mean and std similarity
    assert df_syn["float_col"].mean() == approx(df["float_col"].mean(), abs=1.0)
    assert df_syn["float_col"].std() == approx(df["float_col"].std(), abs=0.5)

    # String column: test category distribution similarity
    orig_counts = df["str_col"].value_counts().sort_index()
    syn_counts = df_syn["str_col"].value_counts().sort_index()

    # Ensure all original categories are present in synthetic data
    for cat in orig_counts.index:
        assert cat in syn_counts.index, f"Category {cat} missing from synthetic data"

    # Test count similarity (allowing for some variation)
    for cat in orig_counts.index:
        orig_pct = orig_counts[cat] / len(df)
        syn_pct = syn_counts[cat] / len(df_syn)
        assert syn_pct == approx(orig_pct, abs=0.01), f"Category {cat} proportion differs too much"

    # 2. Dependency/correlation tests

    # For float-string dependency, use mutual information
    # Discretize float column for mutual information calculation
    orig_float_binned = pd.cut(df["float_col"], bins=10, labels=False)
    syn_float_binned = pd.cut(df_syn["float_col"], bins=10, labels=False)

    orig_str_encoded = pd.Categorical(df["str_col"]).codes
    syn_str_encoded = pd.Categorical(df_syn["str_col"]).codes

    orig_mi = mutual_info_score(orig_float_binned, orig_str_encoded)
    syn_mi = mutual_info_score(syn_float_binned, syn_str_encoded)

    assert syn_mi == approx(orig_mi, rel=0.4), f"Mutual information differs too much: orig={orig_mi}, syn={syn_mi}"


def test_quality_float_float() -> None:
    """Test quality of synthetic data for float-float columns with correlation."""
    np.random.seed(42)

    # Create correlated float columns
    x = np.random.normal(0, 1, 1000)
    y = 0.7 * x + np.random.normal(0, 0.5, 1000)  # Correlation ~0.8

    df = pd.DataFrame({"x": x, "y": y})

    # Generate synthetic data
    df_syn = Synthesizer(df, anonymization_params=NOISELESS_PARAMS).sample()

    # 1. Single-column similarity tests

    # X column
    ks_stat_x, _ = ks_2samp(df["x"], df_syn["x"])
    assert ks_stat_x < 0.15, f"X column distributions too different (KS statistic: {ks_stat_x})"
    assert df_syn["x"].mean() == approx(df["x"].mean(), abs=0.02)
    assert df_syn["x"].std() == approx(df["x"].std(), abs=0.05)

    # Y column
    ks_stat_y, _ = ks_2samp(df["y"], df_syn["y"])
    assert ks_stat_y < 0.15, f"Y column distributions too different (KS statistic: {ks_stat_y})"
    assert df_syn["y"].mean() == approx(df["y"].mean(), abs=0.02)
    assert df_syn["y"].std() == approx(df["y"].std(), abs=0.04)

    # 2. Correlation tests

    orig_corr, _ = pearsonr(df["x"], df["y"])
    syn_corr, _ = pearsonr(df_syn["x"], df_syn["y"])

    assert syn_corr == approx(orig_corr, abs=0.01), f"Correlation differs too much: orig={orig_corr}, syn={syn_corr}"


def test_quality_str_str() -> None:
    """Test quality of synthetic data for string-string columns with dependency."""
    np.random.seed(42)

    # Create two categorical columns with dependency
    categories_a = [f"group_{i}" for i in range(10)]
    categories_b = [f"type_{i}" for i in range(10)]

    # Create dependency: certain groups prefer certain types
    group_prefs = {
        0: [0, 1, 2],  # group_0 prefers type_0, type_1, type_2
        1: [1, 2, 3],  # group_1 prefers type_1, type_2, type_3
        2: [2, 3, 4],  # etc.
        3: [3, 4, 5],
        4: [4, 5, 6],
        5: [5, 6, 7],
        6: [6, 7, 8],
        7: [7, 8, 9],
        8: [8, 9, 0],
        9: [9, 0, 1],
    }

    # Generate data with dependency
    col_a = []
    col_b = []

    for _ in range(1000):
        # Choose group randomly
        group_idx = np.random.randint(0, 10)
        group = categories_a[group_idx]

        # Choose type based on group preference (80% of time) or random (20% of time)
        if np.random.random() < 0.8:
            type_idx = np.random.choice(group_prefs[group_idx])
        else:
            type_idx = np.random.randint(0, 10)

        type_val = categories_b[type_idx]

        col_a.append(group)
        col_b.append(type_val)

    df = pd.DataFrame({"group": col_a, "type": col_b})

    # Generate synthetic data
    df_syn = Synthesizer(df, anonymization_params=NOISELESS_PARAMS).sample()

    # 1. Single-column similarity tests

    # Group column
    orig_group_counts = df["group"].value_counts().sort_index()
    syn_group_counts = df_syn["group"].value_counts().sort_index()

    for group in orig_group_counts.index:
        assert group in syn_group_counts.index, f"Group {group} missing from synthetic data"
        orig_pct = orig_group_counts[group] / len(df)
        syn_pct = syn_group_counts[group] / len(df_syn)
        assert syn_pct == approx(orig_pct, abs=0.01), f"Group {group} proportion differs too much"

    # Type column
    orig_type_counts = df["type"].value_counts().sort_index()
    syn_type_counts = df_syn["type"].value_counts().sort_index()
    print(f"len df: {len(df)}, len df_syn: {len(df_syn)}")
    print(f"Original type counts:\n{orig_type_counts}")
    print(f"Synthetic type counts:\n{syn_type_counts}")

    for type_val in orig_type_counts.index:
        assert type_val in syn_type_counts.index, f"Type {type_val} missing from synthetic data"
        orig_pct = orig_type_counts[type_val] / len(df)
        syn_pct = syn_type_counts[type_val] / len(df_syn)
        print(f"Type {type_val}: orig cnt {orig_type_counts[type_val]}, syn cnt {syn_type_counts[type_val]}")
        assert syn_pct == approx(orig_pct, abs=0.005), f"Type {type_val} proportion differs too much"

    # 2. Dependency tests using contingency table analysis

    # Create contingency tables
    orig_contingency = pd.crosstab(df["group"], df["type"])
    syn_contingency = pd.crosstab(df_syn["group"], df_syn["type"])

    # Ensure synthetic contingency table has same shape as original
    assert orig_contingency.shape[0] <= syn_contingency.shape[0], "Missing groups in synthetic data"
    assert orig_contingency.shape[1] <= syn_contingency.shape[1], "Missing types in synthetic data"

    # Calculate Cramér's V (measure of association between categorical variables)
    def cramers_v(contingency_table: pd.DataFrame) -> float:
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        if min_dim == 0:
            return 0
        return np.sqrt(chi2 / (n * min_dim))

    # Align contingency tables for comparison
    common_groups = set(orig_contingency.index) & set(syn_contingency.index)
    common_types = set(orig_contingency.columns) & set(syn_contingency.columns)

    orig_aligned = orig_contingency.loc[list(common_groups), list(common_types)]
    syn_aligned = syn_contingency.loc[list(common_groups), list(common_types)]

    # Ensure arguments are always DataFrames
    if isinstance(orig_aligned, pd.Series):
        orig_aligned = orig_aligned.to_frame().T
    if isinstance(syn_aligned, pd.Series):
        syn_aligned = syn_aligned.to_frame().T

    orig_cramers_v = cramers_v(orig_aligned)
    syn_cramers_v = cramers_v(syn_aligned)

    assert syn_cramers_v == approx(
        orig_cramers_v, rel=0.4
    ), f"Cramér's V differs too much: orig={orig_cramers_v}, syn={syn_cramers_v}"


def test_synthetic_data_size_consistency() -> None:
    """Test that synthetic data has reasonable size compared to original."""
    np.random.seed(42)

    # Create simple test data
    df = pd.DataFrame({"col1": np.random.normal(0, 1, 1000), "col2": np.random.choice(["A", "B", "C"], 1000)})

    df_syn = Synthesizer(df, anonymization_params=NOISELESS_PARAMS).sample()

    # Synthetic data should have reasonable size (within 50% of original)
    assert len(df_syn) > 0.5 * len(df), "Synthetic data too small"
    assert len(df_syn) < 2.0 * len(df), "Synthetic data too large"

    # Should have same number of columns
    assert len(df_syn.columns) == len(df.columns), "Different number of columns"

    # Should have same column names
    assert list(df_syn.columns) == list(df.columns), "Different column names"
