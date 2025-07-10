import numpy as np
import pandas as pd
from pytest import approx

from syndiffix import Synthesizer

from .conftest import *


def test_noiseless_category_category_dataset() -> None:
    raw_data = pd.DataFrame([["x", True], ["x", False], ["y", True]] * 15)
    syn_data = Synthesizer(raw_data, anonymization_params=NOISELESS_PARAMS).sample()
    assert raw_data.value_counts().to_list() == syn_data.value_counts().to_list()


def test_noisy_category_numeric_dataset() -> None:
    raw_data = pd.DataFrame(
        [
            [True, 0],
            [True, 6],
            [True, 12],
            [True, 7],
            [True, 21],
            [True, 12],
            [True, 17],
            [True, 31],
            [False, 5],
            [False, 2],
            [False, 7],
            [False, 19],
            [False, 24],
            [False, 16],
            [False, 8],
            [False, 2],
            [True, 16],
            [True, 30],
            [False, 4],
            [False, 3],
            [False, 0],
        ]
    )
    syn_data = Synthesizer(raw_data, anonymization_params=NOISELESS_PARAMS).sample()

    # Test categorical column.
    raw_category_counts = raw_data[0].value_counts()
    syn_category_counts = syn_data[0].value_counts()
    assert len(raw_category_counts) == len(syn_category_counts)
    assert syn_category_counts[True] == approx(raw_category_counts[True], abs=5)
    assert syn_category_counts[False] == approx(raw_category_counts[False], abs=5)

    # Test numeric column.
    assert syn_data[1].mean() == approx(raw_data[1].mean(), abs=5)
    assert syn_data[1].std() == approx(raw_data[1].std(), rel=0.35)


def test_string_ranges() -> None:
    raw_data = pd.DataFrame(
        [
            "Leopoldstraße 92",
            "Leopoldstraße 1",
            "Leopoldstraße 108",
            "Leopoldstraße 27",
            "Leopoldstraße 34",
            "Leopoldstraße 15",
            "Leopoldstraße 1",
            "Leopoldstraße 27",
            "Potsdamer Straße 47",
            "Potsdamer Straße 17",
            "Potsdamer Straße 2",
            "Potsdamer Straße 31",
            "Potsdamer Straße 2",
            "Potsdamer Straße 17",
            "Potsdamer Straße 2",
            "Potsdamer Straße 17",
            "Spandauer Str. 84",
            "Spandauer Str. 4",
            "Spandauer Str. 1",
            "Spandauer Str. 21",
            "Spandauer Str. 41",
            "Spandauer Str. 40",
            "Spandauer Str. 44",
            "Spandauer Str. 49",
            "Gerichtstraße 3",
            "Gerichtstraße 1",
            "Gerichtstraße 2a",
            "Gerichtstraße 2b",
            "Gerichtstraße 20",
            "Gerichtstraße 12",
            "Gerichtstraße 9",
            "Gerichtstraße 4",
        ]
    )
    syn_data = Synthesizer(raw_data, anonymization_params=NOISELESS_PARAMS).sample()

    assert len(syn_data) == approx(len(raw_data), rel=0.1)

    syn_prefixes = set()
    for value in syn_data[0]:
        syn_prefixes.add(value[: value.find("*")])
    assert syn_prefixes.issuperset(["Leopoldstraße ", "Potsdamer Straße ", "Spandauer Str. 4", "Gerichtstraße "])


def test_result_consistency() -> None:
    raw_data = pd.DataFrame(
        [
            [1, 0],
            [1, 6],
            [1, 12],
            [1, 7],
            [1, 21],
            [1, 12],
            [1, 17],
            [1, 31],
            [0, 5],
            [0, 2],
            [0, 7],
            [0, 19],
            [0, 24],
            [0, 16],
            [0, 8],
            [0, 2],
            [2, 1],
            [2, 3],
            [2, 4],
            [2, 3],
            [2, 0],
            [2, 1],
            [2, 5],
            [2, 2],
            [2, 0],
            [2, 0],
        ]
    )
    syn_data_1 = Synthesizer(raw_data).sample()
    syn_data_2 = Synthesizer(raw_data).sample()

    pd.testing.assert_frame_equal(syn_data_1, syn_data_2)


def test_normalize_reals() -> None:
    col1_vals = [0.93227, 8.16111, 143.7828783]
    col2_vals = [-31.6776, 0.00011, 20.71131]
    num_rows = 500
    col1_random = np.random.choice(col1_vals, num_rows)
    col2_random = np.random.choice(col2_vals, num_rows)
    df = pd.DataFrame({"col1": col1_random, "col2": col2_random})
    syn_data = Synthesizer(df).sample()
    assert set(syn_data["col1"]) == set(col1_vals)
    assert set(syn_data["col2"]) == set(col2_vals)


def test_normalize_ints() -> None:
    col1_vals = [-6, 0, 1294]
    col2_vals = [-20, 14, 15]
    num_rows = 500
    col1_random = np.random.choice(col1_vals, num_rows)
    col2_random = np.random.choice(col2_vals, num_rows)
    df = pd.DataFrame({"col1": col1_random, "col2": col2_random})
    syn_data = Synthesizer(df).sample()
    assert set(syn_data["col1"]) == set(col1_vals)
    assert set(syn_data["col2"]) == set(col2_vals)


def test_value_safe_columns_integers() -> None:
    # Generate 100 random integers with wide range to minimize duplicates
    np.random.seed(42)  # For reproducible tests
    integer_values = np.random.randint(-1000000, 1000000, size=100)
    df = pd.DataFrame({"safe_col": integer_values, "other_col": np.random.randint(0, 10, size=100)})

    syn_data = Synthesizer(df, value_safe_columns=["safe_col"]).sample()

    # Ensure all synthesized values in safe_col are from original dataset
    original_safe_values = set(df["safe_col"])
    synthesized_safe_values = set(syn_data["safe_col"])
    assert synthesized_safe_values.issubset(original_safe_values)

    # Ensure we still get a reasonable number of rows
    assert len(syn_data) > 0


def test_value_safe_columns_floats() -> None:
    # Generate 100 random floats between -1000 and 1000 with 5 digits precision
    np.random.seed(42)  # For reproducible tests
    float_values = np.round(np.random.uniform(-1000, 1000, size=100), 10)
    df = pd.DataFrame({"safe_col": float_values, "other_col": np.random.uniform(0, 1, size=100)})

    syn_data = Synthesizer(df, value_safe_columns=["safe_col"]).sample()

    # Ensure all synthesized values in safe_col are from original dataset
    original_safe_values = set(df["safe_col"])
    synthesized_safe_values = set(syn_data["safe_col"])
    assert synthesized_safe_values.issubset(original_safe_values)

    # Ensure we still get a reasonable number of rows
    assert len(syn_data) > 0
    original_safe_values = set(df["safe_col"])
    synthesized_safe_values = set(syn_data["safe_col"])
    assert synthesized_safe_values.issubset(original_safe_values)

    # Ensure we still get a reasonable number of rows
    assert len(syn_data) > 0


def test_value_safe_columns_floats_approximate() -> None:
    # Generate 100 random floats between -1000 and 1000 without rounding
    np.random.seed(42)  # For reproducible tests
    float_values = np.random.uniform(-1000, 1000, size=100)
    df = pd.DataFrame({"safe_col": float_values, "other_col": np.random.uniform(0, 1, size=100)})

    syn_data = Synthesizer(df, value_safe_columns=["safe_col"]).sample()

    # Ensure each synthesized value is very close to a value in original dataset
    original_safe_values = df["safe_col"].values
    for syn_value in syn_data["safe_col"]:
        # Check if syn_value is close to any original value (within 1e-10 tolerance)
        assert any(abs(syn_value - orig_value) < 1e-10 for orig_value in original_safe_values)

    # Ensure we still get a reasonable number of rows
    assert len(syn_data) > 0


def test_value_safe_columns_strings() -> None:
    # Generate 100 distinct strings
    np.random.seed(42)  # For reproducible tests
    string_values = [f"string_{i}_{np.random.randint(1000, 9999)}" for i in range(100)]
    df = pd.DataFrame({"safe_col": string_values, "other_col": np.random.choice(["A", "B", "C"], size=100)})

    syn_data = Synthesizer(df, value_safe_columns=["safe_col"]).sample()

    # Ensure all synthesized values in safe_col are from original dataset
    original_safe_values = set(df["safe_col"])
    synthesized_safe_values = set(syn_data["safe_col"])
    assert synthesized_safe_values.issubset(original_safe_values)

    # Ensure we still get a reasonable number of rows
    assert len(syn_data) > 0
