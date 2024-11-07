import argparse
import itertools
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from syndiffix import Synthesizer
from syndiffix.blob import SyndiffixBlobBuilder, SyndiffixBlobReader

"""
To use, assign the env variable BLOB_TEST_PATH to the path of some test directory.
Each directory under BLOB_TEST_PATH contains the tests for a single table.

If you want to test an existing dataset, then place it in the test directory with the
name <dir_name>_table.csv or <dir_name>.table.parquet
"""


def make_dataframe_small(N: int = 300) -> pd.DataFrame:
    a = np.random.randint(100, 200, size=N)
    b = a + np.random.randint(200, 211, size=N)
    c = b + np.random.uniform(2.0, 4.0, size=N)
    d = np.random.choice(["x", "y", "z"], size=N)
    e = np.random.choice(["i", "j", "k"], size=N)
    f = np.random.randint(100, 200, size=N)
    g = a + np.random.randint(200, 211, size=N)
    pid = np.arange(1, N + 1)
    return pd.DataFrame(
        {
            "pid": pid,
            "i1": a,
            "i2": b,
            "f1": c,
            "t1": d,
            "t2": e,
            "i3": f,
            "i4": g,
        }
    )


def get_combinations(lst: List[str], N: int) -> List[Tuple[str, ...]]:
    # Generate all combinations of all possible lengths
    all_combs: List[Tuple[str, ...]] = []
    for r in range(N + 1, len(lst) + 1):
        all_combs.extend(itertools.combinations(lst, r))

    return all_combs


def measure(df_orig: pd.DataFrame, df_blob: pd.DataFrame, df_syn: pd.DataFrame) -> Tuple[float, float, float, float]:
    # Make a copy of df_orig that has only the same columns as df_syn
    df_orig = df_orig[df_syn.columns].copy()

    # Determine column types
    column_types = {}
    for col in df_orig.columns:
        if pd.api.types.is_float_dtype(df_orig[col]):
            column_types[col] = "continuous"
        elif pd.api.types.is_integer_dtype(df_orig[col]):
            if df_orig[col].nunique() <= 9:
                column_types[col] = "categorical"
            else:
                column_types[col] = "continuous"
        elif pd.api.types.is_string_dtype(df_orig[col]):
            column_types[col] = "categorical"
        else:
            raise ValueError(f"Unsupported column type for column {col}")

    # Bin continuous columns
    binned_dfs = []
    for df in [df_orig, df_blob, df_syn]:
        binned_df = df.copy()
        for col, col_type in column_types.items():
            if col_type == "continuous":
                min_val = min(df_orig[col].min(), df_blob[col].min(), df_syn[col].min())
                max_val = max(df_orig[col].max(), df_blob[col].max(), df_syn[col].max())
                bins = np.linspace(min_val, max_val, 11)
                binned_df[col] = pd.cut(df[col], bins=bins, include_lowest=True)
        binned_dfs.append(binned_df)

    binned_df_orig, binned_df_blob, binned_df_syn = binned_dfs

    # Ensure categorical columns have the same categories
    for col, col_type in column_types.items():
        if col_type == "categorical":
            categories = pd.Categorical(df_orig[col]).categories
            binned_df_orig[col] = pd.Categorical(binned_df_orig[col], categories=categories)
            binned_df_blob[col] = pd.Categorical(binned_df_blob[col], categories=categories)
            binned_df_syn[col] = pd.Categorical(binned_df_syn[col], categories=categories)

    # Compute row counts for all unique combinations of values using pd.crosstab
    def compute_counts(df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby(list(df.columns)).size().reset_index(name="count")
        return counts

    counts_orig = compute_counts(binned_df_orig)
    counts_blob = compute_counts(binned_df_blob)
    counts_syn = compute_counts(binned_df_syn)

    # Align the counts DataFrames to ensure they have the same index
    counts_blob = counts_blob.reindex(counts_orig.index, fill_value=0)
    counts_syn = counts_syn.reindex(counts_orig.index, fill_value=0)

    cols = list(df_orig.columns)
    cols.sort()
    print(cols)
    if cols == ["CommHome", "CommToSch"]:
        print(f"Counts orig:\n{counts_orig}")
        print(f"Counts blob:\n{counts_blob}")

    # Compute absolute differences
    error_blob = (counts_orig["count"] - counts_blob["count"]).abs()
    error_syn = (counts_orig["count"] - counts_syn["count"]).abs()

    # Compute average errors
    avg_error_blob = error_blob.mean()
    avg_error_syn = error_syn.mean()

    # Compute average errors
    max_error_blob = error_blob.max()
    max_error_syn = error_syn.max()

    return avg_error_blob, avg_error_syn, max_error_blob, max_error_syn


def do_check(columns: List[str], sbr: SyndiffixBlobReader, df_raw: pd.DataFrame, with_target: bool = False) -> None:
    target = None
    if with_target and len(columns) > 1:
        target = columns[random.randint(0, len(columns) - 1)]
    print("----------------------------------------")
    print(f"Do read blob with columns={columns} and target {target}")
    df_blob = sbr.read(columns=columns, target_column=target)
    if list(df_blob.columns) != columns:
        print("ERROR: Returned columns do not match requested columns")
        print(f"    Columns in blob: {list(df_blob.columns)}")
        print(f"    Columns requested: {columns}")
        quit()
    print("Do Synthesizer sample")
    df_syn = Synthesizer(df_raw[columns], target_column=target).sample()
    if len(columns) <= 3:
        avg_error_blob, avg_error_syn, max_error_blob, max_error_syn = measure(df_raw, df_blob, df_syn)
        print(f"Average error for blob/syn: {avg_error_blob}, {avg_error_syn}")
        print(f"Max error for blob/syn: {max_error_blob}, {max_error_syn}")
        return

    dfs: List[pd.DataFrame] = [df_raw, df_blob, df_syn]
    df_names = ["df_raw", "df_blob", "df_syn"]
    for column in columns:
        print(f"Measures for column={column}")
        df_blob_column_sorted = df_blob[column].sort_values().reset_index(drop=True)
        df_syn_column_sorted = df_syn[column].sort_values().reset_index(drop=True)
        merged_df = pd.merge(
            df_blob_column_sorted, df_syn_column_sorted, left_index=True, right_index=True, suffixes=("_blob", "_syn")
        )
        differences = merged_df[merged_df[f"{column}_blob"] != merged_df[f"{column}_syn"]]
        num_differences = len(differences)
        print(f"    Number of differences between blob and syn: {num_differences}")
        print("Lengths:")
        for i, df in enumerate(dfs):
            print(f"    {df_names[i]}: {len(df[column])}")
        if pd.api.types.is_numeric_dtype(df_raw[column]):
            print("Averages:")
            for i, df in enumerate(dfs):
                print(f"    {df_names[i]}: {df[column].mean()}")
            print("Mins")
            for i, df in enumerate(dfs):
                print(f"    {df_names[i]}: {df[column].min()}")
            print("Maxs:")
            for i, df in enumerate(dfs):
                print(f"    {df_names[i]}: {df[column].max()}")
            print("STDs:")
            for i, df in enumerate(dfs):
                print(f"    {df_names[i]}: {df[column].std()}")


def get_blob_paths(test_dir: str) -> Tuple[Path, Path]:
    blob_test_env = os.environ.get("BLOB_TEST_PATH")
    if blob_test_env is None:
        raise ValueError("BLOB_TEST_PATH environment variable must be set")
    blob_test_path = Path(blob_test_env)
    blob_test_path = blob_test_path.joinpath(test_dir)
    data_path_parquet = blob_test_path.joinpath(f"{test_dir}_table.parquet")
    data_path_csv = blob_test_path.joinpath(f"{test_dir}_table.csv")
    if data_path_parquet.exists() or not data_path_csv.exists():
        data_path = data_path_parquet
    else:
        data_path = data_path_csv
    print(f"Test path is {blob_test_path}, table path is {data_path}")
    return blob_test_path, data_path


def read_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"The file at {str(data_path)} does not exist.")

    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        return df
    elif data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")


def setup(test_dir: str) -> None:
    blob_test_path, data_path = get_blob_paths(test_dir)
    blob_test_path.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        return
    if test_dir == "small_table1":
        df_raw = make_dataframe_small()
        df_raw.to_parquet(data_path)


def write(test_dir: str) -> None:
    setup(test_dir)
    blob_test_path, data_path = get_blob_paths(test_dir)

    df_raw = read_data(data_path)

    start_time = time.time()
    sbb = SyndiffixBlobBuilder(blob_name=test_dir, path_to_dir=blob_test_path, force=True)
    sbb.write(df_raw=df_raw, pids=None)
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(blob_test_path.joinpath("elapsed_time.txt"), "w", encoding="utf-8") as file:
        file.write(f"Elapsed time to build blob: {elapsed_time}")


def read(test_dir: str) -> None:
    setup(test_dir)
    blob_test_path, data_path = get_blob_paths(test_dir)
    df_raw = read_data(data_path)
    print(f"Data path = {data_path}")
    quit()

    sbr = SyndiffixBlobReader(blob_name=test_dir, path_to_dir=blob_test_path, cache_df_in_memory=True, force=True)

    # Test read for single-column tables
    for column in df_raw.columns:
        do_check([column], sbr, df_raw)

    # Test read 2-column tables
    for comb in list(itertools.combinations(df_raw.columns, 2)):
        do_check(list(comb), sbr, df_raw)

    # Test read for a random set of tables that require stitching
    all_combinations = list(sbr.catalog.keys())
    max_length = max([len(t) for t in all_combinations])
    big_combinations = get_combinations(list(df_raw.columns), max_length)
    for _ in range(20):
        comb = random.choice(big_combinations)
        do_check(list(comb), sbr, df_raw, with_target=True)
        do_check(list(comb), sbr, df_raw, with_target=False)

    sbr = SyndiffixBlobReader(blob_name=test_dir, path_to_dir=blob_test_path, cache_df_in_memory=False, force=True)

    # Test read for a random set of tables that are definitely in the blob
    for _ in range(20):
        comb = random.choice(all_combinations)
        do_check(list(comb), sbr, df_raw, with_target=True)
        do_check(list(comb), sbr, df_raw, with_target=False)


def main() -> None:
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some commands.")

    # Add the arguments
    parser.add_argument("test_dir", type=str, help="The test directory")
    parser.add_argument("command", type=str, choices=["write", "read"], help="The command to execute")

    # Parse the arguments
    args = parser.parse_args()

    # Assign the test_dir variable
    test_dir = args.test_dir
    print(f"Test directory: {test_dir}")

    # Execute the corresponding function based on the command
    if args.command == "write":
        write(test_dir)
    elif args.command == "read":
        read(test_dir)


if __name__ == "__main__":
    main()
