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


def do_check(columns: List[str], sbr: SyndiffixBlobReader, df_raw: pd.DataFrame, with_target: bool = False) -> None:
    target = None
    if with_target and len(columns) > 1:
        target = columns[random.randint(0, len(columns) - 1)]
    print(f"Do read blob with columns={columns} and target {target}")
    df_blob = sbr.read(columns=columns, target_column=target)
    if list(df_blob.columns) != columns:
        print("ERROR: Returned columns do not match requested columns")
        print(f"    Columns in blob: {list(df_blob.columns)}")
        print(f"    Columns requested: {columns}")
        quit()
    print("Do Synthesizer sample")
    df_syn = Synthesizer(df_raw[columns], target_column=target).sample()
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
        return pd.read_csv(data_path)
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

    sbr = SyndiffixBlobReader(blob_name=test_dir, path_to_dir=blob_test_path, cache_df_in_memory=True, force=True)

    # Test read for a random set of tables that require stitching
    all_combinations = list(sbr.catalog.keys())
    max_length = max([len(t) for t in all_combinations])
    big_combinations = get_combinations(list(df_raw.columns), max_length)
    for _ in range(20):
        comb = random.choice(big_combinations)
        do_check(list(comb), sbr, df_raw, with_target=True)
        do_check(list(comb), sbr, df_raw, with_target=False)

    sbr = SyndiffixBlobReader(blob_name=test_dir, path_to_dir=blob_test_path, cache_df_in_memory=False, force=True)

    # Test read for single-column tables
    for column in df_raw.columns:
        do_check([column], sbr, df_raw)

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
