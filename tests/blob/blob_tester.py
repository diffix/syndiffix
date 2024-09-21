import os
import shutil
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from syndiffix.blob import SyndiffixBlobWriter, SyndiffixBlobReader
from syndiffix import Synthesizer

'''
To use, asign the env variable BLOB_TEST_PATH to the path of some test directory.
Each directory under BLOB_TEST_PATH contains the tests for a single table.
'''

def make_dataframe_small(N: int = 300) -> pd.DataFrame:
    a = np.random.randint(100, 200, size=N)
    b = a + np.random.randint(200, 211, size=N)
    c = b + np.random.uniform(2.0, 4.0, size=N)
    d = np.random.choice(["x", "y", "z"], size=N)
    e = np.random.choice(["i", "j", "k"], size=N)
    f = np.random.randint(100, 200, size=N)
    g = a + np.random.randint(200, 211, size=N)
    pid = np.arange(1, N + 1)
    return pd.DataFrame({"pid":pid, "i1": a, "i2": b, "f1": c, "t1": d, "t2": e,
                         "i3": f, "i4": g,})


def do_check(columns: list[str], sbr: SyndiffixBlobReader, df_raw: pd.DataFrame) -> None:
    print(f"Do read_blob with columns={columns}")
    df_blob = sbr.read_blob(columns=columns)
    print("Do Synthesizer sample")
    df_syn = Synthesizer(df_raw[columns]).sample()
    dfs: list[pd.DataFrame] = [df_raw, df_blob, df_syn]
    df_names = ['df_raw', 'df_blob', 'df_syn']
    for column in columns:
        print(f"Measures for column={column}")
        df_blob_column_sorted = df_blob[column].sort_values().reset_index(drop=True)
        df_syn_column_sorted = df_syn[column].sort_values().reset_index(drop=True)
        merged_df = pd.merge(df_blob_column_sorted, df_syn_column_sorted, left_index=True, right_index=True, suffixes=('_blob', '_syn'))
        differences = merged_df[merged_df[f'{column}_blob'] != merged_df[f'{column}_syn']]
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

def get_blob_paths(test_dir):
    blob_test_path = os.environ.get('BLOB_TEST_PATH')
    if blob_test_path is None:
        raise ValueError('BLOB_TEST_PATH environment variable must be set')
    blob_test_path = Path(blob_test_path)
    blob_test_path = blob_test_path.joinpath(test_dir)
    data_path = blob_test_path.joinpath(f"{test_dir}_table.parquet")
    return blob_test_path, data_path

def setup(test_dir):
    blob_test_path, data_path = get_blob_paths(test_dir)
    blob_test_path.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        return
    if test_dir == 'small_table1':
        df_raw = make_dataframe_small()
    df_raw.to_parquet(data_path)

def write(test_dir):
    setup(test_dir)
    blob_test_path, data_path = get_blob_paths(test_dir)

    df_raw = pd.read_parquet(data_path)

    sbw = SyndiffixBlobWriter(blob_name=test_dir, path_to_dir=blob_test_path, force=True)

    sbw.write_blob(df_raw=df_raw, pids=None)

def read(test_dir):
    setup(test_dir)
    blob_test_path, data_path = get_blob_paths(test_dir)
    df_raw = pd.read_parquet(data_path)

    sbr = SyndiffixBlobReader(blob_name=test_dir, path_to_dir=blob_test_path)

    df = sbr.read_blob(['i1', 'i2', 'f1', 't1', 't2',], target_column = 't1')

    for column in df_raw.columns:
        do_check([column], sbr, df_raw)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some commands.")
    
    # Add the arguments
    parser.add_argument('test_dir', type=str, help='The test directory')
    parser.add_argument('command', type=str, choices=['write', 'read'], help='The command to execute')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Assign the test_dir variable
    test_dir = args.test_dir
    print(f"Test directory: {test_dir}")
    
    # Execute the corresponding function based on the command
    if args.command == 'write':
        write(test_dir)
    elif args.command == 'read':
        read(test_dir)

if __name__ == "__main__":
    main()