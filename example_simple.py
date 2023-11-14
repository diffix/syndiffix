import os
import sys
import time

import pandas as pd
import psutil

from syndiffix import Synthesizer


# Utility function for loading a CSV file.
def load_csv(path: str) -> pd.DataFrame:
    from pandas.errors import ParserError

    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)

    # Try to infer datetime columns.
    for col in df.columns[df.dtypes == "object"]:
        try:
            df[col] = pd.to_datetime(df[col], format="ISO8601")
        except (ParserError, ValueError):
            pass

    return df


# Simple usage example of the SynDiffix library.
# This script assumes each row belongs to a different protected entity.
# All columns in the input file are processed.


if len(sys.argv) != 3:
    print(f"Usage: py {sys.argv[0]} <input.csv> <output.csv>")
    exit()

input_file = sys.argv[1]
output_file = sys.argv[2]


print(f"Loading data from `{input_file}`...")
input_data = load_csv(input_file)

print(f"Loaded {len(input_data)} rows. Columns:")
for i, (column, dtype) in enumerate(zip(input_data.columns, input_data.dtypes)):
    print(f"{i}: {column} ({dtype})")

start_time = time.time()
process = psutil.Process(os.getpid())
start_memory_usage = process.memory_info().rss

print("\nFitting the synthesizer over the data...")
synthesizer = Synthesizer(input_data)

print("Column clusters:")
print("Initial=", synthesizer.clusters.initial_cluster)
for cluster in synthesizer.clusters.derived_clusters:
    print("Derived=", cluster)

print("\nSampling rows from the synthesizer...")
output_data = synthesizer.sample()

run_time = round(time.time() - start_time)
memory_usage = (process.memory_info().rss - start_memory_usage) // (1024**2)
print(f"Runtime: {run_time} seconds. Memory usage: {memory_usage} MB.")

print(f"\nWriting sampled rows to `{output_file}`...")
output_data.to_csv(output_file, index=False)

print("Done!")
