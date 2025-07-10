## Overview

**SynDiffix** is an open-source tool for generating statistically-accurate and strongly anonymous synthetic data from structured data. Compared to existing open-source and proprietary commercial approaches, SynDiffix is

- many times more accurate,
- has comparable or better ML efficacy,
- runs as fast or faster,
- has stronger anonymization.

SynDiffix takes a **multi-table** approach to synthetic data. Rather than attempt to generate one table with all the columns of the original data, SynDiffix is designed to allow any number of tables with fewer columns to be generated while still maintaining strong anonymity. The benefit of this approach is that tables with fewer columns are more accurate with respect to those columns.

For best performance, one should only synthesize the columns needed for a specific given analytic task. For instance, if one wants only a single-column measure, one should synthesize a table with only that column. If one wants to measure the correlation between two columns, one would make a synthetic table with only those two columns.

## Purpose

Synthetic data has two primary use cases:

1. Descriptive analytics (histograms, heatmaps, column correlations, basic statistics like counting, averages, standard deviations, and so on)
2. Machine learning (building models)

While SynDiffix serves both use cases, it is especially good at descriptive analytics. The quality of descriptive analytics is many times that of other synthetic data products.

## Install

Install with `pip`:

`pip install syndiffix`

Requires python 3.10 or later.

## Usage

SynDiffix has two usage styles:
- Generate individual synthetic tables.
- Generate a "SynDiffix blob". The user of a SynDiffix blob can generate any table from the blob.

The individual table style is best when the data owner knows which tables are needed by users. The blob style is best when the data owner does not know which tables are needed by users, but the user must install SynDiffix in order to generate tables from the blob. The blob style is also appropriate when the data owner must delete the original data, for instance in order to comply with GDPR. The data owner can subsequently generate any synthetic table from the blob.

**Blob Limitations:** Currently the blob does not scale well with more than 10 or so columns. This limitation will be fixed in future releases.

### Generating individual tables

To generate individual tables, the 'Synthesizer' class is used. Usage can be as simple as:

```py
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original).sample()
```

This will create a new Pandas dataframe containing synthetic data, with the same number of columns and nearly the same number of rows as in the input dataframe.

The script [example_simple.py](example_simple.py) gives a simple CSV-in-CSV-out example for generating synthetic data.

### Generating and using a SynDiffix blob

To generate a SynDiffix blob, the `SyndiffixBlobBuilder` class is used:

```py
from syndiffix import SyndiffixBlobBuilder

sbb = SyndiffixBlobBuilder(blob_name='blob_name', path_to_dir='blob_path')
sbb.write(df_raw=df_orig)
```

The creates a SynDiffix blob called `blob_name.sdxblob.zip` in directory `blob_path`. The blob can be safely distributed to the public.

To use a SynDiffix blob, the `SyndiffixBlobReader` class is used:

```py
from syndiffix import SyndiffixBlobReader

sbr = SyndiffixBlobReader(blob_name='blob_name', path_to_dir='blob_path')
# Make a dataframe with columns col1 and col2
df_syn = sbr.read(columns=['col1', 'col2'])
```

Usage examples for blobs can be found in `blob_tester.py`.

### Maximizing data accuracy

Data accuracy is maximized by synthesizing only the columns required for the analytic task. For instance, if the goal is to understand the correlation between two columns `col1` and `col2`, then only those columns should be synthesized:

```py
# Individual table usage
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original[['col1','col2']]).sample()
```

```py
# Blob usage
from syndiffix import SyndiffixBlobReader

sbr = SyndiffixBlobReader(blob_name='blob_name', path_to_dir='blob_path')
df_synthetic = sbr.read(columns=['col1', 'col2'])
```

See the [tutorial notebook](docs/tutorial.ipynb) for an example. Note that anonymity is preserved regardless of how many different synthetic dataframes are generated from any given column.

### Maximizing ML efficacy relative to a given target column

When the use case is producing an ML model for a given target column 'your_target_col', the target column is specified as:

```py
# Individual table usage
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original, target_column='your_target_col')
```

```py
# Blob usage
from syndiffix import SyndiffixBlobReader

sbr = SyndiffixBlobReader(blob_name='blob_name', path_to_dir='blob_path')
df_synthetic = sbr.read(target_column='your_target_col')
```

Note that the quality of the predictive model for the target column will be much better than when not specifying the target column. If a model is needed for a different target column, then a separate synthetic dataframe should be created. See the [tutorial notebook](docs/tutorial.ipynb) for an example.

### Managing protected entities

A *protected entity* is the thing in the dataframe whose anonymity is being protected. Normally this is a person, but could be a device used by a person or really anything else.

If there are multiple rows per protected entity (e.g. event or time-series data), then there must be a column that identifies the protected entity, and this column must be conveyed in its own dataframe. Failing to do so compromises anonymity.

If the column identifying the protected entity is 'pid_col', then it is specified as:

```py
df_pid = df_original[["pid_col"]]
df_original = df_original.drop(columns=["pid_col"])

# Individual table usage
from syndiffix import Synthesizer
df_synthetic = Synthesizer(df_original, pids=df_pid).sample()

# Blob usage
from syndiffix import SyndiffixBlobBuilder
sbb = SyndiffixBlobBuilder(blob_name='blob_name', path_to_dir='blob_path')
sbb.write(df_raw=df_original, pids=df_pid)
```

Note that dropping 'pid_col' from `df_original` is not strictly necessary, but doing so leads to slightly higher quality data and faster execution time.

A dataframe can have multiple protected entities. Examples include sender and receiver in a banking transaction, or patient and doctor in a medical database. If the columns identifying two protected entities are 'pid_col1' and 'pid_col2', then they are specified as:

```py
pid_columns = ["pid_col1", "pid_col2"]
df_pids = df_original[pid_columns]
df_original = df_original.drop(columns=pid_columns)

# Individual table usage
from syndiffix import Synthesizer
df_synthetic = Synthesizer(df_original, pids=df_pid).sample()

# Blob usage
from syndiffix import SyndiffixBlobBuilder
sbb = SyndiffixBlobBuilder(blob_name='blob_name', path_to_dir='blob_path')
sbb.write(df_raw=df_original, pids=df_pid)
```

### Declaring columns with "safe" (publicly known) values

Often a column in a dataset contains values that are publicly known. These values do not need to be suppressed. Such columns can be declared with the `value_safe_columns` parameter. This parameter works with all data types. Only the values in the original column will appear in the corresponding synthetic data column. (In the case of floats with extreme precision, for instance 15 or more decimal places, the synthetic data values will be very close to the original data.)

```py
# Individual table usage
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original, value_safe_columns=['safe_col1', 'safe_col2'])

# Blob usage
from syndiffix import SyndiffixBlobBuilder
sbb = SyndiffixBlobBuilder(blob_name='blob_name', path_to_dir='blob_path')
sbb.write(df_raw=df_original, value_safe_columns=['safe_col1', 'safe_col2'])
```

### Other parameters

There are a wide variety of parameters that control the operation of SynDiffix. They are documented [here](docs/parameters.md).

### Additional information

The [time-series notebook](docs/time-series.ipynb) gives examples of how to obtain accurate statistics from time-series data. The [clustering notebook](docs/clustering.ipynb) gives examples about how to control the underlying clustering algorithm.

A step-by-step description of the algorithm can be found [here](docs/algorithm.md).

There is an API to the stitching function. It is primarily for testing and development purposes. A description can be found [here](docs/stitching-api.md).

A paper describing the design of SynDiffix, its performance, and its anonymity properties can be found [here on ArXiv](https://arxiv.org/abs/2311.09628).

A per-dimension range is internally called an interval (and handled by the `Interval` class), in order to avoid
potential name clashes with the native Python `range` API.

Development instructions are available [here](docs/development.md).
