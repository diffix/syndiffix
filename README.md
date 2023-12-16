## Overview

**SynDiffix** is an open-source tool for generating statistically-accurate and strongly anonymous synthetic data from structured data. Compared to existing open-source and proprietary commercial approaches, SynDiffix is

- many times more accurate,
- has comparable or better ML efficacy,
- runs as fast or faster,
- has stronger anonymization.

## Purpose

Synthetic data has two primary use cases:

1. Descriptive analytics (histograms, heatmaps, column correlations, basic statistics like counting, averages, standard deviations, and so on)
2. Machine learning (building models)

While **SynDiffix** serves both use cases, it is especially good at descriptive analytics. The quality of descriptive analytics is many times that of other synthetic data products.

Obtaining this improvement, however, requires a different usage style compared to other products. The intended usage style of other products is "*one size fits all*": a single synthetic dataset serves all use cases. By contrast, with **SynDiffix**, a different *tailored* synthetic dataset should be produced for each use case.

## Install

Install with `pip`:

`pip install syndiffix`

Requires python 3.10 or later.

## Usage

Usage can be as simple as:

```py
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original).sample()
```

This will create a new Pandas dataframe containing synthetic data, with the same number of columns and the similar number of rows as in the input dataframe.

The script [example_simple.py](example_simple.py) gives a simple CSV-in-CSV-out example for generating synthetic data.

### Maximizing data accuracy

Data accuracy is maximized by synthesizing only the columns required for the analytic task. For instance, if the goal is to understand the correlation between two columns `col1` and `col2`, then only those columns should be synthesized:

```py
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original[['col1','col2']]).sample()
```

See the [tutorial notebook](docs/tutorial.ipynb) for an example. Note that anonymity is preserved regardless of how many different synthetic dataframes are generated from any given column.

### Maximizing ML efficacy relative to a given target column

When the use case is producing an ML model for a given target column 'your_target_col', the target column is specified as:

```py
from syndiffix import Synthesizer

df_synthetic = Synthesizer(df_original, target_column='your_target_col')
```

Note that the quality of the predictive model for the target column will be much better than when not specifying the target column. If a model is needed for a different target column, then a separate synthetic dataframe should be created. See the [tutorial notebook](docs/tutorial.ipynb) for an example.

### Managing protected entities

A *protected entity* is the thing in the dataframe whose anonymity is being protected. Normally this is a person, but could be a device used by a person or really anything else.

If there are multiple rows per protected entity (e.g. event or time-series data), then there must be a column that identifies the protected entity, and this column must be conveyed in its own dataframe. Failing to do so compromises anonymity.

If the column identifying the protected entity is 'pid_col', then it is specified as:

```py
from syndiffix import Synthesizer

df_pid = df_original[["pid_col"]]
df_original = df_original.drop(columns=["pid_col"])
df_synthetic = Synthesizer(df_original, pids=df_pid).sample()
```

Note that dropping 'pid_col' from `df_original` is not strictly necessary, but doing so leads to slightly higher quality data and faster execution time.

A dataframe can have multiple protected entities. Examples include sender and receiver in a banking transaction, or patient and doctor in a medical database. If the columns identifying two protected entities are 'pid_col1' and 'pid_col2', then they are specified as:

```py
from syndiffix import Synthesizer

pid_columns = ["pid_col1", "pid_col2"]
df_pids = df_original[pid_columns]
df_original = df_original.drop(columns=pid_columns)
df_synthetic = Synthesizer(df_original, pids=df_pids).sample()
```

### Other parameters

There are a wide variety of parameters that control the operation of **SynDiffix**. They are documented [here](docs/parameters.md).

### Additional information

The [time-series notebook](docs/time-series.ipynb) gives examples of how to obtain accurate statistics from time-series data. The [clustering notebook](docs/clustering.ipynb) gives examples about how to control the underlying clustering algorithm.

A step-by-step description of the algorithm can be found [here](docs/algorithm.md).

A paper describing the design of **SynDiffix**, its performance, and its anonymity properties can be found
[here on ArXiv](https://arxiv.org/abs/2311.09628).

A per-dimension range is internally called an interval (and handled by the `Interval` class), in order to avoid
potential name clashes with the native Python `range` API.

Development instructions are available [here](docs/development.md).
