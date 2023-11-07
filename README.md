# Overview

**SynDiffix** is a new approach to generating statistically-accurate and strongly anonymous synthetic data from structured
data. Compared to existing open-source and proprietary commercial approaches, SynDiffix is

- many times more accurate,
- has comparable or better ML efficacy,
- runs at least an order of magnitude faster, and
- has equal or stronger anonymization.

## Purpose

This library implements the SynDiffix method for tabular data synthesis in pure Python.

## Implementation details

A step-by-step description of the algorithm can be found [docs/algorithm.md](here).

A per-dimension range is internally called an interval (and handled by the `Interval` class), in order to avoid
potential name clashes with the native Python `range` API.

## Installation

Prerequisites: [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

1. `poetry install`

## Development

1. Activate `poetry` environment: `poetry shell`. (can skip, then prepend `poetry run` to the next commands)
2. Format: `black . && isort .`
3. Test: `pytest .`
4. Lints and checks: `flake8 . && mypy . && black --check . && isort . --check`

## Usage

Usage can be as simple as:

```py
raw_data = load_dataframe()
syn_data = Synthesizer(raw_data).sample()
```

This will create a new Pandas DataFrame containing synthetic data, with the same number of columns and the same
number of rows as in the input data.
Default settings will be used and each row in the original data has to belong to a different protected entity.

The script [example_simple.py](example_simple.py) shows a simple example on how to process all data in a CSV file
that holds a different protected entity in each row.

### Processing data with multiple rows per-entity

If the same entity can have multiple rows belonging to it, then a dataframe with the AID values has to be passed separately:

```py
raw_data = load_dataframe()
aid_columns = ["aid1", "aid2"]
aids = raw_data[aid_columns]
raw_data = raw_data.drop(columns=aid_columns)
syn_data = Synthesizer(raw_data, aids=aids).sample()
```

The AID columns should not be a part of the input data to the synthesizer.

### Changing anonymization parameters

Anonymization parameters determine how much raw data is suppressed and distorted before synthesis.
To modify default parameters, pass a custom instance of the `AnonymizationParams` class
to the `Synthesizer` object. For example:

```py
Synthesizer(raw_data, anonymization_params=AnonymizationParams(layer_noise_sd=1.5))
```

The following parameters are available:

- `salt`: noise salt for the data source; if empty, an automatically generated value is used.

- `low_count_params`: parameters for the low-count filter.

- `outlier_count`: outlier count interval used during flattening.

- `top_count`: top count interval used during flattening.

- `layer_noise_sd`: stddev of each noise layer added to row counts.

### Changing bucketization parameters

Bucketization parameters determine how raw data is aggregated before synthesis.
To modify default parameters, pass a custom instance of the `BucketizationParams` class
to the `Synthesizer` object. For example:

```py
Synthesizer(raw_data, bucketization_params=BucketizationParams(precision_limit_depth_threshold=10))
```

The following parameters are available:

- `singularity_low_threshold`: low threshold for a singularity bucket.

- `range_low_threshold`: low threshold for a range bucket.

- `precision_limit_row_fraction`: the fraction of rows needed for splitting nodes, in addition to passing the
  low-count filter, when the tree depth goes beyond the depth threshold.

- `precision_limit_depth_threshold`: tree depth threshold below which nodes are split only if they pass the low-count filter.

### Changing clustering strategy

Clustering strategy determines how columns are grouped together in the forest of trees used for
anonymization and aggregation. Anonymized buckets are harvested from those trees, synthetic
microdata tables are generated, and are then stitched together into a single output table.
To change the clustering strategy, pass an instance for a sub-class of the `ClusteringStrategy` class
to the `Synthesizer` object. For example:

```py
Synthesizer(raw_data, clustering=NoClustering())
```

The following strategies are available:

- `NoClustering`: - strategy that disables clustering; puts all columns in a single cluster.

- `DefaultClustering`: - general-purpose clustering strategy; columns are grouped together in order to maximize the
  chi-square dependence measurement values of the generated clusters; a main column that gets put into every cluster
  can be specified, in order to improve output quality for that specific column.

- `MLClustering`: - strategy for ML tasks; main feature columns for a target column are automatically detected and
  grouped together with the target column in the order of their ML prediction-test scores.
