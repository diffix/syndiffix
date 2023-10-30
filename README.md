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

### Changing anonymization settings

TODO

### Changing bucketization settings

TODO

### Changing clustering strategy

TODO
