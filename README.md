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
