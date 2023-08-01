# Pure Python implementation of SynDiffix

## Installation

Prerequisites: [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

1. `poetry install`

## Development

1. Activate `poetry` environment: `poetry shell`. (can skip, then prepend `poetry run` to the next commands)
2. Format: `black .`
3. Test: `pytest .`
4. Lints and checks: `flake8 . && mypy . && black --check .`
