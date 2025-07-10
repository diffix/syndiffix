from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, unique
from itertools import combinations
from typing import NewType, Sequence, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

Hash = np.uint64
Hashes = npt.NDArray[Hash]

Value = int | float | str | bool | pd.Timestamp | None
Row = tuple[Value, ...]


@unique
class ColumnType(Enum):
    BOOLEAN = 1
    INTEGER = 2
    REAL = 3
    STRING = 4
    TIMESTAMP = 5


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType


Columns = tuple[Column, ...]


@dataclass(frozen=True)
class FlatteningInterval:
    lower: int = 2
    upper: int = 5


@dataclass(frozen=True)
class SuppressionParams:
    low_threshold: int = 3
    layer_sd: float = 1.0
    low_mean_gap: float = 2.0


@dataclass(frozen=True)
class BucketizationParams:
    singularity_low_threshold: int = 5
    range_low_threshold: int = 15
    precision_limit_row_fraction: int = 10000
    precision_limit_depth_threshold: int = 15


@dataclass(frozen=True)
class AnonymizationParams:
    # Each noise layer seed is salted before being hashed with a cryptographically-strong algorithm.
    # The salt value needs to have at least 64 bits of entropy (equal or higher than that of the seed).
    # If the provided salt is empty, a per-system salt will be generated and used.
    salt: bytes = b""
    low_count_params: SuppressionParams = SuppressionParams()
    outlier_count: FlatteningInterval = FlatteningInterval()
    top_count: FlatteningInterval = FlatteningInterval()
    layer_noise_sd: float = 1.0


@dataclass(frozen=True)
class AnonymizationContext:
    bucket_seed: Hash
    anonymization_params: AnonymizationParams


# Global index of a column.
ColumnId = NewType("ColumnId", int)

Combination = tuple[ColumnId, ...]


def generate_combinations(k: int, n: int) -> Iterable[Combination]:
    return cast(Iterable[Combination], combinations(range(n), k) if k > 0 else [])


T = TypeVar("T")


def get_items_combination(combination: Combination, items: Sequence[T]) -> tuple[T, ...]:
    return tuple(items[index] for index in combination)


def get_items_combination_list(combination: Combination, items: Sequence[T]) -> list[T]:
    return [items[index] for index in combination]


def check_column_names_or_ids(df: pd.DataFrame, columns: int | str | ColumnId | list[int | ColumnId | str]) -> None:
    """
    Validate that column names or IDs are valid for the given DataFrame.

    Args:
        df: The DataFrame to validate against
        columns: Column name(s) or ID(s) to validate

    Raises:
        ValueError: If column names don't exist in DataFrame or ints are out of range
        TypeError: If columns parameter has invalid type
    """
    num_columns = len(df.columns)

    # Handle single values
    if isinstance(columns, str):
        if columns not in df.columns:
            raise ValueError(f"Column name '{columns}' not found in DataFrame. Available columns: {list(df.columns)}")
    elif isinstance(columns, int):
        if not (0 <= columns < num_columns):
            raise ValueError(
                f"ColumnId {columns} is out of range. DataFrame has {num_columns} columns (valid range: 0-{num_columns - 1})"
            )
    # Handle lists
    elif isinstance(columns, list):
        if not columns:
            return  # Empty list is valid

        # Check if all elements are strings
        if all(isinstance(col, str) for col in columns):
            invalid_columns = [col for col in columns if col not in df.columns]
            if invalid_columns:
                raise ValueError(
                    f"Column names {invalid_columns} not found in DataFrame. Available columns: {list(df.columns)}"
                )

        # Check if all elements are ColumnIds (integers)
        elif all(isinstance(col, int) for col in columns):
            invalid_ids = [col for col in columns if isinstance(col, int) and not (0 <= col < num_columns)]
            if invalid_ids:
                raise ValueError(
                    f"ColumnIds {invalid_ids} are out of range. DataFrame has {num_columns} columns (valid range: 0-{num_columns - 1})"
                )

        else:
            raise TypeError("List must contain either all strings or all ColumnIds")

    else:
        raise TypeError("columns must be a string, ColumnId, list of strings, or list of ColumnIds")
