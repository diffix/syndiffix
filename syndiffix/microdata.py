from abc import ABC, abstractmethod
from bisect import bisect_left
from itertools import islice
from os.path import commonprefix
from random import Random
from typing import Generator, Iterable, Literal, Optional, Set, cast

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)
from sklearn.preprocessing import MinMaxScaler

from .bucket import Buckets
from .common import ColumnId, ColumnType, Value, check_column_names_or_ids
from .interval import Interval, Intervals
from .tree import Branch, Leaf, Node

MICRODATA_SYN_VALUE: Literal[0] = 0
MICRODATA_FLOAT_VALUE: Literal[1] = 1

MicrodataValue = tuple[Value, float]
MicrodataRow = list[MicrodataValue]


TIMESTAMP_REFERENCE = pd.Timestamp("1800-01-01T00:00:00")


class DataConvertor(ABC):
    def __init__(self) -> None:
        self.scaler: Optional[MinMaxScaler] = None
        self.value_safe_flag: bool = False

    @abstractmethod
    def column_type(self) -> ColumnType:
        pass

    @abstractmethod
    def to_float(self, value: Value) -> float:
        pass

    @abstractmethod
    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        pass

    def set_value_safe_flag(self, value_safe: bool) -> None:
        self.value_safe_flag = value_safe

    @abstractmethod
    def create_value_safe_set(self, values: pd.Series) -> None:
        pass

    def analyze_tree(self, root: Node) -> None:
        pass

    def denormalize_safe_values(self) -> None:
        pass


class BooleanConvertor(DataConvertor):
    def __init__(self) -> None:
        super().__init__()

    def column_type(self) -> ColumnType:
        return ColumnType.BOOLEAN

    def to_float(self, value: Value) -> float:
        assert isinstance(value, bool) or isinstance(value, np.bool_)
        return 1.0 if value else 0.0

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_random_float(interval, rng) >= 0.5
        return (value, 1.0 if value else 0.0)

    def create_value_safe_set(self, values: pd.Series) -> None:
        # not needed
        pass


class RealConvertor(DataConvertor):
    def __init__(self, values: Iterable[Value]) -> None:
        super().__init__()
        # Fit up to 0.9999 so that the max bucket range is [0-1)
        self.scaler = MinMaxScaler(feature_range=(0.0, 0.9999))  # type: ignore
        # This value-neutral fitting is only for passing unit tests, gets overridden
        # later by fit_transform().
        self.scaler.fit(np.array([[0.0], [0.9999]]))
        self.final_round_precision = _get_round_precision(cast(Iterable[float], values))

    def column_type(self) -> ColumnType:
        return ColumnType.REAL

    def to_float(self, value: Value) -> float:
        assert value is not None and (isinstance(value, float) or isinstance(value, np.floating))
        return round(float(value), self.final_round_precision)

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_random_float(interval, rng)
        if self.value_safe_flag is True:
            value = _convert_to_safe_value(value, self.safe_values)
        assert self.scaler is not None
        value = _inverse_normalize_value(value, self.scaler)
        # This last rounding is cosmetic, to give the synthetic data the look and feel of the original data.
        value = round(value, self.final_round_precision)
        return (value, value)

    def create_value_safe_set(self, values: pd.Series) -> None:
        if self.value_safe_flag is True:
            self.safe_values = sorted(values.unique())


class IntegerConvertor(DataConvertor):
    def __init__(self) -> None:
        super().__init__()
        # Fit up to 0.9999 so that the max bucket range is [0-1)
        self.scaler = MinMaxScaler(feature_range=(0.0, 0.9999))  # type: ignore
        # This value-neutral fitting is only for passing unit tests, gets overridden
        # later by fit_transform().
        self.scaler.fit(np.array([[0.0], [0.9999]]))

    def column_type(self) -> ColumnType:
        return ColumnType.INTEGER

    def to_float(self, value: Value) -> float:
        assert value is not None and (isinstance(value, int) or isinstance(value, np.integer))
        return float(value)

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_random_float(interval, rng)
        if self.value_safe_flag is True:
            value = _convert_to_safe_value(value, self.safe_values)
        assert self.scaler is not None
        value = _inverse_normalize_value(value, self.scaler)
        value = round(value)
        return (value, float(value))

    def create_value_safe_set(self, values: pd.Series) -> None:
        if self.value_safe_flag is True:
            self.safe_values = sorted(values.unique())


class TimestampConvertor(DataConvertor):
    def __init__(self) -> None:
        super().__init__()
        # Fit up to 0.9999 so that the max bucket range is [0-1)
        self.scaler = MinMaxScaler(feature_range=(0.0, 0.9999))  # type: ignore
        # This value-neutral fitting is only for passing unit tests, gets overridden
        # later by fit_transform().
        self.scaler.fit(np.array([[0.0], [0.9999]]))

    def column_type(self) -> ColumnType:
        return ColumnType.TIMESTAMP

    def to_float(self, value: Value) -> float:
        assert isinstance(value, pd.Timestamp)
        # converting date time into second timestamp, counting from reference.
        return float((value - TIMESTAMP_REFERENCE) / pd.Timedelta(1, "s"))

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_random_float(interval, rng)
        if self.value_safe_flag is True:
            value = _convert_to_safe_value(value, self.safe_values)
        assert self.scaler is not None
        value = _inverse_normalize_value(value, self.scaler)
        datetime = TIMESTAMP_REFERENCE + np.timedelta64(round(value), "s")
        return (datetime, value)

    def create_value_safe_set(self, values: pd.Series) -> None:
        if self.value_safe_flag is True:
            self.safe_values = sorted(values.unique())


class StringConvertor(DataConvertor):
    def __init__(self, values: Iterable[Value]) -> None:
        super().__init__()
        unique_values = set(v for v in values if not pd.isna(v))
        for value in unique_values:
            if not isinstance(value, str):
                raise TypeError(f"Not a `str` object in a string dtype column: {value}.")
        self.value_map = sorted(cast(Set[str], unique_values))

        # Note that self.safe_values is only used if self.value_safe_flag is False
        self.safe_values: Set[float] = set()
        # Fit up to 0.9999 so that the max bucket range is [0-1)
        self.scaler = MinMaxScaler(feature_range=(0.0, 0.9999))  # type: ignore
        # This value-neutral fitting is only for passing unit tests, gets overridden
        # later by fit_transform().
        self.scaler.fit(np.array([[0.0], [0.9999]]))

    def column_type(self) -> ColumnType:
        return ColumnType.STRING

    def to_float(self, value: Value) -> float:
        # Note that value here is the string itself, not an index.
        index = bisect_left(self.value_map, cast(str, value))
        assert index >= 0 and index < len(self.value_map)
        return float(index)

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        assert self.scaler is not None
        interval = _find_encapsulated_integer_interval(interval, self.scaler)
        # From here on intervals are integers (cast as float)
        if interval.is_singularity():
            # convert to integer for value_map
            return (self.value_map[int(interval.min)], interval.min)
        else:
            return self._map_interval(interval, rng)

    def _map_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        # If a randomly selected value from the interval is not safe, finds a
        # common prefix of the strings encoded as interval boundaries and
        # appends "*" and a random number to ensure that the count of distinct
        # values approximates that in the original data.
        min_value = int(interval.min)
        # max_value is inclusive
        max_value = min(int(interval.max) - 1, len(self.value_map) - 1)
        # The latter term in the above line can 0 (not sure why TODO: check)
        max_value = max(min_value, max_value)
        value = rng.randint(min_value, max_value)
        if self.value_safe_flag is True or value in self.safe_values:
            return (self.value_map[value], float(value))
        else:
            return (
                commonprefix([self.value_map[min_value], self.value_map[max_value]]) + "*" + str(value),
                float(value),
            )

    def analyze_tree(self, root: Node) -> None:
        def analyze_tree_walk(node: Node) -> None:
            if isinstance(node, Leaf):
                low_threshold = node.context.anonymization_context.anonymization_params.low_count_params.low_threshold
                # Avoid the cost of maintaining safe_values if in any
                # event all values are safe (i.e. self.value_safe_flag is True)
                if self.value_safe_flag is False and node.is_singularity() and node.is_over_threshold(low_threshold):
                    # Note that the values here are normalized
                    self.safe_values.add(float(node.actual_intervals[0].min))
            elif isinstance(node, Branch):
                for child_node in node.children.values():
                    analyze_tree_walk(child_node)

        analyze_tree_walk(root)
        # from .tree import _dump_tree
        # _dump_tree(root)       # Debugging line to see the tree structure

    def denormalize_safe_values(self) -> None:
        assert self.scaler is not None
        if self.value_safe_flag is False and self.safe_values:
            # Convert normalized values back to original integer values
            denormalized_safe_values = set()
            for normalized_value in self.safe_values:
                original_value = _inverse_normalize_value(float(normalized_value), self.scaler)
                denormalized_safe_values.add(int(round(original_value)))
            self.safe_values = denormalized_safe_values

    def create_value_safe_set(self, values: pd.Series) -> None:
        # Not needed
        pass


def _generate_random_float(interval: Interval, rng: Random) -> float:
    return rng.uniform(interval.min, interval.max)


def _get_round_precision(values: Iterable[float]) -> int:
    max_precision = 0
    for value in values:
        precision = _get_precision_from_real(value)
        if precision > max_precision:
            max_precision = precision
    return max_precision


def _get_precision_from_real(value: float) -> int:
    value_str = str(value)
    if "." in value_str:
        decimal_part = value_str.split(".")[1]
        precision = len(decimal_part)
    else:
        precision = 0
    return precision


def _convert_to_safe_value(value: float, safe_value_set: list[float]) -> float:
    """
    Return the value from safe_value_set that is closest to value.

    Args:
        value: The target value to find closest match for
        safe_value_set: Sorted list of safe values in ascending order

    Returns:
        The safe value closest to the input value
    """
    if not safe_value_set:
        raise ValueError("safe_value_set cannot be empty")

    # Find insertion point using bisect
    insert_pos = bisect_left(safe_value_set, value)

    # Handle edge cases
    if insert_pos == 0:
        return safe_value_set[0]
    elif insert_pos == len(safe_value_set):
        return safe_value_set[-1]

    # Compare distances to left and right neighbors
    left_value = safe_value_set[insert_pos - 1]
    right_value = safe_value_set[insert_pos]

    if abs(value - left_value) <= abs(value - right_value):
        return left_value
    else:
        return right_value


def _generate(interval: Interval, convertor: DataConvertor, null_mapping: float, rng: Random) -> MicrodataValue:
    return convertor.from_interval(interval, rng) if interval.min != null_mapping else (None, null_mapping)


def _microdata_row_generator(
    intervals: Intervals, convertors: list[DataConvertor], null_mappings: list[float], rng: Random
) -> Generator[MicrodataRow, None, None]:
    assert len(intervals) == len(convertors)
    assert len(intervals) == len(null_mappings)
    while True:
        yield [_generate(i, c, nm, rng) for i, c, nm in zip(intervals, convertors, null_mappings)]


def get_convertor(df: pd.DataFrame, column: str) -> DataConvertor:
    dtype = df.dtypes[column]
    if is_integer_dtype(dtype):
        return IntegerConvertor()
    elif is_float_dtype(dtype):
        return RealConvertor(df[column])
    elif is_bool_dtype(dtype):
        return BooleanConvertor()
    elif is_datetime64_dtype(dtype):
        return TimestampConvertor()
    elif is_string_dtype(dtype):
        # Note above is `True` for `object` dtype, but `StringConvertor` will assert values are `str`.
        return StringConvertor(df[column])
    else:
        raise TypeError(f"Dtype {dtype} is not supported.")


def _inverse_normalize_value(value: float, scaler: MinMaxScaler) -> float:
    # Inverse of normalize, but for one value at a time
    value_array = np.array([[value]])
    inverse_transformed_array = scaler.inverse_transform(value_array)
    inverse_transformed_value = inverse_transformed_array[0, 0]
    return float(inverse_transformed_value)


def _find_encapsulated_integer_interval(interval: Interval, scaler: MinMaxScaler) -> Interval:
    """
    Find the largest interval within the given interval where the inverse-transformed
    bounds correspond to integers (within machine precision).

    Args:
        interval: The input interval in normalized space
        scaler: The MinMaxScaler used for inverse transformation

    Returns:
        A new interval with bounds that are integer values (cast as floats)
    """
    interval_new = interval.copy()

    # Handle singularity case - bounds are already at the same point
    if interval.is_singularity():
        # Convert the single value to its corresponding integer
        inverse_value = _inverse_normalize_value(interval.min, scaler)
        integer_value = float(round(inverse_value))
        interval_new.min = integer_value
        interval_new.max = integer_value
        return interval_new

    # Find the smallest integer >= the inverse-transformed interval.min
    min_inverse = _inverse_normalize_value(interval.min, scaler)
    min_integer = int(round(min_inverse))

    # If the current min already transforms to an integer (within precision), use it
    if abs(min_inverse - min_integer) < 1e-10:
        interval_new.min = float(min_integer)
    else:
        # Find the next integer
        next_integer = min_integer + 1 if min_inverse > min_integer else min_integer
        interval_new.min = float(next_integer)

    # Find the largest integer <= the inverse-transformed interval.max
    max_inverse = _inverse_normalize_value(interval.max, scaler)
    max_integer = int(round(max_inverse))

    # Note that the max value of an Interval is exclusive, so we need to take care
    if abs(max_inverse - max_integer) < 1e-10:
        # If this is exact, then it will be included in the next higher min_integer
        interval_new.max = float(max_integer)
    else:
        # Find the previous integer
        prev_integer = max_integer - 1 if max_inverse < max_integer else max_integer
        # We add 1.0 because the max value is exclusive
        interval_new.max = float(prev_integer + 1.0)

    # Ensure the new interval is valid (min <= max)
    if interval_new.min > interval_new.max:
        # If no valid integer interval exists within bounds, throw an exception
        raise ValueError(
            f"No valid integer interval exists within bounds. "
            f"Min integer: {interval_new.min}, Max integer: {interval_new.max}"
        )

    return interval_new


def _normalize(values: pd.Series, scaler: Optional[MinMaxScaler]) -> pd.Series:
    if scaler is None:
        # Convertors that don't need normalization
        return values

    # MinMax normalize values, while retaining the NaN values
    values_array = values.to_numpy()
    nan_indices = np.isnan(values_array)
    if nan_indices.all():
        return values
    median_value = np.nanmedian(values_array)
    values_array[nan_indices] = median_value
    values_reshaped = values_array.reshape(-1, 1)
    normalized_values = scaler.fit_transform(values_reshaped).flatten()
    normalized_values[nan_indices] = np.nan
    return pd.Series(normalized_values, index=values.index)


def _apply_convertor(value: Value, convertor: DataConvertor) -> float:
    if pd.isna(value):
        return np.nan
    else:
        return convertor.to_float(value)


def apply_convertors(convertors: list[DataConvertor], raw_data: pd.DataFrame) -> pd.DataFrame:
    converted_columns = [
        raw_data[column].apply(_apply_convertor, convertor=convertor)
        for column, convertor in zip(raw_data.columns, convertors)
    ]

    possibly_normalized_columns = [
        _normalize(column, convertor.scaler) for column, convertor in zip(converted_columns, convertors)
    ]

    for column, convertor in zip(possibly_normalized_columns, convertors):
        convertor.create_value_safe_set(column)

    return pd.DataFrame(dict(zip(raw_data.columns, possibly_normalized_columns)), copy=False)


def generate_microdata(
    buckets: Buckets, convertors: list[DataConvertor], null_mappings: list[float], rng: Random
) -> list[MicrodataRow]:
    # print(buckets)      # Debugging line to see the buckets
    [convertor.denormalize_safe_values() for convertor in convertors]
    microdata_rows: list[MicrodataRow] = []
    for bucket in buckets:
        microdata_rows.extend(
            islice(_microdata_row_generator(bucket.intervals, convertors, null_mappings, rng), bucket.count)
        )

    return microdata_rows


def make_value_safe_columns_array(df: pd.DataFrame, value_safe_columns: list[int | str | ColumnId]) -> list[bool]:
    """
    Create a boolean array indicating which columns are value-safe.

    Args:
        df: The DataFrame to create the array for
        value_safe_columns: List of column names (strings) or column indices (ints)

    Returns:
        List of booleans, same length as number of columns in df. True if the
        corresponding column is marked as value-safe.

    Raises:
        ValueError: If column names don't exist in DataFrame or indices are out of range
        TypeError: If value_safe_columns contains invalid types
    """
    # Validate the input using existing function
    check_column_names_or_ids(df, value_safe_columns)

    # Initialize boolean array with False values
    result = [False] * len(df.columns)

    # Set True for specified columns
    for column in value_safe_columns:
        if isinstance(column, str):
            # Find column index by name
            column_index = df.columns.get_loc(column)
            if isinstance(column_index, int):
                result[column_index] = True
            else:
                raise TypeError(f"Column index for '{column}' is not an integer: {column_index}")
        elif isinstance(column, int):
            # Use column index directly
            result[column] = True

    return result
