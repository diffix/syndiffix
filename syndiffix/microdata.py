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
from .common import ColumnType, Value
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

    @abstractmethod
    def column_type(self) -> ColumnType:
        pass

    @abstractmethod
    def to_float(self, value: Value) -> float:
        pass

    @abstractmethod
    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        pass

    def analyze_tree(self, root: Node) -> None:
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
        value = _generate_float(interval, rng) >= 0.5
        return (value, 1.0 if value else 0.0)


class RealConvertor(DataConvertor):
    def __init__(self, values: Iterable[Value]) -> None:
        super().__init__()
        # Fit up to 0.9999 so that the max bucket range is 0-1
        self.scaler = MinMaxScaler(feature_range=(0, 0.9999))
        # This value-neutral fitting is only for passing unit tests.
        self.scaler.fit(np.array([[0.0], [0.9999]]))
        self.round_precision = _get_round_precision(values)

    def column_type(self) -> ColumnType:
        return ColumnType.REAL

    def to_float(self, value: Value) -> float:
        assert isinstance(value, float) or isinstance(value, np.floating)
        return float(value)

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_float(interval, rng)
        value = _inverse_normalize_value(value, self.scaler)
        value = round(value, self.round_precision)
        return (value, value)


class IntegerConvertor(DataConvertor):
    def __init__(self) -> None:
        super().__init__()
        # Fit up to 0.9999 so that the max bucket range is 0-1
        self.scaler = MinMaxScaler(feature_range=(0, 0.9999))
        # This value-neutral fitting is only for passing unit tests.
        self.scaler.fit(np.array([[0.0], [0.9999]]))

    def column_type(self) -> ColumnType:
        return ColumnType.INTEGER

    def to_float(self, value: Value) -> float:
        assert isinstance(value, int) or isinstance(value, np.integer)
        return float(value)

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_float(interval, rng)
        value = _inverse_normalize_value(value, self.scaler)
        value = round(value)
        return (value, float(value))


class TimestampConvertor(DataConvertor):
    def __init__(self) -> None:
        super().__init__()
        # Fit up to 0.9999 so that the max bucket range is 0-1
        self.scaler = MinMaxScaler(feature_range=(0, 0.9999))
        # This value-neutral fitting is only for passing unit tests.
        self.scaler.fit(np.array([[0.0], [0.9999]]))

    def column_type(self) -> ColumnType:
        return ColumnType.TIMESTAMP

    def to_float(self, value: Value) -> float:
        assert isinstance(value, pd.Timestamp)
        # converting date time into second timestamp, counting from reference.
        return float((value - TIMESTAMP_REFERENCE) / pd.Timedelta(1, "s"))

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        value = _generate_float(interval, rng)
        value = _inverse_normalize_value(value, self.scaler)
        datetime = TIMESTAMP_REFERENCE + np.timedelta64(int(value), "s")
        return (datetime, value)


class StringConvertor(DataConvertor):
    def __init__(self, values: Iterable[Value]) -> None:
        super().__init__()
        unique_values = set(v for v in values if not pd.isna(v))
        for value in unique_values:
            if not isinstance(value, str):
                raise TypeError(f"Not a `str` object in a string dtype column: {value}.")
        self.value_map = sorted(cast(Set[str], unique_values))
        self.safe_values: Set[int] = set()

    def column_type(self) -> ColumnType:
        return ColumnType.STRING

    def to_float(self, value: Value) -> float:
        index = bisect_left(self.value_map, cast(str, value))
        assert index >= 0 and index < len(self.value_map)
        return float(index)

    def from_interval(self, interval: Interval, rng: Random) -> MicrodataValue:
        if interval.is_singularity():
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
        if value in self.safe_values:
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
                if node.is_singularity() and node.is_over_threshold(low_threshold):
                    self.safe_values.add(int(node.actual_intervals[0].min))
            elif isinstance(node, Branch):
                for child_node in node.children.values():
                    analyze_tree_walk(child_node)

        analyze_tree_walk(root)


def _generate_float(interval: Interval, rng: Random) -> float:
    return rng.uniform(interval.min, interval.max)


def _get_round_precision(values: Iterable[Value]) -> int:
    max_precision = 0
    for value in values:
        assert isinstance(value, float) or isinstance(value, np.floating)
        value_str = str(value)
        if "." in value_str:
            decimal_part = value_str.split(".")[1]
            precision = len(decimal_part)
        else:
            precision = 0
        if precision > max_precision:
            max_precision = precision
    return max_precision


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

    return pd.DataFrame(dict(zip(raw_data.columns, possibly_normalized_columns)), copy=False)


def generate_microdata(
    buckets: Buckets, convertors: list[DataConvertor], null_mappings: list[float], rng: Random
) -> list[MicrodataRow]:
    microdata_rows: list[MicrodataRow] = []
    for bucket in buckets:
        microdata_rows.extend(
            islice(_microdata_row_generator(bucket.intervals, convertors, null_mappings, rng), bucket.count)
        )

    return microdata_rows
