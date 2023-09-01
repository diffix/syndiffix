from abc import ABC, abstractmethod
from itertools import islice
from os.path import commonprefix
from random import Random
from typing import Generator, Iterable, Literal, Set, cast

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)

from .bucket import Bucket
from .common import ColumnType, Value
from .interval import Interval

MICRODATA_SYN_VALUE: Literal[0] = 0
MICRODATA_FLOAT_VALUE: Literal[1] = 1

MicrodataValue = tuple[Value, float]
MicrodataRow = list[MicrodataValue]

# This source of randomness isn't sticky, so can only be applied to already anonymized data.
_non_sticky_rng = Random(0)

TIMESTAMP_REFERENCE = pd.Timestamp("1800-01-01T00:00:00")


class DataConvertor(ABC):
    @abstractmethod
    def column_type(self) -> ColumnType:
        pass

    @abstractmethod
    def to_float(self, value: Value) -> float:
        pass

    @abstractmethod
    def from_interval(self, interval: Interval) -> MicrodataValue:
        pass


class BooleanConvertor(DataConvertor):
    def column_type(self) -> ColumnType:
        return ColumnType.BOOLEAN

    def to_float(self, value: Value) -> float:
        assert isinstance(value, bool) or isinstance(value, np.bool_)
        return 1.0 if value else 0.0

    def from_interval(self, interval: Interval) -> MicrodataValue:
        value = _generate_float(interval) >= 0.5
        return (value, 1.0 if value else 0.0)


class RealConvertor(DataConvertor):
    def column_type(self) -> ColumnType:
        return ColumnType.REAL

    def to_float(self, value: Value) -> float:
        assert isinstance(value, float) or isinstance(value, np.floating)
        return value

    def from_interval(self, interval: Interval) -> MicrodataValue:
        value = _generate_float(interval)
        return (value, value)


class IntegerConvertor(DataConvertor):
    def column_type(self) -> ColumnType:
        return ColumnType.INTEGER

    def to_float(self, value: Value) -> float:
        assert isinstance(value, int) or isinstance(value, np.integer)
        return float(value)

    def from_interval(self, interval: Interval) -> MicrodataValue:
        value = int(_generate_float(interval))
        return (value, float(value))


class TimestampConvertor(DataConvertor):
    def column_type(self) -> ColumnType:
        return ColumnType.TIMESTAMP

    def to_float(self, value: Value) -> float:
        assert isinstance(value, pd.Timestamp)
        # converting date time into second timestamp, counting from reference.
        return float((value - TIMESTAMP_REFERENCE) / pd.Timedelta(1, "s"))

    def from_interval(self, interval: Interval) -> MicrodataValue:
        value = _generate_float(interval)
        datetime = TIMESTAMP_REFERENCE + np.timedelta64(int(value), "s")
        return (datetime, value)


class StringConvertor(DataConvertor):
    def __init__(self, values: Iterable[Value]) -> None:
        unique_values = set(v for v in values if not pd.isna(v))
        for value in unique_values:
            if not isinstance(value, str):
                raise TypeError(f"Not a `str` object in a string dtype column: {value}.")
        self.value_map = sorted(cast(Set[str], unique_values))

    def column_type(self) -> ColumnType:
        return ColumnType.STRING

    def to_float(self, value: Value) -> float:
        value = cast(str, value)
        return float(self.value_map.index(value))

    def from_interval(self, interval: Interval) -> MicrodataValue:
        if interval.is_singularity():
            return (self.value_map[int(interval.min)], interval.min)
        else:
            return self._map_interval(interval)

    def _map_interval(self, interval: Interval) -> MicrodataValue:
        # Finds a common prefix of the strings encoded as interval boundaries and appends "*"
        # and a random number to ensure that the count of distinct values approximates that in the original data.
        min_value = self.value_map[int(interval.min)]
        max_value = self.value_map[min(int(interval.max), len(self.value_map) - 1)]
        value = int(_generate_float(interval))

        return (commonprefix([min_value, max_value]) + "*" + str(value), float(value))


def _generate_float(interval: Interval) -> float:
    return _non_sticky_rng.uniform(interval.min, interval.max)


def _generate(interval: Interval, convertor: DataConvertor, null_mapping: float) -> MicrodataValue:
    return convertor.from_interval(interval) if interval.min != null_mapping else (None, null_mapping)


def _microdata_row_generator(
    intervals: list[Interval], convertors: list[DataConvertor], null_mappings: list[float]
) -> Generator[MicrodataRow, None, None]:
    assert len(intervals) == len(convertors)
    assert len(intervals) == len(null_mappings)
    while True:
        yield [_generate(i, c, nm) for i, c, nm in zip(intervals, convertors, null_mappings)]


def get_null_mapping(interval: Interval) -> float:
    if interval.max > 0:
        return 2 * interval.max
    elif interval.min < 0:
        return 2 * interval.min
    else:
        return 1.0


def _get_convertor(df: pd.DataFrame, column: str) -> DataConvertor:
    dtype = df.dtypes[column]
    if is_integer_dtype(dtype):
        return IntegerConvertor()
    elif is_float_dtype(dtype):
        return RealConvertor()
    elif is_bool_dtype(dtype):
        return BooleanConvertor()
    elif is_datetime64_dtype(dtype):
        return TimestampConvertor()
    elif is_string_dtype(dtype):
        # Note above is `True` for `object` dtype, but `StringConvertor` will assert values are `str`.
        return StringConvertor(df[column])
    else:
        raise TypeError(f"Dtype {dtype} is not supported.")


def get_convertors(df: pd.DataFrame) -> list[DataConvertor]:
    return [_get_convertor(df, column) for column in df.columns]


def _apply_convertor(value: Value, convertor: DataConvertor) -> float:
    if pd.isna(value):
        return np.NaN
    else:
        return convertor.to_float(value)


def apply_convertors(convertors: list[DataConvertor], df: pd.DataFrame) -> pd.DataFrame:
    for i, column in enumerate(df):
        df[column] = df[column].apply(_apply_convertor, convertor=convertors[i])

    return df


def generate_microdata(
    buckets: list[Bucket], convertors: list[DataConvertor], null_mappings: list[float]
) -> list[MicrodataRow]:
    microdata_rows: list[MicrodataRow] = []
    for bucket in buckets:
        microdata_rows.extend(
            islice(_microdata_row_generator(bucket.intervals, convertors, null_mappings), bucket.count)
        )

    return microdata_rows