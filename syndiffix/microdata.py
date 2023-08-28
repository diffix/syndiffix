from abc import ABC, abstractmethod
from itertools import islice
from random import Random
from typing import Generator, Iterable, Literal, Set, cast

import numpy as np
import pandas as pd

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
        assert isinstance(value, np.bool_)
        return 1.0 if value else 0.0

    def from_interval(self, interval: Interval) -> MicrodataValue:
        value = _generate_float(interval) >= 0.5
        return (value, 1.0 if value else 0.0)


class RealConvertor(DataConvertor):
    def column_type(self) -> ColumnType:
        return ColumnType.REAL

    def to_float(self, value: Value) -> float:
        assert isinstance(value, np.floating)
        return value

    def from_interval(self, interval: Interval) -> MicrodataValue:
        value = _generate_float(interval)
        return (value, value)


class IntegerConvertor(DataConvertor):
    def column_type(self) -> ColumnType:
        return ColumnType.INTEGER

    def to_float(self, value: Value) -> float:
        assert isinstance(value, np.integer)
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
        unique_values = set(values)
        unique_values.discard(None)
        for value in unique_values:
            assert isinstance(value, str)
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
            return ("*", round(_generate_float(interval), 0))


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


def generate_microdata(
    buckets: list[Bucket], convertors: list[DataConvertor], null_mappings: list[float]
) -> list[MicrodataRow]:
    microdata_rows: list[MicrodataRow] = []
    for bucket in buckets:
        microdata_rows.extend(
            islice(_microdata_row_generator(bucket.intervals, convertors, null_mappings), bucket.count)
        )

    return microdata_rows
