from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Interval:
    # Notice: this won't work for integer values bigger than 2^53 because of floating point limitations.
    min: float
    max: float

    def __post_init__(self) -> None:
        assert self.min <= self.max

    def size(self) -> float:
        return self.max - self.min

    def is_singularity(self) -> bool:
        return self.min == self.max

    def middle(self) -> float:
        return self.min if self.is_singularity() else (self.min + self.max) / 2.0

    # Returns 0 if value is in interval's lower half, 1 if value is in interval's upper half.
    def half_index(self, value: float) -> int:
        return 0 if self.is_singularity() or value < self.middle() else 1

    def lower_half(self) -> Interval:
        return Interval(self.min, self.middle())

    def upper_half(self) -> Interval:
        return Interval(self.middle(), self.max)

    def half(self, index: int) -> Interval:
        match index:
            case 0:
                return self.lower_half()
            case 1:
                return self.upper_half()
            case _:
                raise ValueError("Invalid index for interval half!")

    def contains_interval(self, interval: Interval) -> bool:
        return interval.min >= self.min and interval.max <= self.max

    def contains_value(self, value: float) -> bool:
        return value == self.min or (value > self.min and value < self.max)

    def overlaps(self, interval: Interval) -> bool:
        return interval.min < self.max and interval.max > self.min

    def expand(self, value: float) -> None:
        if value > self.max:
            self.max = value
        elif value < self.min:
            self.min = value

    def copy(self) -> Interval:
        return Interval(self.min, self.max)


Intervals = tuple[Interval, ...]


def _next_pow2(x: float) -> float:
    assert x > 0.0
    return 2.0 ** math.ceil(math.log2(x))


def _floor_by(value: float, amount: float) -> float:
    return math.floor(value / amount) * amount


def snap_interval(interval: Interval) -> Interval:
    snapped_size = _next_pow2(interval.size()) if interval.size() > 0.0 else 1.0
    aligned_min = _floor_by(interval.min, snapped_size / 2.0)
    if aligned_min + snapped_size < interval.max:
        # This snapped interval doesn't fit, so we need to increase it.
        return snap_interval(Interval(aligned_min, interval.max))
    else:
        return Interval(aligned_min, aligned_min + snapped_size)


def get_null_mapping(interval: Interval) -> float:
    if interval.max > 0:
        return 2 * interval.max
    elif interval.min < 0:
        return 2 * interval.min
    else:
        return 1.0
