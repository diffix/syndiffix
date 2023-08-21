from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Range:
    # Notice: this won't work for integer values bigger than 2^53 because of floating point limitations.
    min: float
    max: float

    def size(self) -> float:
        return self.max - self.min

    def is_singularity(self) -> bool:
        return self.min == self.max

    def middle(self) -> float:
        return self.min if self.is_singularity() else (self.min + self.max) / 2.0

    # Returns 0 if value is in range's lower half, 1 if value is in range's upper half.
    def half_index(self, value: float) -> int:
        return 0 if self.is_singularity() or value < self.middle() else 1

    def lower_half(self) -> Range:
        return Range(self.min, self.middle())

    def upper_half(self) -> Range:
        return Range(self.middle(), self.max)

    def half(self, index: int) -> Range:
        match index:
            case 0:
                return self.lower_half()
            case 1:
                return self.upper_half()
            case _:
                raise ValueError("Invalid index for range half!")

    def contains_range(self, range: Range) -> bool:
        return range.min >= self.min and range.max <= self.max

    def contains_value(self, value: float) -> bool:
        return value == self.min or (value > self.min and value < self.max)

    def overlaps(self, range: Range) -> bool:
        return range.min < self.max and range.max > self.min


Ranges = list[Range]


def _next_pow2(x: float) -> float:
    assert x > 0.0
    return 2.0 ** math.ceil(math.log2(x))


def _floor_by(value: float, amount: float) -> float:
    return math.floor(value / amount) * amount


def snap_range(range: Range) -> Range:
    snapped_size = _next_pow2(range.size()) if range.size() > 0.0 else 1.0
    aligned_min = _floor_by(range.min, snapped_size / 2.0)
    if aligned_min + snapped_size < range.max:
        # This snapped range doesn't fit, so we need to increase it.
        return snap_range(Range(aligned_min, range.max))
    else:
        return Range(aligned_min, aligned_min + snapped_size)


def expand_range(range: Range, value: float) -> Range:
    return Range(min(value, range.min), max(value, range.max))
