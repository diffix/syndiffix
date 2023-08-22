from dataclasses import dataclass

from .interval import Interval


@dataclass
class Bucket:
    intervals: list[Interval]
    count: int
