from dataclasses import dataclass

from .range import Range


@dataclass
class Bucket:
    ranges: list[Range]
    count: int
