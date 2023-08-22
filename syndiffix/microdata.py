from itertools import islice
from random import Random
from typing import Generator, Literal, Tuple

from .bucket import Bucket
from .common import Value
from .range import Range

MICRODATA_SYN_VALUE: Literal[0] = 0
MICRODATA_FLOAT_VALUE: Literal[1] = 1

MicrodataRow = list[tuple[Value, float]]

# This source of randomness isn't sticky, so can only be applied to already anonymized data.
_non_sticky_rng = Random(0)


def _real_convertor_from_range(range: Range) -> Tuple[float, float]:
    value = _non_sticky_rng.uniform(range.min, range.max)
    return (value, value)


def _microdata_generator(ranges: list[Range]) -> Generator[MicrodataRow, None, None]:
    while True:
        yield list(map(_real_convertor_from_range, ranges))


def generate_microdata(buckets: list[Bucket]) -> list[MicrodataRow]:
    microdata_rows: list[MicrodataRow] = []
    for bucket in buckets:
        microdata_rows += list(islice(_microdata_generator(bucket.ranges), bucket.count))

    return microdata_rows
