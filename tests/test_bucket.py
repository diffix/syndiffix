from syndiffix.bucket import *
from syndiffix.common import *

from .conftest import *


def test_harvest_singularity() -> None:
    forest = create_forest(DataFrame({"data": [1.0] * 12}), anon_params=NOISELESS_PARAMS)
    tree = forest.get_tree((ColumnId(0),))

    assert harvest(tree) == [Bucket((Interval(1.0, 1.0),), 12)]


def test_harvest_range() -> None:
    forest = create_forest(DataFrame({"data": [float(i) for i in range(12)]}), anon_params=NOISELESS_PARAMS)
    tree = forest.get_tree((ColumnId(0),))

    assert harvest(tree) == [
        Bucket((Interval(0.0, 4.0),), 4),
        Bucket((Interval(4.0, 8.0),), 4),
        Bucket((Interval(8.0, 12.0),), 4),
    ]


def test_refining() -> None:
    forest = create_forest(
        DataFrame({"x": [float(i) for i in range(12)] * 5, "y": [1.0, 2.0, 3.0] * 20}),
        anon_params=NOISELESS_PARAMS,
    )
    tree = forest.get_tree((ColumnId(0), ColumnId(1)))
    intervals = [bucket.intervals for bucket in harvest(tree) for _ in range(bucket.count)]

    for i in range(12):
        x = float(i)
        y = float(i % 3 + 1)
        assert intervals.count((Interval(x, x), Interval(y, y))) == 5
