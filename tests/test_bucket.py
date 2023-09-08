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


def test_refining_1() -> None:
    forest = create_forest(
        DataFrame({"data1": [float(i) for i in range(12)], "data2": [1.0] * 12}), anon_params=NOISELESS_PARAMS
    )
    tree = forest.get_tree((ColumnId(0), ColumnId(1)))

    assert harvest(tree) == [
        Bucket((Interval(0.0, 8.0), Interval(1.0, 1.0)), 8),
        Bucket((Interval(8.0, 16.0), Interval(1.0, 1.0)), 4),
    ]


def test_refining_2() -> None:
    forest = create_forest(
        DataFrame({"data1": [float(i) for i in range(12)] * 3, "data2": [1.0, 2.0, 3.0] * 12}),
        anon_params=NOISELESS_PARAMS,
    )
    tree = forest.get_tree((ColumnId(0), ColumnId(1)))

    assert harvest(tree) == [
        Bucket((Interval(0.0, 8.0), Interval(1.0, 1.0)), 9),
        Bucket((Interval(0.0, 8.0), Interval(2.0, 4.0)), 15),
        Bucket((Interval(8.0, 16.0), Interval(2.0, 4.0)), 9),
        Bucket((Interval(9.0, 9.0), Interval(1.0, 1.0)), 3),
    ]
