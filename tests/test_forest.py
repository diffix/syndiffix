import pytest

from syndiffix.common import *
from syndiffix.counters import UniqueAidCountersFactory
from syndiffix.forest import *


def test_column_type_check() -> None:
    with pytest.raises(AssertionError):
        Forest(
            AnonymizationContext(Hash(0), AnonymizationParams()),
            BucketizationParams(),
            (Column("data", ColumnType.INTEGER),),
            UniqueAidCountersFactory(),
            DataFrame({"aid": ["a"]}),
            # Data needs to be preprocessed with `microdata.apply_convertors` first, so this throws.
            DataFrame({"data": [-2]}),
        )


def test_null_mappings() -> None:
    forest = Forest(
        AnonymizationContext(Hash(0), AnonymizationParams()),
        BucketizationParams(),
        (Column("data1", ColumnType.INTEGER), Column("data2", ColumnType.INTEGER)),
        UniqueAidCountersFactory(),
        DataFrame({"aid": ["a", None, 1, 2, 3]}),
        DataFrame({"data1": [-2.0, 0.0, -1.0, None, np.NaN], "data2": [0.0, 0.0, 6.0, None, np.NaN]}),
    )

    assert forest.null_mappings == (-4.0, 12.0)
    assert forest.data[:, 0].tolist() == [-2.0, 0.0, -1.0, -4.0, -4.0]
    assert forest.data[:, 1].tolist() == [0.0, 0.0, 6.0, 12.0, 12.0]


def test_null_mappings_all_nan_column() -> None:
    forest = Forest(
        AnonymizationContext(Hash(0), AnonymizationParams()),
        BucketizationParams(),
        (Column("data", ColumnType.INTEGER),),
        UniqueAidCountersFactory(),
        DataFrame({"aid": ["a", 1]}),
        DataFrame({"data": [np.NaN, np.NaN]}),
    )

    assert forest.null_mappings == (1.0,)
    assert forest.data[:, 0].tolist() == [1.0, 1.0]


def test_aid_hashing() -> None:
    forest = Forest(
        AnonymizationContext(Hash(0), AnonymizationParams()),
        BucketizationParams(),
        (Column("data", ColumnType.INTEGER),),
        UniqueAidCountersFactory(),
        DataFrame({"aid": ["a", None, 1]}),
        DataFrame({"data": [0.0, 0.0, 0.0]}),
    )
    assert forest.aid_data.tolist() == [
        [Hash(3405396810240292928)],
        [Hash(0)],
        [Hash(18232024504446223411)],
    ]


def test_ranges_anonymization() -> None:
    data = [
        [0.0, 1.0],
        [0.0, 5.0],
        [0.0, 2.0],
        [0.0, 7.0],
        [0.0, 21.0],
        [0.0, 14.0],
        [0.0, 21.0],
        [0.0, 28.0],
        [0.0, 19.0],
        [0.0, 2.0],
        [1.0, 1.0],
        [1.0, 13.0],
        [1.0, 25.0],
        [1.0, 30.0],
        [1.0, 6.0],
        [1.0, 2.0],
        [1.0, 15.0],
        [1.0, 24.0],
        [1.0, 9.0],
        [1.0, 100.0],
        [-5.0, 1.0],
    ]
    df = DataFrame(data, columns=["col1", "col2"])
    forest = Forest(
        AnonymizationContext(Hash(0), AnonymizationParams()),
        BucketizationParams(),
        (Column("col1", ColumnType.INTEGER), Column("col2", ColumnType.INTEGER)),
        UniqueAidCountersFactory(),
        DataFrame(df.index),
        df,
    )

    assert forest.snapped_intervals == (Interval(0.0, 2.0), Interval(0.0, 32.0))
