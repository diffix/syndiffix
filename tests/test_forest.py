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
        ),


def test_null_mappings() -> None:
    forest = Forest(
        AnonymizationContext(Hash(0), AnonymizationParams()),
        BucketizationParams(),
        (Column("data", ColumnType.INTEGER),),
        UniqueAidCountersFactory(),
        DataFrame({"aid": ["a", None, 1, 2, 3]}),
        DataFrame({"data": [-2.0, 0.0, 6.0, None, np.NaN]}),
    )

    assert forest.null_mappings == (-4.0,)
    assert forest.data[:, 0].tolist() == [-2.0, 0.0, 6.0, -4.0, -4.0]


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
