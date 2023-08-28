from syndiffix.common import *
from syndiffix.forest import *
from syndiffix.counters import UniqueAidCountersFactory


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
