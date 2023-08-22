from syndiffix.bucket import Bucket
from syndiffix.microdata import *
from syndiffix.range import Range


def test_generates_real_microdata() -> None:
    buckets = [Bucket([Range(-1.0, 2.0), Range(3.0, 3.0)], 3), Bucket([Range(-11.0, 12.0), Range(13.0, 13.0)], 10)]
    microdata = generate_microdata(buckets)
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (float, float)
            assert value[MICRODATA_SYN_VALUE] == value[MICRODATA_FLOAT_VALUE]
    for row in microdata[:3]:
        assert row[0][MICRODATA_FLOAT_VALUE] >= -1 and row[0][MICRODATA_FLOAT_VALUE] < 2.0
        assert row[1][MICRODATA_FLOAT_VALUE] == 3.0
    for row in microdata[3:]:
        assert row[0][MICRODATA_FLOAT_VALUE] >= -11 and row[0][MICRODATA_FLOAT_VALUE] < 12.0
        assert row[1][MICRODATA_FLOAT_VALUE] == 13.0


def test_empty_bucket_list() -> None:
    assert generate_microdata([]) == []


def test_empty_range_list() -> None:
    assert generate_microdata([Bucket([], 2)]) == [[], []]
