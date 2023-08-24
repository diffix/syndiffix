import pandas as pd

from syndiffix.bucket import Bucket
from syndiffix.interval import Interval
from syndiffix.microdata import *


def test_casts_to_float() -> None:
    data = pd.DataFrame({"r": [5.0], "i": [5], "b": [True], "s": ["text"], "t": [pd.Timestamp("1800-01-02")]})
    result = list(
        map(
            lambda column, convertor: convertor.to_float(data.at[0, column]),
            data,
            [RealConvertor(), IntegerConvertor(), BooleanConvertor(), StringConvertor(["text"]), TimestampConvertor()],
        )
    )
    assert result == [5.0, 5.0, 1.0, 0.0, 86400.0]


def test_generates_real_microdata() -> None:
    buckets = [
        Bucket([Interval(-1.0, 2.0), Interval(3.0, 3.0)], 3),
        Bucket([Interval(-11.0, 12.0), Interval(13.0, 13.0)], 10),
    ]
    microdata = generate_microdata(buckets, [RealConvertor(), RealConvertor()], [1234.0, 1234.0])

    assert len(microdata) == 13

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


def test_generates_bool_microdata() -> None:
    buckets = [
        Bucket([Interval(0.0, 0.0), Interval(1.0, 1.0)], 3),
    ]
    microdata = generate_microdata(buckets, [BooleanConvertor(), BooleanConvertor()], [1234.0, 1234.0])
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (bool, float)
        assert row[0][MICRODATA_SYN_VALUE] is False and row[1][MICRODATA_SYN_VALUE] is True


def test_generates_int_microdata() -> None:
    buckets = [
        Bucket([Interval(1.1, 1.6), Interval(3.0, 3.0)], 3),
    ]
    microdata = generate_microdata(buckets, [IntegerConvertor(), IntegerConvertor()], [1234.0, 1234.0])
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (int, float)
        assert row[0][MICRODATA_SYN_VALUE] == 1 and row[1][MICRODATA_SYN_VALUE] == 3


def test_generates_timestamp_microdata() -> None:
    buckets = [
        Bucket([Interval(0.0, 1.5432), Interval(3.0, 3.0)], 3),
    ]
    microdata = generate_microdata(buckets, [TimestampConvertor(), TimestampConvertor()], [1234.0, 1234.0])
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (pd.Timestamp, float)

        assert row[0][MICRODATA_SYN_VALUE] <= pd.Timestamp("1800-01-01 00:00:01") and row[1][  # type: ignore
            MICRODATA_SYN_VALUE
        ] == pd.Timestamp("1800-01-01 00:00:03")


def test_generates_string_microdata() -> None:
    buckets = [
        Bucket([Interval(0.0, 1.0), Interval(2.0, 2.0), Interval(3.0, 3.0)], 3),
    ]
    convertor = StringConvertor(["a", "b", "c", "d"])
    microdata = generate_microdata(buckets, [convertor] * 3, [1234.0] * 3)
    for row in microdata:
        assert len(row) == 3
        for value in row:
            assert tuple(map(type, value)) == (str, float)
        assert (
            row[0][MICRODATA_SYN_VALUE] == "*"
            and row[1][MICRODATA_SYN_VALUE] == "c"
            and row[2][MICRODATA_SYN_VALUE] == "d"
        )


def test_generates_nulls() -> None:
    buckets = [
        Bucket([Interval(1.2, 1.2)], 3),
    ]
    microdata = generate_microdata(buckets, [IntegerConvertor()], [1.2])
    for row in microdata[:3]:
        assert len(row) == 1
        assert row[0] == (None, 1.2)


def test_empty_bucket_list() -> None:
    assert generate_microdata([], [], []) == []


def test_empty_interval_list() -> None:
    assert generate_microdata([Bucket([], 2)], [], []) == [[], []]


def test_prepares_null_mappings() -> None:
    assert get_null_mapping(Interval(1.2, 1.3)) == 2.6
    assert get_null_mapping(Interval(1.2, 1.2)) == 2.4
    assert get_null_mapping(Interval(-1.0, 3.0)) == 6.0
    assert get_null_mapping(Interval(-4.0, -2.5)) == -8.0
    assert get_null_mapping(Interval(-4.0, -0.0)) == -8.0
    assert get_null_mapping(Interval(0.0, 0.0)) == 1.0
