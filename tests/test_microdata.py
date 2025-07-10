import string
from io import StringIO
from random import Random

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from syndiffix import Synthesizer
from syndiffix.bucket import Bucket
from syndiffix.interval import Interval
from syndiffix.microdata import *
from syndiffix.microdata import _convert_to_safe_value, _normalize

from .conftest import *

_rng = Random(0)


def _make_safe_values_df() -> pd.DataFrame:
    # Each column in this dataframe has 10 instances each of 30 distinct strings.
    # This ensures that distinct string is safe (passes LCF). However since there
    # are 3^30 possible 3dim combinations, there won't be any singluarity 3dim buckets
    columns = ["a", "b", "c"]
    values = []
    for i, column in enumerate(columns):
        values.append([column + str(x) for x in range(1, 31)])
        values[i] = values[i] * 10
        np.random.shuffle(values[i])
    return pd.DataFrame(
        {
            columns[0]: values[0],
            columns[1]: values[1],
            columns[2]: values[2],
        }
    )


def _tweak_safe_values_df(df: pd.DataFrame, values_to_tweak: list[int] = [29]) -> None:
    # This takes one or more distinct values in each column, and changes every
    # instance to a random value, thus ensuring that some 1dim values will
    # fail LCF, producing non-singularity leafs
    def ran_str10() -> str:
        return "".join(random.choice(string.ascii_letters) for i in range(10))

    for column in df.columns:
        for value_to_tweak in values_to_tweak:
            df[column] = df[column].apply(lambda x: ran_str10() if str(x).endswith(str(value_to_tweak)) else x)


def _get_convertors(df: pd.DataFrame) -> list[DataConvertor]:
    return [get_convertor(df, column) for column in df.columns]


def test_casts_to_float() -> None:
    data = pd.DataFrame({"r": [5.0], "i": [6], "b": [True], "s": ["text"], "t": [pd.Timestamp("1800-01-02")]})
    convertors = _get_convertors(data)

    results = apply_convertors(convertors, data)
    assert results.shape == data.shape
    # Because of normalization, values have been changed
    assert results.values[0, :].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_recognizes_types() -> None:
    data = pd.DataFrame({"r": [5.0], "i": [6], "b": [True], "s": ["text"], "t": [pd.Timestamp("1800-01-02")]})
    convertors = _get_convertors(data)
    assert [c.column_type() for c in convertors] == [
        ColumnType.REAL,
        ColumnType.INTEGER,
        ColumnType.BOOLEAN,
        ColumnType.STRING,
        ColumnType.TIMESTAMP,
    ]


def test_objects_in_dataframe_rejected() -> None:
    data = pd.DataFrame({"o": [object()]})
    with pytest.raises(TypeError):
        _get_convertors(data)


def test_casts_data_from_csv() -> None:
    csv = StringIO(
        """
a,b,c,d,e,f,g,h,i
1,1.5,1e-7,'50',NULL,NULL,NULL,1800-01-02,nan
1,1.5,1e-7,'51',1.5,abc,NULL,NULL,1.5
"""
    )
    data = pd.read_csv(csv, index_col=False, parse_dates=["h"])
    results = apply_convertors(_get_convertors(data), data)
    # Because of normalization, some values have been changed
    expected = pd.DataFrame(
        {
            "a": [0.0, 0.0],
            "b": [0.0, 0.0],
            "c": [0.0, 0.0],
            "d": [0.0, 1.0],
            "e": [np.nan, 0.0],
            "f": [np.nan, 0.0],
            "g": [np.nan, np.nan],
            "h": [0.0, np.nan],
            "i": [np.nan, 0.0],
        }
    )
    assert results.equals(expected)


def test_generates_real_microdata() -> None:
    buckets = [
        Bucket((Interval(-1.0, 2.0), Interval(3.0, 3.0)), 3),
        Bucket((Interval(-11.0, 12.0), Interval(13.0, 13.0)), 10),
    ]
    microdata = generate_microdata(buckets, [RealConvertor([1.23]), RealConvertor([1.23])], [1234.0, 1234.0], _rng)

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
        Bucket((Interval(0.0, 0.0), Interval(1.0, 1.0)), 3),
    ]
    microdata = generate_microdata(buckets, [BooleanConvertor(), BooleanConvertor()], [1234.0, 1234.0], _rng)
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (bool, float)
        assert row[0][MICRODATA_SYN_VALUE] is False and row[1][MICRODATA_SYN_VALUE] is True


def test_generates_int_microdata() -> None:
    buckets = [
        Bucket((Interval(1.1, 1.6), Interval(3.0, 3.0)), 3),
    ]
    microdata = generate_microdata(buckets, [IntegerConvertor(), IntegerConvertor()], [1234.0, 1234.0], _rng)
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (int, float)
        assert row[0][MICRODATA_SYN_VALUE] == 1 and row[1][MICRODATA_SYN_VALUE] == 3


def test_generates_timestamp_microdata() -> None:
    buckets = [
        Bucket((Interval(0.0, 1.5432), Interval(3.0, 3.0)), 3),
    ]
    microdata = generate_microdata(buckets, [TimestampConvertor(), TimestampConvertor()], [1234.0, 1234.0], _rng)
    for row in microdata:
        assert len(row) == 2
        for value in row:
            assert tuple(map(type, value)) == (pd.Timestamp, float)

        assert row[0][MICRODATA_SYN_VALUE] <= pd.Timestamp("1800-01-01 00:00:01") and row[1][  # type: ignore
            MICRODATA_SYN_VALUE
        ] == pd.Timestamp("1800-01-01 00:00:03")


def test_generates_string_microdata() -> None:
    buckets = [
        Bucket((Interval(0.0, 3.0), Interval(4.0, 4.0), Interval(5.0, 5.0)), 3),
    ]
    convertor = StringConvertor(["aa", "ab", "ac", "ad", "c", "d"])
    microdata = generate_microdata(buckets, [convertor] * 3, [1234.0] * 3, _rng)
    for row in microdata:
        assert len(row) == 3
        for value in row:
            assert tuple(map(type, value)) == (str, float)
        assert (
            len(row[0][MICRODATA_SYN_VALUE]) == 3  # type: ignore
            and "a*" in row[0][MICRODATA_SYN_VALUE]  # type: ignore
            and row[0][MICRODATA_SYN_VALUE][2] in ["0", "1", "2", "3"]  # type: ignore
            and row[1][MICRODATA_SYN_VALUE] == "c"
            and row[2][MICRODATA_SYN_VALUE] == "d"
        )


def test_generates_nulls() -> None:
    buckets = [
        Bucket((Interval(1.2, 1.2),), 3),
    ]
    microdata = generate_microdata(buckets, [IntegerConvertor()], [1.2], _rng)
    for row in microdata[:3]:
        assert len(row) == 1
        assert row[0] == (None, 1.2)


def test_empty_bucket_list() -> None:
    assert generate_microdata([], [], [], _rng) == []


def test_empty_interval_list() -> None:
    assert generate_microdata([Bucket((), 2)], [], [], _rng) == [[], []]


def test_safe_values_set_all() -> None:
    data = _make_safe_values_df()
    convertors = _get_convertors(data)
    results = apply_convertors(convertors, data)
    assert results.shape == data.shape
    forest = create_forest(results)
    for col_id, converter in enumerate(convertors):
        converter.analyze_tree(forest.get_tree((ColumnId(col_id),)))
    for col_id, column in enumerate(data.columns):
        assert data[column].nunique() == len(cast(StringConvertor, convertors[col_id]).safe_values)


def test_safe_values_set_most() -> None:
    data = _make_safe_values_df()
    nuniques = [data[col].nunique() for col in data.columns]
    _tweak_safe_values_df(data)
    convertors = _get_convertors(data)
    results = apply_convertors(convertors, data)
    assert results.shape == data.shape
    forest = create_forest(results)
    for col_id, converter in enumerate(convertors):
        converter.analyze_tree(forest.get_tree((ColumnId(col_id),)))
    for col_id, column in enumerate(data.columns):
        assert nuniques[col_id] == len(cast(StringConvertor, convertors[col_id]).safe_values) + 1


def test_safe_values_e2e_all() -> None:
    data = _make_safe_values_df()
    syn_data = Synthesizer(data).sample()
    for column in syn_data:
        assert syn_data[column].apply(lambda x: "*" in str(x)).sum() == 0


def test_safe_values_e2e_some() -> None:
    data = _make_safe_values_df()
    # By tweaking multiple distinct values, we ensure that there will be buckets
    # with no safe values, thus forcing "*" values
    _tweak_safe_values_df(data, [20, 21, 22, 23, 24, 25, 26])
    syn_data = Synthesizer(data).sample()
    for column in syn_data:
        assert syn_data[column].apply(lambda x: "*" in str(x)).sum() != 0


def test_normalize_with_scaler() -> None:
    values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    scaler = MinMaxScaler()
    normalized_values = _normalize(values, scaler)
    expected_values = [0.0, 0.25, np.nan, 0.75, 1.0]
    assert np.allclose(
        [v for v in normalized_values if not np.isnan(v)], [v for v in expected_values if not np.isnan(v)]
    )
    assert np.isnan(normalized_values[2])


def test_normalize_without_scaler() -> None:
    values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    normalized_values = _normalize(values, None)
    assert normalized_values.equals(values)


def test_normalize_all_nan() -> None:
    values = pd.Series([np.nan, np.nan, np.nan])
    scaler = MinMaxScaler()
    normalized_values = _normalize(values, scaler)
    assert len(normalized_values) == len(values)
    assert all(np.isnan(v) for v in normalized_values)


def test_normalize_no_nan() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = MinMaxScaler()
    normalized_values = _normalize(values, scaler)
    expected_values = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(normalized_values, expected_values)


def test_value_safe_columns_invalid_string_name() -> None:
    data = _make_safe_values_df()
    with pytest.raises(Exception):
        Synthesizer(data, value_safe_columns=["invalid_column"])


def test_value_safe_columns_out_of_range_id() -> None:
    data = _make_safe_values_df()
    with pytest.raises(Exception):
        Synthesizer(data, value_safe_columns=[5])


def test_value_safe_columns_setting() -> None:
    # Create a dataframe with 4 columns
    data = pd.DataFrame(
        {
            "col1": ["a1", "a2", "a3"] * 10,
            "col2": ["b1", "b2", "b3"] * 10,
            "col3": ["c1", "c2", "c3"] * 10,
            "col4": ["d1", "d2", "d3"] * 10,
        }
    )

    # Test with string column names
    synthesizer = Synthesizer(data, value_safe_columns=["col1", "col3"])
    convertors = synthesizer.column_convertors

    # Verify that col1 and col3 have value_safe_flag=True, others have value_safe_flag=False
    assert convertors[0].value_safe_flag is True  # col1
    assert convertors[1].value_safe_flag is False  # col2
    assert convertors[2].value_safe_flag is True  # col3
    assert convertors[3].value_safe_flag is False  # col4

    # Test with column IDs
    synthesizer2 = Synthesizer(data, value_safe_columns=[0, 2])
    convertors2 = synthesizer2.column_convertors

    # Verify same result with column IDs
    assert convertors2[0].value_safe_flag is True  # col1 (ID 0)
    assert convertors2[1].value_safe_flag is False  # col2 (ID 1)
    assert convertors2[2].value_safe_flag is True  # col3 (ID 2)
    assert convertors2[3].value_safe_flag is False  # col4 (ID 3)


def test_convert_to_safe_value_exact_match() -> None:
    safe_values = [1.0, 3.0, 5.0, 7.0, 9.0]
    assert _convert_to_safe_value(5.0, safe_values) == 5.0
    assert _convert_to_safe_value(1.0, safe_values) == 1.0
    assert _convert_to_safe_value(9.0, safe_values) == 9.0


def test_convert_to_safe_value_closest_match() -> None:
    safe_values = [1.0, 3.0, 5.0, 7.0, 9.0]
    # Test values closer to left neighbor
    assert _convert_to_safe_value(2.0, safe_values) == 1.0
    assert _convert_to_safe_value(1.9, safe_values) == 1.0
    # Test values closer to right neighbor
    assert _convert_to_safe_value(2.1, safe_values) == 3.0
    assert _convert_to_safe_value(4.0, safe_values) == 3.0
    # Test exact midpoint (should prefer left due to <= condition)
    assert _convert_to_safe_value(2.0, safe_values) == 1.0
    assert _convert_to_safe_value(4.0, safe_values) == 3.0


def test_convert_to_safe_value_edge_cases() -> None:
    safe_values = [1.0, 3.0, 5.0, 7.0, 9.0]
    # Test values below minimum
    assert _convert_to_safe_value(-10.0, safe_values) == 1.0
    assert _convert_to_safe_value(0.5, safe_values) == 1.0
    # Test values above maximum
    assert _convert_to_safe_value(15.0, safe_values) == 9.0
    assert _convert_to_safe_value(9.5, safe_values) == 9.0


def test_convert_to_safe_value_single_element() -> None:
    safe_values = [5.0]
    assert _convert_to_safe_value(1.0, safe_values) == 5.0
    assert _convert_to_safe_value(5.0, safe_values) == 5.0
    assert _convert_to_safe_value(10.0, safe_values) == 5.0


def test_convert_to_safe_value_empty_list() -> None:
    with pytest.raises(ValueError, match="safe_value_set cannot be empty"):
        _convert_to_safe_value(5.0, [])


def test_convert_to_safe_value_negative_values() -> None:
    # Note that we won't use _convert_to_safe_value with negative values in practice but we test anyway
    safe_values = [-5.0, -2.0, 0.0, 3.0, 8.0]
    assert _convert_to_safe_value(-3.5, safe_values) == -5.0
    assert _convert_to_safe_value(-3.4, safe_values) == -2.0
    assert _convert_to_safe_value(-1.0, safe_values) == -2.0
    assert _convert_to_safe_value(1.0, safe_values) == 0.0
    assert _convert_to_safe_value(5.5, safe_values) == 3.0
