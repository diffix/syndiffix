import pytest

from syndiffix.common import *
from syndiffix.counters import UniqueAidCountersFactory
from syndiffix.forest import *

from .conftest import *


def _create_forest(
    data_df: DataFrame, aid_df: DataFrame | None = None, anon_params: AnonymizationParams | None = None
) -> Forest:
    aid_df = DataFrame(data_df.index) if aid_df is None else aid_df
    anon_params = anon_params if anon_params is not None else AnonymizationParams()
    return Forest(
        AnonymizationContext(Hash(0), anon_params), BucketizationParams(), UniqueAidCountersFactory(), aid_df, data_df
    )


def test_column_type_check() -> None:
    with pytest.raises(AssertionError):
        _create_forest(
            # Data needs to be preprocessed with `microdata.apply_convertors` first, so this throws.
            DataFrame({"data": [-2]})
        )


def test_null_mappings() -> None:
    forest = _create_forest(
        DataFrame({"data1": [-2.0, 0.0, -1.0, None, np.NaN], "data2": [0.0, 0.0, 6.0, None, np.NaN]})
    )

    assert forest.null_mappings == (-4.0, 12.0)
    assert forest.data[:, 0].tolist() == [-2.0, 0.0, -1.0, -4.0, -4.0]
    assert forest.data[:, 1].tolist() == [0.0, 0.0, 6.0, 12.0, 12.0]


def test_null_mappings_all_nan_column() -> None:
    forest = _create_forest(DataFrame({"data": [np.NaN, np.NaN]}))

    assert forest.null_mappings == (1.0,)
    assert forest.data[:, 0].tolist() == [1.0, 1.0]


def test_aid_hashing() -> None:
    forest = _create_forest(
        DataFrame({"data": [0.0, 0.0, 0.0]}),
        aid_df=DataFrame({"aid": ["a", None, 1]}),
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
    forest = _create_forest(DataFrame(data, columns=["col1", "col2"]))

    assert forest.snapped_intervals == (Interval(0.0, 2.0), Interval(0.0, 32.0))


def test_outliers_are_not_dropped_1() -> None:
    data = [
        [1.0],
        [5.0],
        [2.0],
        [7.0],
        [21.0],
        [4.0],
        [21.0],
        [28.0],
        [19.0],
        [2.0],
        [1.0],
        [13.0],
        [25.0],
        [30.0],
        [6.0],
        [2.0],
        [15.0],
        [24.0],
        [9.0],
        [199.0],
        [0.0],
    ]
    forest = _create_forest(DataFrame(data, columns=["col1"]), anon_params=NOISELESS_PARAMS)
    tree = forest.get_tree((ColumnId(0),))

    assert tree.noisy_count() == 21


def test_outliers_are_not_dropped_2() -> None:
    data = [
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [100.0],
    ]
    forest = _create_forest(DataFrame(data, columns=["col1"]), anon_params=NOISELESS_PARAMS)
    tree = forest.get_tree((ColumnId(0),))

    assert tree.noisy_count() == 9
