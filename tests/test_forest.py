from typing import Any, Optional

import pytest

from syndiffix.common import *

from .conftest import *


def test_column_type_check() -> None:
    with pytest.raises(AssertionError):
        create_forest(
            # Data needs to be preprocessed with `microdata.apply_convertors` first, so this throws.
            DataFrame({"data": [-2]})
        )


def test_null_mappings() -> None:
    forest = create_forest(
        DataFrame({"data1": [-2.0, 0.0, -1.0, None, np.NaN], "data2": [0.0, 0.0, 6.0, None, np.NaN]})
    )

    assert forest.null_mappings == (-4.0, 12.0)
    assert forest.data[:, 0].tolist() == [-2.0, 0.0, -1.0, -4.0, -4.0]
    assert forest.data[:, 1].tolist() == [0.0, 0.0, 6.0, 12.0, 12.0]


def test_null_mappings_all_nan_column() -> None:
    forest = create_forest(DataFrame({"data": [np.NaN, np.NaN]}))

    assert forest.null_mappings == (1.0,)
    assert forest.data[:, 0].tolist() == [1.0, 1.0]


def test_aid_hashing() -> None:
    forest = create_forest(
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
    forest = create_forest(DataFrame(data, columns=["col1", "col2"]))

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
    forest = create_forest(DataFrame(data, columns=["col1"]), anon_params=NOISELESS_PARAMS)
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
    forest = create_forest(DataFrame(data, columns=["col1"]), anon_params=NOISELESS_PARAMS)
    tree = forest.get_tree((ColumnId(0),))

    assert tree.noisy_count() == 9


def test_hashing_of_column_names() -> None:
    data = [
        [0.0, 0.0],
    ]
    forest = create_forest(DataFrame(data, columns=["col1", "col2"]))
    tree1 = forest.get_tree((ColumnId(0),))
    tree2 = forest.get_tree((ColumnId(1),))

    assert tree1.context.anonymization_context.bucket_seed != tree2.context.anonymization_context.bucket_seed


def test_depth_limiting() -> None:
    data = [[float(i)] for i in range(10)]

    forest = create_forest(DataFrame(data, columns=["col"]))
    tree = forest.get_tree((ColumnId(0),))
    assert isinstance(tree, Branch)

    zero_depth_params = BucketizationParams(precision_limit_depth_threshold=0, precision_limit_row_fraction=1)
    forest = create_forest(DataFrame(data, columns=["col"]), bucketization_params=zero_depth_params)
    tree = forest.get_tree((ColumnId(0),))
    assert isinstance(tree, Leaf)


def _node_to_dict(node: Node) -> dict[str, Any]:
    count = node.noisy_count()
    children: Optional[dict[str, Any]] = None

    if isinstance(node, Branch):
        children = {str(key): _node_to_dict(child) for key, child in sorted(node.children.items())}

    return {
        "ranges": [[interval.min, interval.max] for interval in node.snapped_intervals],
        "count": count,
        "children": children,
    }


def test_tree_snapshot() -> None:
    forest = load_forest(
        "tree.csv",
        anon_params=NOISELESS_PARAMS,
    )

    def get_tree(*comb: int) -> dict[str, Any]:
        return _node_to_dict(forest.get_tree(tuple(ColumnId(c) for c in comb)))

    assert get_tree(0) == load_json("tree.0.json")
    assert get_tree(1) == load_json("tree.1.json")
    assert get_tree(2) == load_json("tree.2.json")
    assert get_tree(0, 1) == load_json("tree.0_1.json")
    assert get_tree(0, 1, 2) == load_json("tree.0_1_2.json")
