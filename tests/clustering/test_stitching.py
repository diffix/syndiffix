import numpy as np

from syndiffix.clustering.common import Clusters, MicrodataRow, StitchOwner
from syndiffix.clustering.stitching import StitchingMetadata, build_table
from syndiffix.common import ColumnId

from ..conftest import *


def test_stitching() -> None:
    forest = load_forest("dummy.csv")

    col_a_left = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    col_a_right = [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
    col_b = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
    col_c_left = [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]
    col_c_right = [310, 311, 312, 313, 314, 315, 316, 317, 318, 319]
    col_d = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]

    def build_rows(*cols: list[int]) -> list[MicrodataRow]:
        microdata_cols = [[(val, float(val)) for val in col] for col in cols]
        return [list(row) for row in zip(*microdata_cols)]

    microtables: dict[Combination, list[MicrodataRow]] = {
        (ColumnId(0), ColumnId(1)): build_rows(col_a_left, col_b),
        (ColumnId(0), ColumnId(2)): build_rows(col_a_right, col_c_left),
        (ColumnId(2), ColumnId(3)): build_rows(col_c_right, col_d),
    }

    metadata = StitchingMetadata(
        dimension_is_integral=[True, True, True, True],
        entropy_1dim=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    def materialize_tree(_forest: Forest, columns: list[ColumnId]) -> tuple[list[MicrodataRow], Combination]:
        combination = tuple(sorted(columns))
        return (
            microtables[combination],
            combination,
        )

    clusters = Clusters(
        initial_cluster=[ColumnId(0), ColumnId(1)],
        derived_clusters=[
            (StitchOwner.SHARED, [ColumnId(0)], [ColumnId(2)]),
            (StitchOwner.SHARED, [ColumnId(2)], [ColumnId(3)]),
        ],
    )

    rows, combination = build_table(materialize_tree, forest, metadata, clusters)
    assert combination == (0, 1, 2, 3)
    assert rows == [
        (100, 200, 300, 400),
        (111, 201, 311, 401),
        (102, 202, 302, 402),
        (113, 203, 313, 403),
        (104, 204, 304, 404),
        (115, 205, 315, 405),
        (106, 206, 306, 406),
        (117, 207, 317, 407),
        (108, 208, 308, 408),
        (119, 209, 319, 409),
    ]
