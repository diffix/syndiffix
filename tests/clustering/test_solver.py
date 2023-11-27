import numpy as np

import syndiffix.clustering.solver as solver
from syndiffix.clustering.common import Clusters, ColumnId, StitchOwner

Col = ColumnId


def test_simplify_clusters() -> None:
    clusters = Clusters(
        initial_cluster=[Col(1)],
        derived_clusters=[
            (StitchOwner.SHARED, [Col(1)], [Col(2), Col(3)]),
            (StitchOwner.SHARED, [Col(1), Col(2)], [Col(4), Col(5)]),
        ],
    )

    simplified = solver._simplify_clusters(clusters)

    assert simplified.initial_cluster == [Col(1), Col(2), Col(3)]
    assert simplified.derived_clusters == [(StitchOwner.SHARED, [Col(1), Col(2)], [Col(4), Col(5)])]


def test_solve_with_features() -> None:
    ncols = 8  # 0=target; 1,2,3,4,5=features; 6,7=patches

    clusters_patches = solver.solve_with_features(
        main_column=Col(0),
        main_features=[Col(1), Col(2), Col(3), Col(4), Col(5)],
        max_weight=6.0,
        entropy_1dim=np.array([1.0 for _ in range(ncols)]),
        drop_non_features=False,
    )

    assert clusters_patches.initial_cluster == [Col(0), Col(1), Col(2)]
    assert clusters_patches.derived_clusters == [
        (StitchOwner.SHARED, [Col(0)], [Col(3), Col(4)]),
        (StitchOwner.SHARED, [Col(0)], [Col(5)]),
        (StitchOwner.SHARED, [], [Col(6)]),
        (StitchOwner.SHARED, [], [Col(7)]),
    ]

    clusters_no_patches = solver.solve_with_features(
        main_column=Col(0),
        main_features=[Col(1), Col(2), Col(3), Col(4), Col(5)],
        max_weight=6.0,
        entropy_1dim=np.array([1.0 for _ in range(ncols)]),
        drop_non_features=True,
    )

    assert clusters_no_patches.initial_cluster == [Col(0), Col(1), Col(2)]
    assert clusters_no_patches.derived_clusters == [
        (StitchOwner.SHARED, [Col(0)], [Col(3), Col(4)]),
        (StitchOwner.SHARED, [Col(0)], [Col(5)]),
    ]
