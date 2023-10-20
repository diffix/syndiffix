import syndiffix.clustering.solver as solver
from syndiffix.clustering.common import Clusters, ColumnId, StitchOwner


def test_simplify_clusters() -> None:
    Col = ColumnId

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
