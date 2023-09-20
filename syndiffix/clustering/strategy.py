from abc import ABC, abstractmethod
from typing import Optional

from ..forest import Forest
from . import features, measures, sampling, solver
from .common import *


def _clustering_context(main_column: Optional[ColumnId], forest: Forest) -> ClusteringContext:
    scores = measures.measure_all(forest)

    total_per_column = [
        sum(scores.dependency_matrix[i, j] for j in range(forest.dimensions) if i != j)
        for i in range(forest.dimensions)
    ]

    return ClusteringContext(
        dependency_matrix=scores.dependency_matrix,
        entropy_1dim=scores.entropy_1dim,
        total_dependence_per_column=total_per_column,
        anonymization_params=forest.anonymization_context.anonymization_params,
        bucketization_params=forest.bucketization_params,
        rng=forest.unsafe_rng,
        main_column=main_column,
    )


def _resolve_column_id(forest: Forest, col_name: ColumnId | str) -> ColumnId:
    if not isinstance(col_name, str):
        return col_name

    return ColumnId(forest.orig_data.columns.get_loc(col_name))


def _resolve_column_name(forest: Forest, col_id: ColumnId | str) -> str:
    if isinstance(col_id, str):
        return col_id

    return forest.orig_data.columns[col_id]


class ClusteringStrategy(ABC):
    @abstractmethod
    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        pass


class DefaultClustering(ClusteringStrategy):
    def __init__(self, main_column: Optional[ColumnId | str] = None) -> None:
        self.main_column = main_column

    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        sampled_forest = sampling.sample_forest(forest) if sampling.should_sample(forest) else forest
        main_column = _resolve_column_id(forest, self.main_column) if self.main_column else None
        clustering_context = _clustering_context(main_column=main_column, forest=sampled_forest)
        return solver.solve(clustering_context), clustering_context.entropy_1dim


class MlClustering(ClusteringStrategy):
    def __init__(self, target_column: ColumnId | str, drop_non_features: bool = False) -> None:
        # TODO: Accept ML parameters.
        self.target_column = target_column
        self.drop_non_features = drop_non_features

    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        # TODO: Support forest sampling for faster ML feature detection?

        ml_features = features.select_features_ml(forest.orig_data, _resolve_column_name(forest, self.target_column))
        feature_ids = [_resolve_column_id(forest, feature) for feature in ml_features.k_features]

        entropy_1dim = np.array(
            [
                # Solver does not care about entropy of non-features.
                (measures.measure_entropy(forest.get_tree((ColumnId(i),))) if i in feature_ids else 1.0)
                for i in range(forest.dimensions)
            ],
            dtype=float,
        )

        return (
            solver.solve_with_features(
                _resolve_column_id(forest, self.target_column),
                feature_ids,
                forest,
                entropy_1dim,
            ),
            entropy_1dim,
        )