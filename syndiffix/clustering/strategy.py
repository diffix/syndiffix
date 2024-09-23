from abc import ABC, abstractmethod
from typing import Any, Optional

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
        total_dependence=sum(total_per_column),
        anonymization_params=forest.anonymization_params,
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


class NoClustering(ClusteringStrategy):
    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        # Build one patch cluster per column, don't measure entropy.
        assert forest.dimensions > 0
        patch_clusters = Clusters(
            initial_cluster=[ColumnId(0)],
            derived_clusters=[(StitchOwner.SHARED, [], [ColumnId(i)]) for i in range(1, forest.dimensions)],
        )
        return patch_clusters, np.zeros(forest.dimensions, np.float64)


class SingleClustering(ClusteringStrategy):
    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        # Build and return a cluster that includes everything, don't measure entropy.
        single_cluster = Clusters(initial_cluster=[ColumnId(i) for i in range(forest.dimensions)], derived_clusters=[])
        return single_cluster, np.zeros(forest.dimensions, np.float64)


class DefaultClustering(ClusteringStrategy):
    def __init__(
        self,
        main_column: Optional[ColumnId | str] = None,
        sample_size: int = 1000,
        max_weight: float = 15.0,
        merge_threshold: float = 0.1,
        solver_alpha: float = 1e-2,
    ) -> None:
        self.main_column = main_column
        self.sample_size = sample_size
        self.max_weight = max_weight
        self.merge_threshold = merge_threshold
        self.solver_alpha = solver_alpha

    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        sampled_forest = (
            sampling.sample_forest(forest, self.sample_size)
            if sampling.should_sample(forest, self.sample_size)
            else forest
        )
        main_column = _resolve_column_id(forest, self.main_column) if self.main_column else None
        clustering_context = _clustering_context(main_column=main_column, forest=sampled_forest)
        clusters = solver.solve(clustering_context, self.max_weight, self.merge_threshold, self.solver_alpha)
        return clusters, clustering_context.entropy_1dim


class MlClustering(ClusteringStrategy):
    def __init__(
        self,
        target_column: ColumnId | str,
        drop_non_features: bool = False,
        max_weight: float = 15.0,
        classifier_model: Any | None = None,
        regressor_model: Any | None = None,
    ) -> None:
        # TODO: Accept ML parameters.
        self.target_column = target_column
        self.drop_non_features = drop_non_features
        self.max_weight = max_weight
        self.classifier_model = classifier_model
        self.regressor_model = regressor_model

    def build_clusters(self, forest: Forest) -> tuple[Clusters, Entropy1Dim]:
        # TODO: Support forest sampling for faster ML feature detection?

        ml_features = features.select_features_ml(
            forest.orig_data,
            _resolve_column_name(forest, self.target_column),
            self.classifier_model,
            self.regressor_model,
        )
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
                self.max_weight,
                entropy_1dim,
                self.drop_non_features,
            ),
            entropy_1dim,
        )
