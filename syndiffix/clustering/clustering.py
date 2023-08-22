import math
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..common import *
from .common import *


@dataclass
class ClusteringContext:
    dependency_matrix: list[list[float]]
    entropy_1_dim: list[float]
    total_dependence_per_column: list[float]
    anonymization_params: AnonymizationParams
    bucketization_params: BucketizationParams
    random: Random
    main_column: Optional[ColumnId]

    @property
    def num_columns(self) -> int:
        return len(self.dependency_matrix)


@dataclass
class MutableCluster:
    columns: set[ColumnId]
    total_entropy: float


DERIVED_COLS_MIN = 1
DERIVED_COLS_RESERVED = 0.5
DERIVED_COLS_RATIO = 0.7


def col_weight(entropy: float) -> float:
    return 1.0 + math.sqrt(max(entropy, 1.0))


def floor_by(n: float, x: float) -> float:
    return math.floor(x / n) * n


def build_clusters(context: ClusteringContext, permutation: list[ColumnId]) -> Clusters:
    merge_thresh = context.bucketization_params.clustering_merge_threshold
    max_weight = context.bucketization_params.clustering_max_cluster_weight

    dependency_matrix = context.dependency_matrix
    entropy_1_dim = context.entropy_1_dim

    def col_weight_by_id(col: ColumnId) -> float:
        return col_weight(entropy_1_dim[col])

    clusters: list[MutableCluster] = []

    if context.main_column is not None:
        clusters = [MutableCluster(columns={context.main_column}, total_entropy=col_weight_by_id(context.main_column))]
        permutation = [col for col in permutation if col != context.main_column]

    # For each column in the permutation, we find the "best" cluster that has available space.
    # We judge how suitable a cluster is by average dependence score.
    # If no available cluster is found, we start a new one.
    # After all clusters are built, we fill remaining space of non-initial clusters with stitch columns.

    for col in permutation:
        best_cluster = None
        best_avg_quality = -1.0

        for i, cluster in enumerate(clusters):
            capacity = max_weight if i == 0 else DERIVED_COLS_RATIO * max_weight
            average_quality = sum(dependency_matrix[col][c] for c in cluster.columns) / len(cluster.columns)

            # Skip if below threshold or above weight limit.
            if average_quality < merge_thresh or (
                len(cluster.columns) > DERIVED_COLS_MIN and cluster.total_entropy + col_weight_by_id(col) > capacity
            ):
                continue

            if average_quality > best_avg_quality:
                # Found new best cluster.
                best_cluster = cluster
                best_avg_quality = average_quality

        if best_cluster is not None:
            best_cluster.columns.add(col)
            best_cluster.total_entropy += col_weight_by_id(col)
        else:
            clusters.append(MutableCluster(columns={col}, total_entropy=col_weight_by_id(col)))

    derived_clusters: list[DerivedCluster] = []
    available_columns = clusters[0].columns.copy()

    for i in range(1, len(clusters)):
        cluster = clusters[i]
        total_weight = max(cluster.total_entropy, DERIVED_COLS_RESERVED * max_weight)

        stitch_columns: set[ColumnId] = set()
        derived_columns = list(cluster.columns)

        best_stitch_columns = [
            (
                c_left,
                sum(dependency_matrix[c_left][c_right] for c_right in derived_columns) / len(derived_columns),
                max(dependency_matrix[c_left][c_right] for c_right in derived_columns),
            )
            for c_left in available_columns
        ]

        best_stitch_columns.sort(
            key=lambda x: (math.floor(x[1] / 0.05), math.floor(x[2] / 0.01), context.total_dependence_per_column[x[0]]),
            reverse=True,
        )

        best_stitch_col = context.main_column if context.main_column is not None else best_stitch_columns[0][0]
        stitch_columns.add(best_stitch_col)
        total_weight += col_weight_by_id(best_stitch_col)

        for c_left, _dep_avg, dep_max in best_stitch_columns:
            if c_left != best_stitch_col and dep_max >= merge_thresh:
                weight = col_weight_by_id(c_left)
                if total_weight + weight <= max_weight:
                    stitch_columns.add(c_left)
                    total_weight += weight

        available_columns.update(derived_columns)
        derived_clusters.append((StitchOwner.SHARED, list(stitch_columns), derived_columns))

    initial_cluster = list(clusters[0].columns)

    return Clusters(initial_cluster=initial_cluster, derived_clusters=derived_clusters)


def clustering_quality(context: ClusteringContext, clusters: Clusters) -> float:
    dependency_matrix = context.dependency_matrix

    unsatisfied_dependencies = context.total_dependence_per_column.copy()

    def visit_pairs(columns: list[ColumnId]) -> None:
        for i in range(1, len(columns)):
            col_a = columns[i]

            for j in range(i):
                col_b = columns[j]
                unsatisfied_dependencies[col_a] -= dependency_matrix[col_a][col_b]
                unsatisfied_dependencies[col_b] -= dependency_matrix[col_b][col_a]

    visit_pairs(clusters.initial_cluster)

    for _, stitch_columns, derived_columns in clusters.derived_clusters:
        visit_pairs(stitch_columns + derived_columns)

    return sum(unsatisfied_dependencies) / (2.0 * len(unsatisfied_dependencies))


def clustering_context(main_column: Optional[ColumnId], forest: Forest) -> ClusteringContext:
    total_per_column = [
        sum(forest.dependency_matrix[i][j] for j in range(forest.dimensions) if i != j)
        for i in range(forest.dimensions)
    ]

    return ClusteringContext(
        dependency_matrix=forest.dependency_matrix,
        entropy_1_dim=forest.entropy_1_dim,
        total_dependence_per_column=total_per_column,
        anonymization_params=forest.anonymization_context.anonymization_params,
        bucketization_params=forest.bucketization_params,
        random=forest.random,
        main_column=main_column,
    )


def do_solve(context: ClusteringContext) -> Clusters:
    num_cols = context.num_columns
    random = context.random

    # Constants
    initial_solution = [ColumnId(i) for i in range(num_cols)]

    initial_temperature = 5.0
    min_temperature = 3.5e-3
    alpha = 1.5e-3

    def next_temperature(current_temp: float) -> float:
        return current_temp / (1.0 + alpha * current_temp)

    def mutate(solution: list[ColumnId]) -> list[ColumnId]:
        copy = solution.copy()
        i = random.randint(0, num_cols - 1)
        j = random.randint(0, num_cols - 1)
        while i == j:
            j = random.randint(0, num_cols - 1)

        copy[i], copy[j] = solution[j], solution[i]
        return copy

    def evaluate(solution: list[ColumnId]) -> float:
        clusters = build_clusters(context, solution)
        return clustering_quality(context, clusters)

    # Solver state
    current_solution = initial_solution
    current_energy = evaluate(initial_solution)
    best_solution = initial_solution
    best_energy = current_energy
    temperature = initial_temperature

    # Simulated annealing loop
    while best_energy > 0 and temperature > min_temperature:
        new_solution = mutate(current_solution)
        new_energy = evaluate(new_solution)
        energy_delta = new_energy - current_energy

        if energy_delta <= 0.0 or math.exp(-energy_delta / temperature) > random.random():
            current_solution = new_solution
            current_energy = new_energy

        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy

        temperature = next_temperature(temperature)

    return build_clusters(context, best_solution)


def solve_with_features(main_column: ColumnId, main_features: list[ColumnId], forest: Forest) -> Clusters:
    num_columns = forest.dimensions
    entropy_1_dim = entropy_1_dim = forest.entropy_1_dim
    main_column_weight = col_weight(entropy_1_dim[main_column])
    max_weight = forest.bucketization_params.clustering_max_cluster_weight

    clusters: list[MutableCluster] = []

    def new_cluster() -> MutableCluster:
        cluster = MutableCluster(columns=set(), total_entropy=main_column_weight)
        clusters.append(cluster)
        return cluster

    curr = new_cluster()
    curr.columns.add(main_column)  # Only for the first cluster, in others main is a stitch column.

    for feature in main_features:
        weight = col_weight(entropy_1_dim[feature])

        if len(curr.columns) > 1 and curr.total_entropy + weight > max_weight:
            curr = new_cluster()

        curr.columns.add(feature)
        curr.total_entropy += weight

    initial_cluster = list(clusters[0].columns)
    derived_clusters = [(StitchOwner.SHARED, [main_column], list(cluster.columns)) for cluster in clusters[1:]]

    ml_columns = [main_column] + main_features
    patch_columns: list[DerivedCluster] = [
        (StitchOwner.SHARED, [], [ColumnId(c)]) for c in range(num_columns) if c not in ml_columns
    ]

    return Clusters(initial_cluster=initial_cluster, derived_clusters=derived_clusters + patch_columns)
