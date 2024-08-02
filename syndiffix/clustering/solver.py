import math
from dataclasses import dataclass

from ..common import *
from .common import *


@dataclass
class MutableCluster:
    columns: set[ColumnId]
    total_entropy: float


DERIVED_COLS_MIN = 1
DERIVED_COLS_RESERVED = 0.5
DERIVED_COLS_RATIO = 0.7


def _col_weight(entropy: float) -> float:
    return 1.0 + math.sqrt(max(entropy, 1.0))


def _floor_by(value: float, bin_size: float) -> float:
    return math.floor(value / bin_size) * bin_size


def _build_clusters(
    context: ClusteringContext,
    max_weight: float,
    merge_thresh: float,
    col_weights: list[float],
    permutation: list[ColumnId],
) -> Clusters:
    dependency_matrix = context.dependency_matrix

    clusters: list[MutableCluster] = []
    if context.main_column is not None:
        clusters = [MutableCluster(columns={context.main_column}, total_entropy=col_weights[context.main_column])]
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
            average_quality = sum(dependency_matrix[col, c] for c in cluster.columns) / len(cluster.columns)

            # Skip if below threshold or above weight limit.
            if average_quality < merge_thresh or (
                len(cluster.columns) > DERIVED_COLS_MIN and cluster.total_entropy + col_weights[col] > capacity
            ):
                continue

            if average_quality > best_avg_quality:
                # Found new best cluster.
                best_cluster = cluster
                best_avg_quality = average_quality

        if best_cluster is not None:
            best_cluster.columns.add(col)
            best_cluster.total_entropy += col_weights[col]
        else:
            clusters.append(MutableCluster(columns={col}, total_entropy=col_weights[col]))

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
                sum(dependency_matrix[c_left, c_right] for c_right in derived_columns) / len(derived_columns),
                max(dependency_matrix[c_left, c_right] for c_right in derived_columns),
            )
            for c_left in available_columns
        ]

        best_stitch_columns.sort(
            key=lambda x: (_floor_by(x[1], 0.05), _floor_by(x[2], 0.01), context.total_dependence_per_column[x[0]]),
            reverse=True,
        )

        best_stitch_col = context.main_column if context.main_column is not None else best_stitch_columns[0][0]
        stitch_columns.add(best_stitch_col)
        total_weight += col_weights[best_stitch_col]

        for c_left, _dep_avg, dep_max in best_stitch_columns:
            if c_left != best_stitch_col and dep_max >= merge_thresh:
                weight = col_weights[c_left]
                if total_weight + weight <= max_weight:
                    stitch_columns.add(c_left)
                    total_weight += weight

        available_columns.update(derived_columns)
        derived_clusters.append((StitchOwner.SHARED, list(stitch_columns), derived_columns))

    initial_cluster = list(clusters[0].columns)

    return Clusters(initial_cluster=initial_cluster, derived_clusters=derived_clusters)


def _clustering_quality(context: ClusteringContext, clusters: Clusters) -> float:
    dependency_matrix = context.dependency_matrix

    unsatisfied_dependence = context.total_dependence

    def visit_pairs(columns: list[ColumnId]) -> None:
        nonlocal unsatisfied_dependence

        for i in range(1, len(columns)):
            col_a = columns[i]

            for j in range(i):
                col_b = columns[j]
                unsatisfied_dependence -= dependency_matrix[col_a, col_b]
                unsatisfied_dependence -= dependency_matrix[col_b, col_a]

    visit_pairs(clusters.initial_cluster)

    for _, stitch_columns, derived_columns in clusters.derived_clusters:
        visit_pairs(stitch_columns + derived_columns)

    return unsatisfied_dependence / (2.0 * context.num_columns)


def _simplify_clusters(clusters: Clusters) -> Clusters:
    if len(clusters.derived_clusters) == 0:
        return clusters

    # If we have clusters in shape initial=[a], derived=[[a],[b,c]]..., we merge to initial=[a,b,c].
    derived_cluster = clusters.derived_clusters[0]
    if set(clusters.initial_cluster) == set(derived_cluster[1]):
        return Clusters(
            initial_cluster=derived_cluster[1] + derived_cluster[2], derived_clusters=clusters.derived_clusters[1:]
        )

    return clusters


def _do_solve(context: ClusteringContext, max_weight: float, merge_thresh: float, alpha: float) -> Clusters:
    num_cols = context.num_columns
    rng = context.rng

    # Constants
    initial_solution = [ColumnId(i) for i in range(num_cols)]

    initial_temperature = 5.0
    min_temperature = 3.5e-3

    def next_temperature(current_temp: float) -> float:
        return current_temp / (1.0 + alpha * current_temp)

    def mutate(solution: list[ColumnId]) -> list[ColumnId]:
        copy = solution.copy()
        i = rng.randint(0, num_cols - 1)
        j = rng.randint(0, num_cols - 1)
        while i == j:
            j = rng.randint(0, num_cols - 1)

        copy[i], copy[j] = solution[j], solution[i]
        return copy

    col_weights = list(_col_weight(col) for col in context.entropy_1dim)

    def evaluate(solution: list[ColumnId]) -> float:
        clusters = _build_clusters(context, max_weight, merge_thresh, col_weights, solution)
        return _clustering_quality(context, clusters)

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

        if energy_delta <= 0.0 or math.exp(-energy_delta / temperature) > rng.random():
            current_solution = new_solution
            current_energy = new_energy

        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy

        temperature = next_temperature(temperature)

    return _simplify_clusters(_build_clusters(context, max_weight, merge_thresh, col_weights, best_solution))


def solve(context: ClusteringContext, max_weight: float, merge_thresh: float, alpha: float) -> Clusters:
    assert max_weight > 1.0
    num_cols = context.num_columns

    if num_cols <= 4:
        # Build a cluster that includes everything.
        return Clusters(initial_cluster=[ColumnId(i) for i in range(num_cols)], derived_clusters=[])

    # TODO: Do an exact search up to a number of columns.
    return _do_solve(context, max_weight, merge_thresh, alpha)


def solve_with_features(
    main_column: ColumnId,
    main_features: list[ColumnId],
    max_weight: float,
    entropy_1dim: Entropy1Dim,
    drop_non_features: bool,
) -> Clusters:
    num_columns = len(entropy_1dim)
    main_column_weight = _col_weight(entropy_1dim[main_column])

    clusters: list[MutableCluster] = []

    def new_cluster() -> MutableCluster:
        cluster = MutableCluster(columns=set(), total_entropy=main_column_weight)
        clusters.append(cluster)
        return cluster

    curr = new_cluster()
    curr.columns.add(main_column)  # Only for the first cluster, in others main is a stitch column.

    for feature in main_features:
        weight = _col_weight(entropy_1dim[feature])

        if len(curr.columns) > 1 and curr.total_entropy + weight > max_weight:
            curr = new_cluster()

        curr.columns.add(feature)
        curr.total_entropy += weight

    initial_cluster = list(clusters[0].columns)
    derived_clusters = [(StitchOwner.SHARED, [main_column], list(cluster.columns)) for cluster in clusters[1:]]

    ml_columns = [main_column] + main_features
    non_feature_columns: list[DerivedCluster] = (
        []
        if drop_non_features
        else [(StitchOwner.LEFT, [main_column], [ColumnId(c)]) for c in range(num_columns) if c not in ml_columns]
    )

    return Clusters(initial_cluster=initial_cluster, derived_clusters=derived_clusters + non_feature_columns)
