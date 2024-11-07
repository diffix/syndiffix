import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from ..common import *
from ..forest import Forest
from ..tree import Branch, Leaf, Node
from .common import *


@dataclass(frozen=True)
class Score:
    score: float
    count: float
    node_xy: Optional[Node]
    node_x: Node
    node_y: Node


@dataclass(frozen=True)
class DependenceMeasure:
    columns: tuple[int, int]
    dependence: float
    measure_time: float


@dataclass(frozen=True)
class DependenceMeasures:
    dependency_matrix: npt.NDArray[np.float64]
    entropy_1dim: Entropy1Dim


def measure_entropy(root: Node) -> float:
    num_rows = root.noisy_count()
    entropy = 0.0

    def entropy_walk(node: Node) -> None:
        nonlocal entropy
        count = node.noisy_count()
        if isinstance(node, Leaf):
            entropy = entropy - (count / num_rows * math.log2(count / num_rows))
        elif isinstance(node, Branch):
            for child_node in node.children.values():
                entropy_walk(child_node)

    entropy_walk(root)
    return entropy


def _is_singularity(node: Node) -> bool:
    return node.actual_intervals[0].is_singularity()


def _singularity_value(node: Node) -> float:
    return node.actual_intervals[0].min


def _get_child(node: Optional[Node], child_id: int) -> Optional[Node]:
    if isinstance(node, Branch):
        return node.children.get(child_id)
    else:
        return None


def _find_child(node: Optional[Node], func: Callable[[int, Node], bool]) -> Optional[Node]:
    if isinstance(node, Branch):
        for key, value in node.children.items():
            if func(key, value):
                return value

    return None


def _get_bit(value: int, index: int) -> int:
    return (value >> index) & 1


def measure_dependence(forest: Forest, col_x: ColumnId, col_y: ColumnId) -> DependenceMeasure:
    if col_x >= col_y:
        raise ValueError("Invalid input.")

    num_rows = forest.data.shape[0]
    range_thresh = forest.bucketization_params.range_low_threshold

    scores: list[Score] = []

    def walk(node_xy: Optional[Node], node_x: Node, node_y: Node) -> None:
        count_x = node_x.noisy_count()
        count_y = node_y.noisy_count()

        if count_x < range_thresh or count_y < range_thresh:
            return

        actual_2dim_count = node_xy.noisy_count() if node_xy else 0.0
        expected_2dim_count = count_x * count_y / num_rows

        score = abs(expected_2dim_count - actual_2dim_count) / max(expected_2dim_count, actual_2dim_count)
        scores.append(Score(score=score, count=expected_2dim_count, node_xy=node_xy, node_x=node_x, node_y=node_y))

        # Walk children.
        # Dim 0 (X) is at bit 1.
        # Dim 1 (Y) is at bit 0.
        if _is_singularity(node_x) and _is_singularity(node_y):
            # Stop walk if both 1-dims are singularities.
            return
        elif _is_singularity(node_x):
            x_singularity = _singularity_value(node_x)

            for id_child_y in range(2):
                child_y = _get_child(node_y, id_child_y)
                if child_y is None:
                    continue

                # Find child that matches on dim Y. It can be on either side of dim X.
                child_xy = _find_child(
                    node_xy,
                    lambda key, value: _get_bit(key, 0) == id_child_y
                    and value.snapped_intervals[0].contains_value(x_singularity),
                )
                walk(child_xy, node_x, child_y)
        elif _is_singularity(node_y):
            y_singularity = _singularity_value(node_y)

            for id_child_x in range(2):
                child_x = _get_child(node_x, id_child_x)
                if child_x is None:
                    continue

                # Find child that matches on dim X. It can be on either side of dim Y.
                child_xy = _find_child(
                    node_xy,
                    lambda key, value: _get_bit(key, 1) == id_child_x
                    and value.snapped_intervals[1].contains_value(y_singularity),
                )
                walk(child_xy, child_x, node_y)
        else:
            for id_child_xy in range(4):
                id_child_x = _get_bit(id_child_xy, 1)
                id_child_y = _get_bit(id_child_xy, 0)

                child_x = _get_child(node_x, id_child_x)
                child_y = _get_child(node_y, id_child_y)

                if child_x is None or child_y is None:
                    continue

                child_xy = _get_child(node_xy, id_child_xy)
                walk(child_xy, child_x, child_y)

    root_xy = forest.get_tree((col_x, col_y))
    root_x = forest.get_tree((col_x,))
    root_y = forest.get_tree((col_y,))

    start_time = time.time()

    walk(root_xy, root_x, root_y)

    total_weighted_score, total_count = 0.0, 0.0

    if len(scores) > 1:
        for i in range(1, len(scores)):
            score = scores[i]
            total_weighted_score += score.score * score.count
            total_count += score.count

    elapsed_time = time.time() - start_time

    return DependenceMeasure(
        columns=(col_x, col_y),
        dependence=total_weighted_score / total_count if total_count > 0 else 0.0,
        measure_time=elapsed_time,
    )


def measure_all(forest: Forest) -> DependenceMeasures:
    num_columns = forest.dimensions
    dependency_matrix = np.full((num_columns, num_columns), 1.0, dtype=float)
    entropy_1dim = np.array([measure_entropy(forest.get_tree((ColumnId(i),))) for i in range(num_columns)], dtype=float)

    for comb in generate_combinations(2, num_columns):
        col_x, col_y = comb
        score = measure_dependence(forest, col_x, col_y)

        dependence = score.dependence
        dependency_matrix[col_x, col_y] = dependence
        dependency_matrix[col_y, col_x] = dependence

    return DependenceMeasures(dependency_matrix=dependency_matrix, entropy_1dim=entropy_1dim)
