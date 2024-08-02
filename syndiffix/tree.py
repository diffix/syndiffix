from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Iterator, NewType, Union

import numpy as np
import numpy.typing as npt

from .anonymizer import hash_strings
from .common import *
from .counters import CountersFactory
from .interval import Interval, Intervals

RowId = NewType("RowId", int)


@dataclass(frozen=True)
class Context:
    combination: Combination
    pid_data: npt.NDArray[Hash]
    data: npt.NDArray[np.float64]
    anonymization_context: AnonymizationContext
    bucketization_params: BucketizationParams
    row_limit: int
    counters_factory: CountersFactory


Subnodes = tuple[Union["Node", None], ...]


class Node(ABC):
    def __init__(
        self, context: Context, subnodes: Subnodes, snapped_intervals: Intervals, actual_intervals: Iterator[Interval]
    ) -> None:
        self.context = context
        self.subnodes = subnodes
        self.snapped_intervals = snapped_intervals
        # These get mutated later, we need to create new Interval instances.
        self.actual_intervals = tuple(interval.copy() for interval in actual_intervals)
        # 0-dim subnodes of 1-dim nodes are not considered stubs.
        self.is_stub = len(subnodes) > 0 and all(subnode is None or subnode.is_stub_subnode() for subnode in subnodes)
        self._stub_subnode_cache: bool | None = None
        self._noisy_count_cache = 0
        self.entity_counter = context.counters_factory.create_entity_counter()

    def dimensions(self) -> int:
        return len(self.context.combination)

    def update_pids(self, row: RowId) -> None:
        self.entity_counter.add(self.context.pid_data[row])

        # We need to recompute cached values each time new contributions arrive.
        self._stub_subnode_cache = None
        self._noisy_count_cache = 0

    def update_actual_intervals(self, row: RowId) -> None:
        values = self.context.data[row]
        for interval, value_index in zip(self.actual_intervals, self.context.combination):
            interval.expand(values[value_index])

    def is_singularity(self) -> bool:
        return all(interval.is_singularity() for interval in self.actual_intervals)

    def is_over_threshold(self, low_threshold: int) -> bool:
        anon_params = self.context.anonymization_context.anonymization_params
        suppression_params = SuppressionParams(
            low_threshold, anon_params.low_count_params.layer_sd, anon_params.low_count_params.low_mean_gap
        )
        return not self.entity_counter.is_low_count(anon_params.salt, suppression_params)

    def is_stub_subnode(self) -> bool:
        if self.is_stub:
            return True
        elif self._stub_subnode_cache is not None:
            return self._stub_subnode_cache  # Return previously cached result.
        else:
            stub_low_threshold = (
                self.context.bucketization_params.singularity_low_threshold
                if self.is_singularity()
                else self.context.bucketization_params.range_low_threshold
            )
            is_stub_subnode = not self.is_over_threshold(stub_low_threshold)
            self._stub_subnode_cache = is_stub_subnode  # Cache result for future tests.
            return is_stub_subnode

    # Helper method for `push_down_1dim_root`.
    # Outliers need special handling. They must:
    #   - be added to existing leaves;
    #   - not change the actual ranges of the nodes.
    @abstractmethod
    def _add_1dim_outlier_row(self, row: RowId) -> None:
        pass

    # Pushes down the root of a 1-dim tree as long as one of the children is low-count.
    @abstractmethod
    def push_down_1dim_root(self) -> Node:
        pass

    @abstractmethod
    def add_row(self, depth: int, row: RowId) -> Node:
        pass

    def bucket_intervals(self) -> Iterator[Interval]:
        for snapped, actual in zip(self.snapped_intervals, self.actual_intervals):
            yield actual if actual.is_singularity() else snapped

    # Returns the noisy count of rows matching the current node.
    def noisy_count(self) -> int:
        if self._noisy_count_cache == 0:
            # Use range midpoints as the labels for seeding the per-bucket noise.
            labels_hash = hash_strings(str(interval.middle()) for interval in self.bucket_intervals())
            bucket_seed = self.context.anonymization_context.bucket_seed ^ labels_hash
            anon_context = replace(self.context.anonymization_context, bucket_seed=bucket_seed)

            row_counter = self.context.counters_factory.create_row_counter()
            for row in self._matching_rows():
                row_counter.add(self.context.pid_data[row])

            min_count = anon_context.anonymization_params.low_count_params.low_threshold
            self._noisy_count_cache = max(row_counter.noisy_count(anon_context), min_count)

        return self._noisy_count_cache

    # Helper method for `noisy_count`. Iterates over all of the rows matching the current node.
    @abstractmethod
    def _matching_rows(self) -> Iterator[RowId]:
        pass


class Leaf(Node):
    def __init__(self, context: Context, subnodes: Subnodes, snapped_intervals: Intervals, initial_row: RowId):
        initial_values = get_items_combination(context.combination, context.data[initial_row])
        actual_intervals = (Interval(value, value) for value in initial_values)
        super().__init__(context, subnodes, snapped_intervals, actual_intervals)
        self.rows = [initial_row]

        self.update_pids(initial_row)
        self.update_actual_intervals(initial_row)

    def _should_split(self, depth: int) -> bool:
        low_threshold = self.context.anonymization_context.anonymization_params.low_count_params.low_threshold
        depth_threshold = self.context.bucketization_params.precision_limit_depth_threshold

        return (
            # `row_limit` is 0 for trees higher-dimensional than 1, no need to check dimensions here.
            (depth <= depth_threshold or len(self.rows) >= self.context.row_limit)
            and not self.is_stub
            and not self.is_singularity()
            and self.is_over_threshold(low_threshold)
        )

    def add_row(self, depth: int, row: RowId) -> Node:
        self.update_pids(row)
        self.update_actual_intervals(row)
        self.rows.append(row)

        if self._should_split(depth):
            # Convert current leaf node into a new branch node and insert previously gathered rows down the tree.
            branch: Node = Branch(self)
            for row in self.rows:
                branch = branch.add_row(depth, row)
            return branch
        else:
            return self

    def _add_1dim_outlier_row(self, row: RowId) -> None:
        self.update_pids(row)
        self.rows.append(row)

    def push_down_1dim_root(self) -> Node:
        return self

    def _matching_rows(self) -> Iterator[RowId]:
        yield from self.rows


class Branch(Node):
    def __init__(self, leaf: Leaf):
        super().__init__(leaf.context, leaf.subnodes, leaf.snapped_intervals, iter(leaf.actual_intervals))
        self.children: dict[int, Node] = dict()

    # Each dimension corresponds to a bit in index, with 0 for the lower interval half and
    # 1 for the the upper interval half.
    def _find_child_index(self, row: RowId) -> int:
        values = self.context.data[row]
        child_index = 0
        for value_index, interval in zip(self.context.combination, self.snapped_intervals):
            child_index = (child_index << 1) | interval.half_index(values[value_index])
        return child_index

    @staticmethod
    def _remove_dimension_from_index(position: int, index: int) -> int:
        lower_mask = (1 << position) - 1
        upper_mask = ~((1 << (position + 1)) - 1)
        # Remove missing position bit from index.
        return ((index & upper_mask) >> 1) | (index & lower_mask)

    def _create_child_leaf(self, child_index: int, initial_row: RowId) -> Leaf:
        # Create child's intervals by halfing parent's intervals, using the corresponding bit
        # in the index to select the correct half.
        dimensions = self.dimensions()
        snapped_intervals = tuple(
            interval.half((child_index >> (dimensions - dim_index)) & 1)
            for dim_index, interval in enumerate(self.snapped_intervals, 1)
        )

        # Set child's subnodes to the matching-interval children of the parent's subnodes.
        subnodes = tuple(
            (
                subnode.children.get(Branch._remove_dimension_from_index(dim_index, child_index))
                if isinstance(subnode, Branch)
                else None
            )
            for dim_index, subnode in enumerate(self.subnodes)
        )

        return Leaf(self.context, subnodes, snapped_intervals, initial_row)

    def add_row(self, depth: int, row: RowId) -> Node:
        child_index = self._find_child_index(row)
        child = self.children.get(child_index)
        self.children[child_index] = (
            self._create_child_leaf(child_index, row) if child is None else child.add_row(depth + 1, row)
        )
        self.update_pids(row)
        self.update_actual_intervals(row)
        return self

    def _add_1dim_outlier_row(self, row: RowId) -> None:
        self.update_pids(row)
        child_index = next(iter(self.children)) if len(self.children) == 1 else self._find_child_index(row)
        self.children[child_index]._add_1dim_outlier_row(row)

    # Returns the low-count rows, if any, from the specified child leaf of the current 1-dim branch.
    def _get_low_count_rows_in_child(self, child_index: int) -> list[RowId] | None:
        child = self.children.get(child_index)
        if child is None:
            return []
        elif isinstance(child, Leaf):
            low_threshold = self.context.anonymization_context.anonymization_params.low_count_params.low_threshold
            return None if child.is_over_threshold(low_threshold) else child.rows
        else:
            return None

    def push_down_1dim_root(self) -> Node:
        left_lc_rows = self._get_low_count_rows_in_child(0)
        right_lc_rows = self._get_low_count_rows_in_child(1)
        if left_lc_rows is None and right_lc_rows is not None:
            new_root = self.children[0].push_down_1dim_root()
            for row in right_lc_rows:
                new_root._add_1dim_outlier_row(row)
            return new_root
        elif left_lc_rows is not None and right_lc_rows is None:
            new_root = self.children[1].push_down_1dim_root()
            for row in left_lc_rows:
                new_root._add_1dim_outlier_row(row)
            return new_root
        else:
            return self

    def _matching_rows(self) -> Iterator[RowId]:
        for child in self.children.values():
            yield from child._matching_rows()
