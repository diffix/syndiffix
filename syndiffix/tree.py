from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NewType, cast

import numpy as np
import numpy.typing as npt

from .common import *
from .counters import CountersFactory
from .interval import Interval, Intervals

RowId = NewType("RowId", int)


@dataclass(frozen=True)
class Context:
    combination: Combination
    aid_data: npt.NDArray[np.int64]
    data: npt.NDArray[np.float64]
    anonymization_context: AnonymizationContext
    bucketization_params: BucketizationParams
    row_limit: int
    counters_factory: CountersFactory

    def get_aids(self, row: RowId) -> Hashes:
        return tuple(self.aid_data[row])

    def get_values(self, row: RowId) -> tuple[float, ...]:
        return get_items_combination(self.combination, self.data[row])


Subnodes = tuple["Node" | None, ...]


class Node(ABC):
    def __init__(
        self, context: Context, subnodes: Subnodes, snapped_intervals: Intervals, actual_intervals: Intervals
    ) -> None:
        self.context = context
        self.subnodes = subnodes
        self.snapped_intervals = snapped_intervals
        self.actual_intervals = actual_intervals
        # 0-dim subnodes of 1-dim nodes are not considered stubs.
        self.is_stub = len(subnodes) > 0 and all(subnode is None or subnode.is_stub_subnode() for subnode in subnodes)
        self.stub_subnode_cache: bool | None = None
        self.noisy_count = 0
        self.entity_counter = context.counters_factory.create_entity_counter()

    def update_aids(self, row: RowId) -> None:
        self.entity_counter.add(self.context.get_aids(row))

        # We need to recompute cached values each time new contributions arrive.
        self.stub_subnode_cache = None
        self.noisy_count = 0

    def update_actual_intervals(self, row: RowId) -> None:
        for interval, value in zip(self.actual_intervals, self.context.get_values(row)):
            interval.expand(value)

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
        elif self.stub_subnode_cache is not None:
            return self.stub_subnode_cache  # Return previously cached result.
        else:
            stub_low_threshold = (
                self.context.bucketization_params.singularity_low_threshold
                if self.is_singularity()
                else self.context.bucketization_params.range_low_threshold
            )
            is_stub_subnode = not self.is_over_threshold(stub_low_threshold)
            self.stub_subnode_cache = is_stub_subnode  # Cache result for future tests.
            return is_stub_subnode

    @abstractmethod
    def add_row(self, depth: int, row: RowId) -> Node:
        pass


class Leaf(Node):
    def __init__(self, context: Context, subnodes: Subnodes, snapped_intervals: Intervals, initial_row: RowId):
        actual_intervals = tuple(Interval(value, value) for value in self.context.get_values(initial_row))
        super().__init__(context, subnodes, snapped_intervals, actual_intervals)
        self.rows = [initial_row]

        self.update_aids(initial_row)
        self.update_actual_intervals(initial_row)

    def _should_split(self, depth: int) -> bool:
        low_threshold = self.context.anonymization_context.anonymization_params.low_count_params.low_threshold
        depth_threshold = self.context.bucketization_params.precision_limit_depth_threshold

        return (
            # `RowLimit` is 0 for nodes above 1dim, no need to check dimensions.
            (depth <= depth_threshold or len(self.rows) >= self.context.row_limit)
            and not self.is_stub
            and not self.is_singularity()
            and self.is_over_threshold(low_threshold)
        )

    def add_row(self, depth: int, row: RowId) -> Node:
        self.update_aids(row)
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


class Branch(Node):
    def __init__(self, leaf: Leaf):
        super().__init__(leaf.context, leaf.subnodes, leaf.snapped_intervals, leaf.actual_intervals)
        self.children: dict[int, Node] = dict()

    # Each dimension corresponds to a bit in index, with 0 for the lower interval half and
    # 1 for the the upper interval half.
    def _find_child_index(self, values: tuple[float, ...]) -> int:
        child_index = 0
        for value, interval in zip(values, self.snapped_intervals):
            child_index = (child_index << 1) | interval.half_index(value)
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
        dimensions = len(self.snapped_intervals)
        snapped_intervals = tuple(
            interval.half((child_index >> (dimensions - dim_index)) & 1)
            for dim_index, interval in enumerate(self.snapped_intervals, 1)
        )

        # Set child's subnodes to the matching-interval children of the parent's subnodes.
        subnodes = tuple(
            cast(Branch, subnode).children.get(Branch._remove_dimension_from_index(dim_index, child_index))
            if subnode is Branch
            else None
            for dim_index, subnode in enumerate(self.subnodes)
        )

        return Leaf(self.context, subnodes, snapped_intervals, initial_row)

    def add_row(self, depth: int, row: RowId) -> Node:
        child_index = self._find_child_index(self.context.get_values(row))
        child = self.children.get(child_index)
        self.children[child_index] = (
            self._create_child_leaf(child_index, row) if child is None else child.add_row(depth + 1, row)
        )
        self.update_aids(row)
        self.update_actual_intervals(row)
        return self
