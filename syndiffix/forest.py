from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

from .common import *
from .interval import Interval, snap_interval
from .tree import *


class Forest:
    def __init__(
        self,
        anonymization_context: AnonymizationContext,
        bucketization_params: BucketizationParams,
        columns: Columns,
        counters_factory: CountersFactory,
        aids: DataFrame,
        data: DataFrame,
    ) -> None:
        self.anonymization_context = anonymization_context
        self.bucketization_params = bucketization_params
        self.columns = columns
        self.counters_factory = counters_factory

        assert len(aids) == len(data)
        assert len(data) > 0
        assert len(aids.columns) >= 1
        for dtype in aids.dtypes:
            assert is_string_dtype(dtype) or is_integer_dtype(dtype)
        for dtype in data.dtypes:
            assert is_float_dtype(dtype)
        # TODO: Hash AID values.
        self.aid_data: npt.NDArray[np.int64] = aids.to_numpy()
        self.data: npt.NDArray[np.float_] = data.to_numpy()
        self.dimensions = len(data.columns)

        self.snapped_intervals = tuple(
            snap_interval(Interval(data.iloc[i].min(), data.iloc[i].max())) for i in range(self.dimensions)
        )
        self.tree_cache: dict[Combination, Node] = {}

        # We need to flatten uni-dimensional trees, by pushing the root down as long as one of its halves
        # fails LCF, and update the corresponding dimension's interval, in order to anonymize its boundaries.
        snapped_intervals = list(self.snapped_intervals)
        for i in range(self.dimensions):
            combination = (ColumnId(i),)
            tree = self._build_tree(combination)
            # TODO: flatten tree.
            snapped_intervals[i] = tree.snapped_intervals[0]
            self.tree_cache[combination] = tree  # Cache the flattened version of the tree.
        self.snapped_intervals = tuple(snapped_intervals)

    def _get_subnodes(self, upper_combination: Combination) -> Subnodes:
        dimensions = len(upper_combination)
        sub_combinations = generate_combinations(dimensions - 1, dimensions)
        return tuple(
            self.get_tree(get_items_combination(sub_combination, upper_combination))
            for sub_combination in sub_combinations
        )

    def _build_tree(self, combination: Combination) -> Node:
        subnodes = self._get_subnodes(combination)

        # TODO: hash column names into bucket seed.
        # TODO: compute noisy row limit.
        row_limit = 10

        context = Context(
            combination,
            self.aid_data,
            self.data,
            self.anonymization_context,
            self.bucketization_params,
            row_limit,
            self.counters_factory,
        )

        root_intervals = get_items_combination(combination, self.snapped_intervals)
        tree: Node = Leaf(context, subnodes, root_intervals, RowId(0))
        for index in range(1, len(self.data)):
            tree = tree.add_row(0, RowId(index))
        return tree

    def get_tree(self, combination: Combination) -> Node:
        tree = self.tree_cache.get(combination)
        if tree is None:
            tree = self._build_tree(combination)
            self.tree_cache[combination] = tree
        return tree
