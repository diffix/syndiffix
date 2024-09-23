from __future__ import annotations

import math
import random

import numpy as np
import numpy.typing as npt
from pandas import DataFrame
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

from .anonymizer import hash_pid, noisy_row_limit
from .common import *
from .interval import Interval, get_null_mapping, snap_interval
from .tree import *


def _dimension_interval(data: DataFrame, dimension: int) -> Interval:
    min = data.iloc[:, dimension].min()
    max = data.iloc[:, dimension].max()
    if math.isnan(min):
        assert math.isnan(max)
        min = max = 0.0
    return Interval(min, max)


class Forest:
    def __init__(
        self,
        anonymization_params: AnonymizationParams,
        bucketization_params: BucketizationParams,
        counters_factory: CountersFactory,
        pids: DataFrame,
        data: DataFrame,
    ) -> None:
        self.anonymization_params = anonymization_params
        self.bucketization_params = bucketization_params
        self.counters_factory = counters_factory
        self.orig_pids = pids
        self.orig_data = data
        self.unsafe_rng = random.Random(0)

        assert len(pids) == len(data)
        assert len(data) > 0
        assert len(pids.columns) >= 1
        for dtype in pids.dtypes:
            assert is_string_dtype(dtype) or is_integer_dtype(dtype)
        # We assume `microdata.apply_convertors` has been called on the data.
        for dtype in data.dtypes:
            assert is_float_dtype(dtype)

        self.columns = tuple(data.columns)
        self.dimensions = len(self.columns)

        actual_intervals = tuple(_dimension_interval(data, i) for i in range(self.dimensions))
        self.null_mappings = tuple(get_null_mapping(interval) for interval in actual_intervals)
        for interval, null_mapping in zip(actual_intervals, self.null_mappings):
            interval.expand(null_mapping)
        self.snapped_intervals = tuple(snap_interval(interval) for interval in actual_intervals)

        # Hash and store PID values.
        pids = cast(DataFrame, pids.map(hash_pid))  # type: ignore
        self.pid_data: npt.NDArray[np.uint64] = pids.to_numpy(Hash)

        # Arrange data in a numpy array, applying the null mappings to missing values.
        # `DataFrame.fillna` doesn't seem to accept the fill values by index, must build dict.
        null_mappings_dict = dict(zip(data.columns, self.null_mappings))
        self.data: npt.NDArray[np.float64] = data.fillna(null_mappings_dict, axis=0).to_numpy(np.float64)

        self._tree_cache: dict[Combination, Node] = {}

        # We need to flatten uni-dimensional trees, by pushing the root down as long as one of its halves
        # fails LCF, and update the corresponding dimension's interval, in order to anonymize its boundaries.
        snapped_intervals = list(self.snapped_intervals)
        for i in range(self.dimensions):
            combination = (ColumnId(i),)
            tree = self._build_tree(combination).push_down_1dim_root()
            snapped_intervals[i] = tree.snapped_intervals[0]
            self._tree_cache[combination] = tree  # Cache the flattened version of the tree.
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

        # Hash column names into the tree's base bucket seed.
        base_seed = hash_strings(map(str, get_items_combination(combination, self.columns)))
        anonymization_context = AnonymizationContext(base_seed, self.anonymization_params)

        row_limit = (
            noisy_row_limit(
                anonymization_context.anonymization_params.salt,
                anonymization_context.bucket_seed,
                len(self.data),
                self.bucketization_params.precision_limit_row_fraction,
            )
            if len(combination) == 1
            else 0  # Only 1dim trees are depth limited.
        )

        context = Context(
            combination,
            self.pid_data,
            self.data,
            anonymization_context,
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
        tree = self._tree_cache.get(combination)
        if tree is None:
            tree = self._build_tree(combination)
            self._tree_cache[combination] = tree
        return tree

    def derive_unsafe_rng(self) -> random.Random:
        return random.Random(self.unsafe_rng.random())
