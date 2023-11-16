from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from random import Random
from typing import Any, Callable, Sequence

from ..common import *
from ..interval import Interval
from ..microdata import MICRODATA_FLOAT_VALUE, MicrodataRow
from .common import *


@dataclass
class ColumnLocation:
    source_row: StitchOwner
    column_id: ColumnId
    left_index: ColumnIndex
    right_index: ColumnIndex


@dataclass(frozen=True)
class StitchContext:
    rng: Random
    stitch_owner: StitchOwner
    all_columns: list[ColumnLocation]
    entropy_1dim: Entropy1Dim
    stitch_max_values: list[float]
    stitch_is_integral: list[bool]
    left_stitch_indexes: list[ColumnIndex]
    right_stitch_indexes: list[ColumnIndex]
    result_rows: list[MicrodataRow]

    @property
    def num_stitch_columns(self) -> int:
        return len(self.stitch_is_integral)


@dataclass(frozen=True)
class StitchState:
    depth: int
    stitch_intervals: list[Interval]
    next_sort_column: int
    currently_sorted_by: int
    remaining_sort_attempts: int
    context: StitchContext

    def replace(self, **kwargs: Any) -> StitchState:
        return replace(self, **kwargs)


def _align_length(rng: Random, length: int, microtable: list[MicrodataRow]) -> list[MicrodataRow]:
    curr_length = len(microtable)
    if length == curr_length:
        return microtable
    elif length < curr_length:
        return microtable[:length]
    else:
        microtable_copy = microtable.copy()
        microtable_copy.extend([microtable[rng.randint(0, curr_length - 1)] for _ in range(length - curr_length)])
        return microtable_copy


def _find_indexes(subset: Sequence[ColumnId], superset: Sequence[ColumnId]) -> list[ColumnIndex]:
    assert len(subset) <= len(superset)
    return [ColumnIndex(superset.index(c)) for c in subset]


def _locate_columns(left_combination: Combination, right_combination: Combination) -> list[ColumnLocation]:
    column_locations: dict[ColumnId, ColumnLocation] = defaultdict(
        lambda: ColumnLocation(StitchOwner.LEFT, ColumnId(-1), ColumnIndex(-1), ColumnIndex(-1))
    )

    for column_index, column_id in enumerate(left_combination):
        location = column_locations[column_id]
        location.column_id = column_id
        location.left_index = ColumnIndex(column_index)
        column_locations[column_id] = location

    for column_index, column_id in enumerate(right_combination):
        location = column_locations[column_id]
        location.column_id = column_id
        location.right_index = ColumnIndex(column_index)
        location.source_row = StitchOwner.RIGHT if location.left_index < 0 else StitchOwner.SHARED
        column_locations[column_id] = location

    return sorted(column_locations.values(), key=lambda x: x.column_id)


def _binary_search(rows: list[MicrodataRow], column: ColumnIndex, target: float, start: int, end_exclusive: int) -> int:
    if start >= end_exclusive:
        return -1

    mid = (start + end_exclusive) // 2

    if rows[mid][column][MICRODATA_FLOAT_VALUE] >= target:
        if mid == 0 or rows[mid - 1][column][MICRODATA_FLOAT_VALUE] < target:
            return mid
        else:
            return _binary_search(rows, column, target, start, mid)
    else:
        return _binary_search(rows, column, target, mid + 1, end_exclusive)


DUMMY_VALUE = (0.0, 0.0)


def _merge_row(
    all_columns: list[ColumnLocation], pick_shared_left: bool, left_row: MicrodataRow, right_row: MicrodataRow
) -> MicrodataRow:
    merged_row: MicrodataRow = [DUMMY_VALUE] * len(all_columns)

    for j, col in enumerate(all_columns):
        if col.source_row == StitchOwner.LEFT:
            merged_row[j] = left_row[col.left_index]
        elif col.source_row == StitchOwner.RIGHT:
            merged_row[j] = right_row[col.right_index]
        elif pick_shared_left:
            merged_row[j] = left_row[col.left_index]
        else:
            merged_row[j] = right_row[col.right_index]

    return merged_row


def _row_sort_key(keys: list[ColumnIndex]) -> Callable[[MicrodataRow], Any]:
    if len(keys) == 1:
        key = keys[0]
        return lambda row: row[key][MICRODATA_FLOAT_VALUE]
    else:
        return lambda row: [row[key][MICRODATA_FLOAT_VALUE] for key in keys]


def _merge_microdata(state: StitchState, left_rows: list[MicrodataRow], right_rows: list[MicrodataRow]) -> None:
    if len(left_rows) == 0 or len(right_rows) == 0:
        raise ValueError("Attempted a stitch with no rows.")

    context = state.context
    stitch_owner = context.stitch_owner
    rng = context.rng
    all_columns = context.all_columns

    if stitch_owner == StitchOwner.LEFT:
        num_rows = len(left_rows)
    elif stitch_owner == StitchOwner.RIGHT:
        num_rows = len(right_rows)
    else:
        num_rows = int(round((len(left_rows) + len(right_rows)) / 2.0))

    rng.shuffle(left_rows)
    left_rows = _align_length(rng, num_rows, left_rows)
    left_rows.sort(key=_row_sort_key(context.left_stitch_indexes))

    rng.shuffle(right_rows)
    right_rows = _align_length(rng, num_rows, right_rows)
    right_rows.sort(key=_row_sort_key(context.right_stitch_indexes))

    for i in range(num_rows):
        pick_shared_left = i % 2 == 0 if stitch_owner == StitchOwner.SHARED else stitch_owner == StitchOwner.LEFT
        context.result_rows.append(_merge_row(all_columns, pick_shared_left, left_rows[i], right_rows[i]))


THRESH_REL = 0.7


def _acceptable_distribution(left: int, right: int) -> bool:
    min_val = min(left, right)
    max_val = max(left, right)

    if min_val == 0:
        return False
    else:
        rel_diff = min_val / max_val
        return rel_diff >= THRESH_REL


def _can_split(state: StitchState) -> bool:
    context = state.context
    split_column = state.next_sort_column

    if context.stitch_is_integral[split_column]:
        interval = state.stitch_intervals[split_column]

        if context.stitch_max_values[split_column] == interval.max:
            return interval.size() >= 1.0
        else:
            return interval.size() > 1.0
    else:
        return True


def update_at(lst: list[Any], index: int, new_value: Any) -> list[Any]:
    if index < 0 or index >= len(lst):
        raise IndexError("Index out of range")

    modified_list = lst.copy()
    modified_list[index] = new_value
    return modified_list


def _stitch_rec(state: StitchState, left_rows: list[MicrodataRow], right_rows: list[MicrodataRow]) -> None:
    if state.remaining_sort_attempts == 0 or len(left_rows) == 1 or len(right_rows) == 1:
        _merge_microdata(state, left_rows, right_rows)
    elif _can_split(state):
        context = state.context
        current_sort_column = state.next_sort_column

        left_stitch_index = context.left_stitch_indexes[current_sort_column]
        right_stitch_index = context.right_stitch_indexes[current_sort_column]

        if state.currently_sorted_by != current_sort_column:
            left_rows.sort(key=_row_sort_key([left_stitch_index]))
            right_rows.sort(key=_row_sort_key([right_stitch_index]))

        interval = state.stitch_intervals[current_sort_column]
        interval_middle = interval.middle()

        left_split_point = max(0, _binary_search(left_rows, left_stitch_index, interval_middle, 0, len(left_rows)))
        right_split_point = max(0, _binary_search(right_rows, right_stitch_index, interval_middle, 0, len(right_rows)))

        left_lower = left_rows[:left_split_point]
        right_lower = right_rows[:right_split_point]

        left_upper = left_rows[left_split_point:]
        right_upper = right_rows[right_split_point:]

        if _acceptable_distribution(len(left_lower), len(right_lower)) and _acceptable_distribution(
            len(left_upper), len(right_upper)
        ):
            # Visit lower half.
            _stitch_rec(
                state.replace(
                    depth=state.depth + 1,
                    stitch_intervals=update_at(state.stitch_intervals, current_sort_column, interval.lower_half()),
                    next_sort_column=(current_sort_column + 1) % context.num_stitch_columns,
                    currently_sorted_by=current_sort_column,
                    remaining_sort_attempts=context.num_stitch_columns,
                ),
                left_lower,
                right_lower,
            )

            # Visit upper half.
            _stitch_rec(
                state.replace(
                    depth=state.depth + 1,
                    stitch_intervals=update_at(state.stitch_intervals, current_sort_column, interval.upper_half()),
                    next_sort_column=(current_sort_column + 1) % context.num_stitch_columns,
                    currently_sorted_by=current_sort_column,
                    remaining_sort_attempts=context.num_stitch_columns,
                ),
                left_upper,
                right_upper,
            )
        else:
            next_stitch_intervals = state.stitch_intervals
            if len(left_lower) == 0 and len(right_lower) == 0:
                next_stitch_intervals = update_at(state.stitch_intervals, current_sort_column, interval.upper_half())
            elif len(left_upper) == 0 and len(right_upper) == 0:
                next_stitch_intervals = update_at(state.stitch_intervals, current_sort_column, interval.lower_half())

            # Try next column.
            _stitch_rec(
                state.replace(
                    next_sort_column=(current_sort_column + 1) % context.num_stitch_columns,
                    stitch_intervals=next_stitch_intervals,
                    currently_sorted_by=current_sort_column,
                    remaining_sort_attempts=state.remaining_sort_attempts - 1,
                ),
                left_rows,
                right_rows,
            )
    else:
        # Try next column.
        _stitch_rec(
            state.replace(
                next_sort_column=(state.next_sort_column + 1) % state.context.num_stitch_columns,
                remaining_sort_attempts=state.remaining_sort_attempts - 1,
            ),
            left_rows,
            right_rows,
        )


def _do_stitch(
    forest: Forest,
    metadata: StitchingMetadata,
    left: tuple[list[MicrodataRow], Combination],
    right: tuple[list[MicrodataRow], Combination],
    derived_cluster: DerivedCluster,
) -> tuple[list[MicrodataRow], Combination]:
    (left_rows, left_combination) = left
    (right_rows, right_combination) = right
    (stitch_owner, stitch_columns, derived_columns) = derived_cluster

    if len(left_combination) == 0 or len(stitch_columns) == 0 or len(derived_columns) == 0:
        raise ValueError("Invalid clusters in stitch operation.")

    # Pick lowest entropy column first.
    stitch_columns = sorted(stitch_columns, key=lambda col: (metadata.entropy_1dim[col], col))
    all_columns = _locate_columns(left_combination, right_combination)

    if len(left_rows) == 0 and len(right_rows) == 0:
        return [], tuple(c.column_id for c in all_columns)

    if len(right_rows) == 0:
        raise ValueError(f"Empty sequence in cluster {right_combination}.")

    result_rows: list[MicrodataRow] = []

    root_stitch_intervals = [forest.snapped_intervals[col] for col in stitch_columns]

    stitch_state = StitchState(
        depth=0,
        stitch_intervals=root_stitch_intervals,
        next_sort_column=0,
        currently_sorted_by=-1,
        remaining_sort_attempts=len(stitch_columns),
        context=StitchContext(
            rng=forest.unsafe_rng,
            stitch_owner=stitch_owner,
            all_columns=all_columns,
            entropy_1dim=metadata.entropy_1dim[stitch_columns],
            stitch_max_values=[r.max for r in root_stitch_intervals],
            stitch_is_integral=[metadata.dimension_is_integral[col] for col in stitch_columns],
            left_stitch_indexes=_find_indexes(stitch_columns, left_combination),
            right_stitch_indexes=_find_indexes(stitch_columns, right_combination),
            result_rows=result_rows,
        ),
    )

    _stitch_rec(stitch_state, left_rows, right_rows)

    return result_rows, tuple(c.column_id for c in all_columns)


def _do_patch(
    rng: Random, left: tuple[list[MicrodataRow], Combination], right: tuple[list[MicrodataRow], Combination]
) -> tuple[list[MicrodataRow], Combination]:
    (left_rows, left_combination) = left
    (right_rows, right_combination) = right

    DOESNT_MATTER = True

    all_columns = _locate_columns(left_combination, right_combination)
    num_rows = len(left_rows)

    left_rows = left_rows[:]
    right_rows = right_rows[:]
    rng.shuffle(right_rows)
    right_rows = _align_length(rng, num_rows, right_rows)

    all_rows = [
        _merge_row(all_columns, DOESNT_MATTER, left_row, right_row)
        for left_row, right_row in zip(left_rows, right_rows)
    ]

    all_columns_ids = tuple(c.column_id for c in all_columns)
    return all_rows, all_columns_ids


def _stitch(
    materialize_tree: TreeMaterializer,
    forest: Forest,
    metadata: StitchingMetadata,
    left: tuple[list[MicrodataRow], Combination],
    derived_cluster: DerivedCluster,
) -> tuple[list[MicrodataRow], Combination]:
    (_, stitch_columns, derived_columns) = derived_cluster

    right = materialize_tree(forest, stitch_columns + derived_columns)

    if len(stitch_columns) == 0:
        return _do_patch(forest.unsafe_rng, left, right)
    else:
        return _do_stitch(forest, metadata, left, right, derived_cluster)


def build_table(
    materialize_tree: TreeMaterializer, forest: Forest, metadata: StitchingMetadata, clusters: Clusters
) -> tuple[list[Row], Combination]:
    acc = materialize_tree(forest, clusters.initial_cluster)

    for derived_cluster in clusters.derived_clusters:
        acc = _stitch(materialize_tree, forest, metadata, acc, derived_cluster)

    rows, columns = acc
    return [microdata_row_to_row(row) for row in rows], columns
