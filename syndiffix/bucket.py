from dataclasses import dataclass
from itertools import chain
from math import floor
from random import Random

from .common import *
from .interval import Interval, Intervals
from .tree import Branch, Leaf, Node


@dataclass
class Bucket:
    intervals: Intervals
    count: int


Buckets = list[Bucket]

BucketsCache = dict[Node, Buckets]


def _adjust_counts(buckets: Buckets, current_count: int, target_count: int) -> None:
    ratio = target_count / current_count

    acc_error = 0.0
    for bucket in buckets:
        adjusted_count = bucket.count * ratio
        acc_error += adjusted_count - floor(adjusted_count)

        # Add the previously accumulated errors to the current count, if larger than 1 unit.
        if acc_error > 1.0:
            adjusted_count += 1.0
            acc_error -= 1.0

        bucket.count = int(adjusted_count)


def _get_subbuckets(node: Node, harvested_nodes: BucketsCache, unsafe_rng: Random) -> list[Buckets]:
    subnodes_combinations = generate_combinations(node.dimensions() - 1, node.dimensions())
    subnodes_buckets: list[Buckets] = []

    for subnode, combination in zip(node.subnodes, subnodes_combinations):
        actual_subnode_intervals = get_items_combination(combination, node.actual_intervals)
        if all(interval.is_singularity() for interval in actual_subnode_intervals):
            subnode_buckets = [Bucket(actual_subnode_intervals, node.noisy_count())]
        elif subnode is None:
            subnode_buckets = []
        else:
            subnode_buckets = _harvest_node(subnode, harvested_nodes, unsafe_rng)

        subnodes_buckets.append(subnode_buckets)

    return subnodes_buckets


def _get_smallest_intervals(subbuckets: list[Buckets]) -> Intervals:
    dimensions = len(subbuckets)
    cumulative_intervals: list[list[Interval | None]] = [[None] * dimensions for _ in range(dimensions)]

    for combination_index, combination in enumerate(generate_combinations(dimensions - 1, dimensions)):
        for bucket in subbuckets[combination_index]:
            for interval_index, bucket_interval in enumerate(bucket.intervals):
                dimension_index = combination[interval_index]

                match cumulative_intervals[dimension_index][combination_index]:
                    case None:
                        # This gets mutated later, so we need to create a new Interval instance.
                        cumulative_intervals[dimension_index][combination_index] = bucket_interval.copy()
                    case cumulative_interval:
                        cumulative_interval = cast(Interval, cumulative_interval)  # Needed to silence the type checker.
                        cumulative_interval.min = min(cumulative_interval.min, bucket_interval.min)
                        cumulative_interval.max = max(cumulative_interval.max, bucket_interval.max)

    return tuple(
        min(filter(None, intervals), key=lambda interval: interval.size()) for intervals in cumulative_intervals
    )


def _get_per_dimension_interval_lists(smallest_intervals: Intervals, subbuckets: list[Buckets]) -> list[list[Interval]]:
    dimensions = len(subbuckets)
    per_dimension_interval_lists: list[list[Interval]] = [[] for _ in range(dimensions)]

    for subnode_buckets, subnode_combination in zip(subbuckets, generate_combinations(dimensions - 1, dimensions)):
        for interval_index, dimension_index in enumerate(subnode_combination):
            smallest_interval = smallest_intervals[dimension_index]
            for bucket in subnode_buckets:
                bucket_interval = bucket.intervals[interval_index]
                if smallest_interval.contains_interval(bucket_interval):
                    per_dimension_interval_lists[dimension_index].extend([bucket_interval] * bucket.count)

    return per_dimension_interval_lists


#  Removes the low-count half, if one exists, from the specified interval.
def _compact_node_interval(leaf: Leaf, dimension: int, interval: Interval) -> Interval:
    value_index = leaf.context.combination[dimension]

    # IEntityCounter is mutable, so we need to create a new object for each half.
    per_half_entity_counters = (
        leaf.context.counters_factory.create_entity_counter(),
        leaf.context.counters_factory.create_entity_counter(),
    )

    for row in leaf.rows:
        half_index = interval.half_index(leaf.context.data[row][value_index])
        per_half_entity_counters[half_index].add(leaf.context.pid_data[row])

    anon_params = leaf.context.anonymization_context.anonymization_params
    per_half_low_count = tuple(
        counter.is_low_count(anon_params.salt, anon_params.low_count_params) for counter in per_half_entity_counters
    )
    match per_half_low_count:
        case (False, True):
            return interval.half(0)
        case (True, False):
            return interval.half(1)
        case _:
            return interval


def _compact_node_intervals(node: Node) -> Intervals:
    if isinstance(node, Leaf) and node.is_stub:
        return tuple(
            _compact_node_interval(node, index, interval) for index, interval in enumerate(node.snapped_intervals)
        )
    else:
        return node.snapped_intervals


def _compact_smallest_intervals(node: Node, smallest_intervals: Intervals) -> None:
    compacted_node_intervals = _compact_node_intervals(node)
    for smallest_interval, compacted_node_interval in zip(smallest_intervals, compacted_node_intervals):
        if compacted_node_interval.overlaps(smallest_interval):
            # Drop the sections of the smallest interval which are outside the compacted node interval.
            smallest_interval.min = max(smallest_interval.min, compacted_node_interval.min)
            smallest_interval.max = min(smallest_interval.max, compacted_node_interval.max)
        else:
            # If the intervals have no overlap, expand the smallest interval to include both.
            smallest_interval.min = min(smallest_interval.min, compacted_node_interval.min)
            smallest_interval.max = max(smallest_interval.max, compacted_node_interval.max)


def _get_per_subnode_intervals_lists(smallest_intervals: Intervals, subbuckets: list[Buckets]) -> list[list[Intervals]]:
    assert len(smallest_intervals) == len(subbuckets)
    dimensions = len(smallest_intervals)
    per_subnode_intervals_lists: list[list[Intervals]] = [[] for _ in range(dimensions)]

    for combination_index, combination in enumerate(generate_combinations(dimensions - 1, dimensions)):
        # For every subnode, gather only the subintervals that are contained in the corresponding smallest intervals.
        subsmallest_intervals = get_items_combination(combination, smallest_intervals)
        for subbucket in subbuckets[combination_index]:
            if all(subsmallest_intervals[i].contains_interval(subbucket.intervals[i]) for i in range(dimensions - 1)):
                per_subnode_intervals_lists[combination_index].extend([subbucket.intervals] * subbucket.count)

    return per_subnode_intervals_lists


def _match_subintervals(
    count: int,
    per_dimension_interval_lists: list[list[Interval]],
    per_subnode_intervals_lists: list[list[Intervals]],
    unsafe_rng: Random,
) -> Iterable[Intervals]:
    assert len(per_dimension_interval_lists) == len(per_subnode_intervals_lists)
    dimensions = len(per_dimension_interval_lists)

    for match_counter in range(count):
        # We cycle through all the dimensions and subnodes equally when selecting the sub-intervals to be matched.
        subnode_index = match_counter % dimensions
        dimension_index = dimensions - subnode_index - 1

        subnode_intervals_list = per_subnode_intervals_lists[subnode_index]
        dimension_interval_list = per_dimension_interval_lists[dimension_index]

        # Create a random match between the selected sub-intervals in order to avoid biases in the output.
        subnode_intervals = subnode_intervals_list[unsafe_rng.randint(0, len(subnode_intervals_list) - 1)]
        dimension_interval = dimension_interval_list[unsafe_rng.randint(0, len(dimension_interval_list) - 1)]

        yield (*subnode_intervals[:dimension_index], dimension_interval, *subnode_intervals[dimension_index:])


def _refine_buckets(node: Node, harvested_nodes: BucketsCache, count: int, unsafe_rng: Random) -> Buckets:
    subbuckets = _get_subbuckets(node, harvested_nodes, unsafe_rng)

    if not all(subbuckets):
        return [Bucket(tuple(node.bucket_intervals()), count)]

    smallest_intervals = _get_smallest_intervals(subbuckets)
    _compact_smallest_intervals(node, smallest_intervals)
    per_dimension_interval_lists = _get_per_dimension_interval_lists(smallest_intervals, subbuckets)
    per_subnode_intervals_lists = _get_per_subnode_intervals_lists(smallest_intervals, subbuckets)

    if not all(per_dimension_interval_lists) or not all(per_subnode_intervals_lists):
        # Can't match if there are no intervals for some subnodes or no interval for some dimensions.
        return [Bucket(tuple(node.bucket_intervals()), count)]

    matched_intervals = _match_subintervals(
        count, per_dimension_interval_lists, per_subnode_intervals_lists, unsafe_rng
    )
    return [Bucket(interval, 1) for interval in matched_intervals]


def _harvest_leaf(leaf: Leaf, harvested_nodes: BucketsCache, unsafe_rng: Random) -> Buckets:
    low_threshold = leaf.context.anonymization_context.anonymization_params.low_count_params.low_threshold
    if leaf.is_over_threshold(low_threshold):
        if leaf.is_singularity() or leaf.dimensions() == 1:
            return [Bucket(tuple(leaf.bucket_intervals()), leaf.noisy_count())]
        else:
            return _refine_buckets(leaf, harvested_nodes, leaf.noisy_count(), unsafe_rng)
    else:
        return []


def _harvest_branch(branch: Branch, harvested_nodes: BucketsCache, unsafe_rng: Random) -> Buckets:
    children_buckets = list(
        chain.from_iterable(
            _harvest_node(child_node, harvested_nodes, unsafe_rng) for child_node in branch.children.values()
        )
    )

    children_count = sum(bucket.count for bucket in children_buckets)
    parent_count = branch.noisy_count()

    if 2 * children_count < parent_count:
        if branch.dimensions() == 1:
            children_buckets = [Bucket(tuple(branch.bucket_intervals()), parent_count)]
        else:
            children_buckets += _refine_buckets(branch, harvested_nodes, parent_count - children_count, unsafe_rng)
    else:
        _adjust_counts(children_buckets, children_count, parent_count)

    return children_buckets


def _harvest_node(node: Node, harvested_nodes: BucketsCache, unsafe_rng: Random) -> Buckets:
    if harvested_nodes.get(node) is None:
        harvester = _harvest_leaf if isinstance(node, Leaf) else _harvest_branch
        harvested_nodes[node] = harvester(node, harvested_nodes, unsafe_rng)  # type: ignore

    return harvested_nodes[node]


# The passed RNG isn't safe for anonymization, so it can only be applied to already anonymized data.
# Used for randomly matching subintervals during refining.
def harvest(node: Node, unsafe_rng: Random) -> Buckets:
    # Maps node.NodeData.Id to its already harvested buckets, so it's done only once.
    harvested_nodes: BucketsCache = dict()
    buckets = _harvest_node(node, harvested_nodes, unsafe_rng)
    return [bucket for bucket in buckets if bucket.count > 0]
