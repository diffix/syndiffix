import hashlib
import math
import operator
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Iterator, cast

from .common import *


@dataclass
class PidContributions:
    value_counts: Counter[Hash] = field(default_factory=Counter)
    unaccounted_for: int = 0


@dataclass(frozen=True)
class CountResult:
    anonymized_count: int
    noise_sd: float


# ----------------------------------------------------------------
# Noise
# ----------------------------------------------------------------

# The noise seeds are hash values.
# From each seed we generate a single random value, with either a uniform or a normal distribution.
# Any decent hash function should produce values that are uniformly distributed over the output space.
# Hence, we only need to limit the seed to the requested interval to get a uniform random integer.
# To get a normal random float, we use the Box-Muller method on two uniformly distributed integers.


def _random_uniform(interval: FlatteningInterval, seed: Hash) -> int:
    # While using modulo to bound values produces biased output, we are using very small ranges
    # (typically less than 10), for which the bias is insignificant.
    return int(seed) % (interval.upper - interval.lower + 1) + interval.lower


def _random_normal(sd: float, seed: Hash) -> float:
    u1 = (int(seed) & 0x7FFFFFFF) / 0x7FFFFFFF
    u1 = max(u1, sys.float_info.epsilon)
    u2 = ((int(seed) >> 32) & 0x7FFFFFFF) / 0x7FFFFFFF
    normal = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
    return sd * normal


def _crypto_hash_salted_seed(salt: bytes, seed: Hash) -> Hash:
    hash = hashlib.sha256(salt + seed.tobytes()).digest()
    return Hash(int.from_bytes(hash[:8], "little"))


def _hash_bytes(b: bytes) -> Hash:
    hash = hashlib.blake2b(b, digest_size=8).digest()
    return Hash(int.from_bytes(hash, "little"))


def _hash_string(s: str) -> Hash:
    return _hash_bytes(s.encode())


def _hash_int(i: int) -> Hash:
    return _hash_bytes(i.to_bytes(8, "little"))


def _mix_seed(step_name: str, seed: Hash) -> Hash:
    return _hash_string(step_name) ^ seed


def _generate_noise(salt: bytes, step_name: str, sd: float, noise_layers: tuple[Hash, ...]) -> float:
    noise = 0.0
    for layer_seed in noise_layers:
        noise += _random_normal(sd, _mix_seed(step_name, _crypto_hash_salted_seed(salt, layer_seed)))
    return noise


def _compact_flattening_intervals(
    outlier_count: FlatteningInterval, top_count: FlatteningInterval, total_count: int
) -> tuple[FlatteningInterval, FlatteningInterval] | None:
    if total_count < outlier_count.lower + top_count.lower:
        return None

    total_adjustment = outlier_count.upper + top_count.upper - total_count

    outlier_upper = outlier_count.upper
    top_upper = top_count.upper

    if total_adjustment > 0:
        # NOTE: At this point we know `0 < total_adjustment <= outlier_range + top_range` (*):
        #       `total_adjustment = outlier_count.upper + top_count.upper - total_count
        #                        <= outlier_count.upper + top_count.upper - outlier_count.lower - top_count.lower`.
        outlier_range = outlier_count.upper - outlier_count.lower
        top_range = top_count.upper - top_count.lower
        # `top_adjustment` will be half of `total_adjustment` rounded up, so it takes priority as it should.
        outlier_adjustment = total_adjustment // 2
        top_adjustment = total_adjustment - outlier_adjustment

        # Adjust, depending on how the adjustments "fit" in the ranges.
        match (outlier_range >= outlier_adjustment, top_range >= top_adjustment):
            case (True, True):
                # Both ranges are compacted at same rate.
                outlier_upper -= outlier_adjustment
                top_upper -= top_adjustment
            case (False, True):
                # `outlier_count` is compacted as much as possible by `outlier_range`,
                # `top_count` takes the surplus adjustment.
                outlier_upper = outlier_count.lower
                top_upper -= total_adjustment - outlier_range
            case (True, False):
                # Vice versa.
                outlier_upper -= total_adjustment - top_range
                top_upper = top_count.lower
            case (False, False):
                # Not possible. Otherwise:
                # `outlier_range + top_range < outlier_adjustment + top_adjustment = total_adjustment`,
                # but we knew the opposite was true in (*) above.
                raise RuntimeError("Impossible interval compacting.")

    return replace(outlier_count, upper=outlier_upper), replace(top_count, upper=top_upper)


@dataclass(frozen=True)
class _PidCount:
    flattened_count: float
    flattening: float
    noise_sd: float
    noise: float


def _flatten_contributions(pid_contributions: PidContributions, context: AnonymizationContext) -> _PidCount | None:
    total_count = len(pid_contributions.value_counts)
    anon_params = context.anonymization_params

    flattening_intervals = _compact_flattening_intervals(anon_params.outlier_count, anon_params.top_count, total_count)
    if flattening_intervals is None:
        return None  # Insufficient values for a sensible flattening.

    outlier_interval, top_interval = flattening_intervals

    # Sort contributions by amount and PID value (to ensure consistency in case of equal contributions).
    sorted_value_counts = sorted(
        pid_contributions.value_counts.items(),
        reverse=True,
        key=operator.itemgetter(1, 0),
    )

    flat_seed = seed_from_pid_set(pid for pid, _ in sorted_value_counts[: outlier_interval.upper + top_interval.upper])
    flat_seed = _crypto_hash_salted_seed(anon_params.salt, flat_seed)
    outlier_count = _random_uniform(outlier_interval, _mix_seed("outlier", flat_seed))
    top_count = _random_uniform(top_interval, _mix_seed("top", flat_seed))

    top_group_sum = sum(
        contribution for _, contribution in sorted_value_counts[outlier_count: (outlier_count + top_count)]
    )
    top_group_average = top_group_sum / top_count

    flattening = sum(
        max(contribution - top_group_average, 0) for _, contribution in sorted_value_counts[:outlier_count]
    )

    real_sum = pid_contributions.value_counts.total()
    flattened_unaccounted_for = max(pid_contributions.unaccounted_for - flattening, 0)
    flattened_sum = real_sum - flattening
    flattened_avg = flattened_sum / total_count

    noise_scale = max(flattened_avg, 0.5 * top_group_average)
    noise_sd = anon_params.layer_noise_sd * noise_scale

    pid_seed = seed_from_pid_set(pid_contributions.value_counts)
    noise = _generate_noise(anon_params.salt, "noise", noise_sd, (context.bucket_seed, pid_seed))

    return _PidCount(flattened_sum + flattened_unaccounted_for, flattening, noise_sd, noise)


def _anonymized_sum(pid_counts: Iterable[_PidCount]) -> tuple[float, float]:
    # We might end up with multiple different flattened counts that have the same amount of flattening.
    # This could be the result of some PID values being null for one of the PIDs, while there were still
    # overall enough PIDs to produce a flattened count.
    # In these cases, we want to use the largest flattened count to minimize unnecessary flattening.
    flattening = max(pid_counts, key=lambda count: (count.flattening, count.flattened_count))

    # For determinism, resolve draws using the maximum absolute noise value.
    noise = max(pid_counts, key=lambda count: (count.noise_sd, abs(count.noise)))

    return flattening.flattened_count + noise.noise, noise.noise_sd


MONEY_ROUND_MIN = 1e-10
MONEY_ROUND_DELTA = MONEY_ROUND_MIN / 100.0


# Works with `value` between 1.0 and 10.0.
def _money_round_internal(value: float) -> float:
    if 1.0 <= value < 1.5:
        return 1.0
    elif 1.5 <= value < 3.5:
        return 2.0
    elif 3.5 <= value < 7.5:
        return 5.0
    else:
        return 10.0


def _money_round(value: float) -> float:
    if 0.0 <= value < MONEY_ROUND_MIN:
        return 0.0

    tens = 10.0 ** math.floor(math.log10(value))
    return tens * _money_round_internal(value / tens)


def _money_round_noise(noise_sd: float) -> float:
    if noise_sd == 0.0:
        return 0.0

    rounding_resolution = _money_round(0.05 * noise_sd)
    return math.ceil(noise_sd / rounding_resolution) * rounding_resolution


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------


def hash_strings(strings: Iterator[str]) -> Hash:
    return seed_from_pid_set(_hash_string(string) for string in set(strings))


def hash_pid(pid: object) -> Hash:
    if not pid:
        return Hash(0)
    elif isinstance(pid, int):
        return _hash_int(cast(int, pid))
    elif isinstance(pid, str):
        return _hash_string(cast(str, pid))
    else:
        raise NotImplementedError("Unsupported PID type!")


def seed_from_pid_set(pid_set: Iterable[Hash]) -> Hash:
    return reduce(operator.xor, pid_set, Hash(0))


# Returns whether any of the PID value sets has a low count.
def is_low_count(salt: bytes, params: SuppressionParams, pid_trackers: list[tuple[int, Hash]]) -> bool:
    assert len(pid_trackers) > 0

    for count, seed in pid_trackers:
        if count < params.low_threshold:
            return True

        noise = _generate_noise(salt, "suppress", params.layer_sd, (seed,))

        # `low_mean_gap` is the number of standard deviations between `low_threshold` and desired mean.
        mean = params.low_mean_gap * params.layer_sd + params.low_threshold

        if count < noise + mean:
            return True

    return False


def count_multiple_contributions(
    context: AnonymizationContext, contributions_list: list[PidContributions]
) -> CountResult | None:
    assert len(contributions_list) > 0

    flattened_contributions = [_flatten_contributions(contributions, context) for contributions in contributions_list]

    # If any of the PIDs had insufficient data to produce a sensible flattening, we have to abort anonymization.
    if not all(flattened_contributions):
        return None

    value, noise_sd = _anonymized_sum(cast(list[_PidCount], flattened_contributions))

    return CountResult(round(value), _money_round_noise(noise_sd))


def count_single_contributions(context: AnonymizationContext, count: int, seed: Hash) -> int:
    params = context.anonymization_params
    noise = _generate_noise(params.salt, "noise", params.layer_noise_sd, (context.bucket_seed, seed))
    return round(count + noise)


def noisy_row_limit(salt: bytes, seed: Hash, row_count: int, row_fraction: int) -> int:
    real_row_limit = row_count // row_fraction

    # Select an integer between plus and minus 5% of `real_row_limit`.
    noise_range = real_row_limit // 20
    noise_seed = _mix_seed("precision_limit", _crypto_hash_salted_seed(salt, seed))
    noise = _random_uniform(FlatteningInterval(-noise_range, noise_range), noise_seed)

    return real_row_limit + noise
