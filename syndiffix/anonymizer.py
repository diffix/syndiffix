import hashlib
import math
import operator
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from typing import Iterator, cast

from .common import *


@dataclass
class Contributions:
    per_aid: Counter[Hash] = field(default_factory=Counter)
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
    u2 = ((int(seed) >> 32) & 0x7FFFFFFF) / 0x7FFFFFFF
    normal = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
    return sd * normal


def _crypto_hash_salted_seed(salt: bytes, seed: Hash) -> Hash:
    hash = hashlib.sha256(salt + seed.tobytes()).digest()
    return Hash(int.from_bytes(hash[:8], "little"))


def _seed_from_aid_set(aid_set: Iterable[Hash]) -> Hash:
    return reduce(operator.xor, aid_set, Hash(0))


def _hash_bytes(b: bytes) -> Hash:
    hash = hashlib.blake2b(b, digest_size=8).digest()
    return Hash(int.from_bytes(hash, "little"))


def _hash_string(s: str) -> Hash:
    return _hash_bytes(s.encode())


def _hash_int(i: int) -> Hash:
    return _hash_bytes(i.to_bytes(8, "little"))


def _mix_seed(step_name: str, seed: Hash) -> Hash:
    return _hash_string(step_name) ^ seed


def _generate_noise(salt: bytes, step_name: str, sd: float, noise_layers: Hashes) -> float:
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
    compactIntervals = outlier_count, top_count

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
                outlier_count.upper -= outlier_adjustment
                top_count.upper -= top_adjustment
            case (False, True):
                # `outlier_count` is compacted as much as possible by `outlier_range`,
                # `top_count` takes the surplus adjustment.
                outlier_count.upper = outlier_count.lower
                top_count.upper -= total_adjustment - outlier_range
            case (True, False):
                # Vice versa.
                outlier_count.upper -= total_adjustment - top_range
                top_count.upper = top_count.lower
            case (False, False):
                # Not possible. Otherwise:
                # `outlier_range + top_range < outlier_adjustment + top_adjustment = total_adjustment`,
                # but we knew the opposite was true in (*) above.
                raise RuntimeError("Impossible interval compacting.")

    return compactIntervals


@dataclass(frozen=True)
class _AidCount:
    flattened_sum: float
    flattening: float
    noise_sd: float
    noise: float


def _aid_flattening(contribution: Contributions, context: AnonymizationContext) -> _AidCount | None:
    unaccounted_for = contribution.unaccounted_for
    aid_contributions = list(dict(contribution.per_aid).items())

    total_count = len(aid_contributions)
    anon_params = context.anonymization_params

    flattening_intervals = _compact_flattening_intervals(anon_params.outlier_count, anon_params.top_count, total_count)

    if flattening_intervals is None:
        return None

    outlier_interval, top_interval = flattening_intervals
    sorted_aid_contributions = sorted(
        aid_contributions,
        reverse=True,
        key=lambda aid_and_contribution: (aid_and_contribution[1], aid_and_contribution[0]),
    )

    flat_seed = _seed_from_aid_set(
        aid for aid, _ in sorted_aid_contributions[: outlier_interval.upper + top_interval.upper]
    )
    flat_seed = _crypto_hash_salted_seed(anon_params.salt, flat_seed)
    outlier_count = _random_uniform(outlier_interval, _mix_seed("outlier", flat_seed))
    top_count = _random_uniform(top_interval, _mix_seed("top", flat_seed))

    top_group_sum = sum(
        contribution for _, contribution in sorted_aid_contributions[outlier_count : (outlier_count + top_count)]
    )
    top_group_average = top_group_sum / top_count

    flattening = sum(
        max(contribution - top_group_average, 0) for _, contribution in sorted_aid_contributions[:outlier_count]
    )

    real_sum = sum(contribution for _, contribution in aid_contributions)
    flattened_unaccounted_for = max(unaccounted_for - flattening, 0)
    flattened_sum = real_sum - flattening
    flattened_avg = flattened_sum / total_count

    noise_scale = max(flattened_avg, 0.5 * top_group_average)
    noise_sd = anon_params.layer_noise_sd * noise_scale

    seeds = (context.bucket_seed, _seed_from_aid_set(aid for aid, _ in aid_contributions))
    noise = _generate_noise(anon_params.salt, "noise", noise_sd, seeds)

    return _AidCount(flattened_sum + flattened_unaccounted_for, flattening, noise_sd, noise)


def _anonymized_sum(by_aid_sum: Iterable[_AidCount]) -> tuple[float, float]:
    # We might end up with multiple different flattened sums that have the same amount of flattening.
    # This could be the result of some AID values being null for one of the AIDs, while there were still
    # overall enough AIDs to produce a flattened sum.
    # In these cases, we want to use the largest flattened sum to minimize unnecessary flattening.
    flattening = max(by_aid_sum, key=lambda aggregate: (aggregate.flattening, aggregate.flattened_sum))

    # For determinism, resolve draws using the maximum absolute noise value.
    noise = max(by_aid_sum, key=lambda aggregate: (aggregate.noise_sd, abs(aggregate.noise)))

    return flattening.flattened_sum + noise.noise, noise.noise_sd


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
    return _seed_from_aid_set(_hash_string(string) for string in set(strings))


def hash_aid(aid: object) -> Hash:
    if aid is None or aid == "":
        return Hash(0)
    elif isinstance(aid, int):
        return _hash_int(cast(int, aid))
    elif isinstance(aid, str):
        return _hash_string(cast(str, aid))
    else:
        raise NotImplementedError("Unsupported AID type!")


# Returns whether any of the AID value sets has a low count.
def is_low_count(salt: bytes, params: SuppressionParams, aid_trackers: list[tuple[int, Hash]]) -> bool:
    assert len(aid_trackers) > 0

    for count, seed in aid_trackers:
        if count < params.low_threshold:
            return True

        noise = _generate_noise(salt, "suppress", params.layer_sd, (seed,))

        # `low_mean_gap` is the number of standard deviations between `low_threshold` and desired mean.
        mean = params.low_mean_gap * params.layer_sd + params.low_threshold

        if count < noise + mean:
            return True

    return False


def count_multiple_contributions(
    context: AnonymizationContext, contributions: list[Contributions]
) -> CountResult | None:
    by_aid = [_aid_flattening(c, context) for c in contributions]

    if any((flattened is None for flattened in by_aid)):
        return None

    value, noise_sd = _anonymized_sum(list(filter(None, by_aid)))

    return CountResult(int(value), _money_round_noise(noise_sd))


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
