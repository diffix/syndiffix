import hashlib
import math
from dataclasses import dataclass, field

from syndiffix.common import *

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
    return seed % (interval.upper - interval.lower + 1) + interval.lower


def _random_normal(sd: float, seed: Hash) -> float:
    u1 = (seed & 0x7FFFFFFF) / 0x7FFFFFFF
    u2 = ((seed >> 32) & 0x7FFFFFFF) / 0x7FFFFFFF
    normal = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
    return sd * normal


def _crypto_hash_salted_seed(salt: bytes, seed: Hash) -> Hash:
    hash = hashlib.sha256(salt + seed.to_bytes(8, "little")).digest()
    return int.from_bytes(hash[:8], "little")


def _hash_string(s: str) -> Hash:
    hash = hashlib.md5(s.encode()).digest()
    return int.from_bytes(hash[:8], "little")


def _mix_seed(step_name: str, seed: Hash) -> Hash:
    return _hash_string(step_name) ^ seed


def _generate_noise(salt: bytes, step_name: str, sd: float, noise_layers: list[Hash]) -> float:
    noise = 0.0
    for layer_seed in noise_layers:
        noise += _random_normal(sd, _mix_seed(step_name, _crypto_hash_salted_seed(salt, layer_seed)))
    return noise


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------


# Returns whether any of the AID value sets has a low count.
def is_low_count(salt: bytes, params: SuppressionParams, aid_trackers: list[tuple[int, Hash]]) -> bool:
    assert len(aid_trackers) > 0

    for count, seed in aid_trackers:
        if count < params.low_threshold:
            return True
        else:
            noise = _generate_noise(salt, "suppress", params.layer_sd, [seed])

            # `low_mean_gap` is the number of standard deviations between `low_threshold` and desired mean.
            mean = params.low_mean_gap * params.layer_sd + params.low_threshold

            if count < noise + mean:
                return True

    return False


@dataclass
class Contributions:
    per_aid: dict[Hash, int] = field(default_factory=dict)
    unaccounted_for: int = 0


def count_multiple_contributions(context: AnonymizationParams, contributions: list[Contributions]) -> int:
    raise NotImplementedError("Counting multiple contributions is not implemented yet!")


def count_single_contributions(context: AnonymizationContext, count: int, seed: Hash) -> int:
    params = context.anonymization_params
    noise = _generate_noise(params.salt, "noise", params.layer_noise_sd, [context.bucket_seed, seed])
    return round(count + noise)
