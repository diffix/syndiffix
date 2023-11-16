from dataclasses import replace

from ..common import *
from ..forest import Forest


def should_sample(forest: Forest, sample_size: int) -> bool:
    dimensions = forest.dimensions
    num_rows = len(forest.data)

    if sample_size >= num_rows:
        return False

    sampled_2dim_work = dimensions * dimensions * sample_size
    full_2dim_work = (num_rows * dimensions * 3) // 2

    total_work_with_sample = sampled_2dim_work + full_2dim_work
    total_work_without_sample = dimensions * dimensions * num_rows

    return total_work_without_sample > total_work_with_sample * 2


def sample_forest(forest: Forest, sample_size: int) -> Forest:
    sampling_anon_params = replace(
        forest.anonymization_params,
        low_count_params=SuppressionParams(low_threshold=2, layer_sd=0.5, low_mean_gap=1.0),
        layer_noise_sd=0.0,
    )

    # New RNG prevents `forest.Random` from being affected by sample size.
    rng = forest.derive_unsafe_rng()

    random_indices = rng.sample(range(len(forest.orig_data)), sample_size)

    sampled_pids = forest.orig_pids.iloc[random_indices]
    sampled_data = forest.orig_data.iloc[random_indices]

    return Forest(
        sampling_anon_params,
        forest.bucketization_params,
        forest.counters_factory,
        sampled_pids,
        sampled_data,
    )
