from dataclasses import replace

from ..common import *
from ..forest import Forest


def should_sample(forest: Forest) -> bool:
    dimensions = forest.dimensions
    num_rows = len(forest.data)
    num_samples = forest.bucketization_params.clustering_table_sample_size

    if num_samples >= num_rows:
        return False
    else:
        sampled_2dim_work = dimensions * dimensions * num_samples
        full_2dim_work = (num_rows * dimensions * 3) // 2

        total_work_with_sample = sampled_2dim_work + full_2dim_work
        total_work_without_sample = dimensions * dimensions * num_rows

        return total_work_without_sample > total_work_with_sample * 2


def sample_forest(forest: Forest) -> Forest:
    sampling_anon_context = replace(
        forest.anonymization_context,
        anonymization_params=replace(
            forest.anonymization_context.anonymization_params,
            low_count_params=SuppressionParams(low_threshold=2, layer_sd=0.5, low_mean_gap=1.0),
            layer_noise_sd=0.0,
        ),
    )

    # New RNG prevents `forest.Random` from being affected by sample size.
    random = forest.derive_unsafe_random()

    num_samples = forest.bucketization_params.clustering_table_sample_size
    random_indices = random.sample(range(len(forest.orig_data)), num_samples)

    sampled_aids = forest.orig_aids.iloc[random_indices]
    sampled_data = forest.orig_data.iloc[random_indices]

    return Forest(
        sampling_anon_context,
        forest.bucketization_params,
        forest.counters_factory,
        sampled_aids,
        sampled_data,
    )
