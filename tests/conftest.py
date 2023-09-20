from syndiffix.common import *
from syndiffix.counters import UniqueAidCountersFactory
from syndiffix.forest import *

SALT = bytes([])
NOISELESS_SUPPRESSION = SuppressionParams(layer_sd=0.0)

NOISELESS_PARAMS = AnonymizationParams(
    low_count_params=NOISELESS_SUPPRESSION,
    layer_noise_sd=0.0,
    outlier_count=FlatteningInterval(upper=FlatteningInterval().lower),
    top_count=FlatteningInterval(upper=FlatteningInterval().lower),
)

NOISELESS_CONTEXT = AnonymizationContext(bucket_seed=np.uint64(123), anonymization_params=NOISELESS_PARAMS)


def create_forest(
    data_df: DataFrame,
    aid_df: DataFrame | None = None,
    anon_params: AnonymizationParams | None = None,
    bucketization_params: BucketizationParams | None = None,
) -> Forest:
    aid_df = DataFrame(data_df.index) if aid_df is None else aid_df
    anon_params = anon_params if anon_params is not None else AnonymizationParams()
    bucketization_params = bucketization_params if bucketization_params else BucketizationParams()
    return Forest(
        AnonymizationContext(Hash(0), anon_params), bucketization_params, UniqueAidCountersFactory(), aid_df, data_df
    )
