from syndiffix.common import *
from syndiffix.counters import UniqueAidCountersFactory
from syndiffix.forest import *

SALT = bytes([])
NOISELESS_SUPPRESSION = SuppressionParams()
NOISELESS_SUPPRESSION.layer_sd = 0.0
NOISELESS_PARAMS = AnonymizationParams()
NOISELESS_PARAMS.low_count_params = NOISELESS_SUPPRESSION
NOISELESS_PARAMS.layer_noise_sd = 0.0
NOISELESS_PARAMS.outlier_count.upper = NOISELESS_PARAMS.outlier_count.lower
NOISELESS_PARAMS.top_count.upper = NOISELESS_PARAMS.top_count.lower
NOISELESS_CONTEXT = AnonymizationContext(bucket_seed=np.uint64(123), anonymization_params=NOISELESS_PARAMS)


def create_forest(
    data_df: DataFrame, aid_df: DataFrame | None = None, anon_params: AnonymizationParams | None = None
) -> Forest:
    aid_df = DataFrame(data_df.index) if aid_df is None else aid_df
    anon_params = anon_params if anon_params is not None else AnonymizationParams()
    return Forest(
        AnonymizationContext(Hash(0), anon_params), BucketizationParams(), UniqueAidCountersFactory(), aid_df, data_df
    )
