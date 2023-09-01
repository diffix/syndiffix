from syndiffix.common import *

SALT = bytes([])
NOISELESS_SUPPRESSION = SuppressionParams()
NOISELESS_SUPPRESSION.layer_sd = 0.0
NOISELESS_PARAMS = AnonymizationParams()
NOISELESS_PARAMS.low_count_params = NOISELESS_SUPPRESSION
NOISELESS_PARAMS.layer_noise_sd = 0.0
NOISELESS_PARAMS.outlier_count.upper = NOISELESS_PARAMS.outlier_count.lower
NOISELESS_PARAMS.top_count.upper = NOISELESS_PARAMS.top_count.lower
NOISELESS_CONTEXT = AnonymizationContext(bucket_seed=np.uint64(123), anonymization_params=NOISELESS_PARAMS)
