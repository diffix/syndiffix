import os

from syndiffix.common import *
from syndiffix.counters import UniqueAidCountersFactory
from syndiffix.forest import *
from syndiffix.microdata import apply_convertors, get_convertor

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
    aid_df = DataFrame(range(1, len(data_df) + 1)) if aid_df is None else aid_df
    anon_params = anon_params if anon_params is not None else AnonymizationParams()
    bucketization_params = bucketization_params if bucketization_params else BucketizationParams()
    return Forest(anon_params, bucketization_params, UniqueAidCountersFactory(), aid_df, data_df)


def _load_csv(path: str, columns: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    if columns is not None:
        df = df[columns]
    return apply_convertors([get_convertor(df, column) for column in df.columns], df)


def load_forest(
    filename: str,
    columns: list[str] | None = None,
    anon_params: AnonymizationParams | None = None,
    bucketization_params: BucketizationParams | None = None,
) -> Forest:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_df = _load_csv(os.path.join(current_dir, "data", filename), columns)
    return create_forest(data_df, anon_params=anon_params, bucketization_params=bucketization_params)
