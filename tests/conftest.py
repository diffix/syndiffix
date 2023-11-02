import json
import os
from typing import Any

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
    return Forest(
        AnonymizationContext(Hash(0), anon_params), bucketization_params, UniqueAidCountersFactory(), aid_df, data_df
    )


def _test_file_dir(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "data", filename)


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
    data_df = _load_csv(_test_file_dir(filename), columns)
    return create_forest(data_df, anon_params=anon_params, bucketization_params=bucketization_params)


def load_json(filename: str) -> Any:
    with open(_test_file_dir(filename), "r", encoding="utf-8") as file:
        return json.load(file)
