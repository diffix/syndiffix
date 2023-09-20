from typing import Optional

import pandas as pd

from .bucket import harvest
from .clustering.common import MicrodataRow
from .clustering.stitching import StitchingMetadata, build_table
from .clustering.strategy import ClusteringStrategy, DefaultClustering
from .common import *
from .counters import GenericAidCountersFactory, UniqueAidCountersFactory
from .forest import Forest
from .microdata import apply_convertors, generate_microdata, get_convertor


def _is_integral(col_type: ColumnType) -> bool:
    if col_type == ColumnType.REAL or col_type == ColumnType.TIMESTAMP:
        return False
    else:
        return True


def synthesize(
    df: pd.DataFrame,
    aid_columns: Optional[list[str]] = None,
    data_columns: Optional[list[str]] = None,
    anon_params: AnonymizationParams = AnonymizationParams(),
    bucketization_params: BucketizationParams = BucketizationParams(),
    clustering: ClusteringStrategy = DefaultClustering(),
) -> pd.DataFrame:
    df_raw_data = df

    if aid_columns is None:
        aid_columns = []

    if len(aid_columns) > 0:
        df_aids = df[aid_columns]
        df_raw_data = df.drop(columns=aid_columns)
    else:
        df_aids = pd.DataFrame({"RowIndex": range(1, len(df) + 1)})

    if data_columns is not None:
        assert len(data_columns) > 0
        df_raw_data = df_raw_data[data_columns]

    column_convertors = [get_convertor(df, column) for column in df.columns]
    column_is_integral = [_is_integral(convertor.column_type()) for convertor in column_convertors]
    df_data = apply_convertors(column_convertors, df_raw_data)

    counters_factory = (
        UniqueAidCountersFactory()
        if len(aid_columns) == 0
        else GenericAidCountersFactory(len(aid_columns), bucketization_params.range_low_threshold)
    )

    forest = Forest(
        AnonymizationContext(Hash(0), anon_params),
        bucketization_params,
        counters_factory,
        df_aids,
        df_data,
    )

    clusters, entropy_1dim = clustering.build_clusters(forest)

    def materialize_tree(forest: Forest, columns: list[ColumnId]) -> tuple[list[MicrodataRow], Combination]:
        combination = tuple(sorted(columns))
        tree = forest.get_tree(combination)
        buckets = harvest(tree)
        return (
            generate_microdata(
                buckets,
                get_items_combination_list(combination, column_convertors),
                get_items_combination_list(combination, forest.null_mappings),
            ),
            combination,
        )

    rows, root_combination = build_table(
        materialize_tree,
        forest,
        StitchingMetadata(column_is_integral, entropy_1dim),
        clusters,
    )

    df_syn = pd.DataFrame(rows, columns=get_items_combination_list(root_combination, df_raw_data.columns.tolist()))

    for col in df_syn.columns:
        df_syn[col] = df_syn[col].astype(df_raw_data[col].dtype)

    return df_syn


class Synthesizer(object):
    def __init__(self) -> None:
        pass

    def fit(self, df: pd.DataFrame) -> None:
        self.df = df

    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        if n_samples is not None:
            raise NotImplementedError("Specifying n_samples not implemented yet")
        return self.df
