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
    raw_data: pd.DataFrame,
    aids: Optional[pd.DataFrame] = None,
    anon_params: AnonymizationParams = AnonymizationParams(),
    bucketization_params: BucketizationParams = BucketizationParams(),
    clustering: ClusteringStrategy = DefaultClustering(),
) -> pd.DataFrame:
    if aids is None:
        aids = pd.DataFrame({"RowIndex": range(1, len(raw_data) + 1)})
        counters_factory = UniqueAidCountersFactory()
    else:
        counters_factory = GenericAidCountersFactory(len(aids.columns), bucketization_params.range_low_threshold)

    raw_dtypes = raw_data.dtypes

    column_convertors = [get_convertor(raw_data, column) for column in raw_data.columns]
    column_is_integral = [_is_integral(convertor.column_type()) for convertor in column_convertors]
    # TODO: this changes the input DataFrame; we need to create a new one.
    converted_data = apply_convertors(column_convertors, raw_data)

    forest = Forest(
        AnonymizationContext(Hash(0), anon_params),
        bucketization_params,
        counters_factory,
        aids,
        converted_data,
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

    syn_data = pd.DataFrame(rows, columns=get_items_combination_list(root_combination, raw_data.columns.tolist()))

    for col, dtype in zip(syn_data.columns, raw_dtypes):
        syn_data[col] = syn_data[col].astype(dtype)

    return syn_data


class Synthesizer(object):
    def __init__(self) -> None:
        pass

    def fit(self, df: pd.DataFrame) -> None:
        self.df = df

    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        if n_samples is not None:
            raise NotImplementedError("Specifying n_samples not implemented yet")
        return self.df
