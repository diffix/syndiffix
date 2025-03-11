from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .clustering.common import MicrodataRow, StitchOwner
from .clustering.measures import measure_entropy
from .clustering.stitching import StitchingMetadata, _do_stitch
from .clustering.strategy import NoClustering
from .common import ColumnId, Combination
from .synthesizer import Synthesizer


def _make_synthesizers(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    col_names_stitch: list[str],
) -> Tuple[Synthesizer, Synthesizer, Synthesizer]:
    syn_left = Synthesizer(df_left, clustering=NoClustering())
    syn_right = Synthesizer(df_right, clustering=NoClustering())
    df_left_short = df_left.iloc[[0]]
    df_right_short = df_right.iloc[[0]].copy()
    for col_name in col_names_stitch:
        df_right_short.rename(columns={col_name: f"__fake__{col_name}"}, inplace=True)
    syn_fake = Synthesizer(pd.concat([df_left_short, df_right_short], axis=1), clustering=NoClustering())
    syn_fake.forest.snapped_intervals = syn_left.forest.snapped_intervals + syn_right.forest.snapped_intervals
    return syn_fake, syn_left, syn_right


def _make_microdata(
    df: pd.DataFrame,
    syn: Synthesizer,
    columns: list[ColumnId],
) -> tuple[list[MicrodataRow], Combination]:
    col_names = list(df.columns)
    microdata: list[MicrodataRow] = []

    for i in range(len(df)):
        row = []
        for col_name in col_names:
            row.append((df[col_name].iloc[i], float(syn.forest.orig_data[col_name].iloc[i])))
        microdata.append(row)
    return (microdata, tuple(columns))


def stitch(df_left: pd.DataFrame, df_right: pd.DataFrame, shared: bool = True) -> pd.DataFrame:
    # Make the needed column names with the original names (not later rename)
    col_names_left = list(df_left.columns)
    col_names_right = list(df_right.columns)
    col_names_stitch = list(set(col_names_left) & set(col_names_right))
    col_names_right_minus_stitch = [col for col in col_names_right if col not in col_names_stitch]

    syn_fake, syn_left, syn_right = _make_synthesizers(df_left, df_right, col_names_stitch)
    entropy_1dim_left = np.array(
        [measure_entropy(syn_left.forest.get_tree((ColumnId(i),))) for i in range(len(syn_left.forest.columns))],
        dtype=float,
    )
    entropy_1dim_right = np.array(
        [measure_entropy(syn_right.forest.get_tree((ColumnId(i),))) for i in range(len(syn_right.forest.columns))],
        dtype=float,
    )
    stitching_metadata = StitchingMetadata(
        syn_left.column_is_integral + syn_right.column_is_integral,
        np.concatenate((entropy_1dim_left, entropy_1dim_right)),
    )

    col_names_all = syn_fake.forest.columns
    columns_left = [ColumnId(col_names_all.index(col_name)) for col_name in col_names_left]
    columns_right = [ColumnId(col_names_all.index(col_name)) for col_name in col_names_right]
    columns_right_minus_stitch = [ColumnId(col_names_all.index(col_name)) for col_name in col_names_right_minus_stitch]
    columns_stitch = [ColumnId(col_names_all.index(col_name)) for col_name in col_names_stitch]
    owner = StitchOwner.SHARED if shared else StitchOwner.LEFT
    derived_cluster = (owner, columns_stitch, columns_right_minus_stitch)

    microdata_left = _make_microdata(df_left, syn_left, columns_left)
    microdata_right = _make_microdata(df_right, syn_right, columns_right)

    (microdata, columns) = _do_stitch(
        syn_fake.forest, stitching_metadata, microdata_left, microdata_right, derived_cluster
    )
    col_names = [col_names_all[col_id] for col_id in columns]
    data = [[tup[0] for tup in row] for row in microdata]
    return pd.DataFrame(data, columns=col_names)


def get_cluster(syn: Synthesizer) -> Dict[str, List[Any]]:
    # Returns a friendly representation of the cluster (column names instead of IDs)
    cluster: Dict[str, List[Any]] = {"initial": [], "derived": []}
    for col_id in syn.clusters.initial_cluster:
        cluster["initial"].append(syn.forest.columns[col_id])
    for owner, cols1, cols2 in syn.clusters.derived_clusters:
        cluster["derived"].append(
            {
                "stitch_style": owner,
                "stitch_columns": [syn.forest.columns[col_id] for col_id in cols1],
                "new_columns": [syn.forest.columns[col_id] for col_id in cols2],
            }
        )
    return cluster
