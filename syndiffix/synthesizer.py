import os
import secrets
from dataclasses import replace
from typing import Optional

import pandas as pd
from appdirs import user_config_dir

from .bucket import harvest
from .clustering.common import MicrodataRow
from .clustering.stitching import StitchingMetadata, build_table
from .clustering.strategy import ClusteringStrategy, DefaultClustering
from .common import *
from .counters import (
    CountersFactory,
    GenericAidCountersFactory,
    UniqueAidCountersFactory,
)
from .forest import Forest
from .microdata import apply_convertors, generate_microdata, get_convertor


def _get_default_salt() -> bytes:
    config_dir = user_config_dir("SynDiffix", "OpenDiffix")
    salt_file_path = os.path.join(config_dir, "salt.bin")

    if not os.path.isfile(salt_file_path):
        salt = secrets.randbits(64).to_bytes(8, "little")

        os.makedirs(config_dir, exist_ok=True)
        with open(salt_file_path, "wb") as file:
            file.write(salt)

    with open(salt_file_path, "rb") as file:
        return file.read()


class Synthesizer(object):
    @staticmethod
    def _is_integral(col_type: ColumnType) -> bool:
        if col_type == ColumnType.REAL or col_type == ColumnType.TIMESTAMP:
            return False
        else:
            return True

    def __init__(
        self,
        raw_data: pd.DataFrame,
        aids: Optional[pd.DataFrame] = None,
        anonymization_context: AnonymizationContext = AnonymizationContext(Hash(0), AnonymizationParams()),
        bucketization_params: BucketizationParams = BucketizationParams(),
        clustering: ClusteringStrategy = DefaultClustering(),
    ) -> None:
        if anonymization_context.anonymization_params.salt == b"":
            anonymization_context = replace(
                anonymization_context,
                anonymization_params=replace(anonymization_context.anonymization_params, salt=_get_default_salt()),
            )

        if aids is None:
            aids = pd.DataFrame({"RowIndex": range(1, len(raw_data) + 1)})
            counters_factory: CountersFactory = UniqueAidCountersFactory()
        else:
            counters_factory = GenericAidCountersFactory(len(aids.columns))

        self.raw_dtypes = raw_data.dtypes

        self.column_convertors = [get_convertor(raw_data, column) for column in raw_data.columns]
        self.column_is_integral = [self._is_integral(convertor.column_type()) for convertor in self.column_convertors]

        self.forest = Forest(
            anonymization_context,
            bucketization_params,
            counters_factory,
            aids,
            apply_convertors(self.column_convertors, raw_data),
        )

        self.clusters, self.entropy_1dim = clustering.build_clusters(self.forest)

    def sample(self) -> pd.DataFrame:
        def materialize_tree(forest: Forest, columns: list[ColumnId]) -> tuple[list[MicrodataRow], Combination]:
            combination = tuple(sorted(columns))
            tree = forest.get_tree(combination)
            buckets = harvest(tree)
            return (
                generate_microdata(
                    buckets,
                    get_items_combination_list(combination, self.column_convertors),
                    get_items_combination_list(combination, forest.null_mappings),
                ),
                combination,
            )

        rows, root_combination = build_table(
            materialize_tree,
            self.forest,
            StitchingMetadata(self.column_is_integral, self.entropy_1dim),
            self.clusters,
        )

        syn_data = pd.DataFrame(rows, columns=get_items_combination(root_combination, self.forest.columns))
        # Convert the new columns to their original type. We need to account that some columns might be missing.
        syn_data = syn_data.astype({column: self.raw_dtypes[column] for column in syn_data.columns}, copy=False)

        return syn_data

    @property
    def salt(self) -> bytes:
        return self.forest.anonymization_context.anonymization_params.salt
