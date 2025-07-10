import os
import secrets
import sys
from dataclasses import replace
from typing import Optional

import pandas as pd
from appdirs import user_config_dir

from .bucket import harvest
from .clustering.common import MicrodataRow
from .clustering.stitching import StitchingMetadata, build_table
from .clustering.strategy import ClusteringStrategy, DefaultClustering, MlClustering
from .common import *
from .counters import (
    CountersFactory,
    GenericPidCountersFactory,
    UniquePidCountersFactory,
)
from .forest import Forest
from .microdata import (
    apply_convertors,
    generate_microdata,
    get_convertor,
    make_value_safe_columns_array,
)


def _get_default_salt() -> bytes:
    config_dir = user_config_dir("SynDiffix", "OpenDiffix")
    salt_file_path = os.path.join(config_dir, "salt.bin")

    try:
        if not os.path.isfile(salt_file_path):
            salt = secrets.randbits(64).to_bytes(8, "little")

            os.makedirs(config_dir, exist_ok=True)
            with open(salt_file_path, "wb") as file:
                file.write(salt)

        with open(salt_file_path, "rb") as file:
            return file.read()
    except Exception as e:
        print(
            "Error: could not create or retrieve the default salt value!\n"
            + f"Make sure you have the required rights to access the file `{salt_file_path}` "
            + "or provide a custom salt value.",
            file=sys.stderr,
        )
        raise e


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
        pids: Optional[pd.DataFrame] = None,
        anonymization_params: AnonymizationParams = AnonymizationParams(),
        bucketization_params: BucketizationParams = BucketizationParams(),
        target_column: Optional[ColumnId | str] = None,
        clustering: Optional[ClusteringStrategy] = None,
        value_safe_columns: Optional[list[int | ColumnId | str]] = None,
    ) -> None:
        if target_column is not None:
            check_column_names_or_ids(raw_data, target_column)
            if clustering:
                raise ValueError("Cannot specify both target_column and clustering parameters.")
            clustering = MlClustering(target_column=target_column)
        elif not clustering:
            clustering = DefaultClustering()

        if value_safe_columns is not None:
            if not isinstance(value_safe_columns, list):
                raise TypeError("value_safe_columns must be a list of ColumnIds or column names.")
            self.value_safe_columns_array = make_value_safe_columns_array(raw_data, value_safe_columns)
        else:
            self.value_safe_columns_array = [False] * len(raw_data.columns)

        if anonymization_params.salt == b"":
            anonymization_params = replace(anonymization_params, salt=_get_default_salt())

        if pids is None:
            pids = pd.DataFrame({"RowIndex": range(1, len(raw_data) + 1)})
            counters_factory: CountersFactory = UniquePidCountersFactory()
        else:
            low_count_params = anonymization_params.low_count_params
            # Stop counting entities over 4 standard deviations more than the mean of the range threshold.
            # `low_mean_gap` is the number of standard deviations between `low_threshold` and desired mean.
            max_low_count = bucketization_params.range_low_threshold + int(
                (low_count_params.low_mean_gap + 4.0) * low_count_params.layer_sd
            )
            counters_factory = GenericPidCountersFactory(len(pids.columns), max_low_count)

        self.raw_dtypes = raw_data.dtypes

        self.column_convertors = [get_convertor(raw_data, column) for column in raw_data.columns]
        for col_id, convertor in enumerate(self.column_convertors):
            convertor.set_value_safe_flag(self.value_safe_columns_array[col_id])
        self.column_is_integral = [self._is_integral(convertor.column_type()) for convertor in self.column_convertors]

        self.forest = Forest(
            anonymization_params,
            bucketization_params,
            counters_factory,
            pids,
            apply_convertors(self.column_convertors, raw_data),
        )

        self.clusters, self.entropy_1dim = clustering.build_clusters(self.forest)
        for col_id, converter in enumerate(self.column_convertors):
            converter.analyze_tree(self.forest.get_tree((ColumnId(col_id),)))

    def sample(self) -> pd.DataFrame:
        def materialize_tree(forest: Forest, columns: list[ColumnId]) -> tuple[list[MicrodataRow], Combination]:
            combination = tuple(sorted(columns))
            tree = forest.get_tree(combination)
            buckets = harvest(tree, self.forest.derive_unsafe_rng())
            return (
                generate_microdata(
                    buckets,
                    get_items_combination_list(combination, self.column_convertors),
                    get_items_combination_list(combination, forest.null_mappings),
                    forest.derive_unsafe_rng(),
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
        return self.forest.anonymization_params.salt
