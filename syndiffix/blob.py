from typing import Union, Optional, Any
from pathlib import Path
import random
import string
import json
import shutil
from enum import Enum
from itertools import combinations
import numpy.typing as npt
import numpy as np
import pandas as pd

from .clustering.common import Clusters, ClusteringContext, MicrodataRow, StitchOwner
from .clustering.measures import measure_entropy, measure_all, DependenceMeasures
from .clustering.stitching import StitchingMetadata, _do_stitch
from .clustering.strategy import NoClustering
from .clustering.solver import _do_solve
from .common import ColumnId, AnonymizationParams, SuppressionParams, FlatteningInterval, BucketizationParams
from .synthesizer import Synthesizer


def _shrink_matrix(matrix: npt.NDArray, comb: tuple[int]) -> npt.NDArray:
    return matrix[np.ix_(comb, comb)]

def _shrink_entropy_1dim(entropy_1dim: npt.NDArray[np.float_], comb: tuple[int, ...]) -> npt.NDArray[np.float_]:
    return entropy_1dim[list(comb)]

class BlobFiles(Enum):
    VERSION = "_meta_version.json"
    DEPENDENCE_MATRIX = "_meta_dependence_matrix.npy"
    ENTROPY_1DIM = "_meta_entropy_1dim.npy"
    ANONYMIZATION_PARAMS = "_meta_anonymization_params.json"
    BUCKETIZATION_PARAMS = "_meta_bucketization_params.json"
    CLUSTER_PARAMS = "_meta_cluster_params.json"

class SyndiffixBlob(object):
    def __init__(self,
                 blob_name: str,
                 path_to_dir: Optional[Union[str, Path]] = None,
                 ) -> None:
        self.unsafe_rng = random.Random(0)
        self.blob_name = blob_name
        # This is where the zipped blob is
        self.path_to_dir = None
        self.sample_size = None
        self.max_weight = None
        self.merge_threshold = None
        self.solver_alpha = None
        # This is where the unzipped blob files go
        self.path_to_blob_dir: Path = None

        self.measures: DependenceMeasures = None
        self.context: ClusteringContext = None
        self.anonymization_params: AnonymizationParams = None
        self.bucketization_params: BucketizationParams = None

        self._make_paths(path_to_dir)
    
    def _clustering_context(self, comb: tuple[int]) -> ClusteringContext:
        dependency_matrix = _shrink_matrix(self.measures.dependency_matrix, comb)
        entropy_1dim = _shrink_entropy_1dim(self.measures.entropy_1dim, comb)
        total_per_column = [
            sum(dependency_matrix[i, j] for j in range(len(comb)) if i != j)
            for i in range(len(comb))
        ]
        return ClusteringContext(dependency_matrix=dependency_matrix,
                                    entropy_1dim=entropy_1dim,
                                    total_dependence_per_column=total_per_column,
                                    total_dependence=sum(total_per_column),
                                    anonymization_params=self.anonymization_params,
                                    bucketization_params=self.bucketization_params,
                                    rng=self.unsafe_rng,
                                    main_column=None,
        )


    def _make_paths(self, path_to_dir) -> None:
        if path_to_dir is None:
            self.path_to_dir = Path.cwd()
        else:
            # Try to convert path_to_dir to a Path object
            try:
                if isinstance(path_to_dir, str) or isinstance(path_to_dir, Path):
                    self.path_to_dir = Path(path_to_dir)
                else:
                    raise ValueError("path_to_dir must be a string or a Path object")
            except Exception as e:
                raise ValueError(f"Failed to convert path_to_dir to a Path object: {e}") from e
        if not self.path_to_dir.is_dir():
            raise NotADirectoryError(f"The path {self.path_to_dir} is not an existing directory.")
        self.path_to_blob_dir = self.path_to_dir.joinpath(f".sdx_blob_{self.blob_name}")
        if self.path_to_blob_dir.exists():
            raise FileExistsError(f"Something already exists at temporary working directory {self.path_to_blob_dir}.")
        try:
            self.path_to_blob_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            raise OSError(f"Failed to create directory {self.path_to_blob_dir}: {e}") from e
        
        
    def _data_filename(self, columns: list[str]) -> str:
        columns.sort()
        name = self.blob_name + f".col{len(columns)}."
        if len(columns) <= 10:
            chars_per_col = int(30 / len(columns)) + 1
            for col in columns:
                # substitute whitespace with underscore
                col = col.replace(" ", "_")
                name += col[:chars_per_col] + "_"
            # strip off the last underscore
            name = name[:-1]
        seed = self.blob_name + "__".join(columns)
        random.seed(seed)
        # append random alphanumeric characters to ensure that file name is unique
        name += "." + "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return name

    def _zip_blob(self):
        zip_file_path = self.path_to_dir.joinpath(f"{self.blob_name}.zip")
        shutil.make_archive(zip_file_path.with_suffix(''), 'zip', self.path_to_blob_dir)


    def _unzip_blob(self):
        zip_file_path = self.path_to_dir.joinpath(f"{self.blob_name}.zip")
        shutil.unpack_archive(str(zip_file_path), str(self.path_to_blob_dir), 'zip')


class SyndiffixBlobWriter(SyndiffixBlob):
    def __init__(self,
                 blob_name: str,
                 path_to_dir: Optional[Union[str, Path]] = None,
                 sample_size: Optional[int] = 1000,
                 max_weight: Optional[float] = 15.0,
                 merge_threshold: Optional[float] = 0.1,
                 solver_alpha: Optional[float] = 1e-2,
                 ) -> None:
        super().__init__(blob_name, path_to_dir)  # Call the base class initializer
        self.write_version: int = 1
        self.sample_size = sample_size
        self.max_weight = max_weight
        self.merge_threshold = merge_threshold
        self.solver_alpha = solver_alpha

        self.df_raw: pd.DataFrame = None
        self.pids: pd.DataFrame = None
        self.measures: DependenceMeasures = None
        self.context: ClusteringContext = None
        self.anonymization_params: AnonymizationParams = None
        self.bucketization_params: BucketizationParams = None


    def write_blob(
        self,
        df_raw: pd.DataFrame,
        pids: Optional[pd.DataFrame] = None,
    ) -> None:
        '''
            Parameters:
                df_raw: The raw data to be synthesized in a blob
                blob_name: The root name of the generated blob (will add version and .zip)
                pids: The PIDs to be used in the synthesis
                path_to_dir: The place to put the blob. Defaults to current directory
                sample_size, max_weight, merge_threshold, and solver_alpha control clustering
        '''
        self.df_raw = df_raw
        self.pids = pids

        self._write_version()
        # Build a synthesizer that synthesizes each column independently of the others
        syn = Synthesizer(self.df_raw, pids=self.pids, clustering=NoClustering())
        self.anonymization_params = syn.forest.anonymization_params
        self._write_anonymization_params()
        self.bucketization_params = syn.forest.bucketization_params
        self._write_bucketization_params()
        self._write_cluster_params()
        df_syn_no_clustering = syn.sample()
        # Save all of the 1-column tables
        for column in df_syn_no_clustering.columns:
            df_1col = df_syn_no_clustering[[column]]
            self._parquet_writer(df_1col, f"{self._data_filename([column])}.parquet")
        col_names_all = syn.forest.columns
        # Build all the 2-dim tables
        for comb in combinations(range(len(col_names_all)), 2):
            # By forcing the clusters to be two columns only, we trick syn.smaple() into
            # building exactly the 2dim table
            syn.clusters = Clusters(initial_cluster=[ColumnId(comb[0]), ColumnId(comb[1])],
                                    derived_clusters=[],
            )
            df_2col = syn.sample()
            self._parquet_writer(df_2col, f"{self._data_filename(list(df_2col.columns))}.parquet")
        self.measures = measure_all(syn.forest)
        self._write_measures()
        # Build 3-dim and larger tables, so long as no clusters are formed
        for comb in combinations(range(len(col_names_all)), 3):
            # By forcing the clusters to be two columns only, we trick syn.smaple() into
            # building exactly the 2dim table
            syn.clusters = Clusters(initial_cluster=[ColumnId(comb[0]), ColumnId(comb[1]), ColumnId(comb[2])],
                                    derived_clusters=[],
            )
            df_3col = syn.sample()
            self._parquet_writer(df_3col, f"{self._data_filename(list(df_3col.columns))}.parquet")
            self._build_larger_tables(syn, comb, len(col_names_all)-1)
        # zip up the blob
        self._zip_blob()
        shutil.rmtree(self.path_to_blob_dir)

    def _build_larger_tables(self, syn: Synthesizer, comb: tuple[int], maxval: int) -> bool:
        next_start = max(comb) + 1
        if next_start > maxval:
            return
        for i in range(next_start, maxval+1):
            new_comb = comb + (i,)
            context = self._clustering_context(new_comb)
            clusters = _do_solve(context, self.max_weight, self.merge_threshold, self.solver_alpha)
            if len(clusters.derived_clusters) == 0:
                # There is no clustering here, so we want to generate the table and try larger tables
                # _do_solve has no notion of column indexes. It operates as though the column IDs
                # are in the range 0 to len(new_comb)-1. So we need to convert the column IDs to those
                # in new_comb
                syn.clusters = Clusters(initial_cluster=[ColumnId(c) for c in new_comb],
                                        derived_clusters=[],
                )
                df = syn.sample()
                self._parquet_writer(df, f"{self._data_filename(list(df.columns))}.parquet")
                self._build_larger_tables(syn, new_comb, maxval)
                # At this point, we have built any tables made from new_comb or larger, so the
                # tree for new_comb is no longer needed. Delete to save memory.
                del syn.forest._tree_cache[new_comb]

    def _write_version(self) -> None:
        self._json_writer(self.write_version, BlobFiles.VERSION.value)


    def _write_measures(self) -> None:
        dependency_matrix_path = self.path_to_blob_dir.joinpath(BlobFiles.DEPENDENCE_MATRIX.value)
        entropy_1dim_path = self.path_to_blob_dir.joinpath(BlobFiles.ENTROPY_1DIM.value)
        np.save(dependency_matrix_path, self.measures.dependency_matrix)
        np.save(entropy_1dim_path, self.measures.entropy_1dim)

    def _parquet_writer(self, column_df: pd.DataFrame, filename: str):
        file_path = self.path_to_blob_dir.joinpath(filename)
        try:
            column_df.to_parquet(file_path)
        except Exception as e:
            raise IOError(f"Failed to write to Parquet file {file_path}: {e}") from e

    def _json_writer(self, x: Any, filename: str):
        file_path = self.path_to_blob_dir.joinpath(filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(x, file)
        except Exception as e:
            raise IOError(f"Failed to write to file {file_path}: {e}") from e


    def _write_cluster_params(self) -> None:
        cp = {
            'sample_size': self.sample_size,
            'max_weight': self.max_weight,
            'merge_threshold': self.merge_threshold,
            'solver_alpha': self.solver_alpha,
        }
        self._json_writer(cp, BlobFiles.CLUSTER_PARAMS.value)
    
    def _write_bucketization_params(self) -> None:
        bp = {
            'singularity_low_threshold': self.bucketization_params.singularity_low_threshold,
            'range_low_threshold': self.bucketization_params.range_low_threshold,
            'precision_limit_row_fraction': self.bucketization_params.precision_limit_row_fraction,
            'precision_limit_depth_threshold': self.bucketization_params.precision_limit_depth_threshold,
        }
        self._json_writer(bp, BlobFiles.BUCKETIZATION_PARAMS.value)
    
    def _write_anonymization_params(self) -> None:
        anonymization_params = {
            'low_count_params': {
                'low_threshold': self.anonymization_params.low_count_params.low_threshold,
                'layer_sd': self.anonymization_params.low_count_params.layer_sd,
                'low_mean_gap': self.anonymization_params.low_count_params.low_mean_gap,
            },
            'outlier_count':{
                'lower': self.anonymization_params.outlier_count.lower,
                'upper': self.anonymization_params.outlier_count.upper,
            },
            'top_count':{
                'lower': self.anonymization_params.top_count.lower,
                'upper': self.anonymization_params.top_count.upper,
            },
            'layer_noise_sd': self.anonymization_params.layer_noise_sd,
        }
        self._json_writer(anonymization_params, BlobFiles.ANONYMIZATION_PARAMS.value)

class SyndiffixBlobReader(SyndiffixBlob):
    def __init__(self,
                 blob_name: str,
                 path_to_dir: Optional[Union[str, Path]] = None,
                 sample_size: Optional[int] = None,
                 max_weight: Optional[float] = None,
                 merge_threshold: Optional[float] = None,
                 solver_alpha: Optional[float] = None,
                 cache_df_in_memory: Optional[bool] = True,
        ) -> None:
        super().__init__(blob_name, path_to_dir)  # Call the base class initializer
        self.read_version: int = None
        self.cache_df_in_memory = cache_df_in_memory
        self.catalog = {}
        self._unzip_blob()
        self._read_cluster_params()
        # We overwrite the cluster parameters if they are passed in. Otherwise they
        # are set as in the blob
        if sample_size is not None:
            self.sample_size = sample_size
        if max_weight is not None:
            self.max_weight = max_weight
        if merge_threshold is not None:
            self.merge_threshold = merge_threshold
        if solver_alpha is not None:
            self.solver_alpha = solver_alpha

        self.measures: DependenceMeasures = None
        self.context: ClusteringContext = None
        self.anonymization_params: AnonymizationParams = None
        self.bucketization_params: BucketizationParams = None

        self._read_version()
        self._read_measures()
        self._read_anonymization_params()
        self._read_bucketization_params()
        self._load_catalog()
        pass

    def _load_catalog(self) -> None:
        for file_path in self.path_to_blob_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.parquet':
                df = self._parquet_reader(file_path)
                columns = list(df.columns)
                columns.sort()
                columns = tuple(columns)
                if not self.cache_df_in_memory:
                    df = None
                self.catalog[columns] = {'df': df, 'file_path': file_path}

    def _read_version(self) -> None:
        self.read_version = int(self._json_reader(BlobFiles.VERSION.value))

    
    def _read_measures(self) -> None:
        dependency_matrix_path = self.path_to_blob_dir.joinpath(BlobFiles.DEPENDENCE_MATRIX.value)
        entropy_1dim_path = self.path_to_blob_dir.joinpath(BlobFiles.ENTROPY_1DIM.value)
        self.measures = DependenceMeasures(dependency_matrix=np.load(dependency_matrix_path, allow_pickle=True),
                                 entropy_1dim=np.load(entropy_1dim_path, allow_pickle=True))

    def _parquet_reader(self, file_path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise IOError(f"Failed to read Parquet file {file_path}: {e}") from e

    def _json_reader(self, filename: str) -> Any:
        file_path = self.path_to_blob_dir.joinpath(filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            raise IOError(f"Failed to read from file {file_path}: {e}") from e

    def _read_bucketization_params(self) -> None:
        bp = self._json_reader(BlobFiles.BUCKETIZATION_PARAMS.value)
        self.bucketization_params = BucketizationParams(
            singularity_low_threshold=bp['singularity_low_threshold'],
            range_low_threshold=bp['range_low_threshold'],
            precision_limit_row_fraction=bp['precision_limit_row_fraction'],
            precision_limit_depth_threshold=bp['precision_limit_depth_threshold'],
        )

    def _read_cluster_params(self) -> None:
        cp = self._json_reader(BlobFiles.CLUSTER_PARAMS.value)
        self.sample_size = cp['sample_size']
        self.max_weight = cp['max_weight']
        self.merge_threshold = cp['merge_threshold']
        self.solver_alpha = cp['solver_alpha']

    def _read_anonymization_params(self) -> None:
        ap = self._json_reader(BlobFiles.ANONYMIZATION_PARAMS.value)
        self.anonymization_params = AnonymizationParams(
            salt = b"",
            low_count_params=SuppressionParams(
                low_threshold=ap['low_count_params']['low_threshold'],
                layer_sd=ap['low_count_params']['layer_sd'],
                low_mean_gap=ap['low_count_params']['low_mean_gap'],
            ),
            outlier_count=FlatteningInterval(
                lower=ap['outlier_count']['lower'],
                upper=ap['outlier_count']['upper'],
            ),
            top_count=FlatteningInterval(
                lower=ap['top_count']['lower'],
                upper=ap['top_count']['upper'],
            ),
            layer_noise_sd=ap['layer_noise_sd'],
        )

