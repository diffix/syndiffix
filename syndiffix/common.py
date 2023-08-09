from enum import Enum, unique
from dataclasses import dataclass, field

Hash = int
Value = int | float | str | bool | None
Row = list[Value]


@unique
class ColumnType(Enum):
    BOOLEAN = 1
    INTEGER = 2
    REAL = 3
    STRING = 4
    TIMESTAMP = 5


@dataclass
class Column:
    name: str
    type: ColumnType


Columns = list[Column]


@dataclass
class FlatteningInterval:
    lower: int = 2
    upper: int = 5


@dataclass
class SuppressionParams:
    low_threshold: int = 3
    layer_sd: float = 1.0
    low_mean_gap: float = 2.0


@dataclass
class BucketizationParams:
    singularity_low_threshold: int = 5
    range_low_threshold: int = 15
    clustering_enabled: bool = True
    clustering_table_sample_size: int = 1000
    clustering_max_cluster_weight: float = 15.0
    clustering_merge_threshold: float = 0.1
    precision_limit_row_fraction = 10000
    precision_limit_depth_threshold = 15


@dataclass
class AnonymizationParams:
    aid_columns: list[str] = field(default_factory=list)
    salt: bytes = b""
    low_count_params: SuppressionParams = SuppressionParams()
    outlier_count: FlatteningInterval = FlatteningInterval()
    top_count: FlatteningInterval = FlatteningInterval()
    layer_noise_sd: float = 1.0


@dataclass
class AnonymizationContext:
    bucket_seed: Hash
    anonymization_params: AnonymizationParams
