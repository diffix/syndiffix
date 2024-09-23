import random
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, NewType, Optional

import numpy.typing as npt

from ..common import *
from ..forest import Forest
from ..microdata import MICRODATA_SYN_VALUE, MicrodataRow

# Local index of a column in a microtable.
ColumnIndex = NewType("ColumnIndex", int)


@unique
class StitchOwner(Enum):
    LEFT = 0
    RIGHT = 1
    SHARED = 2

    def __repr__(self) -> str:
        return self.name


# Owner, Stitch columns, Derived columns
DerivedCluster = tuple[StitchOwner, list[ColumnId], list[ColumnId]]


@dataclass(frozen=True)
class Clusters:
    initial_cluster: list[ColumnId]
    derived_clusters: list[DerivedCluster]


TreeMaterializer = Callable[[Forest, list[ColumnId]], tuple[list[MicrodataRow], Combination]]


def microdata_row_to_row(microdata_row: MicrodataRow) -> Row:
    return tuple(value[MICRODATA_SYN_VALUE] for value in microdata_row)


Entropy1Dim = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ClusteringContext:
    dependency_matrix: npt.NDArray[np.float64]
    entropy_1dim: Entropy1Dim
    total_dependence: float
    total_dependence_per_column: list[float]
    anonymization_params: AnonymizationParams
    bucketization_params: BucketizationParams
    rng: random.Random
    main_column: Optional[ColumnId]

    @property
    def num_columns(self) -> int:
        return len(self.dependency_matrix)


@dataclass(frozen=True)
class StitchingMetadata:
    dimension_is_integral: list[bool]
    entropy_1dim: Entropy1Dim
