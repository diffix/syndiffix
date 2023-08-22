from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, NewType

from ..common import *
from ..microdata import MICRODATA_SYN_VALUE, MicrodataRow

# Global index of a column.
ColumnId = NewType("ColumnId", int)

# Local index of a column in a microtable.
ColumnIndex = NewType("ColumnIndex", int)


@unique
class StitchOwner(Enum):
    LEFT = 0
    RIGHT = 1
    SHARED = 2


# Owner, Stitch columns, Derived columns
DerivedCluster = tuple[StitchOwner, list[ColumnId], list[ColumnId]]


@dataclass
class Clusters:
    initial_cluster: list[ColumnId]
    derived_clusters: list[DerivedCluster]


# TODO
Forest = Any

Combination = list[ColumnId]
TreeMaterializer = Callable[[Forest, list[ColumnId]], tuple[list[MicrodataRow], Combination]]


def microdata_row_to_row(microdata_row: MicrodataRow) -> Row:
    return [value[MICRODATA_SYN_VALUE] for value in microdata_row]
