from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from .anonymizer import *
from .common import *

# If AID values are guaranteed to be unique, we don't need to track distinct values when checking
# whether entities are low-count or apply flattening when counting rows, which greatly reduces memory usage.
# In the generic case, tracking entities requires less memory than tracking row contributions,
# so the two counters are separated into different types.


class IEntityCounter(ABC):
    @abstractmethod
    def add(self, aids: Hashes) -> None:
        pass

    @abstractmethod
    def is_low_count(self, salt: bytes, params: SuppressionParams) -> bool:
        pass


class IRowCounter(ABC):
    @abstractmethod
    def add(self, aids: Hashes) -> None:
        pass

    @abstractmethod
    def noisy_count(self, context: AnonymizationContext) -> int:
        pass


class GenericAidEntityCounter(IEntityCounter):
    def __init__(self, dimensions: int, max_low_count: int) -> None:
        self.aid_sets: list[npt.NDArray[Hash]] = [np.zeros(max_low_count, Hash) for _ in range(dimensions)]
        self.aid_counts: list[int] = [0 for _ in range(dimensions)]
        self.max_low_count = max_low_count

    def add(self, aids: Hashes) -> None:
        for dimension, aid in enumerate(aids):
            aid_set = self.aid_sets[dimension]
            count = self.aid_counts[dimension]
            if count < self.max_low_count and aid != 0 and aid not in aid_set:
                aid_set[count] = aid
                self.aid_counts[dimension] = count + 1

    def is_low_count(self, salt: bytes, params: SuppressionParams) -> bool:
        aid_trackers = [
            (count, seed_from_aid_set(aids))
            for count, aids in zip(self.aid_counts, self.aid_sets)
            if count < self.max_low_count
        ]
        if aid_trackers == []:
            return False
        return is_low_count(salt, params, aid_trackers)


class GenericAidRowCounter(IRowCounter):
    def __init__(self, dimensions: int) -> None:
        self.contributions_list = [AidContributions() for _ in range(dimensions)]

    def add(self, aids: Hashes) -> None:
        for i, aid in enumerate(aids):
            contributions = self.contributions_list[i]
            if aid != 0:
                contributions.value_counts[aid] += 1
            else:
                # Missing AID value, add to the unaccounted rows count, so we can flatten them separately.
                contributions.unaccounted_for += 1

    def noisy_count(self, context: AnonymizationContext) -> int:
        result = count_multiple_contributions(context, self.contributions_list)
        return result.anonymized_count if result is not None else 0


class UniqueAidCounter(IEntityCounter, IRowCounter):
    def __init__(self) -> None:
        self.real_count = 0
        self.seed = Hash(0)

    def add(self, aids: Hashes) -> None:
        assert len(aids) == 1

        if aids[0] != 0:
            self.real_count += 1
            self.seed ^= aids[0]

    def is_low_count(self, salt: bytes, params: SuppressionParams) -> bool:
        return is_low_count(salt, params, [(self.real_count, self.seed)])

    def noisy_count(self, context: AnonymizationContext) -> int:
        return count_single_contributions(context, self.real_count, self.seed)


class CountersFactory(ABC):
    @abstractmethod
    def create_entity_counter(self) -> IEntityCounter:
        pass

    @abstractmethod
    def create_row_counter(self) -> IRowCounter:
        pass


class UniqueAidCountersFactory(CountersFactory):
    def create_row_counter(self) -> IRowCounter:
        return UniqueAidCounter()

    def create_entity_counter(self) -> IEntityCounter:
        return UniqueAidCounter()


class GenericAidCountersFactory(CountersFactory):
    def __init__(self, dimensions: int, max_low_count: int) -> None:
        self.dimensions = dimensions
        self.max_low_count = max_low_count

    def create_row_counter(self) -> IRowCounter:
        return GenericAidRowCounter(self.dimensions)

    def create_entity_counter(self) -> IEntityCounter:
        return GenericAidEntityCounter(self.dimensions, self.max_low_count)
