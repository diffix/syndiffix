from abc import ABC, abstractmethod

from pybloom_live import BloomFilter  # type: ignore

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
    def __init__(self, dimensions: int, estimated_count: int) -> None:
        self.aid_sets = [BloomFilter(capacity=estimated_count, error_rate=0.02) for _ in range(dimensions)]
        self.counts = [0] * dimensions
        self.seeds = [Hash(0)] * dimensions

    def add(self, aids: Hashes) -> None:
        for i, aid in enumerate(aids):
            if aid != 0 and aid not in self.aid_sets[i]:
                self.aid_sets[i].add(aid)
                self.counts[i] += 1
                self.seeds[i] ^= aid

    def is_low_count(self, salt: bytes, params: SuppressionParams) -> bool:
        return is_low_count(salt, params, list(zip(self.counts, self.seeds)))


class GenericAidRowCounter(IRowCounter):
    def __init__(self, dimensions: int) -> None:
        self.contributions = [Contributions() for _ in range(dimensions)]

    def add(self, aids: Hashes) -> None:
        for i, aid in enumerate(aids):
            contribution = self.contributions[i]
            if aid != 0:
                contribution.per_aid[aid] += 1
            else:
                # Missing AID value, add to the unaccounted rows count, so we can flatten them separately.
                contribution.unaccounted_for += 1

    def noisy_count(self, context: AnonymizationContext) -> int:
        result = count_multiple_contributions(context, self.contributions)
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
    def __init__(self, dimensions: int, estimated_count: int) -> None:
        self.dimensions = dimensions
        self.estimated_count = estimated_count

    def create_row_counter(self) -> IRowCounter:
        return GenericAidRowCounter(self.dimensions)

    def create_entity_counter(self) -> IEntityCounter:
        return GenericAidEntityCounter(self.dimensions, self.estimated_count)
