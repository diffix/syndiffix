from abc import ABC, abstractmethod

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


# TODO: Implement GenericAidEntityCounter and GenericAidRowCounter.


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
