from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from .anonymizer import *
from .common import *

# If PID values are guaranteed to be unique, we don't need to track distinct values when checking
# whether entities are low-count or apply flattening when counting rows, which greatly reduces memory usage.
# In the generic case, tracking entities requires less memory than tracking row contributions,
# so the two counters are separated into different types.


class IEntityCounter(ABC):
    @abstractmethod
    def add(self, pids: Hashes) -> None:
        pass

    @abstractmethod
    def is_low_count(self, salt: bytes, params: SuppressionParams) -> bool:
        pass


class IRowCounter(ABC):
    @abstractmethod
    def add(self, pids: Hashes) -> None:
        pass

    @abstractmethod
    def noisy_count(self, context: AnonymizationContext) -> int:
        pass


class GenericPidEntityCounter(IEntityCounter):
    def __init__(self, dimensions: int, max_low_count: int) -> None:
        self.pid_sets: list[npt.NDArray[Hash]] = [np.zeros(max_low_count, Hash) for _ in range(dimensions)]
        self.pid_counts: list[int] = [0 for _ in range(dimensions)]
        self.max_low_count = max_low_count

    def add(self, pids: Hashes) -> None:
        for dimension, pid in enumerate(pids):
            pid_set = self.pid_sets[dimension]
            count = self.pid_counts[dimension]
            if count < self.max_low_count and pid != 0 and pid not in pid_set:
                pid_set[count] = pid
                self.pid_counts[dimension] = count + 1

    def is_low_count(self, salt: bytes, params: SuppressionParams) -> bool:
        pid_trackers = [
            (count, seed_from_pid_set(pids))
            for count, pids in zip(self.pid_counts, self.pid_sets)
            if count < self.max_low_count
        ]
        if pid_trackers == []:
            return False
        return is_low_count(salt, params, pid_trackers)


class GenericPidRowCounter(IRowCounter):
    def __init__(self, dimensions: int) -> None:
        self.contributions_list = [PidContributions() for _ in range(dimensions)]

    def add(self, pids: Hashes) -> None:
        for i, pid in enumerate(pids):
            contributions = self.contributions_list[i]
            if pid != 0:
                contributions.value_counts[pid] += 1
            else:
                # Missing PID value, add to the unaccounted rows count, so we can flatten them separately.
                contributions.unaccounted_for += 1

    def noisy_count(self, context: AnonymizationContext) -> int:
        result = count_multiple_contributions(context, self.contributions_list)
        return result.anonymized_count if result is not None else 0


class UniquePidCounter(IEntityCounter, IRowCounter):
    def __init__(self) -> None:
        self.real_count = 0
        self.seed = Hash(0)

    def add(self, pids: Hashes) -> None:
        assert len(pids) == 1

        if pids[0] != 0:
            self.real_count += 1
            self.seed ^= pids[0]

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


class UniquePidCountersFactory(CountersFactory):
    def create_row_counter(self) -> IRowCounter:
        return UniquePidCounter()

    def create_entity_counter(self) -> IEntityCounter:
        return UniquePidCounter()


class GenericPidCountersFactory(CountersFactory):
    def __init__(self, dimensions: int, max_low_count: int) -> None:
        self.dimensions = dimensions
        self.max_low_count = max_low_count

    def create_row_counter(self) -> IRowCounter:
        return GenericPidRowCounter(self.dimensions)

    def create_entity_counter(self) -> IEntityCounter:
        return GenericPidEntityCounter(self.dimensions, self.max_low_count)
