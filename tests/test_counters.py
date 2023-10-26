import numpy as np

from syndiffix.counters import *

from .conftest import *

MAX_LOW_COUNT = 20


def _hash1(i: int) -> Hashes:
    return np.array([Hash(i)])


def _hash2(i1: int, i2: int) -> Hashes:
    return np.array([Hash(i1), Hash(i2)])


def test_unique_aid_low_count() -> None:
    counter = UniqueAidCountersFactory().create_entity_counter()
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    counter.add(_hash1(1))
    counter.add(_hash1(2))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    counter.add(_hash1(3))
    assert not counter.is_low_count(SALT, NOISELESS_SUPPRESSION)


def test_unique_aid_noisy_count() -> None:
    counter = UniqueAidCountersFactory().create_row_counter()
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash1(1))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 1
    counter.add(_hash1(2))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 2


def test_generic_aid_low_count() -> None:
    counter = GenericAidCountersFactory(1, MAX_LOW_COUNT).create_entity_counter()
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    counter.add(_hash1(1))
    counter.add(_hash1(2))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    # Duplicates - no impact.
    counter.add(_hash1(2))
    counter.add(_hash1(2))
    counter.add(_hash1(2))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    # Null AID - no impact.
    counter.add(_hash1(0))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    counter.add(_hash1(3))
    assert not counter.is_low_count(SALT, NOISELESS_SUPPRESSION)


def test_generic_aid_noisy_count() -> None:
    counter = GenericAidCountersFactory(1, MAX_LOW_COUNT).create_row_counter()
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash1(1))
    counter.add(_hash1(2))
    # Flattening not possible.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash1(3))
    counter.add(_hash1(4))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 4
    counter.add(_hash1(4))
    counter.add(_hash1(4))
    # Flattening.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 4
    counter.add(_hash1(2))
    counter.add(_hash1(3))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 6
    # Null AID - becomes flattened unaccounted for.
    counter.add(_hash1(0))
    counter.add(_hash1(0))
    counter.add(_hash1(0))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 7


def test_multi_aid_low_count() -> None:
    counter = GenericAidCountersFactory(2, MAX_LOW_COUNT).create_entity_counter()
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    counter.add(_hash2(1, 1))
    counter.add(_hash2(2, 2))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    # Duplicates - no impact.
    counter.add(_hash2(2, 1))
    counter.add(_hash2(1, 2))
    counter.add(_hash2(1, 1))
    counter.add(_hash2(2, 2))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    # Null AID - no impact.
    counter.add(_hash2(0, 0))
    counter.add(_hash2(1, 0))
    counter.add(_hash2(0, 1))
    assert counter.is_low_count(SALT, NOISELESS_SUPPRESSION)
    counter.add(_hash2(3, 3))
    assert not counter.is_low_count(SALT, NOISELESS_SUPPRESSION)


def test_multi_aid_noisy_count_identical() -> None:
    # Both AIDs are identical - a simple sanity test.
    counter = GenericAidCountersFactory(2, MAX_LOW_COUNT).create_row_counter()
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash2(1, 1))
    counter.add(_hash2(2, 2))
    # Flattening not possible.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash2(3, 3))
    counter.add(_hash2(4, 4))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 4
    counter.add(_hash2(4, 4))
    counter.add(_hash2(4, 4))
    # Flattening.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 4
    counter.add(_hash2(2, 2))
    counter.add(_hash2(3, 3))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 6
    # Null AID - becomes flattened unaccounted for.
    counter.add(_hash2(0, 0))
    counter.add(_hash2(0, 5))
    counter.add(_hash2(6, 0))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 7


def test_multi_aid_noisy_count_divergent() -> None:
    counter = GenericAidCountersFactory(2, MAX_LOW_COUNT).create_row_counter()
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash2(1, 1))
    counter.add(_hash2(2, 2))
    counter.add(_hash2(2, 1))
    counter.add(_hash2(1, 2))
    # Flattening not possible in both AIDs.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash2(3, 3))
    counter.add(_hash2(3, 4))
    # Flattening not possible in first AID.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 0
    counter.add(_hash2(4, 3))
    counter.add(_hash2(4, 4))
    assert counter.noisy_count(NOISELESS_CONTEXT) == 8
    counter.add(_hash2(4, 4))
    counter.add(_hash2(4, 4))
    # Flattening.
    assert counter.noisy_count(NOISELESS_CONTEXT) == 8
