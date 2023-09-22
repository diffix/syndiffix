from pytest import approx

from syndiffix.anonymizer import *

from .conftest import *


def test_is_low_count() -> None:
    assert is_low_count(b"", SuppressionParams(), [(1, Hash(0))])
    assert not is_low_count(b"", SuppressionParams(), [(100, Hash(0))])
    assert is_low_count(b"", SuppressionParams(), [(100, Hash(0)), (1, Hash(0))])


def test_noisy_row_limit() -> None:
    assert noisy_row_limit(b"", Hash(0), 100, 1) == approx(100, abs=5)
    assert noisy_row_limit(b"", Hash(0), 200, 2) == approx(100, abs=5)
    assert noisy_row_limit(b"", Hash(0), 100, 10000) == 0
    assert noisy_row_limit(b"", Hash(0), 10000, 10000) == 1
    assert noisy_row_limit(b"", Hash(0), 1000000, 10000) == approx(100, abs=5)


def test_hash_strings() -> None:
    assert hash_strings(iter([])) == Hash(0)
    assert hash_strings(iter(["a", "b", "a"])) == hash_strings(iter(["b", "a"]))


def test_hash_aid() -> None:
    assert hash_aid(0) == Hash(0)
    assert hash_aid(None) == Hash(0)
    assert hash_aid("") == Hash(0)


def test_multiple_contributions() -> None:
    def count(contributions_list: list[AidContributions]) -> CountResult | None:
        return count_multiple_contributions(NOISELESS_CONTEXT, contributions_list)

    def contributions(counts: list[int], unaccounted_for: int) -> AidContributions:
        counter: Counter[Hash] = Counter()
        for index, count in enumerate(counts):
            counter[Hash(index)] = count
        return AidContributions(counter, unaccounted_for)

    # insufficient data
    assert count([contributions([], 7)]) is None
    assert count([contributions([7], 0)]) is None
    assert count([contributions([7], 7)]) is None

    # unique values
    assert count([contributions([1] * 10, 0)]) == CountResult(10, 0.0)

    # flattening
    assert count([contributions([1] * 10 + [10], 0)]) == CountResult(11, 0.0)

    # flattening of unaccounted_for
    assert count([contributions([1] * 10 + [10], 10)]) == CountResult(12, 0.0)

    # multiple AIDs
    assert count([contributions([1] * 10, 0), contributions([1], 0)]) is None
    assert count([contributions([1] * 10, 0), contributions([1] * 20, 0)]) == CountResult(20, 0.0)
    assert count([contributions([1] * 20 + [10], 0), contributions([1] * 10 + [20], 0)]) == CountResult(11, 0.0)
