from pytest import approx

from syndiffix.anonymizer import *


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
