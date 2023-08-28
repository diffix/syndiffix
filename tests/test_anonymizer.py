from syndiffix.anonymizer import *


def test_is_low_count() -> None:
    assert is_low_count(b"", SuppressionParams(), [(1, Hash(0))])
    assert not is_low_count(b"", SuppressionParams(), [(100, Hash(0))])
    assert is_low_count(b"", SuppressionParams(), [(100, Hash(0)), (1, Hash(0))])
