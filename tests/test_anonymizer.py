from syndiffix.anonymizer import *


def test_is_low_count() -> None:
    assert is_low_count(b"", SuppressionParams(), [(1, 0)])
    assert not is_low_count(b"", SuppressionParams(), [(100, 0)])
    assert is_low_count(b"", SuppressionParams(), [(100, 0), (1, 0)])
