from syndiffix.anonymizer import *


def test_is_low_count() -> None:
    assert is_low_count(b"", SuppressionParams(), [(1, 0)]) == True
    assert is_low_count(b"", SuppressionParams(), [(100, 0)]) == False
    assert is_low_count(b"", SuppressionParams(), [(100, 0), (1, 0)]) == True
