import pytest

from syndiffix.interval import *


@pytest.mark.parametrize(
    "raw, snapped",
    [
        (Interval(1.0, 2.0), Interval(1.0, 2.0)),
        (Interval(3.0, 7.0), Interval(0.0, 8.0)),
        (Interval(11.0, 21.0), Interval(8.0, 24.0)),
        (Interval(11.0, 14.0), Interval(10.0, 14.0)),
        (Interval(-1.0, 2.0), Interval(-2.0, 2.0)),
        (Interval(-3.0, -2.0), Interval(-3.0, -2.0)),
        (Interval(-5.0, -2.0), Interval(-6.0, -2.0)),
        (Interval(21.0, 23.0), Interval(21.0, 23.0)),
        (Interval(-6.0, 2.0), Interval(-8.0, 8.0)),
        (Interval(0.0, 0.0), Interval(0.0, 1.0)),
        (Interval(0.2, 0.4), Interval(0.0, 0.5)),
        (Interval(0.01, 0.1), Interval(0.0, 0.125)),
        (Interval(-1.4, -0.3), Interval(-2.0, -0.0)),
        (Interval(0.333, 0.78), Interval(0.0, 1.0)),
        (Interval(0.66, 0.9), Interval(0.5, 1.0)),
        (Interval(158.88434124351295, 158.94684124353768), Interval(158.875, 159.0)),
        (Interval(0.0, 1e-17), Interval(0.0, 2.0**-56)),
        (Interval(0.0, 2.0**-1073), Interval(0.0, 2.0**-1073)),
        (Interval(0.0, 2.0**-1073 + 2.0**-1074), Interval(0.0, 2.0**-1072)),
    ],
)
def test_snapping(raw: Interval, snapped: Interval) -> None:
    assert snap_interval(raw) == snapped

def test_prepares_null_mappings() -> None:
    assert get_null_mapping(Interval(1.2, 1.3)) == 2.6
    assert get_null_mapping(Interval(1.2, 1.2)) == 2.4
    assert get_null_mapping(Interval(-1.0, 3.0)) == 6.0
    assert get_null_mapping(Interval(-4.0, -2.5)) == -8.0
    assert get_null_mapping(Interval(-4.0, -0.0)) == -8.0
    assert get_null_mapping(Interval(0.0, 0.0)) == 1.0