import pytest

from syndiffix.range import *


@pytest.mark.parametrize(
    "raw, snapped",
    [
        (Range(1.0, 2.0), Range(1.0, 2.0)),
        (Range(3.0, 7.0), Range(0.0, 8.0)),
        (Range(11.0, 21.0), Range(8.0, 24.0)),
        (Range(11.0, 14.0), Range(10.0, 14.0)),
        (Range(-1.0, 2.0), Range(-2.0, 2.0)),
        (Range(-3.0, -2.0), Range(-3.0, -2.0)),
        (Range(-5.0, -2.0), Range(-6.0, -2.0)),
        (Range(21.0, 23.0), Range(21.0, 23.0)),
        (Range(-6.0, 2.0), Range(-8.0, 8.0)),
        (Range(0.0, 0.0), Range(0.0, 1.0)),
        (Range(0.2, 0.4), Range(0.0, 0.5)),
        (Range(0.01, 0.1), Range(0.0, 0.125)),
        (Range(-1.4, -0.3), Range(-2.0, -0.0)),
        (Range(0.333, 0.78), Range(0.0, 1.0)),
        (Range(0.66, 0.9), Range(0.5, 1.0)),
        (Range(158.88434124351295, 158.94684124353768), Range(158.875, 159.0)),
        (Range(0.0, 1e-17), Range(0.0, 2.0**-56)),
        (Range(0.0, 2.0**-1073), Range(0.0, 2.0**-1073)),
        (Range(0.0, 2.0**-1073 + 2.0**-1074), Range(0.0, 2.0**-1072)),
    ],
)
def test_snapping(raw: Range, snapped: Range) -> None:
    assert snap_range(raw) == snapped
