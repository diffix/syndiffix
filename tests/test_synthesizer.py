import numpy as np
import pandas as pd

from syndiffix.synthesizer import Synthesizer


def test_test() -> None:
    s = Synthesizer()
    df = pd.DataFrame(np.random.randn(2, 2))
    s.fit(df)
    pd.testing.assert_frame_equal(df, s.sample())
