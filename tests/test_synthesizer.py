import pandas as pd
import numpy as np

from syndiffix.synthesizer import Synthesizer


def test_test() -> None:
    s = Synthesizer()
    df = pd.DataFrame(np.random.randn(2, 2))
    s.fit(df)
    pd.testing.assert_frame_equal(df, s.sample())
