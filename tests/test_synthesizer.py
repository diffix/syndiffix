import pandas as pd

from syndiffix.synthesizer import Synthesizer

from .conftest import *


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=df.columns.to_list()).reset_index(drop=True)


def test_simple() -> None:
    raw_data = pd.DataFrame([["x", True], ["x", False]] * 15)
    synth = Synthesizer(raw_data, anonymization_context=NOISELESS_CONTEXT)
    syn_data = synth.sample()
    pd.testing.assert_frame_equal(_sort(raw_data), _sort(syn_data))
