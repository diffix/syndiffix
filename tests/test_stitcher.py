import numpy as np
import pandas as pd

from syndiffix import Synthesizer
from syndiffix.stitcher import stitch


def make_dataframe(N: int) -> pd.DataFrame:
    a = np.random.randint(100, 200, size=N)
    b = a + np.random.randint(200, 211, size=N)
    c = b + np.random.uniform(2.0, 4.0, size=N)
    d = np.random.choice(["x", "y", "z"], size=N)
    e = np.random.choice(["i", "j", "k"], size=N)
    return pd.DataFrame({"i1": a, "i2": b, "f": c, "t1": d, "t2": e})


df = make_dataframe(500)


def test_left_stitch1() -> None:
    # Make dataframes with a single stitchable shared column (close but not identical)
    df_left = Synthesizer(df[["i1", "i2", "t1"]]).sample()
    df_right = Synthesizer(df[["i1", "f", "t2"]]).sample()
    # Ensure that the dataframes are of different length
    df_right = df_right.sample(n=len(df_left) - 10, random_state=42)
    df_stitched = stitch(df_left=df_left, df_right=df_right, shared=False)
    assert set(df_left["i1"]) == set(df_stitched["i1"])
    if set(df_left["i1"]) != set(df_right["i1"]):
        assert set(df_right["i1"]) != set(df_stitched["i1"])
    assert len(df_left) == len(df_stitched)


def test_left_stitch2() -> None:
    df_left = Synthesizer(df[["i1", "i2", "t1"]]).sample()
    df_right = Synthesizer(df[["i1", "i2", "t2"]]).sample()
    df_left = df_left.sample(n=len(df_right) - 10, random_state=42)
    df_stitched = stitch(df_left=df_left, df_right=df_right, shared=False)
    for i in ["i1", "i2"]:
        assert set(df_left[i]) == set(df_stitched[i])
        if set(df_left[i]) != set(df_right[i]):
            assert set(df_right[i]) != set(df_stitched[i])
    assert len(df_left) == len(df_stitched)


def test_left_stitch1_1() -> None:
    df_left = Synthesizer(df[["i1"]]).sample()
    df_right = Synthesizer(df[["i1", "f", "t2"]]).sample()
    df_right = df_right.sample(n=len(df_left) - 10, random_state=42)
    df_stitched = stitch(df_left=df_left, df_right=df_right, shared=False)
    assert set(df_left["i1"]) == set(df_stitched["i1"])
    if set(df_left["i1"]) != set(df_right["i1"]):
        assert set(df_right["i1"]) != set(df_stitched["i1"])
    assert set(df_right["f"]) != set(df_stitched["f"])
    assert len(df_left) == len(df_stitched)


def test_shared_stitch1() -> None:
    df_left = Synthesizer(df[["i1", "i2", "t1"]]).sample()
    df_right = Synthesizer(df[["i1", "f", "t2"]]).sample()
    df_right = df_right.sample(n=len(df_left) - 10, random_state=42)
    df_stitched = stitch(df_left=df_left, df_right=df_right, shared=True)
    assert len(df_left) > len(df_stitched)
    assert len(df_right) < len(df_stitched)
