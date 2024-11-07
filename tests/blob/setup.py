import numpy as np
import pandas as pd


def make_dataframe(N: int = 300) -> pd.DataFrame:
    a = np.random.randint(100, 200, size=N)
    b = a + np.random.randint(200, 211, size=N)
    c = b + np.random.uniform(2.0, 4.0, size=N)
    d = np.random.choice(["x", "y", "z"], size=N)
    e = np.random.choice(["i", "j", "k"], size=N)
    f = np.random.randint(100, 200, size=N)
    g = a + np.random.randint(200, 211, size=N)
    pid = np.arange(1, N + 1)
    return pd.DataFrame(
        {
            "pid": pid,
            "i1": a,
            "i2": b,
            "f1": c,
            "t1": d,
            "t2": e,
            "i3": f,
            "i4": g,
        }
    )


df_raw = make_dataframe()
df_raw.to_parquet("df_raw.parquet")
