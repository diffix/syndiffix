import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from syndiffix.blob import SyndiffixBlobWriter

def make_dataframe(N: int) -> pd.DataFrame:
    a = np.random.randint(100, 200, size=N)
    b = a + np.random.randint(200, 211, size=N)
    c = b + np.random.uniform(2.0, 4.0, size=N)
    d = np.random.choice(["x", "y", "z"], size=N)
    e = np.random.choice(["i", "j", "k"], size=N)
    f = np.random.randint(100, 200, size=N)
    g = a + np.random.randint(200, 211, size=N)
    h = b + np.random.uniform(2.0, 4.0, size=N)
    i = np.random.choice(["x", "y", "z"], size=N)
    j = np.random.choice(["i", "j", "k"], size=N)
    pid = np.arange(1, N + 1)
    return pd.DataFrame({"pid":pid, "i1": a, "i2": b, "f1": c, "t1": d, "t2": e,
                         "i3": f, "i4": g, "f2": h, "t3": i, "t4": j})


df_raw = make_dataframe(300)
tests_dir = Path.cwd().joinpath('tests')
blob_test_path = tests_dir.joinpath('.sdx_blob_test_blob')

blob_test_path = tests_dir.joinpath('.sdx_blob_test_blob')
if blob_test_path.exists() and blob_test_path.is_dir():
    shutil.rmtree(blob_test_path)
SyndiffixBlobWriter(blob_name='test_blob', path_to_dir=tests_dir).write_blob(df_raw=df_raw, pids=None)