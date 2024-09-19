import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import shutil

from syndiffix.blob import SyndiffixBlobWriter, SyndiffixBlobReader, _shrink_entropy_1dim, _shrink_matrix

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
current_dir = Path.cwd()

def test_read_blob() -> None:
    blob_test_path = current_dir.joinpath('.sdx_blob_test_blob')
    if blob_test_path.exists() and blob_test_path.is_dir():
        shutil.rmtree(blob_test_path)
    sblob = SyndiffixBlobReader(blob_name='test_blob', path_to_dir=current_dir)
    assert sblob.blob_name == 'test_blob'

def test_bad_params() -> None:
    with pytest.raises(ValueError):
        SyndiffixBlobWriter(blob_name='test_blob', path_to_dir=1)

def test_shrink_matrix():
    N = 5
    matrix = np.array([np.array([i * N + j for j in range(N)]) for i in range(N)], dtype=object)
    comb = (1, 3, 4)
    expected_matrix = np.array([
        [ 6,  8,  9],
        [16, 18, 19],
        [21, 23, 24]
    ])
    shrunk_matrix = _shrink_matrix(matrix, comb)
    assert np.array_equal(shrunk_matrix, expected_matrix), f"Expected {expected_matrix}, but got {shrunk_matrix}"


def test_shrink_entropy_1dim():
    entropy_1dim = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float_)
    comb = (1, 3, 4)
    expected_entropy_1dim = np.array([0.2, 0.4, 0.5], dtype=np.float_)
    new_entropy_1dim = _shrink_entropy_1dim(entropy_1dim, comb)
    assert np.array_equal(new_entropy_1dim, expected_entropy_1dim), f"Expected {expected_entropy_1dim}, but got {new_entropy_1dim}"

def skip_test_overall() -> None:
    #syn = Synthesizer(df_raw)
    blob_test_path = current_dir.joinpath('.sdx_blob_test_blob')
    if blob_test_path.exists() and blob_test_path.is_dir():
        shutil.rmtree(blob_test_path)
    SyndiffixBlobWriter(blob_name='test_blob', path_to_dir=current_dir).write_blob(df_raw=df_raw, pids=None)