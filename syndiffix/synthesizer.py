import pandas as pd
from typing import Optional


class Synthesizer(object):
    def __init__(self) -> None:
        pass

    def fit(self, df: pd.DataFrame) -> None:
        self.df = df

    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        if n_samples is not None:
            raise ValueError("n_samples not supported")
        return self.df
