
import numpy as np
import pandas as pd
import polars as pl
from w2v_text_embed import W2VEmbed


def test_w3v():
    w_pandas = pd.DataFrame({
        'word': ["text", "two"],
        'v0': [1, 2],
        'v1': [3, 4],
    })
    w_polars = pl.DataFrame(w_pandas)
    d_list = ["text one", "text two"]
    d_pandas = pd.DataFrame({'x': d_list, 'v0': [1, 2]})
    d_polars = pl.DataFrame(d_pandas)
    d_numpy_1 = np.array(d_list)
    d_numpy = np.asarray(d_pandas)
    expect = pd.DataFrame({0: [1.0, 1.5], 1: [3.0, 3.5]})
    for dict_def in [w_pandas, w_polars]:
        w = W2VEmbed(dict_def)
        for d in [d_list, d_pandas, d_polars, d_numpy_1, d_numpy]:
            res = w.transform(d)
            assert expect.equals(res)

