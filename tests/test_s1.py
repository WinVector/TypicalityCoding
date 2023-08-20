

import numpy as np
from numpy.linalg import norm
import pandas as pd
from sphering_transform import SpheringTransform


def test_w1():
    rng = np.random.default_rng(2023)
    # build some example data
    n_rows = 2000
    d_train = pd.DataFrame({
        'x': rng.normal(size=n_rows),
        'y': rng.normal(size=n_rows) * 1e-2,
    })
    d_test = pd.DataFrame({
        'x': rng.normal(size=n_rows),
        'y': rng.normal(size=n_rows) * 1e-2,
    })
    st = SpheringTransform()
    st.fit(d_train)
    xformed_train = st.transform(d_train)
    # check centered
    assert np.max(np.abs(np.mean(xformed_train, axis=0))) < 1e-6
    # check expectation
    assert np.abs(1 - np.mean(norm(xformed_train, axis=1)**2)) < 1e-5
    xform_test = st.transform(d_test)
    assert np.max(np.abs(np.mean(xform_test, axis=0))) < 0.1
