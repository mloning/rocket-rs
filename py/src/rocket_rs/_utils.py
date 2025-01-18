import numpy as np


def _check_array(x: np.ndarray, ndim: int, dtype: str | None = None) -> None:
    assert isinstance(x, np.ndarray)
    if dtype:
        assert x.dtype == np.dtype(dtype)
    assert ndim > 0
    assert x.ndim == ndim
    for i in range(ndim):
        assert x.shape[i] > 0
