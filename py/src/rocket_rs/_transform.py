import numpy as np

from rocket_rs._rocket_rs import transform as _transform_rs
from rocket_rs._utils import _check_array


def transform(x: np.ndarray, n_kernels: int = 10_000, seed: int = 0) -> np.ndarray:
    """
    Apply ROCKET transform.

    Parameters
    ----------
    x : np.ndarray
        3-dimensional time-series array with dimensions:
            (n_samples, n_channels, n_timepoints)
    n_kernels : int
        Number of kernels to use.
    seed : int
        Random number generator seed.

    Returns
    -------
    np.ndarray
        2-dimensional transformed time-series array with
        dimensions: (n_samples, 2 * n_kernels)
    """
    _check_array(x, ndim=3)
    assert x.shape[0] > 0
    assert x.shape[1] == 1  # only supports a single channel
    assert x.shape[2] > 0

    assert isinstance(n_kernels, int)
    assert n_kernels > 0

    x = x.astype("float32")
    return _transform_rs(x=x, n_kernels=n_kernels, seed=seed)
