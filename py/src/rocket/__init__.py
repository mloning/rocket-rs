import numpy as np
from rocket_rs import transform as _transform_rs


def transform(x: np.ndarray, n_kernels: int = 10_000) -> np.ndarray:
    """
    Apply ROCKET transform.

    Parameters
    ----------
    x : np.ndarray
        3-dimensional time-series array with dimensions:
            (n_samples, n_channels, n_timepoints)
    n_kernels : int
        Number of kernels to use.

    Returns
    -------
    np.ndarray
        2-dimensional transformed time-series array with
        dimensions: (n_samples, 2 * n_kernels)
    """
    # check input
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    assert x.ndim == 3
    assert x.shape[0] > 0
    assert x.shape[1] == 1  # only supports a single channel
    assert x.shape[2] > 0
    assert isinstance(n_kernels, int)
    assert n_kernels > 0

    # apply transform
    return _transform_rs(x=x, n_kernels=n_kernels)
