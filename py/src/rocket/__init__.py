import numpy as np
from rocket_rs import transform as _transform


def transform(X: np.ndarray) -> np.ndarray:
    """
    Apply ROCKET transform.

    Parameters
    ----------
    X : np.ndarray
        3-dimensional time-series array with dimensions:
            (n_samples, n_channels, n_timepoints)

    Returns
    -------
    np.ndarray
        Transformed time-series array
    """
    # check input
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float64
    assert X.ndim == 3
    assert X.shape[0] > 0
    assert X.shape[1] == 1
    assert X.shape[2] > 0

    # apply transform
    return _transform(X)
