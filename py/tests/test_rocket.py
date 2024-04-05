import time

import numpy as np
from aeon.datasets import load_arrow_head
from rocket import transform


def _check_array(x: np.ndarray, ndim: int) -> None:
    assert isinstance(x, np.ndarray)
    assert ndim > 0
    assert x.dtype == np.float64
    assert x.ndim == ndim
    for i in range(ndim):
        assert x.shape[i] > 0


def test_rocket_transform():
    """Test ROCKET transform."""
    x, _ = load_arrow_head(split="train")
    _check_array(x, ndim=3)
    n_kernels = 1_000
    start = time.perf_counter()
    z = transform(x, n_kernels=n_kernels)
    _elapsed = time.perf_counter() - start
    _check_array(z, ndim=3)
    assert z.shape == (x.shape[0], n_kernels, 2)

    # from aeon.transformations.collection.convolution_based import Rocket
    # t = Rocket(num_kernels=500, n_jobs=-1)
    # y = t.fit_transform(x)
    # print(f"{_elapsed:.2f} s")


if __name__ == "__main__":
    test_rocket_transform()
