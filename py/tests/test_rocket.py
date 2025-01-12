import time

import numpy as np
import pytest
from aeon.datasets import load_arrow_head
from rocket_rs import Kernel, apply_kernels, transform


def _check_array(x: np.ndarray, ndim: int) -> None:
    assert isinstance(x, np.ndarray)
    assert ndim > 0
    assert x.dtype == np.float64
    assert x.ndim == ndim
    for i in range(ndim):
        assert x.shape[i] > 0


@pytest.fixture
def x() -> np.ndarray:
    x, _ = load_arrow_head(split="train")
    _check_array(x, ndim=3)
    return x


def test_transform(x: np.ndarray) -> None:
    """Test ROCKET transform."""
    n_kernels = 100
    start = time.perf_counter()
    z = transform(x, n_kernels=n_kernels)
    _elapsed = time.perf_counter() - start
    print(f"rust: {_elapsed:.2f}s")
    _check_array(z, ndim=3)
    assert z.shape == (x.shape[0], n_kernels, 2)


def test_kernel_init_and_get_attr() -> None:
    kwargs = {
        "len": 3,
        "weights": [1.0, 2.0, 3.0],
        "bias": 0.5,
        "padding": 2,
        "dilation": 2,
    }
    kernel = Kernel(**kwargs)

    assert isinstance(kernel, Kernel)
    for key, value in kwargs.items():
        assert getattr(kernel, key) == value


@pytest.mark.parametrize(
    "kernels",
    [
        # []  TODO handle empty kernels
        [Kernel(len=3, weights=[1.0, 2.0, 3.0], bias=0.5, padding=2, dilation=2)],
        [
            Kernel(len=2, weights=[1.0, 2.0], bias=0.5, padding=2, dilation=2),
            Kernel(len=3, weights=[1.0, 3.0, 5.0], bias=0.1, padding=1, dilation=1),
        ],
        [
            Kernel(len=2, weights=[1.0, 2.0], bias=0.5, padding=2, dilation=2),
            Kernel(len=3, weights=[1.0, 3.0, 5.0], bias=0.1, padding=1, dilation=1),
            Kernel(
                len=4, weights=[1.0, 4.0, 6.0, 1.0], bias=0.3, padding=0, dilation=3
            ),
        ],
    ],
)
def test_apply_kernels(x: np.ndarray, kernels: list[Kernel]) -> None:
    n_kernels = len(kernels)
    z = apply_kernels(x=x, kernels=kernels)

    _check_array(z, ndim=3)

    # last dimension is number of features
    assert z.shape[:2] == (x.shape[0], n_kernels)
