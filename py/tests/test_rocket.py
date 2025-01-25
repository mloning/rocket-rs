import time

import numpy as np
import pytest
from aeon.datasets import load_unit_test
from rocket_rs import Kernel, apply_kernels, generate_kernels, transform
from rocket_rs._utils import _check_array


@pytest.fixture
def x() -> np.ndarray:
    x, _ = load_unit_test(split="train")
    assert isinstance(x, np.ndarray)  # reassure type checker
    _check_array(x, ndim=3, dtype="float64")
    return x.astype("float32")


def test_transform(x: np.ndarray) -> None:
    """Test ROCKET transform."""
    n_kernels = 100
    start = time.perf_counter()
    z = transform(x, n_kernels=n_kernels)
    _elapsed = time.perf_counter() - start
    print(f"rust: {_elapsed:.2f}s")
    _check_array(z, ndim=3, dtype="float32")
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
def test_apply_kernels_return_type(x: np.ndarray, kernels: list[Kernel]) -> None:
    n_kernels = len(kernels)
    z = apply_kernels(x=x, kernels=kernels)

    _check_array(z, ndim=3, dtype="float32")

    # last dimension is number of features
    assert z.shape[:2] == (x.shape[0], n_kernels)


@pytest.mark.parametrize("n_timepoints", [50, 60])
@pytest.mark.parametrize("n_kernels", [3, 10, 13])
def test_generate_kernels_return_type(n_timepoints: int, n_kernels: int) -> None:
    kernels = generate_kernels(n_timepoints=n_timepoints, n_kernels=n_kernels)

    assert isinstance(kernels, list)
    assert len(kernels) == n_kernels


def test_generate_kernels_optional_seed_arg() -> None:
    generate_kernels(n_timepoints=19, n_kernels=1)  # default should be None
    generate_kernels(n_timepoints=21, n_kernels=2, seed=None)
    generate_kernels(n_timepoints=23, n_kernels=3, seed=42)
