from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from aeon.transformations.collection.convolution_based._rocket import (
    _apply_kernels as _apply_kernels_aeon,
)
from pytest_benchmark.fixture import BenchmarkFixture
from rocket_rs import Kernel, apply_kernels, generate_kernels
from rocket_rs._utils import _check_array

SEED = 42


@pytest.fixture
def x() -> np.ndarray:
    # x, _ = load_unit_test(split="train")
    # x, _ = load_arrow_head(split="train")
    x = np.random.normal(size=(1_000, 1, 1_000))
    assert isinstance(x, np.ndarray)  # reassure type checker
    _check_array(x, ndim=3, dtype="float64")
    return x.astype("float32")


_KernelParams = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def _convert_kernels_aeon(kernels: list[Kernel]) -> _KernelParams:
    # in aeon kernels are not represented as a list of some struct, but instead as
    # list of arrays, one for each kernel parameter; we here convert between the
    # representations, to be able to compare the results of kernel computation for
    # the same input kernels
    n_kernels = len(kernels)
    n_weights = sum([kernel.len for kernel in kernels])

    weights = np.zeros(n_weights, dtype=np.float32)
    lengths = np.zeros(n_kernels, dtype=np.int32)
    biases = np.zeros(n_kernels, dtype=np.float32)
    dilations = np.zeros(n_kernels, dtype=np.int32)
    paddings = np.zeros(n_kernels, dtype=np.int32)

    a = 0
    for i, kernel in enumerate(kernels):
        lengths[i] = kernel.len
        b = a + kernel.len
        weights[a:b] = kernel.weights
        biases[i] = kernel.bias
        dilations[i] = kernel.dilation
        paddings[i] = kernel.padding
        a = b

    n_channel_indices = np.ones(n_kernels, dtype=np.int32)
    channel_indices = np.zeros(np.sum(n_channel_indices), dtype=np.int32)
    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        n_channel_indices,
        channel_indices,
    )


def apply_kernels_aeon(
    x: np.ndarray,
    kernels: list[Kernel],
) -> np.ndarray:
    n_samples = x.shape[0]
    n_kernels = len(kernels)
    kernel_kwargs = _convert_kernels_aeon(kernels=kernels)
    out = _apply_kernels_aeon(X=x, kernels=kernel_kwargs)
    return out.reshape(n_samples, n_kernels, -1)


@pytest.mark.parametrize(
    "generate_kernels_func",
    [
        partial(generate_kernels, n_kernels=1, seed=SEED),
        partial(generate_kernels, n_kernels=3, seed=SEED),
        partial(generate_kernels, n_kernels=13, seed=SEED),
        partial(generate_kernels, n_kernels=35, seed=SEED),
    ],
)
def test_apply_kernels_against_aeon_random_kernels(
    x: np.ndarray, generate_kernels_func: Callable
) -> None:
    n_timepoints = x.shape[-1]
    kernels = generate_kernels_func(n_timepoints=n_timepoints)
    a = apply_kernels(x=x, kernels=kernels)
    b = apply_kernels_aeon(x=x, kernels=kernels)
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "kernels",
    [
        # single kernel
        [Kernel(2, [2.5, 3.1], 0.2, 1, 3)],
        [Kernel(3, [1.0, 2.0, 3.0], 0.5, 2, 2)],
        [Kernel(5, [1.0, 2.0, 3.0, 2.0, 0.5], 1, 1, 1)],
        # multiple kernels
        [
            Kernel(2, [2.5, 3.1], 0.2, 1, 3),
            Kernel(5, [1.1, 2.1, 3.3, 2.2, 1.5], 3, 2, 5),
        ],
        [
            Kernel(2, [2.5, 3.1], 0.2, 1, 3),
            Kernel(5, [1.2, 2.0, 3.2, 2.0, 0.5], 2, 3, 2),
            Kernel(3, [1.0, 2.0, 7.0], 0.7, 2, 2),
        ],
    ],
)
def test_apply_kernels_against_aeon_fixed_kernels(
    x: np.ndarray, kernels: list[Kernel]
) -> None:
    a = apply_kernels(x=x, kernels=kernels)
    b = apply_kernels_aeon(x=x, kernels=kernels)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_apply_kernels_benchmark(x: np.ndarray, benchmark: BenchmarkFixture) -> None:
    n_timepoints = x.shape[-1]
    kernels = generate_kernels(n_kernels=1_000, n_timepoints=n_timepoints)
    benchmark(apply_kernels, x=x, kernels=kernels)


def test_apply_kernels_benchmark_aeon(
    x: np.ndarray, benchmark: BenchmarkFixture
) -> None:
    n_timepoints = x.shape[-1]
    kernels = generate_kernels(n_kernels=1_000, n_timepoints=n_timepoints, seed=SEED)
    kernels_aeon = _convert_kernels_aeon(kernels=kernels)
    benchmark(_apply_kernels_aeon, X=x, kernels=kernels_aeon)
