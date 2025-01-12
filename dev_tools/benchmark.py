import sys
import time

import numpy as np
from aeon.datasets import load_arrow_head
from rocket_rs import Kernel, apply_kernels


def _check_array(x: np.ndarray, ndim: int) -> None:
    assert isinstance(x, np.ndarray)
    assert ndim > 0
    assert x.dtype == np.float64
    assert x.ndim == ndim
    for i in range(ndim):
        assert x.shape[i] > 0


def run_benchmark() -> None:
    """Run benchmark."""
    x, _ = load_arrow_head(split="train")
    _check_array(x, ndim=3)
    n_kernels = 100_000

    kwargs = {
        "len": 3,
        "weights": [1.0, 2.0, 3.0],
        "bias": 0.5,
        "padding": 2,
        "dilation": 2,
    }
    kernels = [Kernel(**kwargs) for _ in range(n_kernels)]

    start = time.perf_counter()
    z = apply_kernels(x=x, kernels=kernels)
    _elapsed = time.perf_counter() - start

    _check_array(z, ndim=3)
    assert z.shape == (x.shape[0], n_kernels, 2)

    print(f"rust: {_elapsed:.2f}s")


def main() -> None:
    """Main function."""
    print(f"Python version: {sys.version}")
    run_benchmark()


if __name__ == "__main__":
    main()
