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
    z = transform(x, n_kernels=10)
    _check_array(z, ndim=3)
    np.testing.assert_array_equal(x, z)

    # from aeon.transformations.collection.convolution_based import Rocket
    # t = Rocket(num_kernels=500, n_jobs=-1)
    # y = t.fit_transform(x)
    # breakpoint()


if __name__ == "__main__":
    test_rocket_transform()
