# Rust implementation of ROCKET time-series transform with Python bindings

This repo implements the [ROCKET] time-series transform in Rust with Python bindings. Originally, ROCKET is implemented in Python.

[ROCKET]: https://github.com/angus924/rocket

## Usage

Using the [arrow-head dataset] provided by [aeon]:

```python
from aeon.datasets import load_arrow_head
from rocket import transform

x, _ = load_arrow_head(split="train")
z = transform(x)
```

[arrow-head dataset]: https://timeseriesclassification.com/description.php?Dataset=ArrowHead
[aeon]: https://github.com/aeon-toolkit/aeon

## Development

To build the Rust and Python package, run:

```bash
make build
```

To run the unit tests, run:

```bash
make test
```

## Notes

* For Python bindings for Rust, see https://github.com/PyO3/pyo3
* To use `poetry` with `maturin`, see https://github.com/PyO3/maturin/discussions/1246#discussioncomment-4047386
* For project layout using both Python and Rust, see https://www.maturin.rs/project_layout.html#mixed-rustpython-project, I preferred having separate folders for Rust and Python
* To generate GitHub Action, run `maturin generate-ci github`
* https://github.com/FL33TW00D/rustDTW
* https://github.com/PyO3/rust-numpy for integration between Python Numpy and Rust ndarray arrays
* https://www.aeon-toolkit.org/en/stable/examples/classification/classification.html
