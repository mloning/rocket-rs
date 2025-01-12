# Rust implementation of ROCKET time-series transform with Python bindings

This repo implements the [ROCKET] time-series transform in Rust with Python bindings. Originally, ROCKET is implemented in Python.

[ROCKET]: https://github.com/angus924/rocket

## Usage

Using the [arrow-head dataset] provided by [aeon]:

```python
from aeon.datasets import load_arrow_head
from rocket_rs import transform

x, _ = load_arrow_head(split="train")
z = transform(x)
```

[arrow-head dataset]: https://timeseriesclassification.com/description.php?Dataset=ArrowHead
[aeon]: https://github.com/aeon-toolkit/aeon

## Contributing

### Initial setup

- Install Python, e.g. using [miniforge](https://github.com/conda-forge/miniforge)
- Install [Rust](https://www.rust-lang.org/tools/install)
- Install [maturin](https://www.maturin.rs/)
- Create and activate Python virtual environment (e.g. `conda create -n rocket-rs python=3.11 && conda activate rocket-rs`)
- Run:

```bash
maturin develop --extras=dev
pre-commit install --install-hooks
python -m maturin_import_hook site install --args="--release"
```

### Development

- `maturin develop` to re-build and install the editable development version (`pip install --editable .`)
- `maturin build` to build the Rust package and Python wheel
- `pytest` to rebuild automatically using `maturin_import_hook` and run Python unit tests

### Todo

- [x] separate kernel parameter generation from computation
- [x] generate kernels in Python, pass to Rust function, check results against aeon in pytest
- [ ] flatten nested loop in `apply_kernels`, iterating over kernels and instances jointly
- [ ] compare against Python implementation
- [ ] benchmark & improve

### Notes

- For Python bindings for Rust, see https://github.com/PyO3/pyo3
- https://github.com/PyO3/maturin for packaging and publishing
- To use `poetry` with `maturin`, see https://github.com/PyO3/maturin/discussions/1246#discussioncomment-4047386
- For project layout using both Python and Rust, see https://www.maturin.rs/project_layout.html#mixed-rustpython-project, I preferred having separate folders for Rust and Python
- To generate GitHub Action, run `maturin generate-ci github`
- https://github.com/FL33TW00D/rustDTW
- https://github.com/PyO3/rust-numpy for integration between Python Numpy and Rust ndarray arrays
- https://www.aeon-toolkit.org/en/stable/examples/classification/classification.html
- https://github.com/projectsyn/reclass-rs and https://www.youtube.com/watch?v=N7GMHcX-WdA
- micro-benchmarking with https://github.com/bheisler/criterion.rs
