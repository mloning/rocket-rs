# Rust implementation of ROCKET time-series transform with Python bindings

This repo implements the [ROCKET] time-series transform in Rust with Python bindings.

ROCKET is originally implemented in Python, using [Numba] for performance. 
  
[ROCKET]: https://github.com/angus924/rocket
[Numba]: https://numba.pydata.org/

## Usage

```bash
python -c "from rocket import say_hi; say_hi()"
```

## Development

To build the Rust package, run:

```bash
maturin develop

```
To install the Python package, run:

```bash
poetry install 
```

## Notes

* For Python bindings for Rust, see https://github.com/PyO3/pyo3
* To use `poetry` with `maturin`, see https://github.com/PyO3/maturin/discussions/1246#discussioncomment-4047386
* For project layout using both Python and Rust, see https://www.maturin.rs/project_layout.html#mixed-rustpython-project, I preferred having separate folders for Rust and Python
* To generate GitHub Action, run `maturin generate-ci github`
