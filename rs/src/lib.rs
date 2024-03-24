use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

/// ROCKET Python module implemented in Rust
#[pymodule]
fn rocket_rs(_py: Python, module: &PyModule) -> PyResult<()> {
    #[pyfn(module)]
    #[pyo3(name = "transform")]
    fn transform_py<'py>(py: Python<'py>, x: PyReadonlyArray3<'py, f64>) -> &'py PyArray3<f64> {
        let z = transform(x.as_array());
        z.into_pyarray(py)
    }

    Ok(())
}

/// Rust implementation of ROCKET transform
fn transform(x: ArrayView3<f64>) -> Array3<f64> {
    println!("Array: {:?}", x);
    x.clone().to_owned()
}
