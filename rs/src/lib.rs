use pyo3::prelude::*;

/// ROCKET Python module implemented in Rust
#[pymodule]
fn rocket_rs(_py: Python, module: &PyModule) -> PyResult<()> {
    // export Python functions
    module.add_function(wrap_pyfunction!(say_hi, module)?)?;

    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn say_hi(name: &str) -> PyResult<String> {
    Ok(format!("Hello {}!", name))
}
