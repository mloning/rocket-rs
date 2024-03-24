use std::io::Write;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

/// ROCKET Python module implemented in Rust
#[pymodule]
fn rocket_rs(_py: Python, module: &PyModule) -> PyResult<()> {
    #[pyfn(module)]
    #[pyo3(name = "transform")]
    fn transform_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray3<'py, f64>,
        n_kernels: usize,
    ) -> &'py PyArray3<f64> {
        let z = transform(x.as_array(), n_kernels);
        z.into_pyarray(py)
    }

    Ok(())
}

/// Rust implementation of ROCKET transform
fn transform(x: ArrayView3<f64>, n_kernels: usize) -> Array3<f64> {
    println!("x: {:?}", x.shape());
    println!("n_kernels: {:?}", n_kernels);

    let kernels = generate_kernels(n_kernels);
    apply_kernels(x, kernels)
}

#[derive(Debug)]
struct Kernel {
    len: usize,
    weights: Array1<f64>,
    bias: f64,
    dilation: usize,
    padding: usize,
}

fn generate_kernels(n_kernels: usize) -> Vec<Kernel> {
    let mut kernels = Vec::with_capacity(n_kernels);
    // TODO randomize kernel generation
    for _ in 0..n_kernels {
        let kernel = Kernel {
            len: 7,
            weights: Array1::ones(7),
            bias: 0.,
            dilation: 3,
            padding: 3,
        };
        kernels.push(kernel);
    }
    kernels
}

// TODO we need to return params as part of an ndarray
struct Params {
    ppv: usize,
    max: f64,
}

fn apply_kernel(x: ArrayView1<f64>, kernel: &Kernel) -> Params {
    let n_timepoints = x.len();

    let n_timepoints_out =
        (n_timepoints + (2 * kernel.padding)) - ((kernel.len - 1) * kernel.dilation);

    let start = -(kernel.padding as isize);
    let end = compute_end(&n_timepoints, kernel);

    let mut sum = kernel.bias;
    let mut ppv = 0;
    let mut max = 0.;

    for mut i in start..end {
        for j in 0..kernel.len {
            if i > -1 && i < n_timepoints as isize {
                let xi = x[i as usize];
                sum += xi * kernel.weights[j];
            }
            i += kernel.dilation as isize;
        }
        if sum > max {
            max = sum;
        }
        if sum > 0. {
            ppv += 1;
        }
    }

    Params {
        ppv: ppv / n_timepoints_out,
        max,
    }
}

fn compute_end(n_timepoints: &usize, kernel: &Kernel) -> isize {
    let end = (n_timepoints + kernel.padding) - ((kernel.len - 1) * kernel.dilation);
    end as isize
}

fn apply_kernels(x: ArrayView3<f64>, kernels: Vec<Kernel>) -> Array3<f64> {
    // TODO
    println!("{:?}", kernels.first().expect("empty"));

    let n_samples = x.shape()[0];
    let mut params = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        for kernel in &kernels {
            let param = apply_kernel(x.slice(s![i, 0, ..]), kernel);
            params.push(param)
        }
    }

    // TODO remove
    x.clone().to_owned()
}
