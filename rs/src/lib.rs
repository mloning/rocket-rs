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

fn get_n_timepoints_out(n_timepoints: usize, kernel: &Kernel) -> usize {
    (n_timepoints + (2 * kernel.padding)) - ((kernel.len - 1) * kernel.dilation)
}

fn apply_kernel(x: ArrayView1<f64>, kernel: &Kernel) -> Array1<f64> {
    let n_timepoints = x.len();
    let n_timepoints_out = get_n_timepoints_out(n_timepoints, kernel) as f64;
    let (start, end) = get_start_end(&n_timepoints, kernel);

    let mut sum = kernel.bias;
    let mut ppv = 0.;
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
            ppv += 1.;
        }
    }
    array![max, ppv / n_timepoints_out]
}

fn get_start_end(n_timepoints: &usize, kernel: &Kernel) -> (isize, isize) {
    let start = -(kernel.padding as isize);
    let end = ((n_timepoints + kernel.padding) - ((kernel.len - 1) * kernel.dilation)) as isize;
    (start, end)
}

fn apply_kernels(x: ArrayView3<f64>, kernels: Vec<Kernel>) -> Array3<f64> {
    println!("{:?}", kernels.first().expect("empty"));

    let n_samples = x.shape()[0];
    let n_kernels = kernels.len();
    let n_features = 2;
    let mut y = Array3::zeros((n_samples, n_kernels, n_features));

    // TODO profile
    // TODO parallize
    for i in 0..n_samples {
        for (k, kernel) in kernels.iter().enumerate() {
            let features = apply_kernel(x.slice(s![i, 0, ..]), kernel);
            y.slice_mut(s![i, k, ..]).assign(&features);
        }
    }
    y
}
