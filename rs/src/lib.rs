use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

type Weights = Vec<f32>;

#[derive(Debug, FromPyObject)]
#[pyclass(get_all, frozen)]
struct Kernel {
    len: usize,
    weights: Weights,
    bias: f32,
    dilation: usize,
    padding: usize,
}

#[pymethods]
impl Kernel {
    #[new]
    fn new(len: usize, weights: Weights, bias: f32, dilation: usize, padding: usize) -> Self {
        Kernel {
            len,
            weights,
            bias,
            dilation,
            padding,
        }
    }
}

type Kernels = Vec<Kernel>;

#[pyfunction]
#[pyo3(name = "transform")]
fn transform_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    n_kernels: usize,
) -> &'py PyArray3<f32> {
    let z = transform(x.as_array(), n_kernels);
    z.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "apply_kernels")]
fn apply_kernels_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    kernels: Kernels,
) -> &'py PyArray3<f32> {
    let z = apply_kernels(x.as_array(), &kernels);
    z.into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "generate_kernels")]
fn generate_kernels_py<'py>(_py: Python<'py>, n_timepoints: usize, n_kernels: usize) -> Kernels {
    generate_kernels(n_timepoints, n_kernels)
}
/// ROCKET implemented in Rust
#[pymodule]
fn _rocket_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Kernel>()?;
    m.add_function(wrap_pyfunction!(transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_kernels_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_kernels_py, m)?)?;

    Ok(())
}

/// Rust implementation of ROCKET transform
fn transform(x: ArrayView3<f32>, n_kernels: usize) -> Array3<f32> {
    // println!("x: {:?}", x.shape());
    // println!("n_kernels: {:?}", n_kernels);
    let n_timepoints = x.shape()[2];
    let kernels = generate_kernels(n_timepoints, n_kernels);
    apply_kernels(x, &kernels)
}

fn generate_random_kernel(
    candidate_lengths: [usize; 3],
    weight_distribution: Normal<f32>,
    bias_distribution: Uniform<f32>,
    n_timepoints: usize,
) -> Kernel {
    // TODO refactor to pass in rng to all random functions so that other, seeded rngs can be used
    // for reproducibility
    let mut rng = thread_rng();

    // length
    let len = *candidate_lengths.choose(&mut rng).expect("empty lenghts");

    // dilation
    let high = (((n_timepoints - 1) / (len - 1)) as f32).log2();
    let dilation_dist = Uniform::new(0., high);
    let dilation = 2 * dilation_dist.sample(&mut rng) as usize;

    // weights
    let mut weights = Array1::random_using(len, weight_distribution, &mut rng);
    weights -= weights.mean().expect("no mean");
    let weights = weights.to_vec();

    // bias
    let bias = bias_distribution.sample(&mut rng);

    // padding
    let padding = match rng.gen() {
        true => 0,
        false => (len - 1) * dilation / 2,
    };

    // collect everything into kernel struct
    Kernel {
        len,
        weights,
        bias,
        dilation,
        padding,
    }
}

fn generate_kernels(n_timepoints: usize, n_kernels: usize) -> Kernels {
    let candidate_lengths: [usize; 3] = [7, 9, 11];

    let weigth_distribution = Normal::new(0., 1.).expect("failed normal distribution");
    let bias_distribution = Uniform::new(-1., 1.);

    let mut kernels = Vec::with_capacity(n_kernels);
    // TODO can we not sample first all params outside of the parallel loop?
    (0..n_kernels)
        .into_par_iter()
        .map(|_| {
            generate_random_kernel(
                candidate_lengths,
                weigth_distribution,
                bias_distribution,
                n_timepoints,
            )
        })
        .collect_into_vec(&mut kernels);
    kernels
}

fn get_n_timepoints_out(n_timepoints: usize, kernel: &Kernel) -> usize {
    // println!("{:?}", kernel);
    // println!("{:?}", n_timepoints);
    (n_timepoints + (2 * kernel.padding)) - ((kernel.len - 1) * kernel.dilation)
}

fn apply_kernel(x: ArrayView1<f32>, kernel: &Kernel) -> Array1<f32> {
    let n_timepoints = x.len();
    let n_timepoints_out = get_n_timepoints_out(n_timepoints, kernel) as f32;
    let (start, end) = get_start_end(n_timepoints, kernel);

    let mut ppv = 0.;
    let mut max = f32::NEG_INFINITY;

    for mut i in start..end {
        let mut sum = kernel.bias;
        for j in 0..kernel.len {
            if (i > -1) && (i < n_timepoints as isize) {
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
    array![ppv / n_timepoints_out, max]
}

fn get_start_end(n_timepoints: usize, kernel: &Kernel) -> (isize, isize) {
    let start = -(kernel.padding as isize);
    let end = ((n_timepoints + kernel.padding) - ((kernel.len - 1) * kernel.dilation)) as isize;
    (start, end)
}

/// Apply kernels to time series
fn apply_kernels(x: ArrayView3<f32>, kernels: &Kernels) -> Array3<f32> {
    let n_samples = x.shape()[0];
    let n_kernels = kernels.len();
    let n_features = 2; // depends on `apply_kernel` function
    let mut z = Array3::zeros((n_samples, n_kernels, n_features));

    // we parallelize kernel computation using multi-threading; this create threads for each time
    // series (or row) in the array, and iterates over all kernels inside each thread; this tries
    // to balance thread management overhead with parallel execution; this will work well for larger
    // numbers of longer time series, relative to the number of kernels, but less so for smaller
    // numbers of shorter time series
    //
    // sequential loop
    // for i in 0..n_samples {
    //     for (k, kernel) in kernels.iter().enumerate() {
    //         let features = apply_kernel(x.slice(s![i, 0, ..]), kernel);
    //         y.slice_mut(s![i, k, ..]).assign(&features);
    //     }
    // }
    //
    z.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut yi)| {
            let xi = x.index_axis(Axis(0), i);
            for (k, kernel) in kernels.iter().enumerate() {
                let features = apply_kernel(xi.slice(s![0, ..]), kernel);
                yi.slice_mut(s![k, ..]).assign(&features);
            }
        });
    z
}
