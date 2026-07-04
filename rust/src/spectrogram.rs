// Shake&Tune native module
//
// File: spectrogram.rs
// Description: Kernel 2 - power spectral density spectrogram computation, mirroring
//              shaketune/helpers/spectrogram.py::compute_spectrogram exactly.

use numpy::ndarray::{Array1, Array2, ArrayView2};

use crate::welch::{next_pow2_from_truncated, WelchPlan};

/// Output of [`compute_spectrogram`]: (pdata, t, f).
pub type SpectrogramOutput = (Array2<f64>, Array1<f64>, Array1<f64>);

/// Compute the (pdata, t, f) spectrogram triple from raw accelerometer `data` of shape (N, 4)
/// with columns [time, x, y, z].
pub fn compute_spectrogram(data: ArrayView2<f64>) -> Result<SpectrogramOutput, String> {
    let n = data.nrows();
    if n < 2 {
        return Err("Not enough data samples".to_string());
    }

    let t0 = data[[0, 0]];
    let t_last = data[[n - 1, 0]];
    let fs = n as f64 / (t_last - t0);

    let nperseg = next_pow2_from_truncated(0.5 * fs - 1.0);
    if n < nperseg {
        return Err(format!("Input data too short for nperseg={nperseg}"));
    }

    let plan = WelchPlan::new(n, fs, nperseg);
    let n_segments = plan.n_segments;
    let n_freqs = plan.n_freqs;
    let step = plan.step;

    // Time and frequency arrays.
    let t = Array1::from_shape_fn(n_segments, |i| (i as f64) * (step as f64) / fs + (nperseg as f64) / (2.0 * fs));
    let f = Array1::from_shape_fn(n_freqs, |j| (j as f64) * fs / (nperseg as f64));

    let mut pdata = Array2::<f64>::zeros((n_freqs, n_segments));

    for axis_idx in [1usize, 2, 3] {
        let column: Vec<f64> = data.column(axis_idx).iter().copied().collect();
        let psds = plan.segment_psds(&column);
        for i in 0..n_segments {
            let row = &psds[i * n_freqs..(i + 1) * n_freqs];
            for j in 0..n_freqs {
                pdata[[j, i]] += row[j];
            }
        }
    }

    Ok((pdata, t, f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2 as NdArray2;

    #[test]
    fn too_few_samples_errors() {
        let data = NdArray2::<f64>::zeros((1, 4));
        let res = compute_spectrogram(data.view());
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Not enough data samples");
    }

    #[test]
    fn produces_expected_shapes() {
        // Build a synthetic 2000-sample signal sampled at 1000 Hz containing a 50Hz sine on X.
        let n = 4000;
        let fs = 1000.0_f64;
        let mut data = NdArray2::<f64>::zeros((n, 4));
        for i in 0..n {
            let t = i as f64 / fs;
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * 50.0 * t).sin();
            data[[i, 2]] = 0.0;
            data[[i, 3]] = 0.0;
        }
        let (pdata, t, f) = compute_spectrogram(data.view()).unwrap();
        assert_eq!(pdata.nrows(), f.len());
        assert_eq!(pdata.ncols(), t.len());
        assert!(!t.is_empty());
        // Peak frequency bin should be near 50Hz.
        let mut max_j = 0;
        let mut max_v = f64::MIN;
        for j in 0..f.len() {
            let col_sum: f64 = pdata.row(j).sum();
            if col_sum > max_v {
                max_v = col_sum;
                max_j = j;
            }
        }
        assert!((f[max_j] - 50.0).abs() < 5.0);
    }
}
