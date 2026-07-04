// Shake&Tune native module
//
// File: psd.rs
// Description: Kernel 4 - Klipper-parity Welch PSD computation, mirroring Klipper's
//              shaper_calibrate.py::ShaperCalibrate._psd / calc_freq_response exactly
//              (same window, same normalization, full-length output, no max_freq filtering).

use numpy::ndarray::{Array1, ArrayView2};

use crate::welch::{next_pow2_from_truncated, WelchPlan};

pub struct KlipperPsdResult {
    pub freqs: Array1<f64>,
    pub psd_sum: Array1<f64>,
    pub psd_x: Array1<f64>,
    pub psd_y: Array1<f64>,
    pub psd_z: Array1<f64>,
}

/// Compute Klipper-parity Welch-averaged PSDs for raw accelerometer `data` of shape (N, 4)
/// with columns [time, x, y, z]. WINDOW_T_SEC = 0.5 (Klipper constant) is baked into the
/// nperseg formula, matching `int(SAMPLING_FREQ * 0.5 - 1).bit_length()`.
pub fn compute_klipper_psd(data: ArrayView2<f64>) -> Result<KlipperPsdResult, String> {
    let n = data.nrows();
    if n < 2 {
        return Err("data too short for klipper psd".to_string());
    }

    let t0 = data[[0, 0]];
    let t_last = data[[n - 1, 0]];
    let fs = n as f64 / (t_last - t0);

    let m = next_pow2_from_truncated(fs * 0.5 - 1.0);
    if n <= m {
        return Err("data too short for klipper psd".to_string());
    }

    let plan = WelchPlan::new(n, fs, m);
    let n_freqs = plan.n_freqs;
    let n_segments = plan.n_segments;

    let freqs = Array1::from_shape_fn(n_freqs, |j| (j as f64) * fs / (m as f64));

    let mean_psd_for_axis = |axis_idx: usize| -> Array1<f64> {
        let column: Vec<f64> = data.column(axis_idx).iter().copied().collect();
        let psds = plan.segment_psds(&column);
        Array1::from_shape_fn(n_freqs, |j| {
            let mut sum = 0.0_f64;
            for i in 0..n_segments {
                sum += psds[i * n_freqs + j];
            }
            sum / n_segments as f64
        })
    };

    let psd_x = mean_psd_for_axis(1);
    let psd_y = mean_psd_for_axis(2);
    let psd_z = mean_psd_for_axis(3);
    let psd_sum = &psd_x + &psd_y + &psd_z;

    Ok(KlipperPsdResult {
        freqs,
        psd_sum,
        psd_x,
        psd_y,
        psd_z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    #[test]
    fn too_short_errors() {
        let data = Array2::<f64>::zeros((1, 4));
        let res = compute_klipper_psd(data.view());
        assert!(res.is_err());
    }

    #[test]
    fn full_length_output_and_peak_detection() {
        let n = 4000;
        let fs = 1000.0_f64;
        let mut data = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let t = i as f64 / fs;
            data[[i, 0]] = t;
            data[[i, 1]] = (2.0 * std::f64::consts::PI * 80.0 * t).sin();
            data[[i, 2]] = 0.0;
            data[[i, 3]] = 0.0;
        }
        let res = compute_klipper_psd(data.view()).unwrap();
        assert_eq!(res.freqs.len(), res.psd_sum.len());
        assert_eq!(res.freqs.len(), res.psd_x.len());

        let mut max_j = 0;
        let mut max_v = f64::MIN;
        for (j, &v) in res.psd_x.iter().enumerate() {
            if v > max_v {
                max_v = v;
                max_j = j;
            }
        }
        assert!((res.freqs[max_j] - 80.0).abs() < 5.0);

        // psd_sum must equal psd_x + psd_y + psd_z pointwise.
        for j in 0..res.freqs.len() {
            let expected = res.psd_x[j] + res.psd_y[j] + res.psd_z[j];
            assert!((res.psd_sum[j] - expected).abs() < 1e-9);
        }
    }
}
