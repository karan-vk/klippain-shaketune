// Shake&Tune native module
//
// File: welch.rs
// Description: Shared Welch's method primitive used by both the spectrogram (kernel 2) and
//              the Klipper-parity PSD (kernel 4) computations. Produces, for a single 1-D
//              signal column, the per-segment (un-averaged) one-sided power spectral density
//              matrix, using a Kaiser(beta=6.0) window exactly like Klipper/numpy.

use realfft::num_complex::Complex64;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

use crate::kaiser::kaiser;

/// Parameters describing a Welch decomposition of a signal of length `n` sampled at `fs`,
/// with a given (power-of-two) segment length `nperseg` and 50% overlap.
pub struct WelchPlan {
    pub nperseg: usize,
    pub window: Vec<f64>,
    pub window_norm: f64,
    pub step: usize,
    pub n_segments: usize,
    pub n_freqs: usize,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl WelchPlan {
    /// Build a Welch plan for a signal of length `n`, sampling frequency `fs`, and segment
    /// length `nperseg` (must be a power of two, matching Klipper's convention). `overlap` is
    /// nperseg / 2.
    pub fn new(n: usize, fs: f64, nperseg: usize) -> Self {
        let overlap = nperseg / 2;
        let step = nperseg - overlap;
        let n_segments = 1 + (n - nperseg) / step;
        let n_freqs = nperseg / 2 + 1;

        let window = kaiser(nperseg, 6.0);
        let window_sq_sum: f64 = window.iter().map(|w| w * w).sum();
        let window_norm = 1.0 / (fs * window_sq_sum);

        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nperseg);

        WelchPlan {
            nperseg,
            window,
            window_norm,
            step,
            n_segments,
            n_freqs,
            fft,
        }
    }

    /// Compute the per-segment one-sided PSD matrix for signal `x` (must have at least
    /// `nperseg + (n_segments - 1) * step` samples). Returns a flat row-major buffer of shape
    /// (n_segments, n_freqs): segment `i`'s PSD occupies `result[i * n_freqs .. (i+1) * n_freqs]`.
    pub fn segment_psds(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0_f64; self.n_segments * self.n_freqs];
        let mut indata = self.fft.make_input_vec();
        let mut outdata = self.fft.make_output_vec();

        for i in 0..self.n_segments {
            let start = i * self.step;
            let seg = &x[start..start + self.nperseg];

            let mean: f64 = seg.iter().sum::<f64>() / self.nperseg as f64;
            for k in 0..self.nperseg {
                indata[k] = (seg[k] - mean) * self.window[k];
            }

            self.fft
                .process(&mut indata, &mut outdata)
                .expect("fft process should never fail with correctly sized buffers");

            let row = &mut result[i * self.n_freqs..(i + 1) * self.n_freqs];
            for (j, c) in outdata.iter().enumerate() {
                row[j] = psd_value(*c, self.window_norm);
            }
            // Double all bins except DC (index 0) and Nyquist (last index), since nperseg is even.
            if self.n_freqs >= 2 {
                for v in row[1..self.n_freqs - 1].iter_mut() {
                    *v *= 2.0;
                }
            }
        }

        result
    }
}

#[inline]
fn psd_value(c: Complex64, window_norm: f64) -> f64 {
    (c.re * c.re + c.im * c.im) * window_norm
}

/// Compute `1 << bit_length(v)` where `v = int(x)` truncated toward zero (matching Python's
/// `int(x).bit_length()`), i.e. the "round up to a power of two" idiom used throughout Klipper
/// and Shake&Tune. `bit_length(0) == 0` so this returns 1 for `v <= 0`.
pub fn next_pow2_from_truncated(x: f64) -> usize {
    let v = x.trunc() as i64;
    let v = if v < 0 { 0 } else { v as u64 };
    let bits = bit_length(v);
    1usize << bits
}

fn bit_length(v: u64) -> u32 {
    if v == 0 {
        0
    } else {
        64 - v.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_length_matches_python() {
        // Python: (0).bit_length() == 0, (1).bit_length() == 1, (5).bit_length() == 3
        assert_eq!(bit_length(0), 0);
        assert_eq!(bit_length(1), 1);
        assert_eq!(bit_length(5), 3);
        assert_eq!(bit_length(255), 8);
        assert_eq!(bit_length(256), 9);
    }

    #[test]
    fn next_pow2_examples() {
        // int(0.5*1000 - 1.0) = 499, bit_length = 9, 1<<9 = 512
        assert_eq!(next_pow2_from_truncated(499.0), 512);
        // Negative/zero truncation clamps to bit_length(0) = 0 -> 1<<0 = 1
        assert_eq!(next_pow2_from_truncated(-5.0), 1);
    }

    #[test]
    fn welch_plan_basic_shapes() {
        let n = 4096;
        let fs = 1000.0;
        let nperseg = 512;
        let plan = WelchPlan::new(n, fs, nperseg);
        assert_eq!(plan.n_freqs, 257);
        let overlap = nperseg / 2;
        let expected_segments = 1 + (n - nperseg) / (nperseg - overlap);
        assert_eq!(plan.n_segments, expected_segments);

        // Constant signal (after detrending, everything should be ~0 power).
        let x = vec![3.0_f64; n];
        let psds = plan.segment_psds(&x);
        assert_eq!(psds.len(), plan.n_segments * plan.n_freqs);
        for v in psds {
            assert!(v.abs() < 1e-18);
        }
    }
}
