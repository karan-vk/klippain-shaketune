// Shake&Tune native module
//
// File: vibrations.rs
// Description: Kernel 1 - directional speed spectrogram computation, mirroring
//              shaketune/graph_creators/computations/vibrations_computation.py::
//              VibrationsComputation._compute_dir_speed_spectrogram exactly.

use numpy::ndarray::{Array1, Array2};

const N_ANGLES: usize = 720;

/// numpy-compatible linspace: `step = (b - a) / (n - 1)`, `out[i] = a + i*step` for `i < n-1`,
/// and `out[n-1] = b` exactly (avoids floating point drift on the last element).
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n - 1 {
        out.push(a + i as f64 * step);
    }
    out.push(b);
    out
}

/// `np.searchsorted(speeds, v, side='left')`: the number of elements of the (ascending, sorted)
/// slice `speeds` that are strictly less than `v`.
fn searchsorted_left(speeds: &[f64], v: f64) -> usize {
    speeds.partition_point(|&x| x < v)
}

/// Linear interpolation (with deliberate extrapolation outside the sample range) matching
/// `VibrationsComputation._compute_dir_speed_spectrogram.get_interpolated_vibrations`.
fn interp(vals: &[f64], speeds: &[f64], sp: f64) -> f64 {
    let m = speeds.len();
    let idx = searchsorted_left(speeds, sp).clamp(1, m - 1);
    let lo = speeds[idx - 1];
    let up = speeds[idx];
    let lower_vibrations = vals[idx - 1];
    let upper_vibrations = vals[idx];
    lower_vibrations + (sp - lo) * (upper_vibrations - lower_vibrations) / (up - lo)
}

/// Compute the (spectrum_angles, spectrum_speeds, spectrum_vibrations) triple.
///
/// `measured_speeds`, `vibs_a`, `vibs_b` must all have the same length `m` and `measured_speeds`
/// must be sorted ascending (as produced by the Python caller via `sorted(...)`).
pub fn compute_dir_speed_spectrogram(
    measured_speeds: &[f64],
    vibs_a: &[f64],
    vibs_b: &[f64],
    corexy: bool,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let m = measured_speeds.len();
    let n_speeds = m * 6;

    let spectrum_angles = linspace(0.0, 360.0, N_ANGLES);
    let min_speed = measured_speeds.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_speed = measured_speeds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spectrum_speeds = linspace(min_speed, max_speed, n_speeds);

    let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;

    let cos_vals: Vec<f64> = spectrum_angles.iter().map(|a| (a.to_radians()).cos()).collect();
    let sin_vals: Vec<f64> = spectrum_angles.iter().map(|a| (a.to_radians()).sin()).collect();

    let mut spectrum_vibrations = Array2::<f64>::zeros((N_ANGLES, n_speeds));

    for ai in 0..N_ANGLES {
        let cos_val = cos_vals[ai];
        let sin_val = sin_vals[ai];
        for si in 0..n_speeds {
            let ts = spectrum_speeds[si];
            let (speed_1, speed_2) = if corexy {
                (
                    (ts * (cos_val + sin_val) * sqrt2_inv).abs(),
                    (ts * (cos_val - sin_val) * sqrt2_inv).abs(),
                )
            } else {
                ((ts * cos_val).abs(), (ts * sin_val).abs())
            };

            let v1 = interp(vibs_a, measured_speeds, speed_1);
            let v2 = interp(vibs_b, measured_speeds, speed_2);
            spectrum_vibrations[[ai, si]] = v1 + v2;
        }
    }

    (
        Array1::from_vec(spectrum_angles),
        Array1::from_vec(spectrum_speeds),
        spectrum_vibrations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_numpy_endpoints() {
        let v = linspace(0.0, 360.0, 720);
        assert_eq!(v.len(), 720);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[719], 360.0);
        // step = 360/719
        assert!((v[1] - 360.0 / 719.0).abs() < 1e-12);
    }

    #[test]
    fn linspace_single_point() {
        let v = linspace(5.0, 5.0, 1);
        assert_eq!(v, vec![5.0]);
    }

    #[test]
    fn searchsorted_left_counts_strictly_less() {
        let speeds = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(searchsorted_left(&speeds, 0.5), 0);
        assert_eq!(searchsorted_left(&speeds, 1.0), 0);
        assert_eq!(searchsorted_left(&speeds, 2.5), 2);
        assert_eq!(searchsorted_left(&speeds, 5.0), 4);
    }

    #[test]
    fn interp_extrapolates_below_range() {
        let speeds = [10.0, 20.0, 30.0];
        let vals = [1.0, 2.0, 4.0];
        // sp below min: idx clamps to 1, extrapolates using segment [10,20].
        let v = interp(&vals, &speeds, 0.0);
        // slope = (2-1)/(20-10) = 0.1; v = 1 + (0-10)*0.1 = 0.0
        assert!((v - 0.0).abs() < 1e-12);
        // Exact sample point.
        let v2 = interp(&vals, &speeds, 20.0);
        assert!((v2 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn output_shapes() {
        let speeds = vec![10.0, 20.0, 30.0, 40.0];
        let vibs_a = vec![0.1, 0.2, 0.3, 0.1];
        let vibs_b = vec![0.2, 0.1, 0.2, 0.3];
        let (angles, out_speeds, vib) = compute_dir_speed_spectrogram(&speeds, &vibs_a, &vibs_b, false);
        assert_eq!(angles.len(), 720);
        assert_eq!(out_speeds.len(), speeds.len() * 6);
        assert_eq!(vib.shape(), &[720, speeds.len() * 6]);
    }
}
