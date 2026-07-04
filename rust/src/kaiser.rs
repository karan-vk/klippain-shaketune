// Shake&Tune native module
//
// File: kaiser.rs
// Description: Kaiser window generation matching numpy's `np.kaiser` to within ~1e-13,
//              using the same Cephes-derived polynomial approximation for the modified
//              Bessel function of the first kind, order 0 (i0), that numpy uses internally.

/// Modified Bessel function of the first kind, order 0, computed with the same
/// Chebyshev polynomial approximations (Cephes library) that numpy's `i0` uses.
// The coefficients below are transcribed verbatim (all significant digits) from the Cephes
// `i0.c` source that numpy itself vendors; some digits are beyond f64 precision and are kept
// only for traceability against the reference source.
#[allow(clippy::excessive_precision)]
fn i0(x: f64) -> f64 {
    // Coefficients from Cephes `i0.c`, as used by numpy.i0 / numpy.kaiser.
    const A: [f64; 30] = [
        -4.41534164647933937950e-18,
        3.33079451882223809783e-17,
        -2.43127984654795469359e-16,
        1.71539128555513303061e-15,
        -1.16853328779934516808e-14,
        7.67618549860493561688e-14,
        -4.85644678311192946090e-13,
        2.95505266312963983461e-12,
        -1.72682629144155570723e-11,
        9.67580903537323691224e-11,
        -5.18979560163526290666e-10,
        2.65982372468238665035e-9,
        -1.30002500998624804212e-8,
        6.04699502254191894932e-8,
        -2.67079385394061173391e-7,
        1.11738753912010371815e-6,
        -4.41673835845875056359e-6,
        1.64484480707288970893e-5,
        -5.75419501008210370398e-5,
        1.88502885095841655729e-4,
        -5.76375574538582365885e-4,
        1.63947561694133579842e-3,
        -4.32430999505057594430e-3,
        1.05464603945949983183e-2,
        -2.37374148058994688156e-2,
        4.93052842396707084878e-2,
        -9.49010970480476444210e-2,
        1.71620901522208775349e-1,
        -3.04682672343198398683e-1,
        6.76795274409476084995e-1,
    ];
    const B: [f64; 25] = [
        -7.23318048787475395456e-18,
        -4.83050448594418207126e-18,
        4.46562142029675999901e-17,
        3.46122286769746109310e-17,
        -2.82762398051658348494e-16,
        -3.42548561967721913462e-16,
        1.77256013305652638360e-15,
        3.81168066935262242075e-15,
        -9.55484669882830764870e-15,
        -4.15056934728722208663e-14,
        1.54008621752140982691e-14,
        3.85277838274214270114e-13,
        7.18012445138366623367e-13,
        -1.79417853150680611778e-12,
        -1.32158118404477131188e-11,
        -3.14991652796324136454e-11,
        1.18891471078464383424e-11,
        4.94060238822496958910e-10,
        3.39623202570838634515e-9,
        2.26666899049817806459e-8,
        2.04891858946906374183e-7,
        2.89137052083475648297e-6,
        6.88975834691682398426e-5,
        3.36911647825569408990e-3,
        8.04490411014108831608e-1,
    ];

    // Chebyshev polynomial evaluation, matching Cephes `chbevl` exactly (note the final
    // result uses `b2`, the value of `b1` from *before* the last iteration, not `b1` itself).
    fn chbevl(x: f64, coef: &[f64]) -> f64 {
        let mut b0 = coef[0];
        let mut b1 = 0.0_f64;
        let mut b2 = 0.0_f64;
        for &c in &coef[1..] {
            b2 = b1;
            b1 = b0;
            b0 = x * b1 - b2 + c;
        }
        0.5 * (b0 - b2)
    }

    let ax = x.abs();
    if ax <= 8.0 {
        let y = ax / 2.0 - 2.0;
        chbevl(y, &A) * ax.exp()
    } else {
        let y = 32.0 / ax - 2.0;
        chbevl(y, &B) * ax.exp() / ax.sqrt()
    }
}

/// Kaiser window of length `m` with shape parameter `beta`, matching `np.kaiser(m, beta)`.
pub fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let n = m as f64;
    let alpha = (n - 1.0) / 2.0;
    let i0_beta = i0(beta);
    (0..m)
        .map(|k| {
            let x = (k as f64 - alpha) / alpha;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            i0(arg) / i0_beta
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0_zero_is_one() {
        assert!((i0(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kaiser_endpoints_and_length() {
        let w = kaiser(8, 6.0);
        assert_eq!(w.len(), 8);
        // np.kaiser(8, 6.0) is symmetric.
        for i in 0..w.len() {
            assert!((w[i] - w[w.len() - 1 - i]).abs() < 1e-12);
        }
        // Reference value computed from numpy's exact algorithm: np.kaiser(8, 6.0)[0].
        assert!((w[0] - 0.014_873_337_104_763_207).abs() < 1e-12);
        // Center-ish values near 1.0 for even length (max around center).
        assert!(w[3] > 0.9 && w[3] <= 1.0);
    }

    #[test]
    fn kaiser_single_element() {
        let w = kaiser(1, 6.0);
        assert_eq!(w, vec![1.0]);
    }
}
