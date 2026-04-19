//! Small numerical functions shared across scattering kernels.
//!
//! Mirrors `c/funcs.c`.  Frequencies and temperatures are in THz units
//! throughout.

/// Bose-Einstein occupation `1 / (exp(x / T) - 1)` for `x`, `T` in
/// consistent (THz) units.  Matches `funcs_bose_einstein` in
/// `c/funcs.c`.
#[inline]
pub(crate) fn bose_einstein(x: f64, temperature_thz: f64) -> f64 {
    1.0 / ((x / temperature_thz).exp() - 1.0)
}

/// Gaussian normalized so that its integral is 1.  Matches
/// `funcs_gaussian` in `c/funcs.c`.
#[inline]
pub(crate) fn gaussian(x: f64, sigma: f64) -> f64 {
    (-(x * x) / (2.0 * sigma * sigma)).exp() / ((2.0 * std::f64::consts::PI).sqrt() * sigma)
}

/// `1 / sinh(x / (2 T))` for `x`, `T` in consistent (THz) units.
/// Matches `funcs_inv_sinh_occupation` in `c/funcs.c`.
#[inline]
pub(crate) fn inv_sinh_occupation(x: f64, temperature_thz: f64) -> f64 {
    1.0 / (x / (2.0 * temperature_thz)).sinh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bose_einstein_basic() {
        // At T = x: n = 1 / (e - 1).
        let n = bose_einstein(1.0, 1.0);
        let expected = 1.0 / (std::f64::consts::E - 1.0);
        assert!((n - expected).abs() < 1e-15);
    }

    #[test]
    fn gaussian_peak() {
        let g = gaussian(0.0, 1.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((g - expected).abs() < 1e-15);
    }

    #[test]
    fn inv_sinh_occupation_basic() {
        // At x = 2 T: 1 / sinh(1).
        let n = inv_sinh_occupation(2.0, 1.0);
        let expected = 1.0 / 1.0_f64.sinh();
        assert!((n - expected).abs() < 1e-15);
    }
}
