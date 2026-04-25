//! Isotope scattering strength gamma_isotope.
//!
//! Port of `iso_get_isotope_scattering_strength` (Gaussian smearing)
//! and `iso_get_thm_isotope_scattering_strength` (tetrahedron method)
//! in `c/isotope.c`.  Frequencies are in THz throughout.  The
//! eigenvector indexing convention matches the C code:
//! ``eigenvectors[gp, component, band]`` with ``component = l * 3 + m``
//! where ``l`` ranges over atoms and ``m`` over Cartesian axes.

use rayon::prelude::*;

use crate::common::Cmplx;
use crate::funcs::gaussian;

const M_2PI: f64 = std::f64::consts::TAU;

/// Per-atom component inner product `sum_m conj(e0_lm) * e1_lm` for a
/// fixed atom `l`, accumulated into `(a, b)` with `a = Re`, `b = -Im`.
///
/// Matching the C expression in `c/isotope.c`, `b` has the sign opposite
/// to the imaginary part, but since only `a*a + b*b` is used downstream
/// the sign is irrelevant.
#[inline]
fn atom_overlap(e0: &[Cmplx], e1: &[Cmplx], num_band: usize, k: usize, l: usize) -> (f64, f64) {
    let mut a = 0.0;
    let mut b = 0.0;
    for m in 0..3 {
        let e0_r = e0[l * 3 + m][0];
        let e0_i = e0[l * 3 + m][1];
        let idx = (l * 3 + m) * num_band + k;
        let e1_r = e1[idx][0];
        let e1_i = e1[idx][1];
        a += e0_r * e1_r + e0_i * e1_i;
        b += e0_i * e1_r - e0_r * e1_i;
    }
    (a, b)
}

/// Load `e0[i, j] = eigenvectors[grid_point, j, band_indices[i]]` and
/// `f0[i] = frequencies[grid_point, band_indices[i]]`.
fn load_band0_data(
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    grid_point: i64,
    band_indices: &[i64],
    num_band: usize,
) -> (Vec<f64>, Vec<Cmplx>) {
    let num_band0 = band_indices.len();
    let mut f0 = vec![0.0; num_band0];
    let mut e0: Vec<Cmplx> = vec![[0.0, 0.0]; num_band0 * num_band];
    let gp_base_f = grid_point as usize * num_band;
    let gp_base_e = grid_point as usize * num_band * num_band;
    for i in 0..num_band0 {
        let bi = band_indices[i] as usize;
        f0[i] = frequencies[gp_base_f + bi];
        for j in 0..num_band {
            e0[i * num_band + j] = eigenvectors[gp_base_e + j * num_band + bi];
        }
    }
    (f0, e0)
}

/// Sum of `|conj(e0_i) . e1|^2 * mass_var[l]` over atoms `l` at a
/// fixed band column `k`, given the eigenvector slice `e1` of the
/// scattered grid point (length `num_band * num_band`).
#[inline]
fn atoms_overlap_squared(
    e0_band0: &[Cmplx],
    e1: &[Cmplx],
    mass_variances: &[f64],
    num_band: usize,
    k: usize,
) -> f64 {
    let num_atom = num_band / 3;
    let mut acc = 0.0;
    for l in 0..num_atom {
        let (a, b) = atom_overlap(e0_band0, e1, num_band, k, l);
        acc += (a * a + b * b) * mass_variances[l];
    }
    acc
}

/// Multiply every entry by the C-side prefactor `(2pi) / 4 * f0^2 / 2`.
#[inline]
fn apply_prefactor(gamma: &mut [f64], f0: &[f64]) {
    for (g, &f) in gamma.iter_mut().zip(f0.iter()) {
        *g *= M_2PI / 4.0 * f * f / 2.0;
    }
}

/// Gaussian-smearing isotope strength.  Writes `gamma[0..num_band0]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn isotope_strength(
    gamma: &mut [f64],
    grid_point: i64,
    ir_grid_points: &[i64],
    weights: &[f64],
    mass_variances: &[f64],
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    band_indices: &[i64],
    num_band: usize,
    sigma: f64,
    cutoff_frequency: f64,
) {
    let num_band0 = band_indices.len();
    let (f0, e0) = load_band0_data(
        frequencies,
        eigenvectors,
        grid_point,
        band_indices,
        num_band,
    );

    for g in gamma.iter_mut().take(num_band0) {
        *g = 0.0;
    }

    for i in 0..num_band0 {
        if f0[i] < cutoff_frequency {
            continue;
        }
        let e0_band0 = &e0[i * num_band..(i + 1) * num_band];

        let sum_g: f64 = ir_grid_points
            .par_iter()
            .map(|&gp| {
                let gp_u = gp as usize;
                let e1 =
                    &eigenvectors[gp_u * num_band * num_band..(gp_u + 1) * num_band * num_band];
                let mut sum_k = 0.0;
                for k in 0..num_band {
                    let f = frequencies[gp_u * num_band + k];
                    if f < cutoff_frequency {
                        continue;
                    }
                    let dist = gaussian(f - f0[i], sigma);
                    sum_k +=
                        atoms_overlap_squared(e0_band0, e1, mass_variances, num_band, k) * dist;
                }
                sum_k * weights[gp_u]
            })
            .sum();
        gamma[i] = sum_g;
    }

    apply_prefactor(&mut gamma[..num_band0], &f0);
}

/// Tetrahedron-method isotope strength, using pre-computed
/// ``integration_weights`` of shape ``(num_grid_points, num_band0,
/// num_band)`` (row-major, `gp` as the outermost axis).
#[allow(clippy::too_many_arguments)]
pub(crate) fn thm_isotope_strength(
    gamma: &mut [f64],
    grid_point: i64,
    ir_grid_points: &[i64],
    weights: &[f64],
    mass_variances: &[f64],
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    band_indices: &[i64],
    num_band: usize,
    integration_weights: &[f64],
    cutoff_frequency: f64,
) {
    let num_band0 = band_indices.len();
    let num_grid_points = ir_grid_points.len();
    let (f0, e0) = load_band0_data(
        frequencies,
        eigenvectors,
        grid_point,
        band_indices,
        num_band,
    );
    let f0_ref: &[f64] = &f0;
    let e0_ref: &[Cmplx] = &e0;

    let gamma_ij: Vec<f64> = (0..num_grid_points)
        .into_par_iter()
        .flat_map_iter(|i| {
            let gp = ir_grid_points[i] as usize;
            let e1 = &eigenvectors[gp * num_band * num_band..(gp + 1) * num_band * num_band];
            let w_gp = weights[gp];
            // integration_weights is indexed by BZ grid point, so use `gp`
            // rather than `i`, matching the C convention.
            let iw_gp =
                &integration_weights[gp * num_band0 * num_band..(gp + 1) * num_band0 * num_band];
            let freqs_gp = &frequencies[gp * num_band..(gp + 1) * num_band];
            (0..num_band0).map(move |j| {
                if f0_ref[j] < cutoff_frequency {
                    return 0.0;
                }
                let e0_band0 = &e0_ref[j * num_band..(j + 1) * num_band];
                let mut sum_k = 0.0;
                for k in 0..num_band {
                    if freqs_gp[k] < cutoff_frequency {
                        continue;
                    }
                    let dist = iw_gp[j * num_band + k];
                    sum_k +=
                        atoms_overlap_squared(e0_band0, e1, mass_variances, num_band, k) * dist;
                }
                sum_k * w_gp
            })
        })
        .collect();

    for g in gamma.iter_mut().take(num_band0) {
        *g = 0.0;
    }
    for i in 0..num_grid_points {
        for j in 0..num_band0 {
            gamma[j] += gamma_ij[i * num_band0 + j];
        }
    }

    apply_prefactor(&mut gamma[..num_band0], &f0);
}
