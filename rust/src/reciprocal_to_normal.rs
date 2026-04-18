//! Transform fc3 from reciprocal space to phonon (normal) coordinates.
//!
//! Port of `reciprocal_to_normal_squared` and its static helpers in
//! `c/reciprocal_to_normal.c`.  The C version operates on band-first
//! `fc3_reciprocal` of shape `(num_band, num_band, num_band)` flat
//! complex with `num_band = num_patom * 3`.  Here the input is the
//! atom-first layout `(num_patom, num_patom, num_patom, 3, 3, 3)` flat,
//! matching the output of `crate::real_to_reciprocal` and the Python
//! reference `ReciprocalToNormal`.
//!
//! Eigenvectors are passed un-scaled as `(num_band, num_band)` row-major
//! with index `[component, band]`.  Internally this module transposes to
//! `[band, component]` and multiplies by `1 / sqrt(mass_of(component))`,
//! mirroring the C preamble in `reciprocal_to_normal_squared`.

use rayon::prelude::*;

use crate::common::{cmplx_mul, Cmplx};

/// Pre-scale eigenvectors by `1 / sqrt(mass)` and transpose so the
/// fast axis is the component axis (for a fixed band).
///
/// Input layout: `eigvecs[component * num_band + band]`
/// (`[component, band]` row-major).
///
/// Output layout: `out[band * num_band + component]`
/// (`[band, component]` row-major), with each element multiplied by
/// `1 / sqrt(masses[component / 3])`.
fn scale_and_transpose(eigvecs: &[Cmplx], masses: &[f64], num_band: usize) -> Vec<Cmplx> {
    let mut out = vec![[0.0f64; 2]; num_band * num_band];
    for comp in 0..num_band {
        let atom = comp / 3;
        let inv_sqrt_mass = 1.0 / masses[atom].sqrt();
        for band in 0..num_band {
            let src = eigvecs[comp * num_band + band];
            out[band * num_band + comp] = [src[0] * inv_sqrt_mass, src[1] * inv_sqrt_mass];
        }
    }
    out
}

/// Contract fc3 with `e0` at a single target band.
///
/// `fc3_reciprocal` is atom-first `(num_patom, num_patom, num_patom, 3, 3, 3)`.
/// `e0_band` is the length-`num_band` row of the scaled-and-transposed
/// `e0` at the target band (i.e. `e0_scaled[band_index_0, :]`).
///
/// Produces `out[b, c, m, n] = sum over (a, l) of
/// fc3[a, b, c, l, m, n] * e0_band[a * 3 + l]`, in flat layout
/// `((b * num_patom + c) * 3 + m) * 3 + n`.
fn contract_fc3_e0(
    out: &mut [Cmplx],
    fc3_reciprocal: &[Cmplx],
    e0_band: &[Cmplx],
    num_patom: usize,
) {
    for entry in out.iter_mut() {
        *entry = [0.0, 0.0];
    }
    for a in 0..num_patom {
        for l in 0..3 {
            let e0_al = e0_band[a * 3 + l];
            if e0_al[0] == 0.0 && e0_al[1] == 0.0 {
                continue;
            }
            for b in 0..num_patom {
                for c in 0..num_patom {
                    let fc3_base = ((a * num_patom + b) * num_patom + c) * 27 + l * 9;
                    let out_base = (b * num_patom + c) * 9;
                    for mn in 0..9 {
                        let prod = cmplx_mul(fc3_reciprocal[fc3_base + mn], e0_al);
                        out[out_base + mn][0] += prod[0];
                        out[out_base + mn][1] += prod[1];
                    }
                }
            }
        }
    }
}

/// Inner contraction over `(b, c, m, n)` at a single `(j, k)` band pair.
/// Returns `|sum|^2` where `sum = sum over (b, c, m, n) of
/// fc3_e0[b, c, m, n] * e1[b * 3 + m] * e2[c * 3 + n]`.
fn get_fc3_sum_atomwise(
    fc3_e0: &[Cmplx],
    e1_band: &[Cmplx],
    e2_band: &[Cmplx],
    num_patom: usize,
) -> f64 {
    let mut sum_real = 0.0f64;
    let mut sum_imag = 0.0f64;
    for b in 0..num_patom {
        for c in 0..num_patom {
            let base = (b * num_patom + c) * 9;
            for m in 0..3 {
                let e1_bm = e1_band[b * 3 + m];
                for n in 0..3 {
                    let e2_cn = e2_band[c * 3 + n];
                    let e12 = cmplx_mul(e1_bm, e2_cn);
                    let t = cmplx_mul(fc3_e0[base + m * 3 + n], e12);
                    sum_real += t[0];
                    sum_imag += t[1];
                }
            }
        }
    }
    sum_real * sum_real + sum_imag * sum_imag
}

/// Compute `|M|^2 / (f0 * f1 * f2)` for each `g_pos` entry.
///
/// Inputs:
/// - `fc3_normal_squared`: output array indexed by `g_pos[i][3]`.
/// - `g_pos`: shape `(num_g_pos, 4)` with columns
///   `(band0_index, j, k, dest)`.  `band0_index` indexes into
///   `band_indices`; `j` and `k` are band indices for the second and
///   third triplet vertices.
/// - `fc3_reciprocal`: atom-first `(num_patom, num_patom, num_patom, 3, 3, 3)`
///   flat.
/// - `freqs0`, `freqs1`, `freqs2`: phonon frequencies at the three
///   triplet vertices, length `num_band`.
/// - `eigvecs0`, `eigvecs1`, `eigvecs2`: un-scaled eigenvectors,
///   `(num_band, num_band)` row-major `[component, band]`.
/// - `masses`: atomic masses, length `num_patom`.
/// - `band_indices`: selection of target bands at the first vertex,
///   length `num_band0`.
/// - `cutoff_frequency`: entries with any of `freqs0[bi]`, `freqs1[j]`,
///   `freqs2[k]` not exceeding this are zeroed.
/// - `openmp_per_triplets`: when `true`, loops run sequentially (the
///   caller parallelizes over triplets); when `false`, the per-band0
///   and per-`g_pos` loops are parallelized with rayon.
#[allow(clippy::too_many_arguments)]
pub fn reciprocal_to_normal_squared(
    fc3_normal_squared: &mut [f64],
    g_pos: &[[i64; 4]],
    fc3_reciprocal: &[Cmplx],
    freqs0: &[f64],
    freqs1: &[f64],
    freqs2: &[f64],
    eigvecs0: &[Cmplx],
    eigvecs1: &[Cmplx],
    eigvecs2: &[Cmplx],
    masses: &[f64],
    band_indices: &[i64],
    num_patom: usize,
    cutoff_frequency: f64,
    openmp_per_triplets: bool,
) {
    let num_band = num_patom * 3;
    debug_assert_eq!(fc3_reciprocal.len(), num_patom * num_patom * num_patom * 27);
    debug_assert_eq!(eigvecs0.len(), num_band * num_band);
    debug_assert_eq!(eigvecs1.len(), num_band * num_band);
    debug_assert_eq!(eigvecs2.len(), num_band * num_band);
    debug_assert_eq!(masses.len(), num_patom);
    debug_assert_eq!(freqs0.len(), num_band);
    debug_assert_eq!(freqs1.len(), num_band);
    debug_assert_eq!(freqs2.len(), num_band);

    let e0 = scale_and_transpose(eigvecs0, masses, num_band);
    let e1 = scale_and_transpose(eigvecs1, masses, num_band);
    let e2 = scale_and_transpose(eigvecs2, masses, num_band);

    // fc3_e0 caches the contraction over (a, l) at each target band0
    // index.  Shape: (num_band0, num_patom, num_patom, 3, 3) flat with
    // entry size num_patom * num_patom * 9 complex.
    let num_band0 = band_indices.len();
    let entry_size = num_patom * num_patom * 9;
    let mut fc3_e0 = vec![[0.0f64; 2]; num_band0 * entry_size];

    if openmp_per_triplets {
        for i0 in 0..num_band0 {
            let bi = band_indices[i0] as usize;
            let e0_band = &e0[bi * num_band..(bi + 1) * num_band];
            let out = &mut fc3_e0[i0 * entry_size..(i0 + 1) * entry_size];
            contract_fc3_e0(out, fc3_reciprocal, e0_band, num_patom);
        }
    } else {
        fc3_e0
            .par_chunks_mut(entry_size)
            .enumerate()
            .for_each(|(i0, out)| {
                let bi = band_indices[i0] as usize;
                let e0_band = &e0[bi * num_band..(bi + 1) * num_band];
                contract_fc3_e0(out, fc3_reciprocal, e0_band, num_patom);
            });
    }

    let eval_one = |gp: &[i64; 4]| -> (usize, f64) {
        let i0 = gp[0] as usize;
        let j = gp[1] as usize;
        let k = gp[2] as usize;
        let dest = gp[3] as usize;
        let bi = band_indices[i0] as usize;
        if freqs0[bi] <= cutoff_frequency
            || freqs1[j] <= cutoff_frequency
            || freqs2[k] <= cutoff_frequency
        {
            return (dest, 0.0);
        }
        let fc3_e0_entry = &fc3_e0[i0 * entry_size..(i0 + 1) * entry_size];
        let e1_band = &e1[j * num_band..(j + 1) * num_band];
        let e2_band = &e2[k * num_band..(k + 1) * num_band];
        let sq = get_fc3_sum_atomwise(fc3_e0_entry, e1_band, e2_band, num_patom);
        (dest, sq / (freqs0[bi] * freqs1[j] * freqs2[k]))
    };

    if openmp_per_triplets {
        for gp in g_pos {
            let (dest, val) = eval_one(gp);
            fc3_normal_squared[dest] = val;
        }
    } else {
        // Parallelize over g_pos; each entry writes to a distinct
        // `dest` so we gather pairs and scatter afterwards.
        let results: Vec<(usize, f64)> = g_pos.par_iter().map(eval_one).collect();
        for (dest, val) in results {
            fc3_normal_squared[dest] = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_eigvecs(num_band: usize) -> Vec<Cmplx> {
        let mut e = vec![[0.0f64; 2]; num_band * num_band];
        for i in 0..num_band {
            e[i * num_band + i] = [1.0, 0.0];
        }
        e
    }

    #[test]
    fn scale_and_transpose_identity() {
        // num_patom = 1 (num_band = 3), mass = 1 -> identity output
        // with the columns/rows swapped.  Identity is symmetric, so the
        // result is still the identity.
        let num_band = 3;
        let eigvecs = make_identity_eigvecs(num_band);
        let masses = vec![1.0];
        let out = scale_and_transpose(&eigvecs, &masses, num_band);
        for i in 0..num_band {
            for j in 0..num_band {
                let expected = if i == j { [1.0, 0.0] } else { [0.0, 0.0] };
                assert_eq!(out[i * num_band + j], expected);
            }
        }
    }

    #[test]
    fn scale_and_transpose_mass_scaling() {
        // Single atom with mass = 4 -> 1/sqrt(4) = 0.5 scaling.
        let num_band = 3;
        let eigvecs = make_identity_eigvecs(num_band);
        let masses = vec![4.0];
        let out = scale_and_transpose(&eigvecs, &masses, num_band);
        for i in 0..num_band {
            assert!((out[i * num_band + i][0] - 0.5).abs() < 1e-15);
            assert!(out[i * num_band + i][1].abs() < 1e-15);
        }
    }

    #[test]
    fn reciprocal_to_normal_zero_fc3_is_zero() {
        let num_patom = 1;
        let num_band = 3;
        let fc3_reciprocal = vec![[0.0f64; 2]; num_patom * num_patom * num_patom * 27];
        let freqs0 = vec![1.0; num_band];
        let freqs1 = vec![1.0; num_band];
        let freqs2 = vec![1.0; num_band];
        let eigvecs = make_identity_eigvecs(num_band);
        let masses = vec![1.0];
        let band_indices = vec![0i64, 1, 2];
        let g_pos: Vec<[i64; 4]> = (0..3)
            .flat_map(|i| (0..3).flat_map(move |j| (0..3).map(move |k| [i, j, k, i * 9 + j * 3 + k])))
            .collect();
        let mut out = vec![1.0f64; band_indices.len() * num_band * num_band];
        reciprocal_to_normal_squared(
            &mut out,
            &g_pos,
            &fc3_reciprocal,
            &freqs0,
            &freqs1,
            &freqs2,
            &eigvecs,
            &eigvecs,
            &eigvecs,
            &masses,
            &band_indices,
            num_patom,
            0.0,
            false,
        );
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn reciprocal_to_normal_cutoff_zeros_output() {
        // Setup: single atom, identity eigvecs, fc3 non-zero only at
        // (a=0, b=0, c=0, l=0, m=0, n=0).  With identity eigvecs the
        // only non-zero raw sum comes from g_pos = [0, 0, 0, *].
        // Verify: f1[0] below cutoff zeros the output, above cutoff
        // gives a non-zero value.
        let num_patom = 1;
        let num_band = 3;
        let mut fc3_reciprocal = vec![[0.0f64; 2]; num_patom * num_patom * num_patom * 27];
        fc3_reciprocal[0] = [1.0, 0.0];
        let freqs0 = vec![1.0, 1.0, 1.0];
        let freqs1_below = vec![0.1, 1.0, 1.0]; // band 0 = 0.1 < cutoff 0.5
        let freqs1_above = vec![1.0, 1.0, 1.0];
        let freqs2 = vec![1.0, 1.0, 1.0];
        let eigvecs = make_identity_eigvecs(num_band);
        let masses = vec![1.0];
        let band_indices = vec![0i64];
        let g_pos: Vec<[i64; 4]> = vec![[0, 0, 0, 0]];

        let mut out = vec![-1.0f64; 1];
        reciprocal_to_normal_squared(
            &mut out,
            &g_pos,
            &fc3_reciprocal,
            &freqs0,
            &freqs1_below,
            &freqs2,
            &eigvecs,
            &eigvecs,
            &eigvecs,
            &masses,
            &band_indices,
            num_patom,
            0.5,
            false,
        );
        assert_eq!(out[0], 0.0);

        let mut out = vec![-1.0f64; 1];
        reciprocal_to_normal_squared(
            &mut out,
            &g_pos,
            &fc3_reciprocal,
            &freqs0,
            &freqs1_above,
            &freqs2,
            &eigvecs,
            &eigvecs,
            &eigvecs,
            &masses,
            &band_indices,
            num_patom,
            0.5,
            false,
        );
        assert!(out[0] > 0.0);
    }
}
