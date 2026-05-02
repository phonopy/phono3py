//! Imaginary part of the bubble-diagram self-energy, pre-computed
//! integration weights variant.
//!
//! Port of `ise_get_imag_self_energy_with_g` and
//! `ise_get_detailed_imag_self_energy_with_g` in
//! `c/imag_self_energy_with_g.c`.  The C callers always pass
//! `num_temps = 1`, so these Rust ports are specialized to a single
//! temperature.  Frequencies and Bose-Einstein occupations are in
//! THz units throughout.

use rayon::prelude::*;

use crate::funcs::bose_einstein;

/// Build the non-zero entries of `g_zero` as `(band0, j, k, flat_idx)`
/// tuples.  `g_zero` is `(num_band0, num_band, num_band)` flat for
/// the band-index mode.  Writes the positions where `g_zero[...] == 0`
/// into `out`, preserving capacity for reuse across calls.
///
/// Iteration order is `(band0, k, j)` so entries sharing the same
/// `(band0, k)` pair are contiguous in `out`.  This lets
/// `reciprocal_to_normal_squared` memoize the Phase 2a row-wise
/// contraction once per `(band0, k)` group rather than per g_pos
/// entry.  The pushed quadruple `[band0, j, k, flat_idx]` and the
/// flat_idx layout (`band0 * num_band^2 + j * num_band + k`) are
/// unchanged; only the visit order differs.
pub(crate) fn set_g_pos(
    out: &mut Vec<[i64; 4]>,
    g_zero: &[i8],
    num_band0: usize,
    num_band: usize,
) {
    out.clear();
    out.reserve(g_zero.len());
    for b0 in 0..num_band0 {
        for k in 0..num_band {
            for j in 0..num_band {
                let jkl = b0 * num_band * num_band + j * num_band + k;
                if g_zero[jkl] == 0 {
                    out.push([b0 as i64, j as i64, k as i64, jkl as i64]);
                }
            }
        }
    }
}

/// Frequency-point mode variant.  `g_zero` here is
/// `(num_band, num_band)` flat (shared across the band0 axis), but the
/// fourth entry encodes the (band0, j, k) linear index as if `g_zero`
/// were `(num_band0, num_band, num_band)`, matching the C layout used
/// to index the `fc3_normal_squared` array downstream.
///
/// Iteration order is `(band0, k, j)` to match `set_g_pos` (entries
/// sharing `(band0, k)` are contiguous).
fn set_g_pos_at_frequency_point(g_zero: &[i8], num_band0: usize, num_band: usize) -> Vec<[i64; 4]> {
    let mut out = Vec::with_capacity(num_band0 * num_band * num_band);
    for b0 in 0..num_band0 {
        for k in 0..num_band {
            for j in 0..num_band {
                let kl = j * num_band + k;
                let jkl = b0 * num_band * num_band + kl;
                if g_zero[kl] == 0 {
                    out.push([b0 as i64, j as i64, k as i64, jkl as i64]);
                }
            }
        }
    }
    out
}

/// Populate `n1` and `n2` (length `num_band`) with Bose-Einstein
/// occupations at bands 1 and 2 of the triplet.  Bands below
/// `cutoff_frequency` are marked with `-1`, matching the C sentinel.
fn set_occupations(
    n1: &mut [f64],
    n2: &mut [f64],
    num_band: usize,
    temperature_thz: f64,
    triplet: [i64; 3],
    frequencies: &[f64],
    cutoff_frequency: f64,
) {
    let base1 = triplet[1] as usize * num_band;
    let base2 = triplet[2] as usize * num_band;
    for j in 0..num_band {
        let f1 = frequencies[base1 + j];
        let f2 = frequencies[base2 + j];
        n1[j] = if f1 > cutoff_frequency {
            bose_einstein(f1, temperature_thz)
        } else {
            -1.0
        };
        n2[j] = if f2 > cutoff_frequency {
            bose_einstein(f2, temperature_thz)
        } else {
            -1.0
        };
    }
}

/// Per-triplet inner loop.  Writes `ise_at_triplet` (length `num_band0`,
/// the caller's per-triplet slot in the outer `ise` buffer).
///
/// `fc3_normal_squared`, `g1`, `g2_3` are already sliced at the current
/// triplet.  `g1`/`g2_3` have `num_band * num_band` entries per band0
/// (or shared across band0 in frequency-point mode; that's accounted
/// for via `at_a_frequency_point`).
#[allow(clippy::too_many_arguments)]
pub fn imag_self_energy_at_triplet(
    ise_at_triplet: &mut [f64],
    num_band0: usize,
    num_band: usize,
    fc3_normal_squared: &[f64],
    frequencies: &[f64],
    triplet: [i64; 3],
    triplet_weight: i64,
    g1: &[f64],
    g2_3: &[f64],
    g_pos: &[[i64; 4]],
    temperature_thz: f64,
    cutoff_frequency: f64,
    at_a_frequency_point: bool,
) {
    let mut n1 = vec![0.0f64; num_band];
    let mut n2 = vec![0.0f64; num_band];
    set_occupations(
        &mut n1,
        &mut n2,
        num_band,
        temperature_thz,
        triplet,
        frequencies,
        cutoff_frequency,
    );

    for slot in ise_at_triplet.iter_mut().take(num_band0) {
        *slot = 0.0;
    }

    let weight = triplet_weight as f64;
    for gp in g_pos {
        let band0 = gp[0] as usize;
        let j = gp[1] as usize;
        let k = gp[2] as usize;
        let fc3_idx = gp[3] as usize;
        let g_idx = if at_a_frequency_point {
            fc3_idx % (num_band * num_band)
        } else {
            fc3_idx
        };
        if n1[j] < 0.0 || n2[k] < 0.0 {
            continue;
        }
        let contribution = if temperature_thz > 0.0 {
            ((n1[j] + n2[k] + 1.0) * g1[g_idx] + (n1[j] - n2[k]) * g2_3[g_idx])
                * fc3_normal_squared[fc3_idx]
                * weight
        } else {
            g1[g_idx] * fc3_normal_squared[fc3_idx] * weight
        };
        ise_at_triplet[band0] += contribution;
    }
}

/// Main entry: imaginary self-energy with pre-computed `g` weights.
///
/// Parameters follow `ise_get_imag_self_energy_with_g` in
/// `c/imag_self_energy_with_g.c`, specialized to a single temperature.
///
/// - `imag_self_energy`: output of shape `(num_band0,)`.
/// - `fc3_normal_squared`: flat `(num_triplets, num_band0, num_band, num_band)`.
/// - `frequencies`: flat `(num_grid, num_band)`.
/// - `triplets`: `(num_triplets,)` of `[i64; 3]` grid-point indices.
/// - `triplet_weights`: `(num_triplets,)`.
/// - `g`: flat `(2, num_triplets, dim, num_band, num_band)` where `dim`
///   is `num_band0` (band-index mode) or `num_frequency_points`
///   (frequency-point mode).  Layer 0 is `g1`, layer 1 is `g2 - g3`.
/// - `g_zero`: flat with the same shape as a single layer of `g` (bytes).
/// - `temperature_thz`: temperature in THz units.
/// - `cutoff_frequency`: bands with frequency <= this contribute zero.
/// - `num_frequency_points`: total frequency points (ignored in band mode).
/// - `frequency_point_index`: `< 0` for band-index mode; otherwise the
///   frequency point index to select within `g`/`g_zero`.
#[allow(clippy::too_many_arguments)]
pub fn get_imag_self_energy_with_g(
    imag_self_energy: &mut [f64],
    fc3_normal_squared: &[f64],
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    g: &[f64],
    g_zero: &[i8],
    temperature_thz: f64,
    cutoff_frequency: f64,
    num_frequency_points: i64,
    frequency_point_index: i64,
    num_band0: usize,
    num_band: usize,
) {
    let num_triplets = triplets.len();
    let num_band_prod = num_band0 * num_band * num_band;

    let at_a_frequency_point = frequency_point_index >= 0;
    let (g_index_dims, g_index_shift) = if at_a_frequency_point {
        let stride = num_frequency_points as usize * num_band * num_band;
        let shift = frequency_point_index as usize * num_band * num_band;
        (stride, shift)
    } else {
        (num_band_prod, 0usize)
    };

    let g_layer_stride = num_triplets * g_index_dims;
    debug_assert_eq!(fc3_normal_squared.len(), num_triplets * num_band_prod);
    debug_assert_eq!(g.len(), 2 * g_layer_stride);
    debug_assert_eq!(g_zero.len(), g_layer_stride);

    // Per-triplet partial sums, reduced below.
    let mut ise_per_triplet = vec![0.0f64; num_triplets * num_band0];

    // Max linear index read by `imag_self_energy_at_triplet` into the
    // g/g_zero slices.  In band-index mode the g table is laid out
    // per-triplet as `(num_band0, num_band, num_band)`; in
    // frequency-point mode only the `(num_band, num_band)` sub-block
    // at `frequency_point_index` is used.
    let g_max_idx = if at_a_frequency_point {
        num_band * num_band
    } else {
        num_band_prod
    };

    ise_per_triplet
        .par_chunks_mut(num_band0)
        .enumerate()
        .for_each(|(i, slot)| {
            let fc3_slice = &fc3_normal_squared[i * num_band_prod..(i + 1) * num_band_prod];
            let g_base = i * g_index_dims + g_index_shift;
            let g1 = &g[g_base..g_base + g_max_idx];
            let g2_3 = &g[g_layer_stride + g_base..g_layer_stride + g_base + g_max_idx];
            let gz = &g_zero[g_base..g_base + g_max_idx];
            let g_pos = if at_a_frequency_point {
                set_g_pos_at_frequency_point(gz, num_band0, num_band)
            } else {
                let mut g_pos = Vec::new();
                set_g_pos(&mut g_pos, gz, num_band0, num_band);
                g_pos
            };
            imag_self_energy_at_triplet(
                slot,
                num_band0,
                num_band,
                fc3_slice,
                frequencies,
                triplets[i],
                triplet_weights[i],
                g1,
                g2_3,
                &g_pos,
                temperature_thz,
                cutoff_frequency,
                at_a_frequency_point,
            );
        });

    for slot in imag_self_energy.iter_mut().take(num_band0) {
        *slot = 0.0;
    }
    for i in 0..num_triplets {
        for j in 0..num_band0 {
            imag_self_energy[j] += ise_per_triplet[i * num_band0 + j];
        }
    }
}

/// Per-triplet inner loop for the detailed variant.  Writes the
/// `detailed_ise_at_triplet` block (flat length `num_band0 * num_band *
/// num_band`) and `ise_at_triplet` (length `num_band0`, one summed
/// contribution per band0).
#[allow(clippy::too_many_arguments)]
fn detailed_imag_self_energy_at_triplet(
    detailed_ise_at_triplet: &mut [f64],
    ise_at_triplet: &mut [f64],
    num_band0: usize,
    num_band: usize,
    fc3_normal_squared: &[f64],
    frequencies: &[f64],
    triplet: [i64; 3],
    g1: &[f64],
    g2_3: &[f64],
    g_zero: &[i8],
    temperature_thz: f64,
    cutoff_frequency: f64,
) {
    let mut n1 = vec![0.0f64; num_band];
    let mut n2 = vec![0.0f64; num_band];
    set_occupations(
        &mut n1,
        &mut n2,
        num_band,
        temperature_thz,
        triplet,
        frequencies,
        cutoff_frequency,
    );

    let block = num_band * num_band;
    for band0 in 0..num_band0 {
        let base = band0 * block;
        let mut sum_g = 0.0;
        for ij in 0..block {
            let idx = base + ij;
            detailed_ise_at_triplet[idx] = 0.0;
            if g_zero[idx] != 0 {
                continue;
            }
            let i = ij / num_band;
            let j = ij % num_band;
            if n1[i] < 0.0 || n2[j] < 0.0 {
                continue;
            }
            let val = if temperature_thz > 0.0 {
                ((n1[i] + n2[j] + 1.0) * g1[idx] + (n1[i] - n2[j]) * g2_3[idx])
                    * fc3_normal_squared[idx]
            } else {
                g1[idx] * fc3_normal_squared[idx]
            };
            detailed_ise_at_triplet[idx] = val;
            sum_g += val;
        }
        ise_at_triplet[band0] = sum_g;
    }
}

/// Normal-scattering predicate for a triplet: true iff the BZ grid
/// addresses of the three grid points sum to zero on every axis.
/// Port of `tpl_is_N` in `c/triplet.c`.
fn tpl_is_n(triplet: [i64; 3], bz_grid_addresses: &[[i64; 3]]) -> bool {
    for axis in 0..3 {
        let sum: i64 = (0..3)
            .map(|j| bz_grid_addresses[triplet[j] as usize][axis])
            .sum();
        if sum != 0 {
            return false;
        }
    }
    true
}

/// Main entry for the detailed variant.  Port of
/// `ise_get_detailed_imag_self_energy_with_g`, specialized to a single
/// temperature.
///
/// - `detailed_imag_self_energy` (out): flat
///   `(num_triplets, num_band0, num_band, num_band)`.
/// - `imag_self_energy_n` (out): `(num_band0,)`, Normal-scattering
///   contribution summed over triplets with `triplet_weights`.
/// - `imag_self_energy_u` (out): `(num_band0,)`, Umklapp contribution.
/// - `fc3_normal_squared`, `g`, `g_zero`, `frequencies`, `triplets`,
///   `triplet_weights`: same layouts as the non-detailed entry; `g` is
///   always in band-index mode here (no frequency-point mode in the
///   detailed C entry).
/// - `bz_grid_addresses`: `(num_grid, 3)` integer addresses, used only
///   to classify each triplet as Normal or Umklapp.
#[allow(clippy::too_many_arguments)]
pub fn get_detailed_imag_self_energy_with_g(
    detailed_imag_self_energy: &mut [f64],
    imag_self_energy_n: &mut [f64],
    imag_self_energy_u: &mut [f64],
    fc3_normal_squared: &[f64],
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    bz_grid_addresses: &[[i64; 3]],
    g: &[f64],
    g_zero: &[i8],
    temperature_thz: f64,
    cutoff_frequency: f64,
    num_band0: usize,
    num_band: usize,
) {
    let num_triplets = triplets.len();
    let num_band_prod = num_band0 * num_band * num_band;
    let g_layer_stride = num_triplets * num_band_prod;
    debug_assert_eq!(fc3_normal_squared.len(), num_triplets * num_band_prod);
    debug_assert_eq!(
        detailed_imag_self_energy.len(),
        num_triplets * num_band_prod
    );
    debug_assert_eq!(g.len(), 2 * g_layer_stride);
    debug_assert_eq!(g_zero.len(), g_layer_stride);

    let mut ise_per_triplet = vec![0.0f64; num_triplets * num_band0];

    detailed_imag_self_energy
        .par_chunks_mut(num_band_prod)
        .zip(ise_per_triplet.par_chunks_mut(num_band0))
        .enumerate()
        .for_each(|(i, (detailed_slot, ise_slot))| {
            let fc3_slice = &fc3_normal_squared[i * num_band_prod..(i + 1) * num_band_prod];
            let g_base = i * num_band_prod;
            let g1 = &g[g_base..g_base + num_band_prod];
            let g2_3 = &g[g_layer_stride + g_base..g_layer_stride + g_base + num_band_prod];
            let gz = &g_zero[g_base..g_base + num_band_prod];
            detailed_imag_self_energy_at_triplet(
                detailed_slot,
                ise_slot,
                num_band0,
                num_band,
                fc3_slice,
                frequencies,
                triplets[i],
                g1,
                g2_3,
                gz,
                temperature_thz,
                cutoff_frequency,
            );
        });

    for slot in imag_self_energy_n.iter_mut().take(num_band0) {
        *slot = 0.0;
    }
    for slot in imag_self_energy_u.iter_mut().take(num_band0) {
        *slot = 0.0;
    }
    for i in 0..num_triplets {
        let is_n = tpl_is_n(triplets[i], bz_grid_addresses);
        let weight = triplet_weights[i] as f64;
        for j in 0..num_band0 {
            let val = ise_per_triplet[i * num_band0 + j] * weight;
            if is_n {
                imag_self_energy_n[j] += val;
            } else {
                imag_self_energy_u[j] += val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_g_pos_filters_zeros() {
        // num_band0 = 1, num_band = 2, shape (1, 2, 2) = 4 entries.
        // g_zero = [0, 1, 0, 0] -> keep indices 0, 2, 3.
        let g_zero = vec![0i8, 1, 0, 0];
        let mut pos = Vec::new();
        set_g_pos(&mut pos, &g_zero, 1, 2);
        assert_eq!(pos.len(), 3);
        assert_eq!(pos[0], [0, 0, 0, 0]);
        assert_eq!(pos[1], [0, 1, 0, 2]);
        assert_eq!(pos[2], [0, 1, 1, 3]);
    }

    #[test]
    fn set_g_pos_at_frequency_point_replicates_per_band0() {
        // num_band0 = 2, num_band = 2, g_zero is (2, 2) = 4 entries.
        // g_zero = [0, 0, 1, 0] -> per band0: keep (j=0,k=0), (j=0,k=1), (j=1,k=1).
        let g_zero = vec![0i8, 0, 1, 0];
        let pos = set_g_pos_at_frequency_point(&g_zero, 2, 2);
        assert_eq!(pos.len(), 6);
        // jkl runs over the full (num_band0, num_band, num_band) space.
        assert_eq!(pos[0], [0, 0, 0, 0]);
        assert_eq!(pos[1], [0, 0, 1, 1]);
        assert_eq!(pos[2], [0, 1, 1, 3]);
        assert_eq!(pos[3], [1, 0, 0, 4]);
        assert_eq!(pos[4], [1, 0, 1, 5]);
        assert_eq!(pos[5], [1, 1, 1, 7]);
    }

    #[test]
    fn tpl_is_n_normal_and_umklapp() {
        // Grid addresses for 5 imaginary grid points.
        let addrs: Vec<[i64; 3]> = vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, -1, 0]];
        // (0, 1, -1) sums to (0, 0, 0) on every axis -> Normal.
        assert!(tpl_is_n([0, 1, 2], &addrs));
        // (0, 3, 4) sums on axis 1 to 0+2+(-1) = 1 -> Umklapp.
        assert!(!tpl_is_n([0, 3, 4], &addrs));
    }

    #[test]
    fn detailed_partitions_into_n_and_u() {
        // One Normal triplet and one Umklapp triplet sharing the same
        // per-triplet contribution; check that each ends up in the
        // matching bucket.
        let num_band0 = 1;
        let num_band = 1;
        let num_band_prod = num_band0 * num_band * num_band;
        let num_triplets = 2;
        // Grid 0 at origin; grid 1 at (1,0,0); triplets (0,0,0) N and
        // (0,0,1) U.
        let addrs: Vec<[i64; 3]> = vec![[0, 0, 0], [1, 0, 0]];
        let triplets: Vec<[i64; 3]> = vec![[0, 0, 0], [0, 0, 1]];
        let weights = vec![2i64, 3];
        let fc3 = vec![1.0f64; num_triplets * num_band_prod];
        // Two distinct grid points each have num_band bands.
        let frequencies = vec![5.0f64; 2 * num_band];
        // g layer 0 = 1.0, layer 1 = 0.0, so T=0 contribution is just
        // g1 * fc3 = 1.0 per (band0, j, k).
        let g_layer = num_triplets * num_band_prod;
        let mut g = vec![0.0f64; 2 * g_layer];
        for v in g.iter_mut().take(g_layer) {
            *v = 1.0;
        }
        let g_zero = vec![0i8; g_layer];

        let mut detailed = vec![0.0f64; num_triplets * num_band_prod];
        let mut ise_n = vec![0.0f64; num_band0];
        let mut ise_u = vec![0.0f64; num_band0];
        get_detailed_imag_self_energy_with_g(
            &mut detailed,
            &mut ise_n,
            &mut ise_u,
            &fc3,
            &frequencies,
            &triplets,
            &weights,
            &addrs,
            &g,
            &g_zero,
            0.0,
            0.0,
            num_band0,
            num_band,
        );
        // Each per-triplet sum is 1.0 (single element g1 * fc3).
        // N bucket gets weight 2, U bucket gets weight 3.
        assert!((ise_n[0] - 2.0).abs() < 1e-15);
        assert!((ise_u[0] - 3.0).abs() < 1e-15);
        // Detailed block equals raw per-(band0,j,k) value, no weight.
        for d in &detailed {
            assert!((d - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn detailed_respects_g_zero_and_cutoff() {
        // num_band0 = 1, num_band = 2, one triplet at origin.
        let num_band0 = 1;
        let num_band = 2;
        let num_band_prod = num_band0 * num_band * num_band;
        let addrs: Vec<[i64; 3]> = vec![[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        let triplets: Vec<[i64; 3]> = vec![[0, 1, 2]];
        let weights = vec![1i64];
        let fc3 = vec![1.0f64, 1.0, 1.0, 1.0];
        // Freqs per grid: grid 1 band 0 below cutoff -> n1[0] = -1.
        let frequencies = vec![
            5.0, 5.0, // grid 0
            0.001, 5.0, // grid 1 (used as triplet[1])
            5.0, 5.0, // grid 2
        ];
        // Layer 0 all 1, layer 1 all 0.
        let mut g = vec![0.0f64; 2 * num_band_prod];
        for v in g.iter_mut().take(num_band_prod) {
            *v = 1.0;
        }
        // Mask out (i=0, j=1) -> flat index 1 in band0 block.
        let g_zero = vec![0i8, 1, 0, 0];

        let mut detailed = vec![0.0f64; num_band_prod];
        let mut ise_n = vec![0.0f64; num_band0];
        let mut ise_u = vec![0.0f64; num_band0];
        get_detailed_imag_self_energy_with_g(
            &mut detailed,
            &mut ise_n,
            &mut ise_u,
            &fc3,
            &frequencies,
            &triplets,
            &weights,
            &addrs,
            &g,
            &g_zero,
            0.0,
            0.1,
            num_band0,
            num_band,
        );
        // Entries: (i=0,j=0) kept but n1[0] = -1 -> 0.
        // (i=0,j=1) masked by g_zero -> 0.
        // (i=1,j=0) kept, n1[1] = 0, n2[0] = 0 -> 1.0.
        // (i=1,j=1) kept, n1[1] = 0, n2[1] = 0 -> 1.0.
        assert_eq!(detailed[0], 0.0);
        assert_eq!(detailed[1], 0.0);
        assert!((detailed[2] - 1.0).abs() < 1e-15);
        assert!((detailed[3] - 1.0).abs() < 1e-15);
        assert!((ise_n[0] - 2.0).abs() < 1e-15);
        assert_eq!(ise_u[0], 0.0);
    }

    #[test]
    fn imag_self_energy_zero_pp_strength_gives_zero() {
        // Trivial: all fc3 = 0 -> output must be 0.
        let num_triplets = 2;
        let num_band0 = 2;
        let num_band = 2;
        let num_band_prod = num_band0 * num_band * num_band;
        let fc3 = vec![0.0f64; num_triplets * num_band_prod];
        // frequencies for one grid point triplet vertex.
        let frequencies = vec![1.0f64; 4 * num_band]; // 4 grid points, num_band each
        let triplets: Vec<[i64; 3]> = vec![[0, 1, 2], [0, 2, 3]];
        let weights = vec![1i64, 1];
        let g_layer = num_triplets * num_band_prod;
        let g = vec![1.0f64; 2 * g_layer]; // arbitrary
        let g_zero = vec![0i8; g_layer];
        let mut out = vec![1.0f64; num_band0];
        get_imag_self_energy_with_g(
            &mut out,
            &fc3,
            &frequencies,
            &triplets,
            &weights,
            &g,
            &g_zero,
            1.0,
            0.0,
            1, // unused when band-index mode
            -1,
            num_band0,
            num_band,
        );
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
