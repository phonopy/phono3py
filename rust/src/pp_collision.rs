//! Port of `c/pp_collision.c`.
//!
//! Per-grid-point, low-memory driver that fuses integration-weight,
//! interaction, and imag-self-energy evaluation into a single loop over
//! triplets.  The caller receives gamma already summed over triplets
//! (optionally split into Normal/Umklapp channels).
//!
//! This mirrors `ppc_get_pp_collision` (tetrahedron method) and
//! `ppc_get_pp_collision_with_sigma` (Gaussian smearing).

use rayon::prelude::*;

use crate::common::Cmplx;
use crate::imag_self_energy::{imag_self_energy_at_triplet, set_g_pos};
use crate::interaction::get_interaction_at_triplet;
use crate::real_to_reciprocal::AtomTriplets;
use crate::triplet::{is_n, set_relative_grid_address, RelativeGridAddress};
use crate::triplet_iw::{
    integration_weight_per_triplet, integration_weight_with_sigma_per_triplet, BzGridError,
    BzGridView, TpType,
};

/// Scratch buffers reused across triplets to avoid per-triplet heap
/// churn.  One instance per rayon worker (or one total for the
/// sequential path).
struct CollisionScratch {
    fc3_normal_squared: Vec<f64>,
    g_buf: Vec<f64>,
    g_zero: Vec<i8>,
}

impl CollisionScratch {
    fn new(num_band_prod: usize) -> Self {
        Self {
            fc3_normal_squared: vec![0.0; num_band_prod],
            g_buf: vec![0.0; 2 * num_band_prod],
            g_zero: vec![0; num_band_prod],
        }
    }

    fn reset(&mut self) {
        self.fc3_normal_squared.fill(0.0);
        self.g_buf.fill(0.0);
        self.g_zero.fill(0);
    }
}

/// Per-triplet evaluator: compute interaction, then imag-self-energy
/// for all temperatures at that triplet.  `g_buf` is the `(2, num_band0,
/// num_band, num_band)` integration weight (pre-filled by the caller),
/// `g_zero` is the flag block of matching shape.  `fc3_normal_squared`
/// is a caller-owned scratch buffer (length `num_band0 * num_band *
/// num_band`) that must be zeroed on entry.  Writes `ise_at_triplet`
/// of shape `(num_temps, num_band0)`.
#[allow(clippy::too_many_arguments)]
fn evaluate_collision_at_triplet(
    ise_at_triplet: &mut [f64],
    fc3_normal_squared: &mut [f64],
    triplet: [i64; 3],
    triplet_weight: i64,
    g_buf: &[f64],
    g_zero: &[i8],
    num_band0: usize,
    num_band: usize,
    num_temps: usize,
    temperatures_thz: &[f64],
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    bz_grid_addresses: &[[i64; 3]],
    d_diag: [i64; 3],
    q_mat: [[i64; 3]; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets<'_>,
    masses: &[f64],
    band_indices: &[i64],
    symmetrize_fc3_q: bool,
    cutoff_frequency: f64,
) {
    let num_band_prod = num_band0 * num_band * num_band;

    get_interaction_at_triplet(
        fc3_normal_squared,
        g_zero,
        num_band0,
        num_band,
        frequencies,
        eigenvectors,
        triplet,
        bz_grid_addresses,
        d_diag,
        q_mat,
        fc3,
        is_compact_fc3,
        atom_triplets,
        masses,
        band_indices,
        symmetrize_fc3_q,
        cutoff_frequency,
    );

    let g_pos = set_g_pos(g_zero, num_band0, num_band);
    let (g1, g2_3) = g_buf.split_at(num_band_prod);

    for (t, &temperature_thz) in temperatures_thz.iter().enumerate().take(num_temps) {
        let slot = &mut ise_at_triplet[t * num_band0..(t + 1) * num_band0];
        imag_self_energy_at_triplet(
            slot,
            num_band0,
            num_band,
            fc3_normal_squared,
            frequencies,
            triplet,
            triplet_weight,
            g1,
            g2_3,
            &g_pos,
            temperature_thz,
            cutoff_frequency,
            false,
        );
    }
}

/// Reduce per-triplet contributions into `collisions`.
///
/// Shapes:
/// - `ise_per_triplet`: `(num_triplets, num_temps, num_band0)` flat.
/// - `collisions` when `is_n_u = false`: `(num_temps, num_band0)`.
/// - `collisions` when `is_n_u = true`:  `(2, num_temps, num_band0)`
///   with index 0 = Normal, index 1 = Umklapp.
fn finalize(
    collisions: &mut [f64],
    ise_per_triplet: &[f64],
    triplets: &[[i64; 3]],
    bz_grid_addresses: &[[i64; 3]],
    num_temps: usize,
    num_band0: usize,
    is_n_u: bool,
) {
    let per_triplet_stride = num_temps * num_band0;
    for slot in collisions.iter_mut() {
        *slot = 0.0;
    }
    if is_n_u {
        for (i, triplet) in triplets.iter().enumerate() {
            let offset = if is_n(*triplet, bz_grid_addresses) {
                0
            } else {
                per_triplet_stride
            };
            let src = &ise_per_triplet[i * per_triplet_stride..(i + 1) * per_triplet_stride];
            let dst = &mut collisions[offset..offset + per_triplet_stride];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
    } else {
        for i in 0..triplets.len() {
            let src = &ise_per_triplet[i * per_triplet_stride..(i + 1) * per_triplet_stride];
            for (d, s) in collisions.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
    }
}

/// Tetrahedron-method driver, port of `ppc_get_pp_collision`.
///
/// Writes `collisions` with shape `(num_temps, num_band0)` or
/// `(2, num_temps, num_band0)` when `is_n_u` is true.
///
/// Always parallelizes over triplets with rayon; per-triplet kernels
/// run their own internal rayon loops, and rayon's work-stealing
/// handles the nested case without oversubscribing.
#[allow(clippy::too_many_arguments)]
pub fn get_pp_collision(
    collisions: &mut [f64],
    relative_grid_address: &RelativeGridAddress,
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    bzgrid: &BzGridView<'_>,
    d_diag: [i64; 3],
    q_mat: [[i64; 3]; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets<'_>,
    masses: &[f64],
    band_indices: &[i64],
    temperatures_thz: &[f64],
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    cutoff_frequency: f64,
    num_band0: usize,
    num_band: usize,
) -> Result<(), BzGridError> {
    let num_triplets = triplets.len();
    let num_temps = temperatures_thz.len();
    let num_band_prod = num_band0 * num_band * num_band;
    let per_triplet_stride = num_temps * num_band0;

    let freqs_at_gp = collect_freqs_at_gp(frequencies, triplets[0][0], band_indices, num_band);
    let tp_relative = set_relative_grid_address(relative_grid_address, 2);

    let mut ise_per_triplet = vec![0.0f64; num_triplets * per_triplet_stride];

    let run_one =
        |scratch: &mut CollisionScratch,
         (i, ise_slot): (usize, &mut [f64])|
         -> Result<(), BzGridError> {
            let triplet = triplets[i];
            scratch.reset();
            {
                let (g1_slot, g2_3_slot) = scratch.g_buf.split_at_mut(num_band_prod);
                let mut iw_ch: Vec<&mut [f64]> = vec![g1_slot, g2_3_slot];
                integration_weight_per_triplet(
                    &mut iw_ch,
                    &mut scratch.g_zero,
                    &freqs_at_gp,
                    num_band0 as i64,
                    &tp_relative,
                    triplet,
                    bzgrid,
                    frequencies,
                    num_band as i64,
                    frequencies,
                    num_band as i64,
                    TpType::Type2,
                )?;
            }
            evaluate_collision_at_triplet(
                ise_slot,
                &mut scratch.fc3_normal_squared,
                triplet,
                triplet_weights[i],
                &scratch.g_buf,
                &scratch.g_zero,
                num_band0,
                num_band,
                num_temps,
                temperatures_thz,
                frequencies,
                eigenvectors,
                bzgrid.addresses,
                d_diag,
                q_mat,
                fc3,
                is_compact_fc3,
                atom_triplets,
                masses,
                band_indices,
                symmetrize_fc3_q,
                cutoff_frequency,
            );
            Ok(())
        };

    ise_per_triplet
        .par_chunks_mut(per_triplet_stride)
        .enumerate()
        .try_for_each_init(|| CollisionScratch::new(num_band_prod), run_one)?;

    finalize(
        collisions,
        &ise_per_triplet,
        triplets,
        bzgrid.addresses,
        num_temps,
        num_band0,
        is_n_u,
    );
    Ok(())
}

/// Gaussian-smearing driver, port of `ppc_get_pp_collision_with_sigma`.
///
/// `sigma_cutoff <= 0` disables the cutoff-skip optimisation (matches
/// the C semantics).  Otherwise the Gaussian is skipped when the
/// integrand is outside `sigma * sigma_cutoff` from the central
/// frequency.
#[allow(clippy::too_many_arguments)]
pub fn get_pp_collision_with_sigma(
    collisions: &mut [f64],
    sigma: f64,
    sigma_cutoff: f64,
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    triplets: &[[i64; 3]],
    triplet_weights: &[i64],
    bz_grid_addresses: &[[i64; 3]],
    d_diag: [i64; 3],
    q_mat: [[i64; 3]; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets<'_>,
    masses: &[f64],
    band_indices: &[i64],
    temperatures_thz: &[f64],
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    cutoff_frequency: f64,
    num_band0: usize,
    num_band: usize,
) {
    let num_triplets = triplets.len();
    let num_temps = temperatures_thz.len();
    let num_band_prod = num_band0 * num_band * num_band;
    let per_triplet_stride = num_temps * num_band0;
    let cutoff = sigma * sigma_cutoff;

    let freqs_at_gp = collect_freqs_at_gp(frequencies, triplets[0][0], band_indices, num_band);

    let mut ise_per_triplet = vec![0.0f64; num_triplets * per_triplet_stride];

    let run_one = |scratch: &mut CollisionScratch, (i, ise_slot): (usize, &mut [f64])| {
        let triplet = triplets[i];
        scratch.reset();
        {
            let (g1_slot, g2_3_slot) = scratch.g_buf.split_at_mut(num_band_prod);
            let mut iw_ch: Vec<&mut [f64]> = vec![g1_slot, g2_3_slot];
            integration_weight_with_sigma_per_triplet(
                &mut iw_ch,
                &mut scratch.g_zero,
                sigma,
                cutoff,
                &freqs_at_gp,
                num_band0 as i64,
                triplet,
                frequencies,
                num_band as i64,
                TpType::Type2,
            );
        }
        evaluate_collision_at_triplet(
            ise_slot,
            &mut scratch.fc3_normal_squared,
            triplet,
            triplet_weights[i],
            &scratch.g_buf,
            &scratch.g_zero,
            num_band0,
            num_band,
            num_temps,
            temperatures_thz,
            frequencies,
            eigenvectors,
            bz_grid_addresses,
            d_diag,
            q_mat,
            fc3,
            is_compact_fc3,
            atom_triplets,
            masses,
            band_indices,
            symmetrize_fc3_q,
            cutoff_frequency,
        );
    };

    ise_per_triplet
        .par_chunks_mut(per_triplet_stride)
        .enumerate()
        .for_each_init(|| CollisionScratch::new(num_band_prod), run_one);

    finalize(
        collisions,
        &ise_per_triplet,
        triplets,
        bz_grid_addresses,
        num_temps,
        num_band0,
        is_n_u,
    );
}

/// Build `freqs_at_gp[j] = frequencies[gp0 * num_band + band_indices[j]]`.
fn collect_freqs_at_gp(
    frequencies: &[f64],
    gp0: i64,
    band_indices: &[i64],
    num_band: usize,
) -> Vec<f64> {
    let base = gp0 as usize * num_band;
    band_indices
        .iter()
        .map(|&b| frequencies[base + b as usize])
        .collect()
}
