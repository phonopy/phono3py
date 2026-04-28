//! Port of `c/pp_collision.c`.
//!
//! Per-grid-point, low-memory driver that fuses integration-weight,
//! interaction, and imag-self-energy evaluation into a single loop over
//! triplets.  The caller receives gamma already summed over triplets
//! (optionally split into Normal/Umklapp channels).
//!
//! This mirrors `ppc_get_pp_collision` (tetrahedron method) and
//! `ppc_get_pp_collision_with_sigma` (Gaussian smearing).

use std::cell::RefCell;

use rayon::prelude::*;

use crate::common::Cmplx;
use crate::imag_self_energy::imag_self_energy_at_triplet;
use crate::interaction::{get_interaction_at_triplet, InteractionScratch};
use crate::real_to_reciprocal::AtomTriplets;
use crate::triplet::{is_n, set_relative_grid_address, RelativeGridAddress};
use crate::triplet_iw::{
    integration_weight_per_triplet, integration_weight_with_sigma_per_triplet, BzGridError,
    BzGridView, TpType,
};

/// Scratch buffers reused across triplets to avoid per-triplet heap
/// churn.  Held in a thread-local cell (see `COLLISION_SCRATCH`) so
/// each rayon worker thread allocates its ~250 MiB scratch (at
/// num_patom = 56) once at first use and reuses it for the rest of
/// the program lifetime, across all driver invocations and batches.
/// Compared to `for_each_init`, this fixes the alloc count to
/// `num_workers` rather than scaling with rayon's adaptive job
/// split count.
#[derive(Default)]
struct CollisionScratch {
    fc3_normal_squared: Vec<f64>,
    g_buf: Vec<f64>,
    g_zero: Vec<i8>,
    interaction: InteractionScratch,
}

impl CollisionScratch {
    /// Resize all per-triplet buffers to match the current shape and
    /// zero them.  Called once per triplet at the top of the kernel.
    /// Vec::resize is a no-op when the size is unchanged (the common
    /// case after the first call on each worker), so amortized cost is
    /// just the fill.
    fn reset(&mut self, num_band_prod: usize) {
        self.fc3_normal_squared.resize(num_band_prod, 0.0);
        self.fc3_normal_squared.fill(0.0);
        self.g_buf.resize(2 * num_band_prod, 0.0);
        self.g_buf.fill(0.0);
        self.g_zero.resize(num_band_prod, 0);
        self.g_zero.fill(0);
    }
}

thread_local! {
    /// Per-rayon-worker persistent scratch.  `None` until first use on
    /// each worker; populated lazily via `get_or_insert_with`.
    static COLLISION_SCRATCH: RefCell<Option<CollisionScratch>> =
        const { RefCell::new(None) };
}

/// Acquire the calling thread's `CollisionScratch`, run `f`, and
/// release the borrow.  Panics if called recursively (which would
/// indicate two collision kernels stacked on the same thread).
fn with_scratch<F, R>(num_band_prod: usize, f: F) -> R
where
    F: FnOnce(&mut CollisionScratch) -> R,
{
    COLLISION_SCRATCH.with(|cell| {
        let mut opt = cell.borrow_mut();
        let scratch = opt.get_or_insert_with(CollisionScratch::default);
        scratch.reset(num_band_prod);
        f(scratch)
    })
}

/// Release the per-rayon-worker `CollisionScratch` instances back to
/// the allocator.  Visits every worker thread in the global rayon
/// pool via `rayon::broadcast` and drops the held `Option<Scratch>`.
///
/// **Must not be called from inside a rayon parallel iterator** — it
/// would deadlock waiting for workers that are themselves blocked
/// inside the broadcast.  Intended to be called from Python after a
/// collision kernel has returned, e.g. before LBTE's memory-heavy
/// kappa-solve diagonalization, when the ~250 MiB-per-worker
/// (num_patom = 56) scratch is no longer needed.  The next collision
/// kernel will lazily re-allocate on first use.
pub fn release_scratch() {
    rayon::broadcast(|_ctx| {
        COLLISION_SCRATCH.with(|cell| {
            *cell.borrow_mut() = None;
        });
    });
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
    inner_par: bool,
    interaction_scratch: &mut InteractionScratch,
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
        inner_par,
        interaction_scratch,
    );

    // get_interaction_at_triplet has populated `interaction_scratch.g_pos`
    // (writes via set_g_pos at entry, restores at exit).  Reuse it
    // directly instead of re-running set_g_pos here.
    let g_pos = interaction_scratch.g_pos.as_slice();
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
            g_pos,
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
    let inner_par = num_triplets < rayon::current_num_threads();

    let mut ise_per_triplet = vec![0.0f64; num_triplets * per_triplet_stride];

    let run_one = |scratch: &mut CollisionScratch,
                   (i, ise_slot): (usize, &mut [f64])|
     -> Result<(), BzGridError> {
        let triplet = triplets[i];
        // scratch already reset+sized by `with_scratch` wrapper.
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
            inner_par,
            &mut scratch.interaction,
        );
        Ok(())
    };

    ise_per_triplet
        .par_chunks_mut(per_triplet_stride)
        .enumerate()
        .try_for_each(|item| with_scratch(num_band_prod, |scratch| run_one(scratch, item)))?;

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
    let inner_par = num_triplets < rayon::current_num_threads();

    let mut ise_per_triplet = vec![0.0f64; num_triplets * per_triplet_stride];

    let run_one = |scratch: &mut CollisionScratch, (i, ise_slot): (usize, &mut [f64])| {
        let triplet = triplets[i];
        // scratch already reset+sized by `with_scratch` wrapper.
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
            inner_par,
            &mut scratch.interaction,
        );
    };

    ise_per_triplet
        .par_chunks_mut(per_triplet_stride)
        .enumerate()
        .for_each(|item| with_scratch(num_band_prod, |scratch| run_one(scratch, item)));

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

/// Tetrahedron-method driver over a **batch of grid points**, port of
/// the per-gp `get_pp_collision` broadened to process multiple gps in
/// a single flat rayon par over `(gp, triplet)` pairs.
///
/// The batch's combined triplet count is expected to exceed the
/// rayon thread count; in that regime this collapses the nested-par
/// work-stealing overhead (`do_spin`) that per-gp calls incur when
/// `num_triplets < num_threads`.  `inner_par` is therefore forced
/// to `false` inside the per-triplet kernel.
///
/// Shapes:
/// - `collisions_per_gp[g]` is the single-sigma output for gp `g`
///   (`(num_temps, num_band0)` or `(2, num_temps, num_band0)` flat).
/// - `triplets_per_gp[g]` / `triplet_weights_per_gp[g]`: gp `g`'s
///   triplets and ir-weights.
/// - All other arguments are global across the batch and match the
///   per-gp `get_pp_collision` semantics.
#[allow(clippy::too_many_arguments)]
pub fn get_pp_collision_multi_gp(
    collisions_per_gp: &mut [&mut [f64]],
    triplets_per_gp: &[&[[i64; 3]]],
    triplet_weights_per_gp: &[&[i64]],
    relative_grid_address: &RelativeGridAddress,
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
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
    let num_gps = triplets_per_gp.len();
    assert_eq!(triplet_weights_per_gp.len(), num_gps);
    assert_eq!(collisions_per_gp.len(), num_gps);

    let num_temps = temperatures_thz.len();
    let num_band_prod = num_band0 * num_band * num_band;
    let per_triplet_stride = num_temps * num_band0;

    // Per-gp freqs_at_gp (cheap; gp0 = triplets_per_gp[g][0][0]).
    let freqs_per_gp: Vec<Vec<f64>> = triplets_per_gp
        .iter()
        .map(|ts| {
            if ts.is_empty() {
                Vec::new()
            } else {
                collect_freqs_at_gp(frequencies, ts[0][0], band_indices, num_band)
            }
        })
        .collect();

    // Flat work list: one entry per (gp, triplet) pair.
    let flat_work: Vec<(usize, [i64; 3], i64)> = triplets_per_gp
        .iter()
        .enumerate()
        .flat_map(|(g, ts)| {
            ts.iter()
                .zip(triplet_weights_per_gp[g].iter())
                .map(move |(&t, &w)| (g, t, w))
        })
        .collect();
    let total_triplets = flat_work.len();

    if total_triplets == 0 {
        for out in collisions_per_gp.iter_mut() {
            for v in out.iter_mut() {
                *v = 0.0;
            }
        }
        return Ok(());
    }

    let tp_relative = set_relative_grid_address(relative_grid_address, 2);

    // Per-triplet ISE output, flat across the batch.
    let mut ise_all = vec![0.0f64; total_triplets * per_triplet_stride];

    let run_one = |scratch: &mut CollisionScratch,
                   (flat_idx, ise_slot): (usize, &mut [f64])|
     -> Result<(), BzGridError> {
        let (gp_idx, triplet, weight) = flat_work[flat_idx];
        let freqs_at_gp = &freqs_per_gp[gp_idx];
        // scratch already reset+sized by `with_scratch` wrapper.
        {
            let (g1_slot, g2_3_slot) = scratch.g_buf.split_at_mut(num_band_prod);
            let mut iw_ch: Vec<&mut [f64]> = vec![g1_slot, g2_3_slot];
            integration_weight_per_triplet(
                &mut iw_ch,
                &mut scratch.g_zero,
                freqs_at_gp,
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
            weight,
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
            false, // batched outer par saturates the pool; no inner par.
            &mut scratch.interaction,
        );
        Ok(())
    };

    ise_all
        .par_chunks_mut(per_triplet_stride)
        .enumerate()
        .try_for_each(|item| with_scratch(num_band_prod, |scratch| run_one(scratch, item)))?;

    // Per-gp finalize.  Sequential (cheap; finalize is a O(num_triplets *
    // per_triplet_stride) add-reduction, dwarfed by the kernel work).
    let mut offset = 0usize;
    for g in 0..num_gps {
        let n = triplets_per_gp[g].len();
        let slice = &ise_all[offset * per_triplet_stride..(offset + n) * per_triplet_stride];
        finalize(
            collisions_per_gp[g],
            slice,
            triplets_per_gp[g],
            bzgrid.addresses,
            num_temps,
            num_band0,
            is_n_u,
        );
        offset += n;
    }

    Ok(())
}

/// Gaussian-smearing driver over a **batch of grid points**, mirror of
/// `get_pp_collision_multi_gp` for the `_with_sigma` path.
///
/// Differs from the tetrahedron variant only in the integration-weight
/// kernel and in using `bz_grid_addresses` directly instead of a full
/// `BzGridView` (matches the 1-gp Gaussian API).
#[allow(clippy::too_many_arguments)]
pub fn get_pp_collision_with_sigma_multi_gp(
    collisions_per_gp: &mut [&mut [f64]],
    sigma: f64,
    sigma_cutoff: f64,
    triplets_per_gp: &[&[[i64; 3]]],
    triplet_weights_per_gp: &[&[i64]],
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
    temperatures_thz: &[f64],
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    cutoff_frequency: f64,
    num_band0: usize,
    num_band: usize,
) {
    let num_gps = triplets_per_gp.len();
    assert_eq!(triplet_weights_per_gp.len(), num_gps);
    assert_eq!(collisions_per_gp.len(), num_gps);

    let num_temps = temperatures_thz.len();
    let num_band_prod = num_band0 * num_band * num_band;
    let per_triplet_stride = num_temps * num_band0;
    let cutoff = sigma * sigma_cutoff;

    let freqs_per_gp: Vec<Vec<f64>> = triplets_per_gp
        .iter()
        .map(|ts| {
            if ts.is_empty() {
                Vec::new()
            } else {
                collect_freqs_at_gp(frequencies, ts[0][0], band_indices, num_band)
            }
        })
        .collect();

    let flat_work: Vec<(usize, [i64; 3], i64)> = triplets_per_gp
        .iter()
        .enumerate()
        .flat_map(|(g, ts)| {
            ts.iter()
                .zip(triplet_weights_per_gp[g].iter())
                .map(move |(&t, &w)| (g, t, w))
        })
        .collect();
    let total_triplets = flat_work.len();

    if total_triplets == 0 {
        for out in collisions_per_gp.iter_mut() {
            for v in out.iter_mut() {
                *v = 0.0;
            }
        }
        return;
    }

    let mut ise_all = vec![0.0f64; total_triplets * per_triplet_stride];

    let run_one = |scratch: &mut CollisionScratch, (flat_idx, ise_slot): (usize, &mut [f64])| {
        let (gp_idx, triplet, weight) = flat_work[flat_idx];
        let freqs_at_gp = &freqs_per_gp[gp_idx];
        // scratch already reset+sized by `with_scratch` wrapper.
        {
            let (g1_slot, g2_3_slot) = scratch.g_buf.split_at_mut(num_band_prod);
            let mut iw_ch: Vec<&mut [f64]> = vec![g1_slot, g2_3_slot];
            integration_weight_with_sigma_per_triplet(
                &mut iw_ch,
                &mut scratch.g_zero,
                sigma,
                cutoff,
                freqs_at_gp,
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
            weight,
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
            false,
            &mut scratch.interaction,
        );
    };

    ise_all
        .par_chunks_mut(per_triplet_stride)
        .enumerate()
        .for_each(|item| with_scratch(num_band_prod, |scratch| run_one(scratch, item)));

    let mut offset = 0usize;
    for g in 0..num_gps {
        let n = triplets_per_gp[g].len();
        let slice = &ise_all[offset * per_triplet_stride..(offset + n) * per_triplet_stride];
        finalize(
            collisions_per_gp[g],
            slice,
            triplets_per_gp[g],
            bz_grid_addresses,
            num_temps,
            num_band0,
            is_n_u,
        );
        offset += n;
    }
}
