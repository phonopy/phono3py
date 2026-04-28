//! Ph-ph interaction strength in normal-mode coordinates.
//!
//! Port of `itr_get_interaction` and its static helpers in
//! `c/interaction.c`.  Wires the existing Rust ports of
//! `real_to_reciprocal` and `reciprocal_to_normal_squared` into a
//! per-triplet driver, with an optional 6-way q-index symmetrization.

use rayon::prelude::*;

use crate::common::Cmplx;
use crate::imag_self_energy::set_g_pos;
use crate::real_to_reciprocal::{real_to_reciprocal, AtomTriplets};
use crate::reciprocal_to_normal::{reciprocal_to_normal_squared, R2NScratch};

/// Per-call scratch for the interaction kernel.  Owned by the caller
/// (typically a per-rayon-worker `for_each_init` in the multi-triplet
/// driver) so the large `fc3_reciprocal` and `R2NScratch` buffers are
/// allocated once per worker, then reused across the triplets that
/// worker processes.
#[derive(Default)]
pub struct InteractionScratch {
    /// Atom-first reciprocal-space fc3 buffer.  Length
    /// `num_patom^3 * 27`.  At 56 atoms this is ~76 MiB; per-triplet
    /// re-allocation drove allocator/page-fault overhead on many-core
    /// runs (see RUST_PORTING.md notes for Ca(BO2)2).
    pub fc3_reciprocal: Vec<Cmplx>,
    /// Symmetrized 6-way path scratch: full `(num_band, num_band,
    /// num_band)` real block.  Empty unless `symmetrize_fc3_q` is set.
    pub sym_q_tmp: Vec<f64>,
    /// Non-zero `g_zero` positions for the current triplet.  Capacity
    /// up to `num_band0 * num_band^2` (~ 150 MiB at num_patom = 56).
    /// Populated by `set_g_pos` at the start of each per-triplet kernel
    /// and reused for the downstream `imag_self_energy_at_triplet` call
    /// in `evaluate_collision_at_triplet`.
    pub g_pos: Vec<[i64; 4]>,
    /// Sub-scratch for `reciprocal_to_normal_squared`.
    pub r2n: R2NScratch,
}

impl InteractionScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fixed 6 permutations of (q0, q1, q2) used by `symmetrize_fc3_q`.
const INDEX_EXCHANGE: [[usize; 3]; 6] = [
    [0, 1, 2],
    [2, 0, 1],
    [1, 2, 0],
    [2, 1, 0],
    [0, 2, 1],
    [1, 0, 2],
];

/// Compute `q_vecs[3][3]` for a triplet.  For each vertex `j`,
/// `q_vecs[j] = Q * (address[triplet[j]] / D_diag)` where the division
/// is element-wise and `Q` is the 3x3 integer generator matrix.
fn compute_q_vecs(
    triplet: [i64; 3],
    bz_grid_addresses: &[[i64; 3]],
    d_diag: [i64; 3],
    q_mat: [[i64; 3]; 3],
) -> [[f64; 3]; 3] {
    let mut q_vecs = [[0.0f64; 3]; 3];
    for j in 0..3 {
        let addr = bz_grid_addresses[triplet[j] as usize];
        let raw = [
            addr[0] as f64 / d_diag[0] as f64,
            addr[1] as f64 / d_diag[1] as f64,
            addr[2] as f64 / d_diag[2] as f64,
        ];
        for k in 0..3 {
            q_vecs[j][k] = q_mat[k][0] as f64 * raw[0]
                + q_mat[k][1] as f64 * raw[1]
                + q_mat[k][2] as f64 * raw[2];
        }
    }
    q_vecs
}

/// Per-triplet kernel matching `real_to_normal` in `c/interaction.c`:
/// transform fc3 to reciprocal space then project onto the normal-mode
/// basis.  Writes `fc3_normal_squared` of length `num_band0 * num_band
/// * num_band`.
#[allow(clippy::too_many_arguments)]
fn real_to_normal(
    fc3_normal_squared: &mut [f64],
    g_pos: &[[i64; 4]],
    freqs0: &[f64],
    freqs1: &[f64],
    freqs2: &[f64],
    eigvecs0: &[Cmplx],
    eigvecs1: &[Cmplx],
    eigvecs2: &[Cmplx],
    fc3: &[f64],
    is_compact_fc3: bool,
    q_vecs: [[f64; 3]; 3],
    atom_triplets: &AtomTriplets,
    masses: &[f64],
    band_indices: &[i64],
    num_patom: usize,
    cutoff_frequency: f64,
    inner_par: bool,
    scratch: &mut InteractionScratch,
) {
    let fc3_recip_len = num_patom * num_patom * num_patom * 27;
    scratch.fc3_reciprocal.resize(fc3_recip_len, [0.0, 0.0]);
    // NOTE: always sequential.  Splitting the num_patom^3 block loop with
    // rayon inside the nested-par call path regressed 46.92s -> 50.30s
    // on NaMgF3 128 threads (2026-04-21); each r0_average_block chunk is
    // too small to amortize rayon launch overhead, and the outer
    // par_chunks_mut over triplets + reciprocal_to_normal_squared's
    // inner par already provide enough work-stealing pool.  The
    // real_to_reciprocal `inner_par` parameter is kept for the
    // single-triplet Python entry (py_real_to_reciprocal), which has
    // no outer par and genuinely benefits.
    real_to_reciprocal(
        &mut scratch.fc3_reciprocal,
        &q_vecs,
        fc3,
        is_compact_fc3,
        atom_triplets,
        false,
    );
    reciprocal_to_normal_squared(
        fc3_normal_squared,
        g_pos,
        &scratch.fc3_reciprocal,
        freqs0,
        freqs1,
        freqs2,
        eigvecs0,
        eigvecs1,
        eigvecs2,
        masses,
        band_indices,
        num_patom,
        cutoff_frequency,
        inner_par,
        &mut scratch.r2n,
    );
}

/// Per-triplet kernel with 6-way q-index symmetrization (mirrors
/// `real_to_normal_sym_q` in `c/interaction.c`).  Each of the 6
/// permutations computes a full `(num_band, num_band, num_band)` block;
/// the result is accumulated into `fc3_normal_squared` of shape
/// `(num_band0, num_band, num_band)` at the re-indexed position and
/// divided by 6.
#[allow(clippy::too_many_arguments)]
fn real_to_normal_sym_q(
    fc3_normal_squared: &mut [f64],
    g_pos: &[[i64; 4]],
    freqs: [&[f64]; 3],
    eigvecs: [&[Cmplx]; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    q_vecs: [[f64; 3]; 3],
    atom_triplets: &AtomTriplets,
    masses: &[f64],
    band_indices: &[i64],
    num_patom: usize,
    cutoff_frequency: f64,
    inner_par: bool,
    scratch: &mut InteractionScratch,
) {
    let num_band = num_patom * 3;
    let num_band0 = band_indices.len();
    debug_assert_eq!(fc3_normal_squared.len(), num_band0 * num_band * num_band);

    for slot in fc3_normal_squared.iter_mut() {
        *slot = 0.0;
    }

    // The permuted kernel needs band_indices_ex = 0..num_band and
    // g_pos covering every (b0, k, l) position, since the 6-way
    // mapping can route output into any band triple.
    let full_band_indices: Vec<i64> = (0..num_band as i64).collect();
    let full = build_full_g_pos(num_band);
    scratch
        .sym_q_tmp
        .resize(num_band * num_band * num_band, 0.0);

    for perm in INDEX_EXCHANGE.iter() {
        let mut q_vecs_ex = [[0.0f64; 3]; 3];
        for j in 0..3 {
            q_vecs_ex[j] = q_vecs[perm[j]];
        }
        // Move tmp out of scratch so it can be borrowed mutably while
        // the rest of scratch (fc3_reciprocal / r2n) is also borrowed.
        let mut tmp = std::mem::take(&mut scratch.sym_q_tmp);
        real_to_normal(
            &mut tmp,
            &full,
            freqs[perm[0]],
            freqs[perm[1]],
            freqs[perm[2]],
            eigvecs[perm[0]],
            eigvecs[perm[1]],
            eigvecs[perm[2]],
            fc3,
            is_compact_fc3,
            q_vecs_ex,
            atom_triplets,
            masses,
            &full_band_indices,
            num_patom,
            cutoff_frequency,
            inner_par,
            scratch,
        );

        // Scatter tmp into fc3_normal_squared using band index exchange.
        // Only positions present in the caller's g_pos are touched.
        for gp in g_pos {
            let band0 = gp[0] as usize;
            let k = gp[1] as usize;
            let l = gp[2] as usize;
            let j = band_indices[band0] as usize;
            let bands = [j, k, l];
            let src_idx =
                bands[perm[0]] * num_band * num_band + bands[perm[1]] * num_band + bands[perm[2]];
            let dst_idx = band0 * num_band * num_band + k * num_band + l;
            fc3_normal_squared[dst_idx] += tmp[src_idx] / 6.0;
        }

        scratch.sym_q_tmp = tmp;
    }
}

fn build_full_g_pos(num_band: usize) -> Vec<[i64; 4]> {
    let n = num_band as i64;
    let mut out = Vec::with_capacity((n * n * n) as usize);
    let mut idx: i64 = 0;
    for b0 in 0..n {
        for k in 0..n {
            for l in 0..n {
                out.push([b0, k, l, idx]);
                idx += 1;
            }
        }
    }
    out
}

/// Per-triplet driver, dispatching to the plain or symmetrized kernel.
#[allow(clippy::too_many_arguments)]
pub fn get_interaction_at_triplet(
    fc3_normal_squared: &mut [f64],
    g_zero_triplet: &[i8],
    num_band0: usize,
    num_band: usize,
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    triplet: [i64; 3],
    bz_grid_addresses: &[[i64; 3]],
    d_diag: [i64; 3],
    q_mat: [[i64; 3]; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets,
    masses: &[f64],
    band_indices: &[i64],
    symmetrize_fc3_q: bool,
    cutoff_frequency: f64,
    inner_par: bool,
    scratch: &mut InteractionScratch,
) {
    let num_patom = num_band / 3;
    let q_vecs = compute_q_vecs(triplet, bz_grid_addresses, d_diag, q_mat);
    // Move g_pos out of scratch so the rest of scratch can be borrowed
    // mutably while g_pos is read by the kernels.  Restored at end.
    let mut g_pos = std::mem::take(&mut scratch.g_pos);
    set_g_pos(&mut g_pos, g_zero_triplet, num_band0, num_band);

    if symmetrize_fc3_q {
        let mut freqs_arr: [Vec<f64>; 3] = [
            vec![0.0; num_band],
            vec![0.0; num_band],
            vec![0.0; num_band],
        ];
        let mut evecs_arr: [Vec<Cmplx>; 3] = [
            vec![[0.0, 0.0]; num_band * num_band],
            vec![[0.0, 0.0]; num_band * num_band],
            vec![[0.0, 0.0]; num_band * num_band],
        ];
        for j in 0..3 {
            let gp = triplet[j] as usize;
            freqs_arr[j].copy_from_slice(&frequencies[gp * num_band..(gp + 1) * num_band]);
            evecs_arr[j].copy_from_slice(
                &eigenvectors[gp * num_band * num_band..(gp + 1) * num_band * num_band],
            );
        }
        let freqs = [
            freqs_arr[0].as_slice(),
            freqs_arr[1].as_slice(),
            freqs_arr[2].as_slice(),
        ];
        let eigvecs = [
            evecs_arr[0].as_slice(),
            evecs_arr[1].as_slice(),
            evecs_arr[2].as_slice(),
        ];
        real_to_normal_sym_q(
            fc3_normal_squared,
            &g_pos,
            freqs,
            eigvecs,
            fc3,
            is_compact_fc3,
            q_vecs,
            atom_triplets,
            masses,
            band_indices,
            num_patom,
            cutoff_frequency,
            inner_par,
            scratch,
        );
    } else {
        let base0 = triplet[0] as usize * num_band;
        let base1 = triplet[1] as usize * num_band;
        let base2 = triplet[2] as usize * num_band;
        let ebase0 = triplet[0] as usize * num_band * num_band;
        let ebase1 = triplet[1] as usize * num_band * num_band;
        let ebase2 = triplet[2] as usize * num_band * num_band;
        real_to_normal(
            fc3_normal_squared,
            &g_pos,
            &frequencies[base0..base0 + num_band],
            &frequencies[base1..base1 + num_band],
            &frequencies[base2..base2 + num_band],
            &eigenvectors[ebase0..ebase0 + num_band * num_band],
            &eigenvectors[ebase1..ebase1 + num_band * num_band],
            &eigenvectors[ebase2..ebase2 + num_band * num_band],
            fc3,
            is_compact_fc3,
            q_vecs,
            atom_triplets,
            masses,
            band_indices,
            num_patom,
            cutoff_frequency,
            inner_par,
            scratch,
        );
    }

    scratch.g_pos = g_pos;
}

/// Main entry: port of `itr_get_interaction` in `c/interaction.c`.
///
/// Writes `fc3_normal_squared` of shape `(num_triplets, num_band0,
/// num_band, num_band)` flat.  Parallelizes over triplets with rayon.
/// When `num_triplets < num_threads` the outer loop cannot saturate
/// the thread pool, so the per-triplet kernel also enables its own
/// inner rayon loops via `inner_par = true`.  Rayon's work-stealing
/// scheduler bridges across-triplet granularity gaps (e.g.  Phase 1
/// of one triplet backfills with Phase 2 of another).
#[allow(clippy::too_many_arguments)]
pub fn get_interaction(
    fc3_normal_squared: &mut [f64],
    g_zero: &[i8],
    frequencies: &[f64],
    eigenvectors: &[Cmplx],
    triplets: &[[i64; 3]],
    bz_grid_addresses: &[[i64; 3]],
    d_diag: [i64; 3],
    q_mat: [[i64; 3]; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets,
    masses: &[f64],
    band_indices: &[i64],
    symmetrize_fc3_q: bool,
    cutoff_frequency: f64,
    num_band0: usize,
    num_band: usize,
) {
    let num_triplets = triplets.len();
    let num_band_prod = num_band0 * num_band * num_band;
    debug_assert_eq!(fc3_normal_squared.len(), num_triplets * num_band_prod);
    debug_assert_eq!(g_zero.len(), num_triplets * num_band_prod);

    let inner_par = num_triplets < rayon::current_num_threads();

    fc3_normal_squared
        .par_chunks_mut(num_band_prod)
        .enumerate()
        .for_each_init(InteractionScratch::new, |scratch, (i, slot)| {
            let gz = &g_zero[i * num_band_prod..(i + 1) * num_band_prod];
            get_interaction_at_triplet(
                slot,
                gz,
                num_band0,
                num_band,
                frequencies,
                eigenvectors,
                triplets[i],
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
                scratch,
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_g_pos_covers_all_positions() {
        let g = build_full_g_pos(2);
        assert_eq!(g.len(), 8);
        assert_eq!(g[0], [0, 0, 0, 0]);
        assert_eq!(g[7], [1, 1, 1, 7]);
    }

    #[test]
    fn compute_q_vecs_identity_q() {
        // D_diag = [2,2,2], Q = identity; address [1,0,-1] -> [0.5, 0, -0.5].
        let addrs: Vec<[i64; 3]> = vec![[0, 0, 0], [1, 0, -1]];
        let d = [2i64, 2, 2];
        let q = [[1i64, 0, 0], [0, 1, 0], [0, 0, 1]];
        let qv = compute_q_vecs([0, 1, 0], &addrs, d, q);
        assert_eq!(qv[0], [0.0, 0.0, 0.0]);
        assert!((qv[1][0] - 0.5).abs() < 1e-15);
        assert!((qv[1][2] + 0.5).abs() < 1e-15);
        assert_eq!(qv[2], [0.0, 0.0, 0.0]);
    }
}
