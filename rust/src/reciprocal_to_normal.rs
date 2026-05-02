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

/// Send+Sync raw-pointer wrapper for disjoint-index writes from rayon
/// tasks.  Same pattern as `collision_matrix.rs` / `fc3.rs`.
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
impl<T> SyncMutPtr<T> {
    #[inline]
    unsafe fn write(&self, idx: usize, val: T) {
        std::ptr::write(self.0.add(idx), val);
    }
}

/// Per-call scratch for `reciprocal_to_normal_squared`.  Buffers are
/// resized lazily on first use and reused on subsequent calls with the
/// same shape.  Owned by the caller (typically a per-rayon-worker
/// `for_each_init`) so that allocation cost is paid once per worker
/// rather than once per triplet.
#[derive(Default)]
pub struct R2NScratch {
    /// Eigenvectors at vertex 0, scaled by `1/sqrt(mass)` and transposed
    /// to `[band, component]`.  Length `num_band * num_band`.
    pub e0: Vec<Cmplx>,
    /// Same as `e0` for vertex 1.
    pub e1: Vec<Cmplx>,
    /// Same as `e0` for vertex 2.
    pub e2: Vec<Cmplx>,
    /// Phase 1 cache `(num_band0, num_band, num_band)` flat.
    pub fc3_e0: Vec<Cmplx>,
    /// Phase 2 row buffer of length `num_band` (used only by the
    /// `inner_par = false` branch).
    pub row: Vec<Cmplx>,
}

impl R2NScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Pre-scale eigenvectors by `1 / sqrt(mass)` and transpose so the
/// fast axis is the component axis (for a fixed band).
///
/// Input layout: `eigvecs[component * num_band + band]`
/// (`[component, band]` row-major).
///
/// Output layout: `out[band * num_band + component]`
/// (`[band, component]` row-major), with each element multiplied by
/// `1 / sqrt(masses[component / 3])`.
fn scale_and_transpose(out: &mut Vec<Cmplx>, eigvecs: &[Cmplx], masses: &[f64], num_band: usize) {
    out.resize(num_band * num_band, [0.0f64; 2]);
    for comp in 0..num_band {
        let atom = comp / 3;
        let inv_sqrt_mass = 1.0 / masses[atom].sqrt();
        for band in 0..num_band {
            let src = eigvecs[comp * num_band + band];
            out[band * num_band + comp] = [src[0] * inv_sqrt_mass, src[1] * inv_sqrt_mass];
        }
    }
}

/// Contract fc3 with `e0` at a single target band and a single `b`
/// index.
///
/// This produces one `(3, num_band)` slab of the per-band0 band-pair
/// matrix — the 3 rows whose band index decomposes as `bm = b * 3 + m`.
/// Distinct `b` values write to disjoint row slabs, which is what lets
/// `(band0, b)` be used as an independent parallel axis in Phase 1.
///
/// `out_slab` layout: flat `(3, num_band)` row-major with index
/// `m * num_band + cn`, where `cn = c * 3 + n`.  Computed as
/// `out_slab[m, cn] = sum over (a, l) of fc3[a, b, c, l, m, n] * e0_band[a * 3 + l]`.
fn contract_fc3_e0_slab(
    out_slab: &mut [Cmplx],
    fc3_reciprocal: &[Cmplx],
    e0_band: &[Cmplx],
    b: usize,
    num_patom: usize,
) {
    let num_band = num_patom * 3;
    debug_assert_eq!(out_slab.len(), 3 * num_band);
    for entry in out_slab.iter_mut() {
        *entry = [0.0, 0.0];
    }
    for a in 0..num_patom {
        for l in 0..3 {
            let e0_al = e0_band[a * 3 + l];
            if e0_al[0] == 0.0 && e0_al[1] == 0.0 {
                continue;
            }
            for c in 0..num_patom {
                let fc3_base = ((a * num_patom + b) * num_patom + c) * 27 + l * 9;
                for m in 0..3 {
                    for n in 0..3 {
                        let cn = c * 3 + n;
                        let prod = cmplx_mul(fc3_reciprocal[fc3_base + m * 3 + n], e0_al);
                        out_slab[m * num_band + cn][0] += prod[0];
                        out_slab[m * num_band + cn][1] += prod[1];
                    }
                }
            }
        }
    }
}

/// Contract fc3 with `e0` at a single target band, producing the full
/// `(num_band, num_band)` band-pair matrix.
///
/// Equivalent to concatenating `num_patom` calls to
/// `contract_fc3_e0_slab`, one per `b`.  Used by the sequential
/// Phase 1 path; the parallel path calls `contract_fc3_e0_slab`
/// directly to expose the `(band0, b)` flat parallel axis.
fn contract_fc3_e0(
    out: &mut [Cmplx],
    fc3_reciprocal: &[Cmplx],
    e0_band: &[Cmplx],
    num_patom: usize,
) {
    let num_band = num_patom * 3;
    let slab_size = 3 * num_band;
    debug_assert_eq!(out.len(), num_band * num_band);
    for (b, out_slab) in out.chunks_mut(slab_size).enumerate() {
        contract_fc3_e0_slab(out_slab, fc3_reciprocal, e0_band, b, num_patom);
    }
}

/// Complex-valued dot product `sum_i a[i] * b[i]` with 4-way partial
/// sums to break the IEEE-strict reduction dependency chain.  Each of
/// the four lanes accumulates 1/4 of the terms independently, then the
/// lanes are combined at the end.  This exposes ILP to LLVM that the
/// single-scalar form cannot express without `-ffast-math`.
///
/// `chunks_exact(4)` + fixed-size `&[Cmplx; 4]` binding destructure
/// lets LLVM see the 4-lane structure statically and emits the inner
/// loop with zero bounds checks.  Earlier `a[base + i]` form produced
/// eight `panic_bounds_check` branches per chunk iteration on aarch64.
///
/// `#[inline]` (hint, not force): benchmarked `#[inline(always)]` on
/// NaMgF3 128 threads and it regressed 46.25s -> 46.92s (2026-04-21),
/// likely due to code-size pressure in the Phase 2 par_iter closure
/// that outweighs prologue/epilogue savings on a 60-element loop.
#[inline]
fn cmplx_dot_partial4(a: &[Cmplx], b: &[Cmplx]) -> Cmplx {
    debug_assert_eq!(a.len(), b.len());
    let mut re0 = 0.0f64;
    let mut re1 = 0.0f64;
    let mut re2 = 0.0f64;
    let mut re3 = 0.0f64;
    let mut im0 = 0.0f64;
    let mut im1 = 0.0f64;
    let mut im2 = 0.0f64;
    let mut im3 = 0.0f64;

    let mut ai = a.chunks_exact(4);
    let mut bi = b.chunks_exact(4);
    for (ac, bc) in (&mut ai).zip(&mut bi) {
        let [a0, a1, a2, a3] = <&[Cmplx; 4]>::try_from(ac).unwrap();
        let [b0, b1, b2, b3] = <&[Cmplx; 4]>::try_from(bc).unwrap();
        re0 += a0[0] * b0[0] - a0[1] * b0[1];
        im0 += a0[0] * b0[1] + a0[1] * b0[0];
        re1 += a1[0] * b1[0] - a1[1] * b1[1];
        im1 += a1[0] * b1[1] + a1[1] * b1[0];
        re2 += a2[0] * b2[0] - a2[1] * b2[1];
        im2 += a2[0] * b2[1] + a2[1] * b2[0];
        re3 += a3[0] * b3[0] - a3[1] * b3[1];
        im3 += a3[0] * b3[1] + a3[1] * b3[0];
    }

    let mut re = (re0 + re1) + (re2 + re3);
    let mut im = (im0 + im1) + (im2 + im3);
    for (av, bv) in ai.remainder().iter().zip(bi.remainder().iter()) {
        re += av[0] * bv[0] - av[1] * bv[1];
        im += av[0] * bv[1] + av[1] * bv[0];
    }
    [re, im]
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
///
/// When `inner_par` is true, the per-band0 `contract_fc3_e0` loop and
/// the per-`g_pos` reduction loop are parallelized with rayon.  This is
/// intended for the regime where the caller's outer triplet-level
/// parallelism cannot saturate the available cores (typical on
/// many-core machines with small `num_triplets`).  When `inner_par` is
/// false both loops run sequentially so the caller's own rayon
/// parallelism owns all threads.
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
    inner_par: bool,
    scratch: &mut R2NScratch,
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

    // Destructure scratch up front so each field has its own mutable
    // borrow.  This lets Phase 1's `&mut fc3_e0` coexist with Phase 2's
    // `&mut row` (and with the immutable reads of `e0/e1/e2`) without
    // tripping the borrow checker.
    let R2NScratch {
        e0,
        e1,
        e2,
        fc3_e0,
        row,
    } = scratch;
    scale_and_transpose(e0, eigvecs0, masses, num_band);
    scale_and_transpose(e1, eigvecs1, masses, num_band);
    scale_and_transpose(e2, eigvecs2, masses, num_band);
    let e0: &[Cmplx] = e0.as_slice();
    let e1: &[Cmplx] = e1.as_slice();
    let e2: &[Cmplx] = e2.as_slice();

    // fc3_e0 caches the contraction over (a, l) at each target band0
    // index as a (num_band, num_band) band-pair matrix.  Shape:
    // (num_band0, num_band, num_band) flat.  entry_size equals
    // num_band * num_band (= num_patom * num_patom * 9).
    let num_band0 = band_indices.len();
    let entry_size = num_band * num_band;
    fc3_e0.resize(num_band0 * entry_size, [0.0f64; 2]);

    if inner_par {
        // Flat (band0, b) parallel axis: num_band0 * num_patom tasks
        // (vs. num_band0 for a per-band0 chunk).  On NaMgF3 20-atom
        // this lifts Phase 1 tasks from 60 to 1200, so 128 threads no
        // longer run with idle workers waiting to steal Phase 2 work
        // from other triplets.  Each slab covers the 3 rows whose
        // band index is `b * 3 + m`; distinct b values write to
        // disjoint slots, so slab-level par is safe.
        let slab_size = 3 * num_band;
        fc3_e0
            .par_chunks_mut(slab_size)
            .enumerate()
            .for_each(|(task_idx, out_slab)| {
                let i0 = task_idx / num_patom;
                let b = task_idx % num_patom;
                let bi = band_indices[i0] as usize;
                let e0_band = &e0[bi * num_band..(bi + 1) * num_band];
                contract_fc3_e0_slab(out_slab, fc3_reciprocal, e0_band, b, num_patom);
            });
    } else {
        for (i0, out) in fc3_e0.chunks_mut(entry_size).enumerate() {
            let bi = band_indices[i0] as usize;
            let e0_band = &e0[bi * num_band..(bi + 1) * num_band];
            contract_fc3_e0(out, fc3_reciprocal, e0_band, num_patom);
        }
    }

    // Downgrade fc3_e0 to a shared slice for Phase 2 so the par_iter
    // closure captures by `&[Cmplx]` (Sync) instead of `&mut [Cmplx]`
    // (not Sync).
    let fc3_e0: &[Cmplx] = fc3_e0.as_slice();

    // Phase 2a (`scratch[bm] = dot(fc3_e0[i0, bm, :], e2[k, :])`)
    // depends on `(i0, k)` but not on the q1 band `j`, so g_pos
    // entries that share the same `(i0, k)` can reuse one Phase 2a
    // result and only differ in Phase 2b (`dot(scratch, e1[j, :])`).
    // `set_g_pos` orders entries so each `(i0, k)` run is a
    // contiguous slice of `g_pos`; `find_b0_q2_runs` scans for the
    // run boundaries in a single linear pass.
    //
    // Empirical run length on NaMgF3 20-atom is ~4.2 entries on
    // average (max ~38), so Phase 2a calls are reduced by roughly 4x
    // versus the per-g_pos baseline.
    let b0_q2_runs = find_b0_q2_runs(g_pos);

    if inner_par {
        // SAFETY: each g_pos entry writes to a distinct `dest` index in
        // fc3_normal_squared (the caller's contract: g_pos encodes a
        // one-to-one mapping from (i0, j, k) to dest).  Rayon workers
        // therefore never collide on the same byte.  The raw-pointer
        // wrapper sidesteps the borrow checker's whole-slice &mut
        // uniqueness rule that par_iter on g_pos would otherwise break.
        let out_ptr = SyncMutPtr(fc3_normal_squared.as_mut_ptr());
        b0_q2_runs.par_iter().for_each_init(
            || vec![[0.0f64; 2]; num_band],
            |row_scratch, &(start, end)| {
                eval_v_squared_at_b0_q2(
                    row_scratch,
                    &g_pos[start..end],
                    fc3_e0,
                    e1,
                    e2,
                    freqs0,
                    freqs1,
                    freqs2,
                    band_indices,
                    num_band,
                    entry_size,
                    cutoff_frequency,
                    |dest, val| unsafe { out_ptr.write(dest, val) },
                );
            },
        );
    } else {
        // Phase 2a row buffer; reused across `(i0, k)` runs via the
        // caller-provided scratch.
        row.resize(num_band, [0.0f64; 2]);
        for &(start, end) in &b0_q2_runs {
            eval_v_squared_at_b0_q2(
                row,
                &g_pos[start..end],
                fc3_e0,
                e1,
                e2,
                freqs0,
                freqs1,
                freqs2,
                band_indices,
                num_band,
                entry_size,
                cutoff_frequency,
                |dest, val| fc3_normal_squared[dest] = val,
            );
        }
    }
}

/// Scan `g_pos` for runs of consecutive entries sharing the same
/// `(i0, k)` = `(gp[0], gp[2])` pair.  Returns one `(start, end)` per
/// run, where `g_pos[start..end]` is the contiguous slice.  Relies on
/// `set_g_pos` ordering entries so each `(i0, k)` pair forms a single
/// contiguous run.
fn find_b0_q2_runs(g_pos: &[[i64; 4]]) -> Vec<(usize, usize)> {
    let mut runs = Vec::with_capacity(g_pos.len() / 4 + 1);
    if g_pos.is_empty() {
        return runs;
    }
    let mut start = 0usize;
    for i in 1..g_pos.len() {
        if g_pos[i][0] != g_pos[start][0] || g_pos[i][2] != g_pos[start][2] {
            runs.push((start, i));
            start = i;
        }
    }
    runs.push((start, g_pos.len()));
    runs
}

/// Evaluate `|V|^2 / (f0 f1 f2)` for the `q1`-fan at fixed
/// `(b0, q2_band) = (entries[0][0], entries[0][2])`.  All `entries`
/// must share the same `(b0, q2_band)`; only the `q1_band` index
/// `entries[i][1]` varies.
///
/// `row_scratch` (length `num_band`) caches Phase 2a
/// (`scratch[bm] = dot(fc3_e0[b0, bm, :], e2[q2_band, :])`).  This is
/// computed once for the whole call and reused across all entries;
/// each entry then evaluates Phase 2b
/// (`|dot(scratch, e1[q1_band, :])|^2 / (f0 f1 f2)`).
///
/// `write` receives `(dest, val)` per entry — the sequential path
/// writes into a `&mut [f64]` slice directly; the parallel path
/// writes through a `SyncMutPtr` to bypass the borrow checker.
#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_v_squared_at_b0_q2<W: FnMut(usize, f64)>(
    row_scratch: &mut Vec<Cmplx>,
    entries: &[[i64; 4]],
    fc3_e0: &[Cmplx],
    e1: &[Cmplx],
    e2: &[Cmplx],
    freqs0: &[f64],
    freqs1: &[f64],
    freqs2: &[f64],
    band_indices: &[i64],
    num_band: usize,
    entry_size: usize,
    cutoff_frequency: f64,
    mut write: W,
) {
    let first = entries[0];
    let i0 = first[0] as usize;
    let k = first[2] as usize;
    let bi = band_indices[i0] as usize;
    let f0 = freqs0[bi];
    let f2 = freqs2[k];
    row_scratch.resize(num_band, [0.0f64; 2]);

    // Phase 2a: row-wise dot fc3_e0[i0, bm, :] . e2[k, :], once for
    // the whole (i0, k) call.
    let fc3_e0_entry = &fc3_e0[i0 * entry_size..(i0 + 1) * entry_size];
    let e2_band = &e2[k * num_band..(k + 1) * num_band];
    for (bm, slot) in row_scratch.iter_mut().enumerate() {
        let row = &fc3_e0_entry[bm * num_band..(bm + 1) * num_band];
        *slot = cmplx_dot_partial4(row, e2_band);
    }

    // Phase 2b per entry.  f0 and f2 are call-invariant; only f1
    // (and the cutoff branch on it) varies with the q1 band j.
    for gp in entries {
        let j = gp[1] as usize;
        let dest = gp[3] as usize;
        let f1 = freqs1[j];
        let val = if f0 <= cutoff_frequency || f1 <= cutoff_frequency || f2 <= cutoff_frequency {
            0.0
        } else {
            let e1_band = &e1[j * num_band..(j + 1) * num_band];
            let sum = cmplx_dot_partial4(row_scratch, e1_band);
            (sum[0] * sum[0] + sum[1] * sum[1]) / (f0 * f1 * f2)
        };
        write(dest, val);
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
        let mut out = Vec::new();
        scale_and_transpose(&mut out, &eigvecs, &masses, num_band);
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
        let mut out = Vec::new();
        scale_and_transpose(&mut out, &eigvecs, &masses, num_band);
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
            .flat_map(|i| {
                (0..3).flat_map(move |j| (0..3).map(move |k| [i, j, k, i * 9 + j * 3 + k]))
            })
            .collect();
        let mut out = vec![1.0f64; band_indices.len() * num_band * num_band];
        let mut scratch = R2NScratch::new();
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
            &mut scratch,
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

        let mut scratch = R2NScratch::new();
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
            &mut scratch,
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
            &mut scratch,
        );
        assert!(out[0] > 0.0);
    }
}
