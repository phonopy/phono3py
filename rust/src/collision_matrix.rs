//! Port of `c/collision_matrix.c`.
//!
//! Two per-grid-point driver functions used by the direct LBTE
//! solver:
//!
//! - `collision_matrix`: irreducible path; sums over k-star
//!   rotations per ir-grid-point.
//! - `reducible_collision_matrix`: no k-star reduction.
//!
//! Both loop over grid points at the outer level and reduce over the
//! third-phonon band index on the inside.  Within a single call, each
//! outer-loop iteration writes to a disjoint set of entries in
//! `collision_matrix`, so rayon dispatches the outer loop and the
//! per-task writes go through a local `SyncMutPtr`.
//!
//! Frequencies and temperatures are in THz units throughout.

use rayon::prelude::*;

use crate::funcs::inv_sinh_occupation;

/// `*mut T` wrapper opting into Send + Sync for rayon.  Only used in
/// parallel kernels where the offsets touched by each task are
/// manually verified to be disjoint.
#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    /// Method-style accessor for the raw pointer.  Calling this method
    /// (rather than touching the `.0` field) forces the 2021 edition's
    /// disjoint-capture analysis to capture the whole wrapper into a
    /// closure, preserving its Send + Sync impls.
    fn ptr(self) -> *mut T {
        self.0
    }
}

/// Per-grid-point to per-triplet index map.  Mirrors
/// `create_gp2tp_map` in `c/collision_matrix.c`: position `i` receives
/// the running ir-triplet index whenever `triplets_map[i] == i`, and
/// -1 otherwise (never read by the main loop).
fn create_gp2tp_map(triplets_map: &[i64]) -> Vec<i64> {
    let mut out = vec![-1i64; triplets_map.len()];
    let mut num_ir: i64 = 0;
    for (i, &m) in triplets_map.iter().enumerate() {
        if m == i as i64 {
            out[i] = num_ir;
            num_ir += 1;
        }
    }
    out
}

/// Fill `inv_sinh[0..num_band]` with `1 / sinh(f / (2 T))` for
/// frequencies at the q2 grid point of the triplet.  Returns whether
/// q1 and q2 are permuted in the triplet representation.  Mirrors
/// `get_inv_sinh` in C.
fn fill_inv_sinh(
    inv_sinh: &mut [f64],
    gp: usize,
    triplet: [i64; 3],
    triplets_map: &[i64],
    map_q: &[i64],
    frequencies: &[f64],
    num_band: usize,
    temperature_thz: f64,
    cutoff_frequency: f64,
) -> bool {
    // triplets_map[gp] == map_q[gp] in the canonical triplet ordering;
    // otherwise q1 and q2 are permuted.
    let swapped = triplets_map[gp] != map_q[gp];
    let gp2 = if swapped {
        triplet[1] as usize
    } else {
        triplet[2] as usize
    };
    let row = &frequencies[gp2 * num_band..(gp2 + 1) * num_band];
    for i in 0..num_band {
        let f = row[i];
        inv_sinh[i] = if f > cutoff_frequency {
            inv_sinh_occupation(f, temperature_thz)
        } else {
            0.0
        };
    }
    swapped
}

/// Port of `col_get_reducible_collision_matrix`.
///
/// Shapes (row-major):
/// - `collision_matrix`: `(num_band0, num_gp, num_band)`, accumulated.
/// - `fc3_normal_squared`: `(num_triplets, num_band0, num_band, num_band)`.
/// - `frequencies`: `(num_grid, num_band)`.
/// - `triplets`: `(num_triplets,)` of `[gp0, gp1, gp2]`.
/// - `triplets_map`, `map_q`: `(num_gp,)`.
/// - `g`: type-3 slab of the integration weights, same shape as
///   `fc3_normal_squared`; the caller passes it pre-offset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reducible_collision_matrix(
    collision_matrix: &mut [f64],
    fc3_normal_squared: &[f64],
    num_band0: usize,
    num_band: usize,
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplets_map: &[i64],
    map_q: &[i64],
    g: &[f64],
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) {
    let num_gp = triplets_map.len();
    let gp2tp = create_gp2tp_map(triplets_map);
    let stride_tri = num_band0 * num_band * num_band;
    let stride_b0 = num_band * num_band;

    let cm_ptr = SyncMutPtr(collision_matrix.as_mut_ptr());

    (0..num_gp).into_par_iter().for_each(|i| {
        let ti_i64 = gp2tp[triplets_map[i] as usize];
        debug_assert!(ti_i64 >= 0);
        let ti = ti_i64 as usize;
        let mut inv_sinh = vec![0.0f64; num_band];
        let swapped = fill_inv_sinh(
            &mut inv_sinh,
            i,
            triplets[ti],
            triplets_map,
            map_q,
            frequencies,
            num_band,
            temperature_thz,
            cutoff_frequency,
        );

        let base = ti * stride_tri;
        let cm = cm_ptr;
        for j in 0..num_band0 {
            let bj = base + j * stride_b0;
            let fc3_j = &fc3_normal_squared[bj..bj + stride_b0];
            let g_j = &g[bj..bj + stride_b0];
            for k in 0..num_band {
                let mut collision = 0.0f64;
                for l in 0..num_band {
                    let idx = if swapped {
                        l * num_band + k
                    } else {
                        k * num_band + l
                    };
                    collision += fc3_j[idx] * g_j[idx] * inv_sinh[l];
                }
                collision *= unit_conversion_factor;
                // collision_matrix[j, i, k]
                let out_idx = j * num_gp * num_band + i * num_band + k;
                // SAFETY: `i` is fixed per task and enters as the middle
                // axis; entries [*, i, *] written by one task are
                // disjoint from those written by any other task.
                unsafe {
                    *cm.ptr().add(out_idx) += collision;
                }
            }
        }
    });
}

/// Port of `col_get_collision_matrix`.
///
/// Shapes (row-major):
/// - `collision_matrix`: `(num_band0, 3, num_ir_gp, num_band, 3)`,
///   accumulated.
/// - `fc3_normal_squared`, `g`: `(num_triplets, num_band0, num_band, num_band)`.
/// - `frequencies`: `(num_grid, num_band)`.
/// - `triplets`: `(num_triplets,)` of `[gp0, gp1, gp2]`.
/// - `triplets_map`, `map_q`: `(num_gp,)`.
/// - `rot_grid_points`: `(num_ir_gp, num_rot)`.
/// - `rotations_cartesian`: `(num_rot, 3, 3)` row-major.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collision_matrix(
    collision_matrix: &mut [f64],
    fc3_normal_squared: &[f64],
    num_band0: usize,
    num_band: usize,
    frequencies: &[f64],
    triplets: &[[i64; 3]],
    triplets_map: &[i64],
    map_q: &[i64],
    rot_grid_points: &[i64],
    num_ir_gp: usize,
    num_rot: usize,
    rotations_cartesian: &[f64],
    g: &[f64],
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) {
    let gp2tp = create_gp2tp_map(triplets_map);
    let stride_tri = num_band0 * num_band * num_band;
    let stride_b0 = num_band * num_band;

    // Output strides for axes [band0, m_out, ir_gp, band, n].
    let s_b0 = 3 * num_ir_gp * num_band * 3;
    let s_mo = num_ir_gp * num_band * 3;
    let s_igp = num_band * 3;
    let s_l = 3;

    let cm_ptr = SyncMutPtr(collision_matrix.as_mut_ptr());

    (0..num_ir_gp).into_par_iter().for_each(|i| {
        let mut inv_sinh = vec![0.0f64; num_band];
        let cm = cm_ptr;
        for j in 0..num_rot {
            let r_gp = rot_grid_points[i * num_rot + j] as usize;
            let ti_i64 = gp2tp[triplets_map[r_gp] as usize];
            debug_assert!(ti_i64 >= 0);
            let ti = ti_i64 as usize;
            let swapped = fill_inv_sinh(
                &mut inv_sinh,
                r_gp,
                triplets[ti],
                triplets_map,
                map_q,
                frequencies,
                num_band,
                temperature_thz,
                cutoff_frequency,
            );
            let rot = &rotations_cartesian[j * 9..(j + 1) * 9];
            let base = ti * stride_tri;
            for k in 0..num_band0 {
                let bk = base + k * stride_b0;
                let fc3_k = &fc3_normal_squared[bk..bk + stride_b0];
                let g_k = &g[bk..bk + stride_b0];
                for l in 0..num_band {
                    let mut collision = 0.0f64;
                    for m in 0..num_band {
                        let idx = if swapped {
                            m * num_band + l
                        } else {
                            l * num_band + m
                        };
                        collision += fc3_k[idx] * g_k[idx] * inv_sinh[m];
                    }
                    collision *= unit_conversion_factor;
                    // collision_matrix[k, m_out, i, l, n] +=
                    //     collision * rotations_cartesian[j, m_out, n]
                    for m_out in 0..3 {
                        let r_row = &rot[m_out * 3..(m_out + 1) * 3];
                        let partial = [
                            collision * r_row[0],
                            collision * r_row[1],
                            collision * r_row[2],
                        ];
                        let row_base = k * s_b0 + m_out * s_mo + i * s_igp + l * s_l;
                        // SAFETY: `i` is fixed per task and enters as
                        // the ir_gp axis; the 5-tuple (k, m_out, i, l, n)
                        // is unique within a task, and the i coordinate
                        // makes ranges from different tasks disjoint.
                        unsafe {
                            *cm.ptr().add(row_base) += partial[0];
                            *cm.ptr().add(row_base + 1) += partial[1];
                            *cm.ptr().add(row_base + 2) += partial[2];
                        }
                    }
                }
            }
        }
    });
}

/// Port of `ph3py_symmetrize_collision_matrix`.
///
/// In-place symmetrization `(A + A^T) / 2` applied to each
/// `(num_column, num_column)` slice of a flat buffer laid out as
/// `(num_sigma, num_temp, num_column, num_column)` row-major.  The
/// caller decides `num_column` based on the collision-matrix ndim
/// (`num_grid_points * num_band` for 6D, times 3 for 8D).
///
/// The parallel outer loop over rows `k` is safe because the writes
/// from task `k` land in row `k` and column `k` only, and for any
/// pair `k != k'` those positions are disjoint (see the proof sketch
/// in comments below).
pub(crate) fn symmetrize_collision_matrix(
    collision_matrix: &mut [f64],
    num_column: usize,
    num_temp: usize,
    num_sigma: usize,
) {
    let n2 = num_column * num_column;
    let cm_ptr = SyncMutPtr(collision_matrix.as_mut_ptr());

    for i in 0..num_sigma {
        for j in 0..num_temp {
            let adrs_shift = i * num_temp * n2 + j * n2;
            // Task k writes positions (k, l) and (l, k) for l > k.
            // For k1 < k2, task k1 touches row k1 and column k1 only;
            // task k2 touches row k2 and column k2 only.  Rows and
            // columns are distinct, so the two tasks never collide.
            (0..num_column).into_par_iter().for_each(|k| {
                let cm = cm_ptr;
                for l in (k + 1)..num_column {
                    let a = adrs_shift + k * num_column + l;
                    let b = adrs_shift + l * num_column + k;
                    // SAFETY: positions (k, l) and (l, k) are unique
                    // to this task (see comment above).
                    unsafe {
                        let va = *cm.ptr().add(a);
                        let vb = *cm.ptr().add(b);
                        let v = 0.5 * (va + vb);
                        *cm.ptr().add(a) = v;
                        *cm.ptr().add(b) = v;
                    }
                }
            });
        }
    }
}

/// Port of `ph3py_expand_collision_matrix`.
///
/// Expands a collision matrix computed for ir-grid-point rows to every
/// grid point by k-star symmetry.  For each ir-gp k:
///  1. Copy row `ir_grid_points[k]` out, dividing by the stabilizer
///     multiplicity, and zero it in place.
///  2. For each rotation `l`, accumulate the divided copy into row
///     `gp_r = rot_grid_points[l, ir_gp]`, with the column index also
///     rotated by `rot_grid_points[l, n]`.
///
/// Shapes (row-major):
/// - `collision_matrix`: `(num_sigma, num_temp, num_grid_points,
///   num_band, num_grid_points, num_band)`.
/// - `rot_grid_points`: `(num_rot, num_grid_points)`.
/// - `ir_grid_points`: `(num_ir_gp,)`.
///
/// The outer loop over ir-gps runs in parallel under rayon.  Different
/// ir-gps lie in disjoint k-orbits, so the sets of rows
/// `{gp_r}` written by different tasks are disjoint and pointer
/// accesses via `SyncMutPtr` are race-free.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expand_collision_matrix(
    collision_matrix: &mut [f64],
    rot_grid_points: &[i64],
    ir_grid_points: &[i64],
    num_grid_points: usize,
    num_rot: usize,
    num_sigma: usize,
    num_temp: usize,
    num_band: usize,
) {
    let num_column = num_grid_points * num_band;
    let num_bgb = num_band * num_grid_points * num_band;
    let n2 = num_column * num_column;

    // multi[k] = number of rotations that stabilize ir_grid_points[k].
    let multi: Vec<i64> = ir_grid_points
        .iter()
        .map(|&ir_gp_i| {
            let ir_gp = ir_gp_i as usize;
            (0..num_rot)
                .filter(|&j| rot_grid_points[j * num_grid_points + ir_gp] == ir_gp as i64)
                .count() as i64
        })
        .collect();

    let cm_ptr = SyncMutPtr(collision_matrix.as_mut_ptr());

    for i in 0..num_sigma {
        for j in 0..num_temp {
            let adrs_shift = i * n2 * num_temp + j * n2;
            ir_grid_points
                .par_iter()
                .enumerate()
                .for_each(|(k, &ir_gp_i)| {
                    let ir_gp = ir_gp_i as usize;
                    let adrs_shift_plus = adrs_shift + ir_gp * num_bgb;
                    let cm = cm_ptr;
                    let m_k = multi[k] as f64;

                    let mut colmat_copy = vec![0.0f64; num_bgb];
                    // SAFETY: row `ir_gp` is owned by this task because
                    // `ir_grid_points` holds distinct k-star
                    // representatives.
                    unsafe {
                        for l in 0..num_bgb {
                            let p = cm.ptr().add(adrs_shift_plus + l);
                            colmat_copy[l] = *p / m_k;
                            *p = 0.0;
                        }
                    }

                    for l in 0..num_rot {
                        let gp_r = rot_grid_points[l * num_grid_points + ir_gp] as usize;
                        let base = adrs_shift + gp_r * num_bgb;
                        for m in 0..num_band {
                            let row_off = m * num_grid_points * num_band;
                            for n in 0..num_grid_points {
                                let gp_c = rot_grid_points[l * num_grid_points + n] as usize;
                                let out_base = base + row_off + gp_c * num_band;
                                let src_base = row_off + n * num_band;
                                // SAFETY: all writes by this task land
                                // in rows within the k-orbit of
                                // `ir_gp`, which is disjoint from other
                                // tasks' orbits.
                                unsafe {
                                    for p in 0..num_band {
                                        *cm.ptr().add(out_base + p) += colmat_copy[src_base + p];
                                    }
                                }
                            }
                        }
                    }
                });
        }
    }
}
