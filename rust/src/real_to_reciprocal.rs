//! Transform fc3 from real space to reciprocal space.
//!
//! Port of `r2r_real_to_reciprocal` and its static helpers in
//! `c/real_to_reciprocal.c`.  The output layout here is atom-first
//! `(num_patom, num_patom, num_patom, 3, 3, 3)` flat, matching the
//! existing Python reference in `phono3py/phonon3/real_to_reciprocal.py`.
//! The C version uses band-first `(num_band, num_band, num_band)` with
//! `num_band = num_patom * 3`; a Python-side transpose bridges the two
//! when crossing into the `reciprocal_to_normal_squared` path.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::common::{cmplx_mul, Cmplx, Vec3D};

/// Atom-triplet metadata shared across the real-to-reciprocal transform.
pub struct AtomTriplets<'a> {
    /// Shortest supercell vectors as `(n_svec, 3)`.
    pub svecs: &'a [Vec3D],
    pub num_satom: usize,
    pub num_patom: usize,
    /// `multiplicity[satom * num_patom + patom]` = `[count, start]`
    /// where `count` is the number of shortest vectors and `start` is
    /// the offset into `svecs`.
    pub multiplicity: &'a [[i64; 2]],
    pub p2s_map: &'a [i64],
    pub s2p_map: &'a [i64],
    pub make_r0_average: bool,
    /// Shape `(num_patom, num_satom, num_satom)` flat.
    pub all_shortest: &'a [i8],
    /// Shape `(num_rows, num_satom, num_satom)` flat where `num_rows`
    /// is `num_patom` for compact fc3 or `num_satom` for full fc3.
    pub nonzero_indices: &'a [i8],
}

/// Summed phase factor over the multiplicity of a single (satom, patom) pair.
fn get_phase_factor(q: Vec3D, svecs: &[Vec3D], multi: [i64; 2]) -> Cmplx {
    let count = multi[0] as usize;
    let base = multi[1] as usize;
    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;
    for i in 0..count {
        let s = svecs[base + i];
        let phase = (q[0] * s[0] + q[1] * s[1] + q[2] * s[2]) * 2.0 * PI;
        let (sin_p, cos_p) = phase.sin_cos();
        sum_real += cos_p;
        sum_imag += sin_p;
    }
    let inv = 1.0 / count as f64;
    [sum_real * inv, sum_imag * inv]
}

/// Pre-phase factor `exp(2 pi i * r_i * (q0 + q1 + q2))`.  Uses the
/// self-multiplicity entry `multi[p2s_map[i_patom] * num_patom + 0]`.
fn get_pre_phase_factor(
    i_patom: usize,
    q_vecs: &[Vec3D; 3],
    atom_triplets: &AtomTriplets,
) -> Cmplx {
    let svecs_adrs = atom_triplets.p2s_map[i_patom] as usize * atom_triplets.num_patom;
    let multi = atom_triplets.multiplicity[svecs_adrs];
    let svec = atom_triplets.svecs[multi[1] as usize];
    let mut phase = 0.0;
    for j in 0..3 {
        phase += svec[j] * (q_vecs[0][j] + q_vecs[1][j] + q_vecs[2][j]);
    }
    phase *= 2.0 * PI;
    let (sin_p, cos_p) = phase.sin_cos();
    [cos_p, sin_p]
}

/// Accumulate the 3x3x3 fc3 element block at primitive triplet
/// `(pi0, pi1, pi2)`.  `phase_factor1` and `phase_factor2` are the
/// rows of the phase-factor table at `pi0` (each of length
/// `num_satom`).  `leg_index` selects the variant:
///
/// - `0`: legacy single-leg (no all_shortest interaction).
/// - `1`: r0-average leg 1 — multiply fc3 by 3 when all_shortest flag is set.
/// - `2`, `3`: r0-average legs 2 and 3 — skip elements where all_shortest
///   flag is set.
#[allow(clippy::too_many_arguments)]
fn real_to_reciprocal_elements(
    fc3_rec_elem: &mut [Cmplx; 27],
    phase_factor1: &[Cmplx],
    phase_factor2: &[Cmplx],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets,
    pi0: usize,
    pi1: usize,
    pi2: usize,
    leg_index: i64,
) {
    let num_satom = atom_triplets.num_satom;
    let mut fc3_rec_real = [0.0f64; 27];
    let mut fc3_rec_imag = [0.0f64; 27];

    let i = if is_compact_fc3 {
        pi0
    } else {
        atom_triplets.p2s_map[pi0] as usize
    };

    let p2s_pi1 = atom_triplets.p2s_map[pi1];
    let p2s_pi2 = atom_triplets.p2s_map[pi2];

    for j in 0..num_satom {
        if atom_triplets.s2p_map[j] != p2s_pi1 {
            continue;
        }
        for k in 0..num_satom {
            if atom_triplets.s2p_map[k] != p2s_pi2 {
                continue;
            }
            let all_shortest_flag =
                atom_triplets.all_shortest[pi0 * num_satom * num_satom + j * num_satom + k];
            if leg_index > 1 && all_shortest_flag != 0 {
                continue;
            }
            let adrs_shift_atoms = i * num_satom * num_satom + j * num_satom + k;
            if atom_triplets.nonzero_indices[adrs_shift_atoms] == 0 {
                continue;
            }
            let adrs_shift_fc3 = adrs_shift_atoms * 27;
            let phase = cmplx_mul(phase_factor1[j], phase_factor2[k]);

            let mult = if leg_index == 1 && all_shortest_flag != 0 {
                3.0
            } else {
                1.0
            };

            // Hoist phase * mult out of the inner loop, and slice fc3 up
            // front so the inner loop iterates over fixed-length chunks.
            // This lets LLVM drop bounds checks and emit a fused-multiply
            // loop over the 27 elements.
            let pr = phase[0] * mult;
            let pi = phase[1] * mult;
            let fc3_block = &fc3[adrs_shift_fc3..adrs_shift_fc3 + 27];
            for ((rr, ri), &f) in fc3_rec_real
                .iter_mut()
                .zip(fc3_rec_imag.iter_mut())
                .zip(fc3_block.iter())
            {
                *rr += pr * f;
                *ri += pi * f;
            }
        }
    }

    for l in 0..27 {
        fc3_rec_elem[l] = [fc3_rec_real[l], fc3_rec_imag[l]];
    }
}

/// Legacy single-leg path.  Output block at `(pi0, pi1, pi2)` is
/// `fc3_rec_elem[l, m, n] * pre_phase[pi0]` (overwrite semantics).
#[allow(clippy::too_many_arguments)]
fn legacy_block(
    block: &mut [Cmplx],
    ijk: usize,
    num_patom: usize,
    num_satom: usize,
    pre_phase_factors: &[Cmplx],
    phase_factor1: &[Cmplx],
    phase_factor2: &[Cmplx],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets,
) {
    let i = ijk / (num_patom * num_patom);
    let j = (ijk - i * num_patom * num_patom) / num_patom;
    let k = ijk % num_patom;
    let mut fc3_rec_elem = [[0.0f64; 2]; 27];
    real_to_reciprocal_elements(
        &mut fc3_rec_elem,
        &phase_factor1[i * num_satom..],
        &phase_factor2[i * num_satom..],
        fc3,
        is_compact_fc3,
        atom_triplets,
        i,
        j,
        k,
        0,
    );
    for l in 0..3 {
        for m in 0..3 {
            for n in 0..3 {
                block[l * 9 + m * 3 + n] =
                    cmplx_mul(fc3_rec_elem[l * 9 + m * 3 + n], pre_phase_factors[i]);
            }
        }
    }
}

/// r0-average three-leg path.  Accumulates three contributions into
/// `block` (which must be zero on entry).
#[allow(clippy::too_many_arguments)]
fn r0_average_block(
    block: &mut [Cmplx],
    ijk: usize,
    num_patom: usize,
    num_satom: usize,
    pre_phase_factors: &[Cmplx],
    phase_factor0: &[Cmplx],
    phase_factor1: &[Cmplx],
    phase_factor2: &[Cmplx],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets,
) {
    let i = ijk / (num_patom * num_patom);
    let j = (ijk - i * num_patom * num_patom) / num_patom;
    let k = ijk % num_patom;
    let mut fc3_rec_elem = [[0.0f64; 2]; 27];

    // Leg 1: (i, j, k), phase1 * phase2, atom anchor = i.
    real_to_reciprocal_elements(
        &mut fc3_rec_elem,
        &phase_factor1[i * num_satom..],
        &phase_factor2[i * num_satom..],
        fc3,
        is_compact_fc3,
        atom_triplets,
        i,
        j,
        k,
        1,
    );
    for l in 0..3 {
        for m in 0..3 {
            for n in 0..3 {
                let add = cmplx_mul(fc3_rec_elem[l * 9 + m * 3 + n], pre_phase_factors[i]);
                block[l * 9 + m * 3 + n][0] += add[0];
                block[l * 9 + m * 3 + n][1] += add[1];
            }
        }
    }

    // Leg 2: (j, i, k), phase0 * phase2, atom anchor = j.  Element
    // swap l <-> m.
    real_to_reciprocal_elements(
        &mut fc3_rec_elem,
        &phase_factor0[j * num_satom..],
        &phase_factor2[j * num_satom..],
        fc3,
        is_compact_fc3,
        atom_triplets,
        j,
        i,
        k,
        2,
    );
    for l in 0..3 {
        for m in 0..3 {
            for n in 0..3 {
                let add = cmplx_mul(fc3_rec_elem[m * 9 + l * 3 + n], pre_phase_factors[j]);
                block[l * 9 + m * 3 + n][0] += add[0];
                block[l * 9 + m * 3 + n][1] += add[1];
            }
        }
    }

    // Leg 3: (k, j, i), phase1 * phase0, atom anchor = k.  Element
    // swap l <-> n.
    real_to_reciprocal_elements(
        &mut fc3_rec_elem,
        &phase_factor1[k * num_satom..],
        &phase_factor0[k * num_satom..],
        fc3,
        is_compact_fc3,
        atom_triplets,
        k,
        j,
        i,
        3,
    );
    for l in 0..3 {
        for m in 0..3 {
            for n in 0..3 {
                let add = cmplx_mul(fc3_rec_elem[n * 9 + m * 3 + l], pre_phase_factors[k]);
                block[l * 9 + m * 3 + n][0] += add[0];
                block[l * 9 + m * 3 + n][1] += add[1];
            }
        }
    }
}

/// Build `fc3_reciprocal` at the q-triplet `q_vecs`.
///
/// Output layout: atom-first `(num_patom, num_patom, num_patom, 3, 3, 3)`
/// flat.  `q_vecs` is the fractional q-triplet `[q0, q1, q2]`.
/// When `inner_par` is false, the `pi0, pi1, pi2` loop runs
/// sequentially so an outer triplet-level parallel caller owns all
/// thread-level parallelism.  When `inner_par` is true (typically
/// when the batch size is smaller than the thread count), the loop
/// parallelizes over the `num_patom^3` blocks directly.
pub fn real_to_reciprocal(
    fc3_reciprocal: &mut [Cmplx],
    q_vecs: &[Vec3D; 3],
    fc3: &[f64],
    is_compact_fc3: bool,
    atom_triplets: &AtomTriplets,
    inner_par: bool,
) {
    let num_patom = atom_triplets.num_patom;
    let num_satom = atom_triplets.num_satom;
    let n_blocks = num_patom * num_patom * num_patom;
    debug_assert_eq!(fc3_reciprocal.len(), n_blocks * 27);

    let mut pre_phase_factors = vec![[0.0f64; 2]; num_patom];
    for (i, pf) in pre_phase_factors.iter_mut().enumerate() {
        *pf = get_pre_phase_factor(i, q_vecs, atom_triplets);
    }

    let mut phase_factor0 = vec![[0.0f64; 2]; num_patom * num_satom];
    let mut phase_factor1 = vec![[0.0f64; 2]; num_patom * num_satom];
    let mut phase_factor2 = vec![[0.0f64; 2]; num_patom * num_satom];
    for i in 0..num_patom {
        for j in 0..num_satom {
            let adrs_vec = j * num_patom + i;
            phase_factor0[i * num_satom + j] = get_phase_factor(
                q_vecs[0],
                atom_triplets.svecs,
                atom_triplets.multiplicity[adrs_vec],
            );
            phase_factor1[i * num_satom + j] = get_phase_factor(
                q_vecs[1],
                atom_triplets.svecs,
                atom_triplets.multiplicity[adrs_vec],
            );
            phase_factor2[i * num_satom + j] = get_phase_factor(
                q_vecs[2],
                atom_triplets.svecs,
                atom_triplets.multiplicity[adrs_vec],
            );
        }
    }

    // Zero the output unconditionally: the r0-average path
    // accumulates via `+=`, and zeroing before the legacy path too
    // keeps the API contract uniform with negligible overhead
    // relative to the main loop.
    for c in fc3_reciprocal.iter_mut() {
        *c = [0.0, 0.0];
    }

    if atom_triplets.make_r0_average {
        if inner_par {
            fc3_reciprocal
                .par_chunks_mut(27)
                .enumerate()
                .for_each(|(ijk, block)| {
                    r0_average_block(
                        block,
                        ijk,
                        num_patom,
                        num_satom,
                        &pre_phase_factors,
                        &phase_factor0,
                        &phase_factor1,
                        &phase_factor2,
                        fc3,
                        is_compact_fc3,
                        atom_triplets,
                    );
                });
        } else {
            for (ijk, block) in fc3_reciprocal.chunks_mut(27).enumerate() {
                r0_average_block(
                    block,
                    ijk,
                    num_patom,
                    num_satom,
                    &pre_phase_factors,
                    &phase_factor0,
                    &phase_factor1,
                    &phase_factor2,
                    fc3,
                    is_compact_fc3,
                    atom_triplets,
                );
            }
        }

        // Divide by 3 to average the three legs.
        if inner_par {
            fc3_reciprocal.par_iter_mut().for_each(|c| {
                c[0] /= 3.0;
                c[1] /= 3.0;
            });
        } else {
            for c in fc3_reciprocal.iter_mut() {
                c[0] /= 3.0;
                c[1] /= 3.0;
            }
        }
    } else if inner_par {
        fc3_reciprocal
            .par_chunks_mut(27)
            .enumerate()
            .for_each(|(ijk, block)| {
                legacy_block(
                    block,
                    ijk,
                    num_patom,
                    num_satom,
                    &pre_phase_factors,
                    &phase_factor1,
                    &phase_factor2,
                    fc3,
                    is_compact_fc3,
                    atom_triplets,
                );
            });
    } else {
        for (ijk, block) in fc3_reciprocal.chunks_mut(27).enumerate() {
            legacy_block(
                block,
                ijk,
                num_patom,
                num_satom,
                &pre_phase_factors,
                &phase_factor1,
                &phase_factor2,
                fc3,
                is_compact_fc3,
                atom_triplets,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_factor_single_svec_zero_q_is_one() {
        let svecs = vec![[0.0, 0.0, 0.0]];
        let p = get_phase_factor([0.0, 0.0, 0.0], &svecs, [1, 0]);
        assert!((p[0] - 1.0).abs() < 1e-15);
        assert!(p[1].abs() < 1e-15);
    }

    #[test]
    fn phase_factor_single_svec_q_half() {
        // q . s = 0.5 -> phase = pi -> exp(i pi) = -1.
        let svecs = vec![[1.0, 0.0, 0.0]];
        let p = get_phase_factor([0.5, 0.0, 0.0], &svecs, [1, 0]);
        assert!((p[0] + 1.0).abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
    }

    #[test]
    fn phase_factor_two_svecs_average() {
        // Two svecs with q.s = 0 and q.s = 0.5 -> (1 + (-1)) / 2 = 0.
        let svecs = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let p = get_phase_factor([0.5, 0.0, 0.0], &svecs, [2, 0]);
        assert!(p[0].abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
    }
}
