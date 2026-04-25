//! Dynamical matrix construction at a single q-point.
//!
//! Port of `dym_*` functions from `c/dynmat.c`.  No LAPACK calls
//! live here; diagonalization is performed in Python with
//! `numpy.linalg.eigh`.  This module currently provides the
//! single-q dynamical-matrix builder; batched-over-q-points and
//! NAC dipole-dipole helpers will follow.

#![allow(dead_code)]

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::common::{matvec_dd, matvec_di, Cmplx, MatD, Vec3D, Vec3I};

/// Build the dynamical matrix at a single q-point.
///
/// Mirrors `dym_get_dynamical_matrix_at_q` in `c/dynmat.c`.  Fills
/// the `[num_patom*3, num_patom*3]` row-major output buffer in
/// place and then symmetrizes it to be Hermitian.
///
/// `fc` first axis may be either `num_satom` (full fc) or
/// `num_patom` (compact fc); `p2s_map[i]` is the row index used
/// into `fc`, so callers must pass the appropriate mapping for
/// their fc layout.  `s2p_map[k]` and `p2s_map[j]` are compared to
/// gate the inner loop, so they must come from the same atom
/// numbering scheme (full satom indices for full fc, or primitive
/// indices for compact fc).
///
/// `charge_sum` carries the Wang-NAC correction; pass `None` for
/// the no-NAC and Gonze-Lee-NAC paths.
#[allow(clippy::too_many_arguments)]
pub fn get_dynamical_matrix_at_q(
    dynamical_matrix: &mut [Cmplx],
    fc: &[f64],
    q: Vec3D,
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    p2s_map: &[i64],
    charge_sum: Option<&[[[f64; 3]; 3]]>,
    num_patom: usize,
    num_satom: usize,
) {
    let num_band = num_patom * 3;
    debug_assert_eq!(dynamical_matrix.len(), num_band * num_band);
    debug_assert_eq!(mass.len(), num_patom);
    debug_assert_eq!(s2p_map.len(), num_satom);
    debug_assert_eq!(p2s_map.len(), num_patom);
    debug_assert_eq!(multi.len(), num_satom * num_patom);
    if let Some(cs) = charge_sum {
        debug_assert_eq!(cs.len(), num_patom * num_patom);
    }

    for i in 0..num_patom {
        for j in 0..num_patom {
            fill_dynmat_block_ij(
                dynamical_matrix,
                fc,
                q,
                svecs,
                multi,
                mass,
                s2p_map,
                p2s_map,
                charge_sum,
                num_patom,
                num_satom,
                i,
                j,
            );
        }
    }

    make_hermitian(dynamical_matrix, num_band);
}

/// Fill the 3x3 block at primitive-atom pair `(i, j)`.  Mirrors
/// the static `get_dynmat_ij` in `c/dynmat.c`.
#[allow(clippy::too_many_arguments)]
fn fill_dynmat_block_ij(
    dynamical_matrix: &mut [Cmplx],
    fc: &[f64],
    q: Vec3D,
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    p2s_map: &[i64],
    charge_sum: Option<&[[[f64; 3]; 3]]>,
    num_patom: usize,
    num_satom: usize,
    i: usize,
    j: usize,
) {
    let mass_sqrt = (mass[i] * mass[j]).sqrt();
    let mut block = [[[0.0f64; 2]; 3]; 3]; // block[k][l][(real, imag)]

    for k in 0..num_satom {
        if s2p_map[k] != p2s_map[j] {
            continue;
        }
        accumulate_block(
            &mut block, fc, q, svecs, multi, p2s_map, charge_sum, num_patom, num_satom, i, j, k,
        );
    }

    let stride = num_patom * 3;
    for k in 0..3 {
        for l in 0..3 {
            let adrs = (i * 3 + k) * stride + j * 3 + l;
            dynamical_matrix[adrs][0] = block[k][l][0] / mass_sqrt;
            dynamical_matrix[adrs][1] = block[k][l][1] / mass_sqrt;
        }
    }
}

/// Phase-summation kernel for a single satom `k` mapped to
/// primitive atom `j`.  Mirrors the static `get_dm` in
/// `c/dynmat.c`.
#[allow(clippy::too_many_arguments)]
fn accumulate_block(
    block: &mut [[[f64; 2]; 3]; 3],
    fc: &[f64],
    q: Vec3D,
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    p2s_map: &[i64],
    charge_sum: Option<&[[[f64; 3]; 3]]>,
    num_patom: usize,
    num_satom: usize,
    i: usize,
    j: usize,
    k: usize,
) {
    let pair_index = k * num_patom + i;
    let m_pair = multi[pair_index][0];
    let svec_base = multi[pair_index][1] as usize;

    let mut cos_phase = 0.0;
    let mut sin_phase = 0.0;
    let inv_m_pair = 1.0 / m_pair as f64;
    for l in 0..m_pair as usize {
        let s = svecs[svec_base + l];
        let phase = (q[0] * s[0] + q[1] * s[1] + q[2] * s[2]) * 2.0 * PI;
        cos_phase += phase.cos() * inv_m_pair;
        sin_phase += phase.sin() * inv_m_pair;
    }

    let fc_row = p2s_map[i] as usize;
    let fc_block_base = fc_row * num_satom * 9 + k * 9;
    let cs_block = charge_sum.map(|cs| &cs[i * num_patom + j]);
    for l in 0..3 {
        for m in 0..3 {
            let fc_elem = match cs_block {
                Some(b) => fc[fc_block_base + l * 3 + m] + b[l][m],
                None => fc[fc_block_base + l * 3 + m],
            };
            block[l][m][0] += fc_elem * cos_phase;
            block[l][m][1] += fc_elem * sin_phase;
        }
    }
}

/// Build the Wang-NAC charge sum.
///
/// Mirrors `dym_get_charge_sum` in `c/dynmat.c`.  Computes
///   q_born[i][a]            = sum_k q_cart[k] * born[i][k][a]
///   charge_sum[i*np+j][a][b] = q_born[i][a] * q_born[j][b] * factor
///
/// `factor` is the 4*pi/V * unit-conversion / denominator
/// prefactor assembled on the Python side.  The output buffer must
/// be `num_patom * num_patom` blocks of 3x3.
pub fn get_charge_sum(
    charge_sum: &mut [[[f64; 3]; 3]],
    factor: f64,
    q_cart: Vec3D,
    born: &[[[f64; 3]; 3]],
) {
    let num_patom = born.len();
    debug_assert_eq!(charge_sum.len(), num_patom * num_patom);

    let mut q_born = vec![[0.0f64; 3]; num_patom];
    for i in 0..num_patom {
        for a in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += q_cart[k] * born[i][k][a];
            }
            q_born[i][a] = s;
        }
    }

    for i in 0..num_patom {
        for j in 0..num_patom {
            let block = &mut charge_sum[i * num_patom + j];
            for a in 0..3 {
                for b in 0..3 {
                    block[a][b] = q_born[i][a] * q_born[j][b] * factor;
                }
            }
        }
    }
}

/// q^T eps q.  Mirrors `get_dielectric_part` in `c/dynmat.c`.
fn dielectric_part(q_cart: Vec3D, dielectric: &MatD) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += q_cart[i] * dielectric[i][j] * q_cart[j];
        }
    }
    s
}

/// Per-G accumulation into the [i, alpha, j, beta] dipole-dipole
/// buffer.  Mirrors `get_dd_at_g` in `c/dynmat.c`.  The phase uses
/// G (not G+q) to give the C-type dynamical matrix.
fn dd_at_g(
    dd_part: &mut [Cmplx],
    i: usize,
    j: usize,
    g_vec: Vec3D,
    num_patom: usize,
    pos: &[Vec3D],
    kk: &MatD,
) {
    let mut phase = 0.0;
    for k in 0..3 {
        phase += (pos[i][k] - pos[j][k]) * g_vec[k];
    }
    phase *= 2.0 * PI;
    let cos_phase = phase.cos();
    let sin_phase = phase.sin();
    for k in 0..3 {
        for l in 0..3 {
            let adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
            dd_part[adrs][0] += kk[k][l] * cos_phase;
            dd_part[adrs][1] += kk[k][l] * sin_phase;
        }
    }
}

/// Build the dipole-dipole reciprocal sum into the
/// [i, alpha, j, beta] (real, imag) flat buffer.  Mirrors `get_dd`
/// in `c/dynmat.c`.  When `q_direction_cart` is `None` and
/// `q_cart` is at G=0, the K=0 term contributes zero (used for
/// the q0 path here); otherwise the q-direction limit is used.
#[allow(clippy::too_many_arguments)]
fn get_dd(
    dd_part: &mut [Cmplx],
    g_list: &[Vec3D],
    num_patom: usize,
    q_cart: Vec3D,
    q_direction_cart: Option<Vec3D>,
    dielectric: &MatD,
    pos: &[Vec3D],
    lambda: f64,
    tolerance: f64,
) {
    let l2 = 4.0 * lambda * lambda;
    let mut kk: Vec<MatD> = vec![[[0.0; 3]; 3]; g_list.len()];
    for (g, kk_g) in g_list.iter().zip(kk.iter_mut()) {
        let mut q_k = [0.0; 3];
        let mut norm = 0.0;
        for i in 0..3 {
            q_k[i] = g[i] + q_cart[i];
            norm += q_k[i] * q_k[i];
        }
        if norm.sqrt() < tolerance {
            match q_direction_cart {
                None => {
                    // already zero
                }
                Some(qd) => {
                    let dpart = dielectric_part(qd, dielectric);
                    for i in 0..3 {
                        for j in 0..3 {
                            kk_g[i][j] = qd[i] * qd[j] / dpart;
                        }
                    }
                }
            }
        } else {
            let dpart = dielectric_part(q_k, dielectric);
            let damp = (-dpart / l2).exp();
            for i in 0..3 {
                for j in 0..3 {
                    kk_g[i][j] = q_k[i] * q_k[j] / dpart * damp;
                }
            }
        }
    }

    // Sequential to avoid races on dd_part (matches the C comment).
    for (g, kk_g) in g_list.iter().zip(kk.iter()) {
        for i in 0..num_patom {
            for j in 0..num_patom {
                dd_at_g(dd_part, i, j, *g, num_patom, pos, kk_g);
            }
        }
    }
}

/// Per-(i, j) Born-charge multiplication for the reciprocal
/// dipole-dipole sum.  Mirrors `multiply_borns_at_ij`.
fn multiply_borns_at_ij(
    dd: &mut [Cmplx],
    i: usize,
    j: usize,
    dd_in: &[Cmplx],
    num_patom: usize,
    born: &[[[f64; 3]; 3]],
) {
    for k in 0..3 {
        for l in 0..3 {
            let adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
            let mut acc = [0.0f64; 2];
            for m in 0..3 {
                for n in 0..3 {
                    let adrs_in = i * num_patom * 9 + m * num_patom * 3 + j * 3 + n;
                    let zz = born[i][m][k] * born[j][n][l];
                    acc[0] += dd_in[adrs_in][0] * zz;
                    acc[1] += dd_in[adrs_in][1] * zz;
                }
            }
            dd[adrs][0] += acc[0];
            dd[adrs][1] += acc[1];
        }
    }
}

/// Apply Born charges to the [i, alpha, j, beta] buffer.  Mirrors
/// `multiply_borns` in `c/dynmat.c`.  Each (i, j) entry of `dd` is
/// independent so this is naturally parallel; we keep it
/// sequential here to match the existing scalar path.
fn multiply_borns(dd: &mut [Cmplx], dd_in: &[Cmplx], num_patom: usize, born: &[[[f64; 3]; 3]]) {
    for i in 0..num_patom {
        for j in 0..num_patom {
            multiply_borns_at_ij(dd, i, j, dd_in, num_patom, born);
        }
    }
}

/// Build the q=0 dipole-dipole correction `dd_q0` used by the
/// Gonze-Lee NAC.  Mirrors `dym_get_recip_dipole_dipole_q0`.
///
/// `dd_q0` has shape `[num_patom, 3, 3]` complex (flat
/// `num_patom * 9`).  `g_list` is the reciprocal lattice vectors
/// to sum over (already cartesian).  `pos` is the primitive-atom
/// positions (cartesian).  `lambda` is the Ewald parameter.
///
/// Output is symmetrized (real part) and antisymmetrized (imag
/// part) over (alpha, beta) per atom.
#[allow(clippy::too_many_arguments)]
pub fn get_recip_dipole_dipole_q0(
    dd_q0: &mut [Cmplx],
    g_list: &[Vec3D],
    num_patom: usize,
    born: &[[[f64; 3]; 3]],
    dielectric: &MatD,
    pos: &[Vec3D],
    lambda: f64,
    tolerance: f64,
) {
    debug_assert_eq!(dd_q0.len(), num_patom * 9);
    debug_assert_eq!(born.len(), num_patom);
    debug_assert_eq!(pos.len(), num_patom);

    let n2 = num_patom * num_patom * 9;
    let mut dd_tmp1: Vec<Cmplx> = vec![[0.0; 2]; n2];
    let mut dd_tmp2: Vec<Cmplx> = vec![[0.0; 2]; n2];

    let zero_vec = [0.0, 0.0, 0.0];
    get_dd(
        &mut dd_tmp1,
        g_list,
        num_patom,
        zero_vec,
        None,
        dielectric,
        pos,
        lambda,
        tolerance,
    );

    multiply_borns(&mut dd_tmp2, &dd_tmp1, num_patom, born);

    for c in dd_q0.iter_mut() {
        *c = [0.0, 0.0];
    }

    for i in 0..num_patom {
        for k in 0..3 {
            for l in 0..3 {
                let adrs = i * 9 + k * 3 + l;
                for j in 0..num_patom {
                    let adrs_tmp = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
                    dd_q0[adrs][0] += dd_tmp2[adrs_tmp][0];
                    dd_q0[adrs][1] += dd_tmp2[adrs_tmp][1];
                }
            }
        }
    }

    // Per-atom (alpha, beta) Hermitian symmetrization of the 3x3 block.
    for i in 0..num_patom {
        for k in 0..3 {
            for l in k..3 {
                let adrs = i * 9 + k * 3 + l;
                let adrs_t = i * 9 + l * 3 + k;
                let re = (dd_q0[adrs][0] + dd_q0[adrs_t][0]) * 0.5;
                let im = (dd_q0[adrs][1] - dd_q0[adrs_t][1]) * 0.5;
                dd_q0[adrs][0] = re;
                dd_q0[adrs][1] = im;
                dd_q0[adrs_t][0] = re;
                dd_q0[adrs_t][1] = -im;
            }
        }
    }
}

/// Build the Gonze-Lee reciprocal dipole-dipole correction at a
/// single q-point.  Mirrors `dym_get_recip_dipole_dipole`.
///
/// `dd` (output) has shape `[num_patom, 3, num_patom, 3]` complex
/// (flat `num_patom^2 * 9`).  `dd_q0` is the q=0 correction from
/// `get_recip_dipole_dipole_q0` and is subtracted on the diagonal
/// (i, i) blocks.  `factor` is `4*pi/V * unit_conversion` applied
/// to the entire output.  `q_direction_cart` is used only when the
/// K=0 term is reached at q=0.
#[allow(clippy::too_many_arguments)]
pub fn get_recip_dipole_dipole(
    dd: &mut [Cmplx],
    dd_q0: &[Cmplx],
    g_list: &[Vec3D],
    num_patom: usize,
    q_cart: Vec3D,
    q_direction_cart: Option<Vec3D>,
    born: &[[[f64; 3]; 3]],
    dielectric: &MatD,
    pos: &[Vec3D],
    factor: f64,
    lambda: f64,
    tolerance: f64,
) {
    let n2 = num_patom * num_patom * 9;
    debug_assert_eq!(dd.len(), n2);
    debug_assert_eq!(dd_q0.len(), num_patom * 9);
    debug_assert_eq!(born.len(), num_patom);
    debug_assert_eq!(pos.len(), num_patom);

    for c in dd.iter_mut() {
        *c = [0.0, 0.0];
    }
    let mut dd_tmp: Vec<Cmplx> = vec![[0.0; 2]; n2];

    get_dd(
        &mut dd_tmp,
        g_list,
        num_patom,
        q_cart,
        q_direction_cart,
        dielectric,
        pos,
        lambda,
        tolerance,
    );

    multiply_borns(dd, &dd_tmp, num_patom, born);

    for i in 0..num_patom {
        for k in 0..3 {
            for l in 0..3 {
                let adrs = i * num_patom * 9 + k * num_patom * 3 + i * 3 + l;
                let adrs_sum = i * 9 + k * 3 + l;
                dd[adrs][0] -= dd_q0[adrs_sum][0];
                dd[adrs][1] -= dd_q0[adrs_sum][1];
            }
        }
    }

    for c in dd.iter_mut() {
        c[0] *= factor;
        c[1] *= factor;
    }
}

/// In-place Hermitian symmetrization: `mat <- (mat + mat^H) / 2`.
/// Mirrors `make_Hermitian` in `c/dynmat.c`.  Diagonal entries lose
/// their imaginary part (as they should for a Hermitian matrix).
fn make_hermitian(mat: &mut [Cmplx], num_band: usize) {
    debug_assert_eq!(mat.len(), num_band * num_band);
    for i in 0..num_band {
        for j in i..num_band {
            let adrs = i * num_band + j;
            let adrs_t = j * num_band + i;
            let re = (mat[adrs][0] + mat[adrs_t][0]) * 0.5;
            let im = (mat[adrs][1] - mat[adrs_t][1]) * 0.5;
            mat[adrs][0] = re;
            mat[adrs][1] = im;
            mat[adrs_t][0] = re;
            mat[adrs_t][1] = -im;
        }
    }
}

/// Wang-NAC inputs shared across q-points.
///
/// `q_direction` (fractional) is applied when a non-None direction
/// is needed for the Gamma-limit.  `nac_factor` is the raw
/// unit-conversion prefactor; the dielectric and `num_satom /
/// num_patom` divisions are applied internally.
pub struct WangNacParams<'a> {
    pub born: &'a [[[f64; 3]; 3]],
    pub dielectric: MatD,
    pub reciprocal_lattice: MatD,
    pub q_direction: Option<Vec3D>,
    pub nac_factor: f64,
}

/// Gonze-Lee inputs shared across q-points.  `pos` is the primitive
/// cell positions in cartesian coordinates, matching the convention
/// used in `get_recip_dipole_dipole`.
pub struct GonzeNacParams<'a> {
    pub born: &'a [[[f64; 3]; 3]],
    pub dielectric: MatD,
    pub reciprocal_lattice: MatD,
    pub q_direction: Option<Vec3D>,
    pub nac_factor: f64,
    pub pos: &'a [Vec3D],
    pub dd_q0: &'a [Cmplx],
    pub g_list: &'a [Vec3D],
    pub lambda: f64,
}

/// Decide whether the NAC contribution must be applied at this
/// grid point.  Mirrors `needs_nac` in `c/phonon.c`: no `born`
/// means no NAC; otherwise NAC is applied unless we are at
/// Gamma with no q-direction set.
fn needs_nac(has_born: bool, grid_address: Vec3I, has_q_direction: bool) -> bool {
    if !has_born {
        return false;
    }
    if !has_q_direction && grid_address == [0, 0, 0] {
        return false;
    }
    true
}

/// Wang-NAC charge-sum with the dielectric/supercell-ratio factor
/// assembled.  Mirrors the static `get_charge_sum` in `c/phonon.c`.
fn wang_charge_sum_with_factor(
    charge_sum: &mut [[[f64; 3]; 3]],
    num_patom: usize,
    num_satom: usize,
    q: Vec3D,
    params: &WangNacParams,
) {
    let q_for_cart = params.q_direction.unwrap_or(q);
    let q_cart = matvec_dd(&params.reciprocal_lattice, q_for_cart);
    let eps_q_q = dielectric_part(q_cart, &params.dielectric);
    let factor = params.nac_factor / eps_q_q / (num_satom as f64) * (num_patom as f64);
    get_charge_sum(charge_sum, factor, q_cart, params.born);
}

/// Build the Wang- or no-NAC dynamical matrix at a single q-point.
/// Mirrors `get_dynamical_matrix` (static) in `c/phonon.c`.
#[allow(clippy::too_many_arguments)]
fn build_dm_with_wang_nac_at_q(
    dm: &mut [Cmplx],
    q: Vec3D,
    grid_address: Vec3I,
    fc: &[f64],
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    p2s_map: &[i64],
    num_patom: usize,
    num_satom: usize,
    wang: Option<&WangNacParams>,
) {
    let is_nac = match wang {
        None => false,
        Some(p) => needs_nac(true, grid_address, p.q_direction.is_some()),
    };
    if is_nac {
        let params = wang.unwrap();
        let mut charge_sum = vec![[[0.0f64; 3]; 3]; num_patom * num_patom];
        wang_charge_sum_with_factor(&mut charge_sum, num_patom, num_satom, q, params);
        get_dynamical_matrix_at_q(
            dm,
            fc,
            q,
            svecs,
            multi,
            mass,
            s2p_map,
            p2s_map,
            Some(&charge_sum),
            num_patom,
            num_satom,
        );
    } else {
        get_dynamical_matrix_at_q(
            dm, fc, q, svecs, multi, mass, s2p_map, p2s_map, None, num_patom, num_satom,
        );
    }
}

/// Build the Gonze-Lee NAC dynamical matrix at a single q-point.
/// Mirrors the static `get_gonze_phonons` in `c/phonon.c`: build
/// the no-NAC dynamical matrix, compute the reciprocal dipole-dipole
/// correction, add it divided by sqrt(m_i m_j) to each 3x3 block.
#[allow(clippy::too_many_arguments)]
fn build_dm_with_gonze_nac_at_q(
    dm: &mut [Cmplx],
    q: Vec3D,
    fc: &[f64],
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    p2s_map: &[i64],
    num_patom: usize,
    num_satom: usize,
    gonze: &GonzeNacParams,
) {
    get_dynamical_matrix_at_q(
        dm, fc, q, svecs, multi, mass, s2p_map, p2s_map, None, num_patom, num_satom,
    );

    let q_cart = matvec_dd(&gonze.reciprocal_lattice, q);
    let q_dir_cart = gonze
        .q_direction
        .map(|qd| matvec_dd(&gonze.reciprocal_lattice, qd));

    let mut dd: Vec<Cmplx> = vec![[0.0; 2]; num_patom * num_patom * 9];
    get_recip_dipole_dipole(
        &mut dd,
        gonze.dd_q0,
        gonze.g_list,
        num_patom,
        q_cart,
        q_dir_cart,
        gonze.born,
        &gonze.dielectric,
        gonze.pos,
        gonze.nac_factor,
        gonze.lambda,
        1e-5,
    );

    let stride = num_patom * 3;
    for i in 0..num_patom {
        for j in 0..num_patom {
            let mm = (mass[i] * mass[j]).sqrt();
            for k in 0..3 {
                for l in 0..3 {
                    let dm_adrs = (i * 3 + k) * stride + j * 3 + l;
                    let dd_adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
                    dm[dm_adrs][0] += dd[dd_adrs][0] / mm;
                    dm[dm_adrs][1] += dd[dd_adrs][1] / mm;
                }
            }
        }
    }
}

/// Peel off disjoint `num_band^2` slices of `dynmats` at the gp
/// positions listed in `undone_grid_points`.
///
/// `undone_grid_points` must be strictly ascending (sorted and
/// unique) and every entry must be in `0..num_phonons`.  The caller
/// is responsible for enforcing this; we only check it under
/// `debug_assertions`.  Returns pairs in gp order.
fn split_dynmats_per_gp<'a>(
    dynmats: &'a mut [Cmplx],
    undone_grid_points: &[i64],
    num_band: usize,
    num_phonons: usize,
) -> Vec<(usize, &'a mut [Cmplx])> {
    let stride = num_band * num_band;
    assert_eq!(dynmats.len(), num_phonons * stride);

    debug_assert!(
        undone_grid_points.windows(2).all(|w| w[0] < w[1]),
        "undone_grid_points must be strictly ascending (sorted and unique)"
    );
    if let Some(&last) = undone_grid_points.last() {
        debug_assert!(last >= 0, "undone_grid_points must be non-negative");
        debug_assert!(
            (last as usize) < num_phonons,
            "undone_grid_points must be < num_phonons"
        );
    }

    let mut out: Vec<(usize, &mut [Cmplx])> = Vec::with_capacity(undone_grid_points.len());
    let mut rest: &mut [Cmplx] = dynmats;
    let mut consumed = 0usize;
    for &gp_i64 in undone_grid_points {
        let gp = gp_i64 as usize;
        let skip = (gp - consumed) * stride;
        let (_head, tail) = rest.split_at_mut(skip);
        let (chunk, tail2) = tail.split_at_mut(stride);
        out.push((gp, chunk));
        consumed = gp + 1;
        rest = tail2;
    }
    out
}

/// Build dynamical matrices at the listed grid points (Wang- or
/// no-NAC path).  Parallel over grid points with rayon.
///
/// `dynmats` is a `[num_phonons, num_band, num_band]` complex
/// buffer; only blocks at `undone_grid_points` are touched.  The
/// q-point at grid point `gp` is `QDinv * grid_address[gp]`.
/// `undone_grid_points` must be strictly ascending (sorted and
/// unique) with every entry in `0..num_phonons`.
#[allow(clippy::too_many_arguments)]
pub fn dynamical_matrices_at_gridpoints(
    dynmats: &mut [Cmplx],
    undone_grid_points: &[i64],
    grid_addresses: &[Vec3I],
    qd_inv: &MatD,
    fc: &[f64],
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    p2s_map: &[i64],
    s2p_map: &[i64],
    num_patom: usize,
    num_satom: usize,
    wang: Option<&WangNacParams>,
) {
    let num_band = num_patom * 3;
    let num_phonons = grid_addresses.len();
    let chunks = split_dynmats_per_gp(dynmats, undone_grid_points, num_band, num_phonons);

    chunks.into_par_iter().for_each(|(gp, dm)| {
        let grid_address = grid_addresses[gp];
        let q = matvec_di(qd_inv, grid_address);
        build_dm_with_wang_nac_at_q(
            dm,
            q,
            grid_address,
            fc,
            svecs,
            multi,
            mass,
            s2p_map,
            p2s_map,
            num_patom,
            num_satom,
            wang,
        );
    });
}

/// Build dynamical matrices at the listed grid points (Gonze-Lee
/// NAC path).  Parallel over grid points with rayon.
/// `undone_grid_points` must be strictly ascending (sorted and
/// unique) with every entry in `0..num_phonons`.
#[allow(clippy::too_many_arguments)]
pub fn dynamical_matrices_at_gridpoints_gonze(
    dynmats: &mut [Cmplx],
    undone_grid_points: &[i64],
    grid_addresses: &[Vec3I],
    qd_inv: &MatD,
    fc: &[f64],
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    p2s_map: &[i64],
    s2p_map: &[i64],
    num_patom: usize,
    num_satom: usize,
    gonze: &GonzeNacParams,
) {
    let num_band = num_patom * 3;
    let num_phonons = grid_addresses.len();
    let chunks = split_dynmats_per_gp(dynmats, undone_grid_points, num_band, num_phonons);

    chunks.into_par_iter().for_each(|(gp, dm)| {
        let q = matvec_di(qd_inv, grid_addresses[gp]);
        build_dm_with_gonze_nac_at_q(
            dm, q, fc, svecs, multi, mass, s2p_map, p2s_map, num_patom, num_satom, gonze,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_hermitian_zeroes_diagonal_imaginary() {
        let mut mat: Vec<Cmplx> = vec![[1.0, 0.5], [2.0, 1.0], [4.0, -1.0], [5.0, -0.5]];
        make_hermitian(&mut mat, 2);
        assert_eq!(mat[0], [1.0, 0.0]);
        assert_eq!(mat[3], [5.0, 0.0]);
        // Off-diagonal: (2 + 4)/2 + i(1 - (-1))/2 = 3 + 1 i for [0][1],
        // and conjugate for [1][0].
        assert_eq!(mat[1], [3.0, 1.0]);
        assert_eq!(mat[2], [3.0, -1.0]);
    }

    #[test]
    fn charge_sum_matches_explicit_outer_product() {
        let q_cart = [0.1, -0.2, 0.3];
        let born = vec![
            [[1.0, 0.2, 0.0], [0.2, 1.5, -0.1], [0.0, -0.1, 0.8]],
            [[-1.0, -0.2, 0.0], [-0.2, -1.5, 0.1], [0.0, 0.1, -0.8]],
        ];
        let factor = 0.7;
        let mut cs = vec![[[0.0f64; 3]; 3]; born.len() * born.len()];
        get_charge_sum(&mut cs, factor, q_cart, &born);

        let np = born.len();
        for i in 0..np {
            let mut qb_i = [0.0; 3];
            for a in 0..3 {
                for k in 0..3 {
                    qb_i[a] += q_cart[k] * born[i][k][a];
                }
            }
            for j in 0..np {
                let mut qb_j = [0.0; 3];
                for b in 0..3 {
                    for k in 0..3 {
                        qb_j[b] += q_cart[k] * born[j][k][b];
                    }
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let expected = qb_i[a] * qb_j[b] * factor;
                        assert!((cs[i * np + j][a][b] - expected).abs() < 1e-15);
                    }
                }
            }
        }
    }

    #[test]
    fn single_atom_gamma_point_no_nac_matches_fc_sum() {
        // num_patom = num_satom = 1, q = 0, single multiplicity.
        // Then dynamical_matrix[k][l] == fc[0,0,k,l] / mass.
        let num_patom = 1usize;
        let num_satom = 1usize;
        let mass = vec![2.0];
        let s2p = vec![0i64];
        let p2s = vec![0i64];
        let multi = vec![[1i64, 0]];
        let svecs = vec![[0.0, 0.0, 0.0]];
        let mut fc = vec![0.0f64; 9];
        // Build a Hermitian-friendly fc: symmetric 3x3.
        fc[0] = 1.0;
        fc[4] = 2.0;
        fc[8] = 3.0;
        fc[1] = 0.5;
        fc[3] = 0.5;
        fc[2] = -0.25;
        fc[6] = -0.25;
        fc[5] = 0.1;
        fc[7] = 0.1;

        let mut dm: Vec<Cmplx> = vec![[0.0, 0.0]; 9];
        get_dynamical_matrix_at_q(
            &mut dm,
            &fc,
            [0.0, 0.0, 0.0],
            &svecs,
            &multi,
            &mass,
            &s2p,
            &p2s,
            None,
            num_patom,
            num_satom,
        );

        for k in 0..3 {
            for l in 0..3 {
                let adrs = k * 3 + l;
                assert!((dm[adrs][0] - fc[k * 3 + l] / mass[0]).abs() < 1e-12);
                assert!(dm[adrs][1].abs() < 1e-12);
            }
        }
    }
}
