//! Irreducible triplet search on a generalized regular grid.
//!
//! Port of `tpk_get_ir_triplets_at_q` and helpers from `c/triplet_grid.c`.

use crate::bzgrid;
use crate::common::{mat_i_to_d, matmul_d, matvec_dd, matvec_ii, MatD, MatI, Vec3I};
use crate::grgrid;
use crate::recip_rotations::{self, ReciprocalRotationsError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BzTripletsError {
    /// Unsupported `bz_grid_type` (must be 1 or 2).
    BadGridType,
}

/// Result of `ir_triplets_at_q`.
pub struct IrTripletsAtQ {
    pub map_triplets: Vec<i64>,
    pub map_q: Vec<i64>,
    pub num_ir: i64,
}

/// Filter rotations to those that stabilize `grid_point`
/// (i.e. R * q === q mod D_diag).
fn get_stabilizer_rotations(rotations: &[MatI], d_diag: Vec3I, grid_point: i64) -> Vec<MatI> {
    let adrs = grgrid::grid_address_from_index(grid_point, d_diag);
    let mut result = Vec::new();

    for rot in rotations {
        let adrs_rot = matvec_ii(rot, adrs);
        let gp_rot = grgrid::grid_index_from_address(adrs_rot, d_diag);
        if gp_rot == grid_point {
            result.push(*rot);
        }
    }

    result
}

/// Swappable case: q1 and q2 can be exchanged, reducing the number
/// of irreducible triplets further.
fn ir_triplets_at_q_perm_q1q2(map_q: &[i64], grid_point: i64, d_diag: Vec3I) -> (Vec<i64>, i64) {
    let num_grid = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;
    let adrs0 = grgrid::grid_address_from_index(grid_point, d_diag);
    let mut map_triplets = vec![0i64; num_grid];
    let mut num_ir: i64 = 0;

    // First pass: process only irreducible q-points.
    for gp1 in 0..num_grid {
        if map_q[gp1] == gp1 as i64 {
            let adrs1 = grgrid::grid_address_from_index(gp1 as i64, d_diag);
            let adrs2 = [
                -adrs0[0] - adrs1[0],
                -adrs0[1] - adrs1[1],
                -adrs0[2] - adrs1[2],
            ];
            // If map_q[gp2] is smaller than current gp1, map_q[gp2] should
            // equal a previous gp1 for which map_triplets is already filled.
            let gp2 = grgrid::grid_index_from_address(adrs2, d_diag) as usize;
            if map_q[gp2] < gp1 as i64 {
                map_triplets[gp1] = map_q[gp2];
            } else {
                map_triplets[gp1] = gp1 as i64;
                num_ir += 1;
            }
        }
    }

    // Second pass: fill elements for non-irreducible q-points.
    for gp1 in 0..num_grid {
        if map_q[gp1] != gp1 as i64 {
            // map_q[gp1] is one of the ir-gp1, so it is already filled.
            map_triplets[gp1] = map_triplets[map_q[gp1] as usize];
        }
    }

    (map_triplets, num_ir)
}

/// Non-swappable case: q1 and q2 cannot be exchanged.
fn ir_triplets_at_q_noperm(map_q: &[i64], d_diag: Vec3I) -> (Vec<i64>, i64) {
    let num_grid = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;
    let mut map_triplets = vec![0i64; num_grid];
    let mut num_ir: i64 = 0;

    for gp1 in 0..num_grid {
        if map_q[gp1] == gp1 as i64 {
            map_triplets[gp1] = gp1 as i64;
            num_ir += 1;
        } else {
            map_triplets[gp1] = map_triplets[map_q[gp1] as usize];
        }
    }

    (map_triplets, num_ir)
}

/// Search symmetry-reduced triplets at a fixed q-point.
///
/// Given reciprocal-space rotations (q' = R q) and a fixed grid
/// point, find the irreducible set of triplets (q0, q1, q2) with
/// q0 + q1 + q2 = G.
///
/// Returns `map_triplets` (mapping all q-points to ir-representatives),
/// `map_q` (ir grid map under the stabilizer of q0), and the count
/// of irreducible triplets.
pub fn ir_triplets_at_q(
    grid_point: i64,
    d_diag: Vec3I,
    rec_rotations: &[MatI],
    is_time_reversal: bool,
    swappable: bool,
) -> Result<IrTripletsAtQ, ReciprocalRotationsError> {
    let rotations =
        recip_rotations::get_reciprocal_point_group(rec_rotations, is_time_reversal, false)?;
    let stabilizer = get_stabilizer_rotations(&rotations, d_diag, grid_point);

    let ps = [0i64; 3];
    let ir_map = grgrid::ir_grid_map(&stabilizer, d_diag, ps);
    let map_q = ir_map.map;

    let (map_triplets, num_ir) = if swappable {
        ir_triplets_at_q_perm_q1q2(&map_q, grid_point, d_diag)
    } else {
        ir_triplets_at_q_noperm(&map_q, d_diag)
    };

    Ok(IrTripletsAtQ {
        map_triplets,
        map_q,
        num_ir,
    })
}

// ---------- BZ triplets at q ----------

/// Compute `LQD_inv = reclat * Q * diag(1 / D_diag)`.
fn get_lqd_inv(reclat: &MatD, q: &MatI, d_diag: Vec3I) -> MatD {
    let q_d = mat_i_to_d(q);
    let mut lqd = matmul_d(reclat, &q_d);
    for i in 0..3 {
        for k in 0..3 {
            lqd[i][k] /= d_diag[k] as f64;
        }
    }
    lqd
}

fn squared_distance(g: Vec3I, lqd_inv: &MatD) -> f64 {
    let g_d = [g[0] as f64, g[1] as f64, g[2] as f64];
    let d = matvec_dd(lqd_inv, g_d);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn add3(a: Vec3I, b: Vec3I, c: Vec3I) -> Vec3I {
    [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]
}

/// Sparse-layout (type 1) BZ image search.  For each ir-q1, scan all
/// translationally-equivalent BZ images of (q0, q1, q2 = -q0 - q1)
/// and keep the triple whose total G is smallest.
fn bz_triplets_at_q_type1(
    triplets: &mut [[i64; 3]],
    grid_point: i64,
    bz_grid_addresses: &[Vec3I],
    bz_map: &[i64],
    ir_q1_gps: &[i64],
    d_diag: Vec3I,
    lqd_inv: &MatD,
    tolerance: f64,
) {
    let bz_adrs0 = bz_grid_addresses[grid_point as usize];
    let num_gp = (d_diag[0] * d_diag[1] * d_diag[2]) as i64;
    let num_bzgp = (num_gp * 8) as usize;
    let gp0_idx = grid_point as usize;

    for (i, &q1_gp) in ir_q1_gps.iter().enumerate() {
        let q1_idx = q1_gp as usize;
        let bz_adrs1 = bz_grid_addresses[q1_idx];
        let bz_adrs2 = [
            -bz_adrs0[0] - bz_adrs1[0],
            -bz_adrs0[1] - bz_adrs1[1],
            -bz_adrs0[2] - bz_adrs1[2],
        ];
        let gp2 = grgrid::grid_index_from_address(bz_adrs2, d_diag);
        let gp2_idx = gp2 as usize;

        let n0 = bz_map[num_bzgp + gp0_idx + 1] - bz_map[num_bzgp + gp0_idx] + 1;
        let n1 = bz_map[num_bzgp + q1_idx + 1] - bz_map[num_bzgp + q1_idx] + 1;
        let n2 = bz_map[num_bzgp + gp2_idx + 1] - bz_map[num_bzgp + gp2_idx] + 1;

        let mut min_d2: f64 = -1.0;

        'outer: for bz0 in 0..n0 {
            let bzgp0 = if bz0 == 0 {
                grid_point
            } else {
                num_gp + bz_map[num_bzgp + gp0_idx] + bz0 - 1
            };
            for bz1 in 0..n1 {
                let bzgp1 = if bz1 == 0 {
                    q1_gp
                } else {
                    num_gp + bz_map[num_bzgp + q1_idx] + bz1 - 1
                };
                for bz2 in 0..n2 {
                    let bzgp2 = if bz2 == 0 {
                        gp2
                    } else {
                        num_gp + bz_map[num_bzgp + gp2_idx] + bz2 - 1
                    };
                    let g = add3(
                        bz_grid_addresses[bzgp0 as usize],
                        bz_grid_addresses[bzgp1 as usize],
                        bz_grid_addresses[bzgp2 as usize],
                    );
                    if g == [0, 0, 0] {
                        triplets[i] = [bzgp0, bzgp1, bzgp2];
                        break 'outer;
                    }
                    let d2 = squared_distance(g, lqd_inv);
                    if min_d2 < 0.0 || d2 < min_d2 - tolerance {
                        min_d2 = d2;
                        triplets[i] = [bzgp0, bzgp1, bzgp2];
                    }
                }
            }
        }
    }
}

/// Dense-layout (type 2) BZ image search.  `bz_map[g]..bz_map[g+1]`
/// directly enumerates the BZ images of GR-grid point `g`.
fn bz_triplets_at_q_type2(
    triplets: &mut [[i64; 3]],
    grid_point: i64,
    bz_grid_addresses: &[Vec3I],
    bz_map: &[i64],
    ir_q1_gps: &[i64],
    d_diag: Vec3I,
    lqd_inv: &MatD,
    tolerance: f64,
) {
    let bz_adrs0 = bz_grid_addresses[grid_point as usize];
    let gp0 = grgrid::grid_index_from_address(bz_adrs0, d_diag);
    let gp0_idx = gp0 as usize;

    for (i, &q1_gp) in ir_q1_gps.iter().enumerate() {
        let q1_idx = q1_gp as usize;
        let bz_adrs1 = bz_grid_addresses[bz_map[q1_idx] as usize];
        let bz_adrs2 = [
            -bz_adrs0[0] - bz_adrs1[0],
            -bz_adrs0[1] - bz_adrs1[1],
            -bz_adrs0[2] - bz_adrs1[2],
        ];
        let gp2 = grgrid::grid_index_from_address(bz_adrs2, d_diag);
        let gp2_idx = gp2 as usize;

        let r0 = bz_map[gp0_idx]..bz_map[gp0_idx + 1];
        let r1 = bz_map[q1_idx]..bz_map[q1_idx + 1];
        let r2 = bz_map[gp2_idx]..bz_map[gp2_idx + 1];

        let mut min_d2: f64 = -1.0;

        'outer: for bzgp0 in r0.clone() {
            for bzgp1 in r1.clone() {
                for bzgp2 in r2.clone() {
                    let g = add3(
                        bz_grid_addresses[bzgp0 as usize],
                        bz_grid_addresses[bzgp1 as usize],
                        bz_grid_addresses[bzgp2 as usize],
                    );
                    if g == [0, 0, 0] {
                        triplets[i] = [bzgp0, bzgp1, bzgp2];
                        break 'outer;
                    }
                    let d2 = squared_distance(g, lqd_inv);
                    if min_d2 < 0.0 || d2 < min_d2 - tolerance {
                        min_d2 = d2;
                        triplets[i] = [bzgp0, bzgp1, bzgp2];
                    }
                }
            }
        }
    }
}

/// Find symmetry-reduced BZ triplets `(q0, q1, q2)` with `q0+q1+q2 = G`.
///
/// `map_triplets` is the output of `ir_triplets_at_q`; ir-q1 grid
/// points are those with `map_triplets[i] == i`.  For each such q1,
/// the routine searches BZ-equivalent images of `(q0, q1, q2)` and
/// keeps the triple whose total reciprocal vector `G` has the
/// smallest squared length under `LQD_inv`.
///
/// Returns the triplet array of shape `(num_ir, 3)`.
pub fn bz_triplets_at_q(
    grid_point: i64,
    bz_grid_addresses: &[Vec3I],
    bz_map: &[i64],
    map_triplets: &[i64],
    d_diag: Vec3I,
    q: MatI,
    reciprocal_lattice: MatD,
    bz_grid_type: i64,
) -> Result<Vec<[i64; 3]>, BzTripletsError> {
    if bz_grid_type != 1 && bz_grid_type != 2 {
        return Err(BzTripletsError::BadGridType);
    }

    let ir_q1_gps: Vec<i64> = map_triplets
        .iter()
        .enumerate()
        .filter(|(i, &v)| v == *i as i64)
        .map(|(i, _)| i as i64)
        .collect();

    let mut triplets: Vec<[i64; 3]> = vec![[-1; 3]; ir_q1_gps.len()];
    let lqd_inv = get_lqd_inv(&reciprocal_lattice, &q, d_diag);
    let tolerance = bzgrid::bz_tolerance(&reciprocal_lattice, d_diag);

    if bz_grid_type == 1 {
        bz_triplets_at_q_type1(
            &mut triplets,
            grid_point,
            bz_grid_addresses,
            bz_map,
            &ir_q1_gps,
            d_diag,
            &lqd_inv,
            tolerance,
        );
    } else {
        bz_triplets_at_q_type2(
            &mut triplets,
            grid_point,
            bz_grid_addresses,
            bz_map,
            &ir_q1_gps,
            d_diag,
            &lqd_inv,
            tolerance,
        );
    }

    Ok(triplets)
}
