//! Brillouin-zone grid address construction.
//!
//! Port of `bzg_get_bz_grid_addresses` and friends in `c/bzgrid.c`.

use crate::common::{matvec_dd, matvec_id, matvec_ii, nint, MatD, MatI, Vec3I};
use crate::grgrid::{double_grid_address, grid_address_from_index, grid_index_from_address};

const GRID_TOLERANCE_FACTOR: f64 = 0.01;

/// 125-point search space used to locate BZ-equivalent images of
/// each GR-grid point.  Copied verbatim from `bz_search_space` in
/// `c/bzgrid.c`.
const BZ_SEARCH_SPACE: [Vec3I; 125] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 2],
    [0, 0, -2],
    [0, 0, -1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 2],
    [0, 1, -2],
    [0, 1, -1],
    [0, 2, 0],
    [0, 2, 1],
    [0, 2, 2],
    [0, 2, -2],
    [0, 2, -1],
    [0, -2, 0],
    [0, -2, 1],
    [0, -2, 2],
    [0, -2, -2],
    [0, -2, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, -1, 2],
    [0, -1, -2],
    [0, -1, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 2],
    [1, 0, -2],
    [1, 0, -1],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, -2],
    [1, 1, -1],
    [1, 2, 0],
    [1, 2, 1],
    [1, 2, 2],
    [1, 2, -2],
    [1, 2, -1],
    [1, -2, 0],
    [1, -2, 1],
    [1, -2, 2],
    [1, -2, -2],
    [1, -2, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, -1, 2],
    [1, -1, -2],
    [1, -1, -1],
    [2, 0, 0],
    [2, 0, 1],
    [2, 0, 2],
    [2, 0, -2],
    [2, 0, -1],
    [2, 1, 0],
    [2, 1, 1],
    [2, 1, 2],
    [2, 1, -2],
    [2, 1, -1],
    [2, 2, 0],
    [2, 2, 1],
    [2, 2, 2],
    [2, 2, -2],
    [2, 2, -1],
    [2, -2, 0],
    [2, -2, 1],
    [2, -2, 2],
    [2, -2, -2],
    [2, -2, -1],
    [2, -1, 0],
    [2, -1, 1],
    [2, -1, 2],
    [2, -1, -2],
    [2, -1, -1],
    [-2, 0, 0],
    [-2, 0, 1],
    [-2, 0, 2],
    [-2, 0, -2],
    [-2, 0, -1],
    [-2, 1, 0],
    [-2, 1, 1],
    [-2, 1, 2],
    [-2, 1, -2],
    [-2, 1, -1],
    [-2, 2, 0],
    [-2, 2, 1],
    [-2, 2, 2],
    [-2, 2, -2],
    [-2, 2, -1],
    [-2, -2, 0],
    [-2, -2, 1],
    [-2, -2, 2],
    [-2, -2, -2],
    [-2, -2, -1],
    [-2, -1, 0],
    [-2, -1, 1],
    [-2, -1, 2],
    [-2, -1, -2],
    [-2, -1, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 0, 2],
    [-1, 0, -2],
    [-1, 0, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [-1, 1, 2],
    [-1, 1, -2],
    [-1, 1, -1],
    [-1, 2, 0],
    [-1, 2, 1],
    [-1, 2, 2],
    [-1, 2, -2],
    [-1, 2, -1],
    [-1, -2, 0],
    [-1, -2, 1],
    [-1, -2, 2],
    [-1, -2, -2],
    [-1, -2, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, -1, 2],
    [-1, -1, -2],
    [-1, -1, -1],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BzGridAddressesError {
    /// `Q` is not unimodular (|det(Q)| != 1).
    NotUnimodularQ,
    /// Unsupported `bz_grid_type` (must be 1 or 2).
    BadGridType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotateBzGridError {
    /// Unsupported `bz_grid_type` (must be 1 or 2).
    BadGridType,
}

/// Output of `bz_grid_addresses`.
pub struct BzGridAddresses {
    /// Addresses in BZ layout, length = `num_gp`.
    pub addresses: Vec<Vec3I>,
    /// Map from GR-grid mesh (type 2) or 2x mesh (type 1) to BZ index.
    pub bz_map: Vec<i64>,
    /// Map from BZ index back to GR grid index.
    pub bzg2grg: Vec<i64>,
}

fn inverse_unimodular(m: &MatI) -> Option<MatI> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() != 1 {
        return None;
    }
    let c = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[2][1] * m[0][2] - m[2][2] * m[0][1]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[2][2] * m[0][0] - m[2][0] * m[0][2]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[2][0] * m[0][1] - m[2][1] * m[0][0]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    Some(c)
}

/// Tolerance for BZ reduction, in units of squared length.
pub(crate) fn bz_tolerance(rec_lattice: &MatD, d_diag: Vec3I) -> f64 {
    let mut max_len = 0.0f64;
    for i in 0..3 {
        let mut len = 0.0;
        for j in 0..3 {
            len += rec_lattice[j][i] * rec_lattice[j][i];
        }
        len /= (d_diag[i] * d_diag[i]) as f64;
        if len > max_len {
            max_len = len;
        }
    }
    max_len * GRID_TOLERANCE_FACTOR
}

/// Compute distances to 125 candidate BZ-equivalent points and
/// identify the minimum.  Returns `(min_distance, nint, distances)`
/// matching the C helper `get_bz_distances`.
fn bz_distances(
    grid_address: Vec3I,
    d_diag: Vec3I,
    ps: Vec3I,
    q: &MatI,
    rec_lattice: &MatD,
    tolerance: f64,
) -> (f64, Vec3I, [f64; 125]) {
    let dadrs = double_grid_address(grid_address, ps);
    let mut q_red = [
        dadrs[0] as f64 / (2.0 * d_diag[0] as f64),
        dadrs[1] as f64 / (2.0 * d_diag[1] as f64),
        dadrs[2] as f64 / (2.0 * d_diag[2] as f64),
    ];
    q_red = matvec_id(q, q_red);
    let mut nint_v = [0i64; 3];
    for i in 0..3 {
        nint_v[i] = nint(q_red[i]);
        q_red[i] -= nint_v[i] as f64;
    }

    let mut distances = [0.0f64; 125];
    for (i, g) in BZ_SEARCH_SPACE.iter().enumerate() {
        let v = [
            q_red[0] + g[0] as f64,
            q_red[1] + g[1] as f64,
            q_red[2] + g[2] as f64,
        ];
        let vc = matvec_dd(rec_lattice, v);
        distances[i] = vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2];
    }

    let mut min_distance = distances[0];
    for i in 1..125 {
        if distances[i] < min_distance - tolerance {
            min_distance = distances[i];
        }
    }
    (min_distance, nint_v, distances)
}

fn set_bz_address(
    bz_index: usize,
    grid_address: Vec3I,
    d_diag: Vec3I,
    nint_v: Vec3I,
    q_inv: &MatI,
) -> Vec3I {
    let delta_g = [
        BZ_SEARCH_SPACE[bz_index][0] - nint_v[0],
        BZ_SEARCH_SPACE[bz_index][1] - nint_v[1],
        BZ_SEARCH_SPACE[bz_index][2] - nint_v[2],
    ];
    let delta_g = matvec_ii(q_inv, delta_g);
    [
        grid_address[0] + delta_g[0] * d_diag[0],
        grid_address[1] + delta_g[1] * d_diag[1],
        grid_address[2] + delta_g[2] * d_diag[2],
    ]
}

/// Build BZ grid addresses, map, and GR mapping.
///
/// `bz_grid_type` mirrors the C API: `1` = sparse layout, `2` = dense layout.
pub fn bz_grid_addresses(
    d_diag: Vec3I,
    q: MatI,
    ps: Vec3I,
    rec_lattice: MatD,
    bz_grid_type: i64,
) -> Result<BzGridAddresses, BzGridAddressesError> {
    debug_assert!(d_diag.iter().all(|&d| d >= 1));
    if bz_grid_type != 1 && bz_grid_type != 2 {
        return Err(BzGridAddressesError::BadGridType);
    }
    let q_inv = inverse_unimodular(&q).ok_or(BzGridAddressesError::NotUnimodularQ)?;
    let tolerance = bz_tolerance(&rec_lattice, d_diag);

    let total_num_gp = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;

    if bz_grid_type == 2 {
        // Dense layout: addresses appear in order, gp_map holds the
        // starting BZ index for each GR-grid point (length = total+1).
        let mut addresses: Vec<Vec3I> = Vec::with_capacity(total_num_gp * 8);
        let mut bzg2grg: Vec<i64> = Vec::with_capacity(total_num_gp * 8);
        let mut bz_map = vec![0i64; total_num_gp + 1];

        for i in 0..total_num_gp {
            let gr_adrs = grid_address_from_index(i as i64, d_diag);
            let (min_distance, nint_v, distances) =
                bz_distances(gr_adrs, d_diag, ps, &q, &rec_lattice, tolerance);
            for j in 0..125 {
                if distances[j] < min_distance + tolerance {
                    let adrs = set_bz_address(j, gr_adrs, d_diag, nint_v, &q_inv);
                    addresses.push(adrs);
                    bzg2grg.push(i as i64);
                }
            }
            bz_map[i + 1] = addresses.len() as i64;
        }
        addresses.shrink_to_fit();
        bzg2grg.shrink_to_fit();
        return Ok(BzGridAddresses {
            addresses,
            bz_map,
            bzg2grg,
        });
    }

    // Sparse layout (type 1).
    let bzmesh: Vec3I = [d_diag[0] * 2, d_diag[1] * 2, d_diag[2] * 2];
    let num_bzmesh = (bzmesh[0] * bzmesh[1] * bzmesh[2]) as usize;

    let mut addresses: Vec<Vec3I> = vec![[0; 3]; total_num_gp * 8];
    let mut bzg2grg: Vec<i64> = vec![0; total_num_gp * 8];
    let mut bz_map = vec![num_bzmesh as i64; num_bzmesh + total_num_gp + 1];
    bz_map[num_bzmesh] = 0;

    let mut boundary_num_gp: i64 = 0;
    let mut id_shift: i64 = 0;

    for i in 0..total_num_gp as i64 {
        let gr_adrs = grid_address_from_index(i, d_diag);
        let (min_distance, nint_v, distances) =
            bz_distances(gr_adrs, d_diag, ps, &q, &rec_lattice, tolerance);
        let mut count: i64 = 0;
        for j in 0..125 {
            if distances[j] < min_distance + tolerance {
                let gp = if count == 0 {
                    i
                } else {
                    let g = boundary_num_gp + total_num_gp as i64;
                    boundary_num_gp += 1;
                    g
                };
                count += 1;
                let adrs = set_bz_address(j, gr_adrs, d_diag, nint_v, &q_inv);
                addresses[gp as usize] = adrs;
                // Equivalent to C's grg_get_double_grid_index on the
                // 2x mesh: `(adrs*2 + ps - ps)/2` collapses to `adrs`,
                // after which `grid_index_from_address` applies the
                // Euclidean modulo against `bzmesh`.
                let bzgp = grid_index_from_address(adrs, bzmesh);
                bz_map[bzgp as usize] = gp;
                bzg2grg[gp as usize] = i;
            }
        }
        id_shift += count - 1;
        bz_map[num_bzmesh + (i as usize) + 1] = id_shift;
    }
    let size = (boundary_num_gp + total_num_gp as i64) as usize;
    addresses.truncate(size);
    bzg2grg.truncate(size);
    Ok(BzGridAddresses {
        addresses,
        bz_map,
        bzg2grg,
    })
}

/// Rotate a BZ-grid point and return the rotated point's BZ index.
///
/// Port of `bzg_rotate_grid_index` in `c/bzgrid.c`.  Searches the
/// translationally-equivalent BZ images listed in `bz_map` for the
/// one whose address exactly matches the rotated address.  Falls
/// back to the first image (the C code's "ill-defined bzgrid"
/// branch) when no match is found.
pub fn rotate_bz_grid_index(
    bz_grid_index: i64,
    rotation: MatI,
    bz_grid_addresses: &[Vec3I],
    bz_map: &[i64],
    d_diag: Vec3I,
    ps: Vec3I,
    bz_grid_type: i64,
) -> Result<i64, RotateBzGridError> {
    if bz_grid_type != 1 && bz_grid_type != 2 {
        return Err(RotateBzGridError::BadGridType);
    }

    let start_adrs = bz_grid_addresses[bz_grid_index as usize];
    let dadrs = double_grid_address(start_adrs, ps);
    let dadrs_rot = matvec_ii(&rotation, dadrs);
    // `(dadrs_rot - PS) / 2` uses truncation toward zero, matching C.
    let adrs_rot = [
        (dadrs_rot[0] - ps[0]) / 2,
        (dadrs_rot[1] - ps[1]) / 2,
        (dadrs_rot[2] - ps[2]) / 2,
    ];
    let gp = grid_index_from_address(adrs_rot, d_diag);

    if bz_grid_type == 1 {
        if bz_grid_addresses[gp as usize] == adrs_rot {
            return Ok(gp);
        }
        let num_grgp = d_diag[0] * d_diag[1] * d_diag[2];
        let num_bzgp = num_grgp * 8;
        let start = bz_map[(num_bzgp + gp) as usize] + num_grgp;
        let end = bz_map[(num_bzgp + gp + 1) as usize] + num_grgp;
        for i in start..end {
            if bz_grid_addresses[i as usize] == adrs_rot {
                return Ok(i);
            }
        }
        // Ill-defined bzgrid fallback.
        Ok(bz_map[gp as usize])
    } else {
        let start = bz_map[gp as usize];
        let end = bz_map[(gp + 1) as usize];
        for i in start..end {
            if bz_grid_addresses[i as usize] == adrs_rot {
                return Ok(i);
            }
        }
        Ok(bz_map[gp as usize])
    }
}
