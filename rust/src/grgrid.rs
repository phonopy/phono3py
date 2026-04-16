//! Generalized regular grid helpers.
//!
//! Port of selected functions from `c/grgrid.c`.

/// Map a grid address to its single-grid-point index.
///
/// Mirrors `grg_get_grid_index` in `c/grgrid.c`: each address
/// component is reduced to `[0, d_diag[i])` using Euclidean
/// modulo, then the components are combined in the default
/// (non `GRID_ORDER_XYZ`) layout.
pub fn grid_index_from_address(address: [i64; 3], d_diag: [i64; 3]) -> i64 {
    debug_assert!(d_diag.iter().all(|&d| d >= 1));
    let r0 = address[0].rem_euclid(d_diag[0]);
    let r1 = address[1].rem_euclid(d_diag[1]);
    let r2 = address[2].rem_euclid(d_diag[2]);
    r2 * d_diag[0] * d_diag[1] + r1 * d_diag[0] + r0
}

/// Inverse of `grid_index_from_address` for a canonical
/// `[0, d_diag[i])` address.  Mirrors `get_grid_address_from_index`
/// in `c/grgrid.c` (non `GRID_ORDER_XYZ` branch).
pub(crate) fn grid_address_from_index(grid_index: i64, d_diag: [i64; 3]) -> [i64; 3] {
    let nn = d_diag[0] * d_diag[1];
    let a2 = grid_index / nn;
    let a1 = (grid_index - a2 * nn) / d_diag[0];
    let a0 = grid_index % d_diag[0];
    [a0, a1, a2]
}

/// Double-grid address `2 * address + PS`.
pub(crate) fn double_grid_address(address: [i64; 3], ps: [i64; 3]) -> [i64; 3] {
    [
        address[0] * 2 + ps[0],
        address[1] * 2 + ps[1],
        address[2] * 2 + ps[2],
    ]
}

/// Map a double-grid address to its single-grid index.
///
/// Integer division `(x - ps) / 2` uses truncation toward zero,
/// matching the C implementation's signed-int behaviour.
fn double_grid_index(address_double: [i64; 3], d_diag: [i64; 3], ps: [i64; 3]) -> i64 {
    let address = [
        (address_double[0] - ps[0]) / 2,
        (address_double[1] - ps[1]) / 2,
        (address_double[2] - ps[2]) / 2,
    ];
    grid_index_from_address(address, d_diag)
}

/// Apply an integer rotation to a grid point and return the index
/// of the rotated point.  Mirrors `grg_rotate_grid_index` in C.
fn rotate_grid_index(
    grid_index: i64,
    rotation: [[i64; 3]; 3],
    d_diag: [i64; 3],
    ps: [i64; 3],
) -> i64 {
    let adrs = grid_address_from_index(grid_index, d_diag);
    let dadrs = double_grid_address(adrs, ps);
    let dadrs_rot = [
        rotation[0][0] * dadrs[0] + rotation[0][1] * dadrs[1] + rotation[0][2] * dadrs[2],
        rotation[1][0] * dadrs[0] + rotation[1][1] * dadrs[1] + rotation[1][2] * dadrs[2],
        rotation[2][0] * dadrs[0] + rotation[2][1] * dadrs[1] + rotation[2][2] * dadrs[2],
    ];
    double_grid_index(dadrs_rot, d_diag, ps)
}

/// Irreducible grid map plus count of irreducible points.
pub struct IrGridMap {
    pub map: Vec<i64>,
    pub num_ir: i64,
}

/// Build the mapping from each GR grid point to its irreducible
/// representative.  The representative is always the smallest
/// symmetrically equivalent index, so a single forward pass over
/// `gp` suffices.
///
/// Mirrors `grg_get_ir_grid_map` + the irreducible-count loop in
/// `recgrid_get_ir_grid_map` from `c/recgrid.c`.
pub fn ir_grid_map(
    rotations: &[[[i64; 3]; 3]],
    d_diag: [i64; 3],
    ps: [i64; 3],
) -> IrGridMap {
    debug_assert!(d_diag.iter().all(|&d| d >= 1));
    let num_gp = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;
    let sentinel = num_gp as i64;
    let mut map = vec![sentinel; num_gp];

    for gp in 0..num_gp as i64 {
        for rot in rotations {
            let r_gp = rotate_grid_index(gp, *rot, d_diag, ps);
            if r_gp < gp {
                map[gp as usize] = map[r_gp as usize];
                break;
            }
        }
        if map[gp as usize] == sentinel {
            map[gp as usize] = gp;
        }
    }

    let num_ir = map
        .iter()
        .enumerate()
        .filter(|(i, &v)| v == *i as i64)
        .count() as i64;
    IrGridMap { map, num_ir }
}

/// Enumerate all grid addresses in GR-grid-index order.
///
/// Mirrors `grg_get_all_grid_addresses` in `c/grgrid.c`.
/// The loop nesting (k outer, i inner) ensures the pushed
/// index matches `grid_index_from_address([i, j, k], d_diag)`
/// so the output is indexed by grid point.
pub fn all_grid_addresses(d_diag: [i64; 3]) -> Vec<[i64; 3]> {
    debug_assert!(d_diag.iter().all(|&d| d >= 1));
    let n = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;
    let mut out = Vec::with_capacity(n);
    for k in 0..d_diag[2] {
        for j in 0..d_diag[1] {
            for i in 0..d_diag[0] {
                out.push([i, j, k]);
            }
        }
    }
    out
}
