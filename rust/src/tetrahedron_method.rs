//! Tetrahedron method for Brillouin-zone integration weights.
//!
//! Port of `c/tetrahedron_method.c`.  The unit cell is split into
//! six tetrahedra; the four-times-six = 24-tetrahedra version
//! used here also accounts for the shortest main diagonal of the
//! reciprocal lattice (`get_main_diagonal`), giving four
//! pre-tabulated relative-address tables.

#![allow(dead_code)]

use crate::common::{matvec_di, MatD, Vec3D, Vec3I};

/// `THM_EPSILON=1e-10` is unconditionally set in `CMakeLists.txt`
/// for every build that links `tetrahedron_method.c`, so port the
/// epsilon-guarded branches as the live code path here.
const THM_EPSILON: f64 = 1e-10;

///      6-------7
///     /|      /|
///    / |     / |
///   4-------5  |
///   |  2----|--3
///   | /     | /
///   |/      |/
///   0-------1
///
///  i: vec        neighbours
///  0: O          1, 2, 4
///  1: a          0, 3, 5
///  2: b          0, 3, 6
///  3: a + b      1, 2, 7
///  4: c          0, 5, 6
///  5: c + a      1, 4, 7
///  6: c + b      2, 4, 7
///  7: c + a + b  3, 5, 6

const MAIN_DIAGONALS: [Vec3I; 4] = [
    [1, 1, 1],  // 0-7
    [-1, 1, 1], // 1-6
    [1, -1, 1], // 2-5
    [1, 1, -1], // 3-4
];

/// Per-tetrahedron, per-vertex relative grid address.
pub type RelativeGridAddress = [[Vec3I; 4]; 24];

/// All four main-diagonal variants stacked together.
pub type AllRelativeGridAddress = [RelativeGridAddress; 4];

const DB_RELATIVE_GRID_ADDRESS: AllRelativeGridAddress = [
    [
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [-1, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [-1, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, -1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [0, -1, 0]],
        [[0, 0, 0], [0, 0, 1], [-1, -1, 0], [0, -1, 0]],
        [[0, 0, 0], [0, 0, 1], [-1, -1, 0], [-1, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, -1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, -1]],
        [[0, 0, 0], [0, 1, 0], [-1, 0, -1], [0, 0, -1]],
        [[0, 0, 0], [0, 1, 0], [-1, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, -1, -1], [0, 0, -1]],
        [[0, 0, 0], [1, 0, 0], [0, -1, -1], [0, -1, 0]],
        [[0, 0, 0], [-1, -1, -1], [0, -1, -1], [0, 0, -1]],
        [[0, 0, 0], [-1, -1, -1], [0, -1, -1], [0, -1, 0]],
        [[0, 0, 0], [-1, -1, -1], [-1, 0, -1], [0, 0, -1]],
        [[0, 0, 0], [-1, -1, -1], [-1, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [-1, -1, -1], [-1, -1, 0], [0, -1, 0]],
        [[0, 0, 0], [-1, -1, -1], [-1, -1, 0], [-1, 0, 0]],
    ],
    [
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]],
        [[0, 0, 0], [-1, 1, 0], [-1, 1, 1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 0, 1], [-1, 1, 1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 1, 0], [-1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [-1, 1, 1], [0, 1, 1]],
        [[0, 0, 0], [-1, 0, 1], [0, 0, 1], [-1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [-1, 1, 1], [0, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [0, -1, 0], [1, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, -1, 0]],
        [[0, 0, 0], [-1, 0, 1], [0, -1, 0], [-1, 0, 0]],
        [[0, 0, 0], [-1, 0, 1], [0, 0, 1], [0, -1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, -1]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, -1]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 1, 0], [0, 0, -1]],
        [[0, 0, 0], [0, -1, -1], [1, -1, -1], [0, 0, -1]],
        [[0, 0, 0], [0, -1, -1], [1, -1, -1], [0, -1, 0]],
        [[0, 0, 0], [1, -1, -1], [0, 0, -1], [1, 0, -1]],
        [[0, 0, 0], [1, 0, 0], [1, -1, -1], [1, 0, -1]],
        [[0, 0, 0], [1, -1, -1], [0, -1, 0], [1, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, -1, -1], [1, -1, 0]],
        [[0, 0, 0], [0, -1, -1], [0, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [0, -1, -1], [0, -1, 0], [-1, 0, 0]],
    ],
    [
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [1, -1, 1], [0, -1, 0], [1, -1, 0]],
        [[0, 0, 0], [0, -1, 1], [1, -1, 1], [0, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, -1, 1], [1, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, -1, 1], [1, 0, 1]],
        [[0, 0, 0], [0, -1, 1], [1, -1, 1], [0, 0, 1]],
        [[0, 0, 0], [1, -1, 1], [0, 0, 1], [1, 0, 1]],
        [[0, 0, 0], [0, -1, 1], [0, -1, 0], [-1, 0, 0]],
        [[0, 0, 0], [0, -1, 1], [0, 0, 1], [-1, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, -1]],
        [[0, 0, 0], [-1, 0, -1], [0, 0, -1], [-1, 1, -1]],
        [[0, 0, 0], [-1, 0, -1], [-1, 1, -1], [-1, 0, 0]],
        [[0, 0, 0], [0, 0, -1], [-1, 1, -1], [0, 1, -1]],
        [[0, 0, 0], [0, 1, 0], [-1, 1, -1], [0, 1, -1]],
        [[0, 0, 0], [-1, 1, 0], [-1, 1, -1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 1, 0], [-1, 1, -1]],
        [[0, 0, 0], [0, 0, -1], [0, -1, 0], [1, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, -1], [1, -1, 0]],
        [[0, 0, 0], [-1, 0, -1], [0, 0, -1], [0, -1, 0]],
        [[0, 0, 0], [-1, 0, -1], [0, -1, 0], [-1, 0, 0]],
    ],
    [
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [-1, 0, 1], [-1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [-1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 1]],
        [[0, 0, 0], [-1, -1, 1], [-1, -1, 0], [0, -1, 0]],
        [[0, 0, 0], [-1, -1, 1], [-1, -1, 0], [-1, 0, 0]],
        [[0, 0, 0], [-1, -1, 1], [0, -1, 1], [0, -1, 0]],
        [[0, 0, 0], [-1, -1, 1], [-1, 0, 1], [-1, 0, 0]],
        [[0, 0, 0], [-1, -1, 1], [0, -1, 1], [0, 0, 1]],
        [[0, 0, 0], [-1, -1, 1], [-1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 0, -1], [1, 0, -1], [1, 1, -1]],
        [[0, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1]],
        [[0, 0, 0], [1, 0, 0], [1, 0, -1], [1, 1, -1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, -1], [1, 1, -1]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, -1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, -1]],
        [[0, 0, 0], [0, 0, -1], [0, 1, -1], [-1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, -1], [-1, 0, 0]],
        [[0, 0, 0], [0, 0, -1], [1, 0, -1], [0, -1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, -1], [0, -1, 0]],
        [[0, 0, 0], [0, 0, -1], [-1, -1, 0], [0, -1, 0]],
        [[0, 0, 0], [0, 0, -1], [-1, -1, 0], [-1, 0, 0]],
    ],
];

/// Selector for `integration_weight` — port of the `'I'`/other
/// `char` argument of `thm_get_integration_weight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFunction {
    /// `'I'` branch: piecewise function `I` paired with weight `g`.
    I,
    /// non-`'I'` branch: piecewise `J` paired with weight `n`.
    J,
}

/// Pick the relative-grid-address table whose main diagonal is
/// shortest in the given reciprocal lattice.  Returns the table
/// and the diagonal index.  Mirrors `thm_get_relative_grid_address`.
pub fn relative_grid_address(rec_lattice: MatD) -> (RelativeGridAddress, i64) {
    let main_diag_index = get_main_diagonal(rec_lattice);
    (
        DB_RELATIVE_GRID_ADDRESS[main_diag_index as usize],
        main_diag_index,
    )
}

/// All four relative-grid-address tables.  Mirrors
/// `thm_get_all_relative_grid_address`.
pub fn all_relative_grid_address() -> AllRelativeGridAddress {
    DB_RELATIVE_GRID_ADDRESS
}

/// Tetrahedron-method integration weight for a single `omega`.
/// Mirrors `thm_get_integration_weight`.
pub fn integration_weight(
    omega: f64,
    tetrahedra_omegas: &[[f64; 4]; 24],
    function: WeightFunction,
) -> f64 {
    match function {
        WeightFunction::I => get_integration_weight(omega, tetrahedra_omegas, _g, _i),
        WeightFunction::J => get_integration_weight(omega, tetrahedra_omegas, _n, _j),
    }
}

fn get_integration_weight(
    omega: f64,
    tetrahedra_omegas: &[[f64; 4]; 24],
    gn: fn(i64, f64, &[f64; 4]) -> f64,
    ij: fn(i64, i64, f64, &[f64; 4]) -> f64,
) -> f64 {
    let mut sum = 0.0;
    for i in 0..24 {
        let mut v = tetrahedra_omegas[i];
        let ci = sort_omegas(&mut v);
        // The chained `else if` exactly preserves the C code's
        // strict-inequality guards: at boundary equality (e.g.
        // omega == v[0]) no branch contributes.
        if omega < v[0] {
            sum += ij(0, ci, omega, &v) * gn(0, omega, &v);
        } else if v[0] < omega && omega < v[1] {
            sum += ij(1, ci, omega, &v) * gn(1, omega, &v);
        } else if v[1] < omega && omega < v[2] {
            sum += ij(2, ci, omega, &v) * gn(2, omega, &v);
        } else if v[2] < omega && omega < v[3] {
            sum += ij(3, ci, omega, &v) * gn(3, omega, &v);
        } else if v[3] < omega {
            sum += ij(4, ci, omega, &v) * gn(4, omega, &v);
        }
    }
    sum / 6.0
}

/// Sort `v` ascending in place; return the case index `ci ∈ {0,1,2,3}`
/// describing which input element ended up in position 1 / 2.
/// Mirrors `sort_omegas` in C.
fn sort_omegas(v: &mut [f64; 4]) -> i64 {
    let mut i: i64 = 0;
    let mut w = [0.0; 4];

    if v[0] > v[1] {
        w[0] = v[1];
        w[1] = v[0];
        i = 1;
    } else {
        w[0] = v[0];
        w[1] = v[1];
    }

    if v[2] > v[3] {
        w[2] = v[3];
        w[3] = v[2];
    } else {
        w[2] = v[2];
        w[3] = v[3];
    }

    if w[0] > w[2] {
        v[0] = w[2];
        v[1] = w[0];
        if i == 0 {
            i = 4;
        }
    } else {
        v[0] = w[0];
        v[1] = w[2];
    }

    if w[1] > w[3] {
        v[3] = w[1];
        v[2] = w[3];
        if i == 1 {
            i = 3;
        }
    } else {
        v[3] = w[3];
        v[2] = w[1];
        if i == 1 {
            i = 5;
        }
    }

    if v[1] > v[2] {
        let tmp = v[1];
        v[1] = v[2];
        v[2] = tmp;
        if i == 4 {
            i = 2;
        }
        if i == 5 {
            i = 1;
        }
    } else {
        if i == 4 {
            i = 1;
        }
        if i == 5 {
            i = 2;
        }
    }
    i
}

fn get_main_diagonal(rec_lattice: MatD) -> i64 {
    let mut shortest: i64 = 0;
    let mut min_length = norm_squared(matvec_di(&rec_lattice, MAIN_DIAGONALS[0]));
    for i in 1..4 {
        let length = norm_squared(matvec_di(&rec_lattice, MAIN_DIAGONALS[i]));
        if min_length > length {
            min_length = length;
            shortest = i as i64;
        }
    }
    shortest
}

fn norm_squared(a: Vec3D) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

// ------------------------------------------------------------------
// Piecewise math helpers `_f`, `_n`, `_g`, `_J`, `_I` (private).
// ------------------------------------------------------------------

/// `f(n, m) = (omega - omega_m) / (omega_n - omega_m)`,
/// guarded by THM_EPSILON to avoid 0/0 at degenerate vertices.
fn _f(n: usize, m: usize, omega: f64, vertices_omegas: &[f64; 4]) -> f64 {
    let delta = vertices_omegas[n] - vertices_omegas[m];
    if delta.abs() < THM_EPSILON {
        return 0.0;
    }
    (omega - vertices_omegas[m]) / delta
}

fn _n(i: i64, omega: f64, vertices_omegas: &[f64; 4]) -> f64 {
    match i {
        0 => 0.0,
        1 => _n_1(omega, vertices_omegas),
        2 => _n_2(omega, vertices_omegas),
        3 => _n_3(omega, vertices_omegas),
        4 => 1.0,
        _ => unreachable!("_n received case index {}", i),
    }
}

fn _g(i: i64, omega: f64, vertices_omegas: &[f64; 4]) -> f64 {
    match i {
        0 => 0.0,
        1 => _g_1(omega, vertices_omegas),
        2 => _g_2(omega, vertices_omegas),
        3 => _g_3(omega, vertices_omegas),
        4 => 0.0,
        _ => unreachable!("_g received case index {}", i),
    }
}

fn _j(i: i64, ci: i64, omega: f64, vertices_omegas: &[f64; 4]) -> f64 {
    match (i, ci) {
        (0, _) => 0.0,
        (1, 0) => _j_10(omega, vertices_omegas),
        (1, 1) => _j_11(omega, vertices_omegas),
        (1, 2) => _j_12(omega, vertices_omegas),
        (1, 3) => _j_13(omega, vertices_omegas),
        (2, 0) => _j_20(omega, vertices_omegas),
        (2, 1) => _j_21(omega, vertices_omegas),
        (2, 2) => _j_22(omega, vertices_omegas),
        (2, 3) => _j_23(omega, vertices_omegas),
        (3, 0) => _j_30(omega, vertices_omegas),
        (3, 1) => _j_31(omega, vertices_omegas),
        (3, 2) => _j_32(omega, vertices_omegas),
        (3, 3) => _j_33(omega, vertices_omegas),
        (4, _) => 0.25,
        _ => unreachable!("_j received (i, ci) = ({}, {})", i, ci),
    }
}

fn _i(i: i64, ci: i64, omega: f64, vertices_omegas: &[f64; 4]) -> f64 {
    match (i, ci) {
        (0, _) => 0.0,
        (1, 0) => _i_10(omega, vertices_omegas),
        (1, 1) => _i_11(omega, vertices_omegas),
        (1, 2) => _i_12(omega, vertices_omegas),
        (1, 3) => _i_13(omega, vertices_omegas),
        (2, 0) => _i_20(omega, vertices_omegas),
        (2, 1) => _i_21(omega, vertices_omegas),
        (2, 2) => _i_22(omega, vertices_omegas),
        (2, 3) => _i_23(omega, vertices_omegas),
        (3, 0) => _i_30(omega, vertices_omegas),
        (3, 1) => _i_31(omega, vertices_omegas),
        (3, 2) => _i_32(omega, vertices_omegas),
        (3, 3) => _i_33(omega, vertices_omegas),
        (4, _) => 0.0,
        _ => unreachable!("_i received (i, ci) = ({}, {})", i, ci),
    }
}

fn _n_1(omega: f64, v: &[f64; 4]) -> f64 {
    _f(1, 0, omega, v) * _f(2, 0, omega, v) * _f(3, 0, omega, v)
}

fn _n_2(omega: f64, v: &[f64; 4]) -> f64 {
    _f(3, 1, omega, v) * _f(2, 1, omega, v)
        + _f(3, 0, omega, v) * _f(1, 3, omega, v) * _f(2, 1, omega, v)
        + _f(3, 0, omega, v) * _f(2, 0, omega, v) * _f(1, 2, omega, v)
}

fn _n_3(omega: f64, v: &[f64; 4]) -> f64 {
    1.0 - _f(0, 3, omega, v) * _f(1, 3, omega, v) * _f(2, 3, omega, v)
}

fn _g_1(omega: f64, v: &[f64; 4]) -> f64 {
    3.0 * _f(1, 0, omega, v) * _f(2, 0, omega, v) / (v[3] - v[0])
}

fn _g_2(omega: f64, v: &[f64; 4]) -> f64 {
    3.0 / (v[3] - v[0])
        * (_f(1, 2, omega, v) * _f(2, 0, omega, v) + _f(2, 1, omega, v) * _f(1, 3, omega, v))
}

fn _g_3(omega: f64, v: &[f64; 4]) -> f64 {
    3.0 * _f(1, 3, omega, v) * _f(2, 3, omega, v) / (v[3] - v[0])
}

fn _j_10(omega: f64, v: &[f64; 4]) -> f64 {
    (1.0 + _f(0, 1, omega, v) + _f(0, 2, omega, v) + _f(0, 3, omega, v)) / 4.0
}

fn _j_11(omega: f64, v: &[f64; 4]) -> f64 {
    _f(1, 0, omega, v) / 4.0
}

fn _j_12(omega: f64, v: &[f64; 4]) -> f64 {
    _f(2, 0, omega, v) / 4.0
}

fn _j_13(omega: f64, v: &[f64; 4]) -> f64 {
    _f(3, 0, omega, v) / 4.0
}

fn _j_20(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_2(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (_f(3, 1, omega, v) * _f(2, 1, omega, v)
        + _f(3, 0, omega, v) * _f(1, 3, omega, v) * _f(2, 1, omega, v) * (1.0 + _f(0, 3, omega, v))
        + _f(3, 0, omega, v)
            * _f(2, 0, omega, v)
            * _f(1, 2, omega, v)
            * (1.0 + _f(0, 3, omega, v) + _f(0, 2, omega, v)))
        / 4.0
        / n
}

fn _j_21(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_2(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (_f(3, 1, omega, v) * _f(2, 1, omega, v) * (1.0 + _f(1, 3, omega, v) + _f(1, 2, omega, v))
        + _f(3, 0, omega, v)
            * _f(1, 3, omega, v)
            * _f(2, 1, omega, v)
            * (_f(1, 3, omega, v) + _f(1, 2, omega, v))
        + _f(3, 0, omega, v) * _f(2, 0, omega, v) * _f(1, 2, omega, v) * _f(1, 2, omega, v))
        / 4.0
        / n
}

fn _j_22(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_2(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (_f(3, 1, omega, v) * _f(2, 1, omega, v) * _f(2, 1, omega, v)
        + _f(3, 0, omega, v) * _f(1, 3, omega, v) * _f(2, 1, omega, v) * _f(2, 1, omega, v)
        + _f(3, 0, omega, v)
            * _f(2, 0, omega, v)
            * _f(1, 2, omega, v)
            * (_f(2, 1, omega, v) + _f(2, 0, omega, v)))
        / 4.0
        / n
}

fn _j_23(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_2(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (_f(3, 1, omega, v) * _f(2, 1, omega, v) * _f(3, 1, omega, v)
        + _f(3, 0, omega, v)
            * _f(1, 3, omega, v)
            * _f(2, 1, omega, v)
            * (_f(3, 1, omega, v) + _f(3, 0, omega, v))
        + _f(3, 0, omega, v) * _f(2, 0, omega, v) * _f(1, 2, omega, v) * _f(3, 0, omega, v))
        / 4.0
        / n
}

fn _j_30(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_3(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (1.0 - _f(0, 3, omega, v) * _f(0, 3, omega, v) * _f(1, 3, omega, v) * _f(2, 3, omega, v))
        / 4.0
        / n
}

fn _j_31(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_3(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (1.0 - _f(0, 3, omega, v) * _f(1, 3, omega, v) * _f(1, 3, omega, v) * _f(2, 3, omega, v))
        / 4.0
        / n
}

fn _j_32(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_3(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (1.0 - _f(0, 3, omega, v) * _f(1, 3, omega, v) * _f(2, 3, omega, v) * _f(2, 3, omega, v))
        / 4.0
        / n
}

fn _j_33(omega: f64, v: &[f64; 4]) -> f64 {
    let n = _n_3(omega, v);
    if n < THM_EPSILON {
        return 0.0;
    }
    (1.0 - _f(0, 3, omega, v)
        * _f(1, 3, omega, v)
        * _f(2, 3, omega, v)
        * (1.0 + _f(3, 0, omega, v) + _f(3, 1, omega, v) + _f(3, 2, omega, v)))
        / 4.0
        / n
}

fn _i_10(omega: f64, v: &[f64; 4]) -> f64 {
    (_f(0, 1, omega, v) + _f(0, 2, omega, v) + _f(0, 3, omega, v)) / 3.0
}

fn _i_11(omega: f64, v: &[f64; 4]) -> f64 {
    _f(1, 0, omega, v) / 3.0
}

fn _i_12(omega: f64, v: &[f64; 4]) -> f64 {
    _f(2, 0, omega, v) / 3.0
}

fn _i_13(omega: f64, v: &[f64; 4]) -> f64 {
    _f(3, 0, omega, v) / 3.0
}

fn _i_20(omega: f64, v: &[f64; 4]) -> f64 {
    let f12_20 = _f(1, 2, omega, v) * _f(2, 0, omega, v);
    let g = f12_20 + _f(2, 1, omega, v) * _f(1, 3, omega, v);
    if g < THM_EPSILON {
        return 0.0;
    }
    (_f(0, 3, omega, v) + _f(0, 2, omega, v) * f12_20 / g) / 3.0
}

fn _i_21(omega: f64, v: &[f64; 4]) -> f64 {
    let f13_21 = _f(1, 3, omega, v) * _f(2, 1, omega, v);
    let g = _f(1, 2, omega, v) * _f(2, 0, omega, v) + f13_21;
    if g < THM_EPSILON {
        return 0.0;
    }
    (_f(1, 2, omega, v) + _f(1, 3, omega, v) * f13_21 / g) / 3.0
}

fn _i_22(omega: f64, v: &[f64; 4]) -> f64 {
    let f12_20 = _f(1, 2, omega, v) * _f(2, 0, omega, v);
    let g = f12_20 + _f(2, 1, omega, v) * _f(1, 3, omega, v);
    if g < THM_EPSILON {
        return 0.0;
    }
    (_f(2, 1, omega, v) + _f(2, 0, omega, v) * f12_20 / g) / 3.0
}

fn _i_23(omega: f64, v: &[f64; 4]) -> f64 {
    let f13_21 = _f(1, 3, omega, v) * _f(2, 1, omega, v);
    let g = _f(1, 2, omega, v) * _f(2, 0, omega, v) + f13_21;
    if g < THM_EPSILON {
        return 0.0;
    }
    (_f(3, 0, omega, v) + _f(3, 1, omega, v) * f13_21 / g) / 3.0
}

fn _i_30(omega: f64, v: &[f64; 4]) -> f64 {
    _f(0, 3, omega, v) / 3.0
}

fn _i_31(omega: f64, v: &[f64; 4]) -> f64 {
    _f(1, 3, omega, v) / 3.0
}

fn _i_32(omega: f64, v: &[f64; 4]) -> f64 {
    _f(2, 3, omega, v) / 3.0
}

fn _i_33(omega: f64, v: &[f64; 4]) -> f64 {
    (_f(3, 0, omega, v) + _f(3, 1, omega, v) + _f(3, 2, omega, v)) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_rec_lattice(a: f64) -> MatD {
        [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    }

    #[test]
    fn main_diagonal_is_zero_for_cubic() {
        // All four diagonals have equal length in a cubic cell;
        // ties resolve to the first (index 0).
        assert_eq!(get_main_diagonal(cubic_rec_lattice(1.0)), 0);
    }

    #[test]
    fn main_diagonal_picks_shortest() {
        // Stretch x so that diagonals containing -x or +x with
        // matching magnitude differ from the all-positive one.
        // Diagonal 1 = (-1, 1, 1) and diagonal 0 = (1, 1, 1)
        // give the same |.| in any orthorhombic cell, so use a
        // sheared lattice: rec[0] = (2, 1, 0) cancels out for
        // diag (-1, 1, 1) — minimal length.
        let rec = [[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // diag 0: (1,1,1) -> (2+1+0, 1, 1) = (3, 1, 1) len^2=11
        // diag 1: (-1,1,1) -> (-2+1+0, 1, 1) = (-1, 1, 1) len^2=3
        // diag 2: (1,-1,1) -> (2-1+0, -1, 1) = (1, -1, 1) len^2=3
        // diag 3: (1,1,-1) -> (2+1+0, 1, -1) = (3, 1, -1) len^2=11
        // first occurring minimum (diag 1) wins
        assert_eq!(get_main_diagonal(rec), 1);
    }

    #[test]
    fn relative_grid_address_returns_correct_table() {
        let rec = [[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (table, idx) = relative_grid_address(rec);
        assert_eq!(idx, 1);
        assert_eq!(table[0][0], [0, 0, 0]);
        assert_eq!(table[0][1], [1, 0, 0]);
        assert_eq!(table[0][3], [0, 1, 1]);
    }

    #[test]
    fn sort_omegas_already_sorted() {
        let mut v = [1.0, 2.0, 3.0, 4.0];
        let ci = sort_omegas(&mut v);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ci, 0);
    }

    #[test]
    fn sort_omegas_reverse() {
        let mut v = [4.0, 3.0, 2.0, 1.0];
        let _ci = sort_omegas(&mut v);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn integration_weight_below_lowest_is_zero() {
        // omega well below all vertex omegas: every tetrahedron
        // contributes via the case-0 branch (gn=0).  Sum = 0.
        let mut to = [[0.0; 4]; 24];
        for i in 0..24 {
            to[i] = [1.0, 2.0, 3.0, 4.0];
        }
        let w = integration_weight(0.0, &to, WeightFunction::J);
        assert!(w.abs() < 1e-12);
        let w = integration_weight(0.0, &to, WeightFunction::I);
        assert!(w.abs() < 1e-12);
    }

    #[test]
    fn integration_weight_above_highest_j_is_one_quarter() {
        // omega above all vertices: case 4.  J branch: gn=1.0,
        // ij=0.25 per tetrahedron → sum = 24 * 0.25, divided by 6
        // gives 1.0.
        let mut to = [[0.0; 4]; 24];
        for i in 0..24 {
            to[i] = [1.0, 2.0, 3.0, 4.0];
        }
        let w = integration_weight(5.0, &to, WeightFunction::J);
        assert!((w - 1.0).abs() < 1e-12);
        // I branch: gn = 0.0 above highest → weight = 0.
        let w = integration_weight(5.0, &to, WeightFunction::I);
        assert!(w.abs() < 1e-12);
    }
}
