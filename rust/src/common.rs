//! Shared types, constants, and small 3x3 linear-algebra helpers.
//!
//! Operates on fixed-size 3x3 matrices and 3-vectors only.  Per the
//! crate convention, integer (i64) and double (f64) flavors are
//! provided as separate functions rather than via generics.

pub type Vec3I = [i64; 3];
pub type Vec3D = [f64; 3];
pub type MatI = [[i64; 3]; 3];
pub type MatD = [[f64; 3]; 3];

/// Complex value as `[real, imag]`, mirroring the `double[2]`
/// layout used throughout the C code (e.g. `c/dynmat.c`).
pub type Cmplx = [f64; 2];

/// Complex multiplication.  Mirrors `phonoc_complex_prod` in
/// `c/lapack_wrapper.c`.
#[inline(always)]
pub fn cmplx_mul(a: Cmplx, b: Cmplx) -> Cmplx {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

pub const IDENTITY: MatI = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
pub const INVERSION: MatI = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]];

/// Round-half-away-from-zero, matching the C `(int)(x + 0.5)` idiom
/// extended to negative inputs.
pub fn nint(x: f64) -> i64 {
    if x < 0.0 {
        (x - 0.5) as i64
    } else {
        (x + 0.5) as i64
    }
}

pub fn transpose_i(m: &MatI) -> MatI {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub fn negate_i(m: &MatI) -> MatI {
    [
        [-m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], -m[2][2]],
    ]
}

pub fn det_i(m: &MatI) -> i64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub fn det_d(m: &MatD) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Returns `a * b`.  Aliasing-safe (returned by value).
pub fn matmul_i(a: &MatI, b: &MatI) -> MatI {
    let mut c = [[0i64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

pub fn matmul_d(a: &MatD, b: &MatD) -> MatD {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

pub fn matvec_ii(a: &MatI, b: Vec3I) -> Vec3I {
    [
        a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2],
        a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2],
        a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2],
    ]
}

pub fn matvec_dd(a: &MatD, b: Vec3D) -> Vec3D {
    [
        a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2],
        a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2],
        a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2],
    ]
}

/// Integer matrix times float vector.
pub fn matvec_id(a: &MatI, b: Vec3D) -> Vec3D {
    [
        a[0][0] as f64 * b[0] + a[0][1] as f64 * b[1] + a[0][2] as f64 * b[2],
        a[1][0] as f64 * b[0] + a[1][1] as f64 * b[1] + a[1][2] as f64 * b[2],
        a[2][0] as f64 * b[0] + a[2][1] as f64 * b[1] + a[2][2] as f64 * b[2],
    ]
}

/// Float matrix times integer vector.
pub fn matvec_di(a: &MatD, b: Vec3I) -> Vec3D {
    [
        a[0][0] * b[0] as f64 + a[0][1] * b[1] as f64 + a[0][2] * b[2] as f64,
        a[1][0] * b[0] as f64 + a[1][1] * b[1] as f64 + a[1][2] * b[2] as f64,
        a[2][0] * b[0] as f64 + a[2][1] * b[1] as f64 + a[2][2] * b[2] as f64,
    ]
}

/// Cast each element of an integer matrix to f64.
pub fn mat_i_to_d(m: &MatI) -> MatD {
    [
        [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64],
        [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64],
        [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64],
    ]
}
