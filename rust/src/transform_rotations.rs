//! Transform reciprocal-space rotations into GR-grid rotations.
//!
//! Port of `grg_transform_rotations` in `c/grgrid.c`.  Computes
//! `D * Q^-1 * R * Q * D^-1` for each input rotation `R`, where
//! `D = diag(d_diag)`, and verifies the result is integer-valued.

type MatI = [[i64; 3]; 3];
type MatD = [[f64; 3]; 3];

const IDENTITY_TOL: f64 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformRotationsError {
    /// `Q` is singular (should never happen in practice).
    SingularQ,
    /// A transformed rotation is not integer-valued within tolerance,
    /// meaning the grid breaks the crystal symmetry.
    SymmetryBroken,
}

fn to_f64(m: &MatI) -> MatD {
    [
        [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64],
        [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64],
        [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64],
    ]
}

fn det_f64(m: &MatD) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse_f64(m: &MatD) -> Option<MatD> {
    let d = det_f64(m);
    if d == 0.0 {
        return None;
    }
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    c[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
    c[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    c[0][1] = (m[2][1] * m[0][2] - m[2][2] * m[0][1]) / d;
    c[1][1] = (m[2][2] * m[0][0] - m[2][0] * m[0][2]) / d;
    c[2][1] = (m[2][0] * m[0][1] - m[2][1] * m[0][0]) / d;
    c[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    c[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    c[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    Some(c)
}

fn matmul_f64(a: &MatD, b: &MatD) -> MatD {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn nint(x: f64) -> i64 {
    if x < 0.0 {
        (x - 0.5) as i64
    } else {
        (x + 0.5) as i64
    }
}

/// Transform each rotation by `D * Q^-1 * R * Q * D^-1` and return
/// the resulting integer matrices.  Returns `SymmetryBroken` if any
/// entry of the transformed matrix is not an integer within
/// `IDENTITY_TOL`.
pub fn transform_rotations(
    rotations: &[MatI],
    d_diag: [i64; 3],
    q: MatI,
) -> Result<Vec<MatI>, TransformRotationsError> {
    debug_assert!(d_diag.iter().all(|&d| d >= 1));

    let q_d = to_f64(&q);
    let q_inv = inverse_f64(&q_d).ok_or(TransformRotationsError::SingularQ)?;

    let mut out = Vec::with_capacity(rotations.len());
    for r in rotations {
        let r_d = to_f64(r);
        let qr = matmul_f64(&q_inv, &r_d);
        let mut m = matmul_f64(&qr, &q_d);
        for j in 0..3 {
            for k in 0..3 {
                m[j][k] = m[j][k] * (d_diag[j] as f64) / (d_diag[k] as f64);
            }
        }
        let rounded: MatI = [
            [nint(m[0][0]), nint(m[0][1]), nint(m[0][2])],
            [nint(m[1][0]), nint(m[1][1]), nint(m[1][2])],
            [nint(m[2][0]), nint(m[2][1]), nint(m[2][2])],
        ];
        for j in 0..3 {
            for k in 0..3 {
                if ((rounded[j][k] as f64) - m[j][k]).abs() > IDENTITY_TOL {
                    return Err(TransformRotationsError::SymmetryBroken);
                }
            }
        }
        out.push(rounded);
    }
    Ok(out)
}
