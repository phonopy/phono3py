//! Transform reciprocal-space rotations into GR-grid rotations.
//!
//! Port of `grg_transform_rotations` in `c/grgrid.c`.  Computes
//! `D * Q^-1 * R * Q * D^-1` for each input rotation `R`, where
//! `D = diag(d_diag)`, and verifies the result is integer-valued.

use crate::common::{det_d, mat_i_to_d, matmul_d, nint, MatD, MatI};

const IDENTITY_TOL: f64 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformRotationsError {
    /// `Q` is singular (should never happen in practice).
    SingularQ,
    /// A transformed rotation is not integer-valued within tolerance,
    /// meaning the grid breaks the crystal symmetry.
    SymmetryBroken,
}

fn inverse_f64(m: &MatD) -> Option<MatD> {
    let d = det_d(m);
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

    let q_d = mat_i_to_d(&q);
    let q_inv = inverse_f64(&q_d).ok_or(TransformRotationsError::SingularQ)?;

    let mut out = Vec::with_capacity(rotations.len());
    for r in rotations {
        let r_d = mat_i_to_d(r);
        let qr = matmul_d(&q_inv, &r_d);
        let mut m = matmul_d(&qr, &q_d);
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
