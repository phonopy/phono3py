//! Reciprocal-space point group construction.
//!
//! Port of `grg_get_reciprocal_point_group` in `c/grgrid.c`.

use crate::common::{negate_i, transpose_i, MatI, INVERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReciprocalRotationsError {
    /// More than 48 unique input rotations collected.
    TooManyRotations,
    /// Time-reversal requested but existing count exceeds 24, so
    /// doubling with inversion would overflow the 48-slot limit.
    TooManyForInversion,
}

/// Core logic: deduplicate rotations, optionally transpose, optionally
/// add time-reversal (inversion).
///
/// - `is_transpose`: when `true`, each unique rotation is transposed
///   before further processing.  The `_recgrid.cpp` path passes
///   real-space rotations and needs transposition; the triplet path
///   passes reciprocal-space rotations directly and does not.
/// - `is_time_reversal`: when `true`, negated copies are appended if
///   the inversion matrix is not already present.
pub fn get_reciprocal_point_group(
    rotations: &[MatI],
    is_time_reversal: bool,
    is_transpose: bool,
) -> Result<Vec<MatI>, ReciprocalRotationsError> {
    let mut unique: Vec<MatI> = Vec::with_capacity(48);
    for r in rotations {
        if unique.iter().any(|u| u == r) {
            continue;
        }
        if unique.len() == 48 {
            return Err(ReciprocalRotationsError::TooManyRotations);
        }
        unique.push(*r);
    }

    if is_transpose {
        for r in unique.iter_mut() {
            *r = transpose_i(r);
        }
    }

    if is_time_reversal {
        let has_inversion = unique.iter().any(|u| u == &INVERSION);
        if !has_inversion {
            if unique.len() > 24 {
                return Err(ReciprocalRotationsError::TooManyForInversion);
            }
            let n = unique.len();
            for i in 0..n {
                unique.push(negate_i(&unique[i]));
            }
        }
    }

    Ok(unique)
}

/// Build the reciprocal-space point group from a set of real-space
/// rotations (with transposition).
///
/// Convenience wrapper used by the `_recgrid.cpp` Python path.
pub fn reciprocal_rotations(
    rotations: &[MatI],
    is_time_reversal: bool,
) -> Result<Vec<MatI>, ReciprocalRotationsError> {
    get_reciprocal_point_group(rotations, is_time_reversal, true)
}
