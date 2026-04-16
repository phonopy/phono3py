//! Reciprocal-space point group construction.
//!
//! Port of `grg_get_reciprocal_point_group` in `c/grgrid.c`.

type Mat = [[i64; 3]; 3];

const INVERSION: Mat = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReciprocalRotationsError {
    /// More than 48 unique input rotations collected.
    TooManyRotations,
    /// Time-reversal requested but existing count exceeds 24, so
    /// doubling with inversion would overflow the 48-slot limit.
    TooManyForInversion,
}

fn transpose(m: &Mat) -> Mat {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn negate(m: &Mat) -> Mat {
    [
        [-m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], -m[2][2]],
    ]
}

/// Build the reciprocal-space point group from a set of real-space
/// rotations.  The Python-facing path in `c/_recgrid.cpp` always
/// passes `is_transpose = 1`, so transposition is hard-coded here.
pub fn reciprocal_rotations(
    rotations: &[Mat],
    is_time_reversal: bool,
) -> Result<Vec<Mat>, ReciprocalRotationsError> {
    let mut unique: Vec<Mat> = Vec::with_capacity(48);
    for r in rotations {
        if unique.iter().any(|u| u == r) {
            continue;
        }
        if unique.len() == 48 {
            return Err(ReciprocalRotationsError::TooManyRotations);
        }
        unique.push(*r);
    }

    for r in unique.iter_mut() {
        *r = transpose(r);
    }

    if is_time_reversal {
        let has_inversion = unique.iter().any(|u| u == &INVERSION);
        if !has_inversion {
            if unique.len() > 24 {
                return Err(ReciprocalRotationsError::TooManyForInversion);
            }
            let n = unique.len();
            for i in 0..n {
                unique.push(negate(&unique[i]));
            }
        }
    }

    Ok(unique)
}
