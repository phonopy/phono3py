//! Triplet-level wrappers.
//!
//! Port of `c/triplet.c`: small helpers (`tpl_is_N`,
//! `tpl_set_relative_grid_address`) plus the multi-triplet
//! integration-weight drivers (`tpl_get_integration_weight`,
//! `tpl_get_integration_weight_with_sigma`) that loop over triplets
//! and dispatch to the per-triplet kernels in `triplet_iw`.

#![allow(dead_code)]

use rayon::prelude::*;

use crate::common::Vec3I;
use crate::triplet_iw::{
    integration_weight_per_triplet, integration_weight_per_triplet_inner_par,
    integration_weight_with_sigma_per_triplet, integration_weight_with_sigma_per_triplet_inner_par,
    BzGridError, BzGridView, TpType,
};

/// Tetrahedron-method relative addresses: 24 tetrahedra,
/// 4 vertices each, 3 spatial components.
pub type RelativeGridAddress = [[Vec3I; 4]; 24];

/// Test whether a triplet is a normal (N) process: the sum of the
/// three BZ grid addresses is zero in every component.
///
/// Mirrors `tpl_is_N` in `c/triplet.c`.
pub fn is_n(triplet: [i64; 3], bz_grid_addresses: &[Vec3I]) -> bool {
    for i in 0..3 {
        let mut sum_q: i64 = 0;
        for j in 0..3 {
            sum_q += bz_grid_addresses[triplet[j] as usize][i];
        }
        if sum_q != 0 {
            return false;
        }
    }
    true
}

/// Build the per-channel relative grid addresses for a triplet
/// integration-weight calculation.
///
/// Mirrors `tpl_set_relative_grid_address` in `c/triplet.c`:
///   - channel 0 always uses the input addresses unchanged.
///   - channel 1 uses negated addresses when `tp_type` is 2 or 3
///     (the q1+q2+q3=G branch where q3 must shift opposite to q2);
///     otherwise it uses the unchanged addresses (tp_type == 4 case).
pub fn set_relative_grid_address(
    relative_grid_address: &RelativeGridAddress,
    tp_type: i64,
) -> [RelativeGridAddress; 2] {
    let sign1: i64 = if tp_type == 2 || tp_type == 3 { -1 } else { 1 };

    let mut out = [[[[0i64; 3]; 4]; 24]; 2];
    for j in 0..24 {
        for k in 0..4 {
            for l in 0..3 {
                let v = relative_grid_address[j][k][l];
                out[0][j][k][l] = v;
                out[1][j][k][l] = v * sign1;
            }
        }
    }
    out
}

/// Multi-triplet tetrahedron-method integration weights.
///
/// Mirrors `tpl_get_integration_weight`.  Builds the per-channel
/// relative grid addresses once and dispatches one kernel call per
/// triplet.
///
/// `iw` flat layout: `[channel][triplet][freq][b1][b2]` C-order,
/// length = `tp_type.num_channels() * num_triplets * num_band0 *
/// num_band1 * num_band2`.
/// `iw_zero` flat layout: `[triplet][freq][b1][b2]` C-order,
/// length = `num_triplets * num_band0 * num_band1 * num_band2`.
pub fn integration_weight(
    iw: &mut [f64],
    iw_zero: &mut [i8],
    frequency_points: &[f64],
    relative_grid_address: &RelativeGridAddress,
    triplets: &[[i64; 3]],
    bzgrid: &BzGridView,
    frequencies1: &[f64],
    num_band1: i64,
    frequencies2: &[f64],
    num_band2: i64,
    tp_type: TpType,
) -> Result<(), BzGridError> {
    let num_band0 = frequency_points.len() as i64;
    let num_triplets = triplets.len();
    let nb_per_triplet = (num_band0 * num_band1 * num_band2) as usize;

    let tp_relative = set_relative_grid_address(relative_grid_address, tp_type_as_i64(tp_type));
    let per_triplet_iw =
        split_iw_per_triplet(iw, tp_type.num_channels(), num_triplets, nb_per_triplet);
    let iwz_chunks: Vec<&mut [i8]> = iw_zero.chunks_mut(nb_per_triplet).collect();

    if outer_parallel(num_triplets) {
        per_triplet_iw
            .into_par_iter()
            .zip(iwz_chunks.into_par_iter())
            .zip(triplets.par_iter())
            .try_for_each(|((mut iw_blocks, iwz), triplet)| {
                integration_weight_per_triplet(
                    &mut iw_blocks,
                    iwz,
                    frequency_points,
                    num_band0,
                    &tp_relative,
                    *triplet,
                    bzgrid,
                    frequencies1,
                    num_band1,
                    frequencies2,
                    num_band2,
                    tp_type,
                )
            })
    } else {
        for ((mut iw_blocks, iwz), triplet) in per_triplet_iw
            .into_iter()
            .zip(iwz_chunks.into_iter())
            .zip(triplets.iter())
        {
            integration_weight_per_triplet_inner_par(
                &mut iw_blocks,
                iwz,
                frequency_points,
                num_band0,
                &tp_relative,
                *triplet,
                bzgrid,
                frequencies1,
                num_band1,
                frequencies2,
                num_band2,
                tp_type,
            )?;
        }
        Ok(())
    }
}

/// Multi-triplet Gaussian-smeared integration weights.
///
/// Mirrors `tpl_get_integration_weight_with_sigma`.  The C-side
/// `cutoff = sigma * sigma_cutoff` is computed here, so callers
/// pass the unitless `sigma_cutoff` (set negative to disable).
pub fn integration_weight_with_sigma(
    iw: &mut [f64],
    iw_zero: &mut [i8],
    sigma: f64,
    sigma_cutoff: f64,
    frequency_points: &[f64],
    triplets: &[[i64; 3]],
    frequencies: &[f64],
    num_band: i64,
    tp_type: TpType,
) {
    let num_band0 = frequency_points.len() as i64;
    let num_triplets = triplets.len();
    let nb_per_triplet = (num_band0 * num_band * num_band) as usize;
    let cutoff = sigma * sigma_cutoff;

    let per_triplet_iw =
        split_iw_per_triplet(iw, tp_type.num_channels(), num_triplets, nb_per_triplet);
    let iwz_chunks: Vec<&mut [i8]> = iw_zero.chunks_mut(nb_per_triplet).collect();

    if outer_parallel(num_triplets) {
        per_triplet_iw
            .into_par_iter()
            .zip(iwz_chunks.into_par_iter())
            .zip(triplets.par_iter())
            .for_each(|((mut iw_blocks, iwz), triplet)| {
                integration_weight_with_sigma_per_triplet(
                    &mut iw_blocks,
                    iwz,
                    sigma,
                    cutoff,
                    frequency_points,
                    num_band0,
                    *triplet,
                    frequencies,
                    num_band,
                    tp_type,
                );
            });
    } else {
        for ((mut iw_blocks, iwz), triplet) in per_triplet_iw
            .into_iter()
            .zip(iwz_chunks.into_iter())
            .zip(triplets.iter())
        {
            integration_weight_with_sigma_per_triplet_inner_par(
                &mut iw_blocks,
                iwz,
                sigma,
                cutoff,
                frequency_points,
                num_band0,
                *triplet,
                frequencies,
                num_band,
                tp_type,
            );
        }
    }
}

/// Choose outer (per-triplet) parallelism when there are at least as
/// many triplets as worker threads; otherwise the caller falls back to
/// inner (per-`b12`) parallelism inside each triplet kernel.
fn outer_parallel(num_triplets: usize) -> bool {
    num_triplets >= rayon::current_num_threads()
}

/// Split a channel-major flat output buffer into per-triplet, per-channel
/// disjoint mutable slices.  Returns a `Vec` of length `num_triplets`
/// where each entry is a `Vec<&mut [f64]>` of length `num_channels`.
///
/// Layout assumed: `iw[c, t, ...]` flat, with channel-stride
/// `num_triplets * nb_per_triplet`.  Each leaf slice covers
/// `nb_per_triplet` elements.
fn split_iw_per_triplet(
    iw: &mut [f64],
    num_channels: usize,
    num_triplets: usize,
    nb_per_triplet: usize,
) -> Vec<Vec<&mut [f64]>> {
    let channel_stride = num_triplets * nb_per_triplet;
    let mut channel_chunks: Vec<Vec<&mut [f64]>> = Vec::with_capacity(num_channels);
    let mut rest: &mut [f64] = iw;
    for _ in 0..num_channels {
        let (head, tail) = rest.split_at_mut(channel_stride);
        let chunks: Vec<&mut [f64]> = head.chunks_mut(nb_per_triplet).collect();
        debug_assert_eq!(chunks.len(), num_triplets);
        channel_chunks.push(chunks);
        rest = tail;
    }

    let mut per_triplet: Vec<Vec<&mut [f64]>> = (0..num_triplets)
        .map(|_| Vec::with_capacity(num_channels))
        .collect();
    for ch in channel_chunks {
        for (i, chunk) in ch.into_iter().enumerate() {
            per_triplet[i].push(chunk);
        }
    }
    per_triplet
}

fn tp_type_as_i64(tp_type: TpType) -> i64 {
    match tp_type {
        TpType::Type2 => 2,
        TpType::Type3 => 3,
        TpType::Type4 => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_n_true_when_addresses_sum_to_zero() {
        let bz = vec![[1, -2, 3], [-4, 5, -6], [3, -3, 3]];
        assert!(is_n([0, 1, 2], &bz));
    }

    #[test]
    fn is_n_false_when_any_component_nonzero() {
        let bz = vec![[1, 0, 0], [0, 0, 0], [0, 0, 0]];
        assert!(!is_n([0, 1, 2], &bz));
    }

    #[test]
    fn set_relative_grid_address_tp2_negates_channel1() {
        let mut input = [[[0i64; 3]; 4]; 24];
        input[0][0] = [1, 2, 3];
        input[5][2] = [-7, 8, -9];
        let out = set_relative_grid_address(&input, 2);
        assert_eq!(out[0][0][0], [1, 2, 3]);
        assert_eq!(out[1][0][0], [-1, -2, -3]);
        assert_eq!(out[0][5][2], [-7, 8, -9]);
        assert_eq!(out[1][5][2], [7, -8, 9]);
    }

    #[test]
    fn set_relative_grid_address_tp4_keeps_channel1() {
        let mut input = [[[0i64; 3]; 4]; 24];
        input[1][1] = [4, -5, 6];
        let out = set_relative_grid_address(&input, 4);
        assert_eq!(out[0][1][1], [4, -5, 6]);
        assert_eq!(out[1][1][1], [4, -5, 6]);
    }
}
