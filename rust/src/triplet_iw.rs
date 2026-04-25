//! Triplet integration-weight kernels.
//!
//! Port of `c/triplet_iw.c`.  These are the per-triplet workers that
//! compute the tetrahedron-method or Gaussian-smeared integration
//! weights consumed by `tpl_get_integration_weight*` (in
//! `c/triplet.c`).

#![allow(dead_code)]

use rayon::prelude::*;

use crate::common::Vec3I;
use crate::grgrid::grid_index_from_address;
use crate::tetrahedron_method::{integration_weight, WeightFunction};

/// `*mut T` wrapper opting into Send + Sync for rayon.  Used only inside
/// parallel kernels where the Rust author has manually verified that
/// the offsets touched by each task are disjoint.
#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    /// Method-style accessor for the raw pointer.  Calling this method
    /// (rather than touching the `.0` field) is what forces the 2021
    /// edition's disjoint-capture analysis to capture the whole
    /// wrapper into a closure, preserving its Send + Sync impls.
    fn ptr(self) -> *mut T {
        self.0
    }
}

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Per-channel relative grid addresses: 2 sign channels (q2, q3),
/// 24 tetrahedra, 4 vertices, 3 spatial components.  Built by
/// `triplet::set_relative_grid_address`.
pub type TpRelativeGridAddress = [[[Vec3I; 4]; 24]; 2];

/// Triplet integration-weight type.
///
/// * `Type2`: q1+q2+q3=G, ph-ph lifetime — outputs `(g[2], g[0]-g[1])`.
/// * `Type3`: q1+q2+q3=G, collision matrix — outputs
///   `(g[2], g[0]-g[1], g[0]+g[1]+g[2])`.
/// * `Type4`: q+k_i-k_f=G, el-ph phonon decay — outputs `(g[0])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpType {
    Type2,
    Type3,
    Type4,
}

impl TpType {
    /// Map the C-side integer encoding (`tp_type`) onto the enum.
    pub fn try_from_i64(tp_type: i64) -> Result<Self, BzGridError> {
        match tp_type {
            2 => Ok(TpType::Type2),
            3 => Ok(TpType::Type3),
            4 => Ok(TpType::Type4),
            _ => Err(BzGridError::BadTpType),
        }
    }

    /// Number of g channels written into `iw` (== `g.shape[0]` on the
    /// Python side for Type2/Type3, 1 for Type4).
    pub fn num_channels(self) -> usize {
        match self {
            TpType::Type2 => 2,
            TpType::Type3 => 3,
            TpType::Type4 => 1,
        }
    }
}

/// Borrowed view onto a BZ grid: matches the fields read by C's
/// `RecgridConstBZGrid` from `tpi_*` callsites.
pub struct BzGridView<'a> {
    pub d_diag: Vec3I,
    pub addresses: &'a [Vec3I],
    pub gp_map: &'a [i64],
    /// 1 = sparse (gp_map indexes 2x mesh), 2 = dense (gp_map[g]..gp_map[g+1]).
    pub bz_grid_type: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BzGridError {
    /// `bz_grid_type` not 1 or 2.
    BadGridType,
    /// `tp_type` not 2, 3, or 4.
    BadTpType,
}

/// Public: tetrahedron-method integration weights for a single triplet.
///
/// Mirrors `tpi_get_integration_weight`.  `iw_ch` is a slice of
/// per-channel output blocks for this triplet; its length must equal
/// `tp_type.num_channels()` and each inner slice covers
/// `num_band0 * num_band1 * num_band2` elements.  `iw_zero` is the
/// (single-channel) flag block of the same per-block size.  Disjoint
/// slices across triplets make this safe to call from rayon.
pub fn integration_weight_per_triplet(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    frequency_points: &[f64],
    num_band0: i64,
    tp_relative_grid_address: &TpRelativeGridAddress,
    triplet: [i64; 3],
    bzgrid: &BzGridView,
    frequencies1: &[f64],
    num_band1: i64,
    frequencies2: &[f64],
    num_band2: i64,
    tp_type: TpType,
) -> Result<(), BzGridError> {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band1 * num_band2) as usize;
    let nb0 = num_band0 as usize;

    let vertices = triplet_tetrahedra_vertices(tp_relative_grid_address, triplet, bzgrid)?;

    let max_i = max_tetra_channels(tp_type);
    for b12 in 0..nbb {
        let b1 = (b12 as i64) / num_band2;
        let b2 = (b12 as i64) % num_band2;
        let freq_vertices = build_freq_vertices(
            &vertices,
            frequencies1,
            frequencies2,
            num_band1,
            num_band2,
            b1,
            b2,
            tp_type,
        );
        let bboxes = freq_vertices_bboxes(&freq_vertices, max_i);
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_tetra_channels(f0, &freq_vertices, &bboxes, tp_type);
            iw_zero[adrs] = iwz;
            for (k, iw_ch_k) in iw_ch.iter_mut().enumerate() {
                iw_ch_k[adrs] = ch[k];
            }
        }
    }
    Ok(())
}

/// Public: tetrahedron-method integration weights for a single triplet,
/// with the inner `b12` loop parallelised over rayon.  Use this variant
/// when the outer triplet loop is too small to feed all threads.  The
/// per-channel slice contract is identical to the serial version.
pub fn integration_weight_per_triplet_inner_par(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    frequency_points: &[f64],
    num_band0: i64,
    tp_relative_grid_address: &TpRelativeGridAddress,
    triplet: [i64; 3],
    bzgrid: &BzGridView,
    frequencies1: &[f64],
    num_band1: i64,
    frequencies2: &[f64],
    num_band2: i64,
    tp_type: TpType,
) -> Result<(), BzGridError> {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band1 * num_band2) as usize;
    let nb0 = num_band0 as usize;

    let vertices = triplet_tetrahedra_vertices(tp_relative_grid_address, triplet, bzgrid)?;

    let ch_ptrs: Vec<SyncMutPtr<f64>> = iw_ch
        .iter_mut()
        .map(|s| SyncMutPtr(s.as_mut_ptr()))
        .collect();
    let iwz_ptr = SyncMutPtr(iw_zero.as_mut_ptr());

    let max_i = max_tetra_channels(tp_type);
    // SAFETY: for each b12 in 0..nbb the inner j loop writes to offsets
    // { j * nbb + b12 : j in 0..nb0 } in every per-channel slice and in
    // iw_zero.  These index sets are pairwise disjoint across different
    // b12 values, so concurrent writes from different rayon tasks do not
    // race.  Slice lengths are nb0 * nbb so all offsets are in-bounds.
    (0..nbb).into_par_iter().for_each(|b12| {
        let b1 = (b12 as i64) / num_band2;
        let b2 = (b12 as i64) % num_band2;
        let freq_vertices = build_freq_vertices(
            &vertices,
            frequencies1,
            frequencies2,
            num_band1,
            num_band2,
            b1,
            b2,
            tp_type,
        );
        let bboxes = freq_vertices_bboxes(&freq_vertices, max_i);
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_tetra_channels(f0, &freq_vertices, &bboxes, tp_type);
            unsafe {
                *iwz_ptr.ptr().add(adrs) = iwz;
                for (k, ch_ptr) in ch_ptrs.iter().enumerate() {
                    *ch_ptr.ptr().add(adrs) = ch[k];
                }
            }
        }
    });

    Ok(())
}

/// Public: Gaussian-smeared integration weights for a single triplet.
///
/// Mirrors `tpi_get_integration_weight_with_sigma`.  `iw_ch` is a slice
/// of per-channel output blocks (length = `tp_type.num_channels()`,
/// each `num_band0 * num_band * num_band` long).  `cutoff <= 0`
/// disables the cutoff-skip optimisation (matches C semantics).
pub fn integration_weight_with_sigma_per_triplet(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    sigma: f64,
    cutoff: f64,
    frequency_points: &[f64],
    num_band0: i64,
    triplet: [i64; 3],
    frequencies: &[f64],
    num_band: i64,
    tp_type: TpType,
) {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band * num_band) as usize;
    let nb0 = num_band0 as usize;

    for b12 in 0..nbb {
        let b1 = (b12 as i64) / num_band;
        let b2 = (b12 as i64) % num_band;
        let f1 = frequencies[(triplet[1] * num_band + b1) as usize];
        let f2 = frequencies[(triplet[2] * num_band + b2) as usize];
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_sigma_channels(f0, f1, f2, sigma, cutoff, tp_type);
            iw_zero[adrs] = iwz;
            for (k, iw_ch_k) in iw_ch.iter_mut().enumerate() {
                iw_ch_k[adrs] = ch[k];
            }
        }
    }
}

/// Public: Gaussian-smeared integration weights for a single triplet,
/// with the inner `b12` loop parallelised over rayon.  See the
/// non-`_inner_par` variant for the slice contract.
pub fn integration_weight_with_sigma_per_triplet_inner_par(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    sigma: f64,
    cutoff: f64,
    frequency_points: &[f64],
    num_band0: i64,
    triplet: [i64; 3],
    frequencies: &[f64],
    num_band: i64,
    tp_type: TpType,
) {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band * num_band) as usize;
    let nb0 = num_band0 as usize;

    let ch_ptrs: Vec<SyncMutPtr<f64>> = iw_ch
        .iter_mut()
        .map(|s| SyncMutPtr(s.as_mut_ptr()))
        .collect();
    let iwz_ptr = SyncMutPtr(iw_zero.as_mut_ptr());

    // SAFETY: see the tetrahedron inner_par variant above; the same
    // disjointedness argument holds (offsets j * nbb + b12 are pairwise
    // disjoint across b12).
    (0..nbb).into_par_iter().for_each(|b12| {
        let b1 = (b12 as i64) / num_band;
        let b2 = (b12 as i64) % num_band;
        let f1 = frequencies[(triplet[1] * num_band + b1) as usize];
        let f2 = frequencies[(triplet[2] * num_band + b2) as usize];
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_sigma_channels(f0, f1, f2, sigma, cutoff, tp_type);
            unsafe {
                *iwz_ptr.ptr().add(adrs) = iwz;
                for (k, ch_ptr) in ch_ptrs.iter().enumerate() {
                    *ch_ptr.ptr().add(adrs) = ch[k];
                }
            }
        }
    });
}

/// Public: BZ-grid neighbours of `grid_point` along a list of relative
/// grid addresses.  Mirrors `tpi_get_neighboring_grid_points`.
///
/// Returns one BZ grid index per relative address, in input order.
pub fn neighboring_grid_points(
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<Vec<i64>, BzGridError> {
    let mut out = vec![0i64; relative_grid_address.len()];
    fill_neighboring_grid_points(&mut out, grid_point, relative_grid_address, bzgrid)?;
    Ok(out)
}

/// Public: parallel batch of `neighboring_grid_points` over many
/// `grid_points`.  Mirrors `ph3py_get_neighboring_gird_points`.  `out`
/// is `num_grid_points * relative_grid_address.len()` long in row-major
/// order: chunk `i` holds the neighbours of `grid_points[i]`.
///
/// Caller must ensure `out.len() == grid_points.len() * relative_grid_address.len()`.
pub fn neighboring_grid_points_many(
    out: &mut [i64],
    grid_points: &[i64],
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    let num_rga = relative_grid_address.len();
    out.par_chunks_mut(num_rga)
        .zip(grid_points.par_iter())
        .try_for_each(|(chunk, &gp)| {
            fill_neighboring_grid_points(chunk, gp, relative_grid_address, bzgrid)
        })
}

/// Public: tetrahedron-method integration weights for many grid points.
/// Mirrors `ph3py_get_thm_integration_weights_at_grid_points`.
///
/// `iw` is `(num_gp, num_fp, num_band)` in C-contiguous layout.
/// `relative_grid_address` is the 24-tetrahedra vertex offset table.
/// `frequencies` is `(num_ir, num_band)` flat; `gp2irgp_map` maps each
/// BZ-grid index to its row in `frequencies`.  Parallelised over grid
/// points (output chunks are disjoint).
pub fn integration_weights_at_grid_points(
    iw: &mut [f64],
    frequency_points: &[f64],
    relative_grid_address: &[[Vec3I; 4]; 24],
    grid_points: &[i64],
    frequencies: &[f64],
    num_band: usize,
    bzgrid: &BzGridView,
    gp2irgp_map: &[i64],
    function: WeightFunction,
) -> Result<(), BzGridError> {
    let num_fp = frequency_points.len();
    let chunk_size = num_fp * num_band;

    iw.par_chunks_mut(chunk_size)
        .zip(grid_points.par_iter())
        .try_for_each(|(iw_chunk, &gp)| -> Result<(), BzGridError> {
            let mut vertices = [[0i64; 4]; 24];
            for (j, tet) in relative_grid_address.iter().enumerate() {
                fill_neighboring_grid_points(&mut vertices[j], gp, tet, bzgrid)?;
            }
            let mut freq_vertices = [[0.0f64; 4]; 24];
            for bi in 0..num_band {
                for j in 0..24 {
                    for k in 0..4 {
                        let ir = gp2irgp_map[vertices[j][k] as usize] as usize;
                        freq_vertices[j][k] = frequencies[ir * num_band + bi];
                    }
                }
                for j in 0..num_fp {
                    iw_chunk[j * num_band + bi] =
                        integration_weight(frequency_points[j], &freq_vertices, function);
                }
            }
            Ok(())
        })
}

fn fill_neighboring_grid_points(
    out: &mut [i64],
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    match bzgrid.bz_grid_type {
        1 => fill_neighboring_grid_points_type1(out, grid_point, relative_grid_address, bzgrid),
        2 => fill_neighboring_grid_points_type2(out, grid_point, relative_grid_address, bzgrid),
        _ => Err(BzGridError::BadGridType),
    }
}

fn fill_neighboring_grid_points_type1(
    out: &mut [i64],
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    let bzmesh: Vec3I = [
        bzgrid.d_diag[0] * 2,
        bzgrid.d_diag[1] * 2,
        bzgrid.d_diag[2] * 2,
    ];
    let prod_bz_mesh = bzmesh[0] * bzmesh[1] * bzmesh[2];
    let base = bzgrid.addresses[grid_point as usize];

    for (i, rel) in relative_grid_address.iter().enumerate() {
        let bz_address: Vec3I = [base[0] + rel[0], base[1] + rel[1], base[2] + rel[2]];
        let bz_gp = bzgrid.gp_map[grid_index_from_address(bz_address, bzmesh) as usize];
        if bz_gp == prod_bz_mesh {
            out[i] = grid_index_from_address(bz_address, bzgrid.d_diag);
        } else {
            out[i] = bz_gp;
        }
    }
    Ok(())
}

fn fill_neighboring_grid_points_type2(
    out: &mut [i64],
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    let base = bzgrid.addresses[grid_point as usize];
    for (i, rel) in relative_grid_address.iter().enumerate() {
        let bz_address: Vec3I = [base[0] + rel[0], base[1] + rel[1], base[2] + rel[2]];
        let gp = grid_index_from_address(bz_address, bzgrid.d_diag);
        let lo = bzgrid.gp_map[gp as usize];
        let hi = bzgrid.gp_map[(gp + 1) as usize];
        out[i] = lo;
        if hi - lo > 1 {
            for j in lo..hi {
                let a = bzgrid.addresses[j as usize];
                if a == bz_address {
                    out[i] = j;
                    break;
                }
            }
        }
    }
    Ok(())
}

/// Build the `[2][24][4]` per-channel BZ-grid vertex indices for a
/// triplet.  Mirrors `get_triplet_tetrahedra_vertices`.
fn triplet_tetrahedra_vertices(
    tp_relative_grid_address: &TpRelativeGridAddress,
    triplet: [i64; 3],
    bzgrid: &BzGridView,
) -> Result<[[[i64; 4]; 24]; 2], BzGridError> {
    let mut vertices = [[[0i64; 4]; 24]; 2];
    for i in 0..2 {
        for j in 0..24 {
            let mut row = [0i64; 4];
            fill_neighboring_grid_points(
                &mut row,
                triplet[i + 1],
                &tp_relative_grid_address[i][j],
                bzgrid,
            )?;
            vertices[i][j] = row;
        }
    }
    Ok(vertices)
}

/// Build the `[3][24][4]` per-channel frequency vertices for a single
/// `(b1, b2)` band pair.  Mirrors `set_freq_vertices`.
///
/// For Type2/Type3 the three channels are `-f1+f2`, `f1-f2`,
/// `f1+f2` (negative input frequencies are clamped to 0).
/// For Type4 only channel 0 (`-f1+f2`) is populated.
fn build_freq_vertices(
    vertices: &[[[i64; 4]; 24]; 2],
    frequencies1: &[f64],
    frequencies2: &[f64],
    num_band1: i64,
    num_band2: i64,
    b1: i64,
    b2: i64,
    tp_type: TpType,
) -> [[[f64; 4]; 24]; 3] {
    let mut out = [[[0.0f64; 4]; 24]; 3];
    for i in 0..24 {
        for j in 0..4 {
            let mut f1 = frequencies1[(vertices[0][i][j] * num_band1 + b1) as usize];
            let mut f2 = frequencies2[(vertices[1][i][j] * num_band2 + b2) as usize];
            match tp_type {
                TpType::Type2 | TpType::Type3 => {
                    if f1 < 0.0 {
                        f1 = 0.0;
                    }
                    if f2 < 0.0 {
                        f2 = 0.0;
                    }
                    out[0][i][j] = -f1 + f2;
                    out[1][i][j] = f1 - f2;
                    out[2][i][j] = f1 + f2;
                }
                TpType::Type4 => {
                    out[0][i][j] = -f1 + f2;
                }
            }
        }
    }
    out
}

/// Number of tetrahedron channels to compute for a given `tp_type`.
fn max_tetra_channels(tp_type: TpType) -> usize {
    match tp_type {
        TpType::Type2 | TpType::Type3 => 3,
        TpType::Type4 => 1,
    }
}

/// Per-channel (fmin, fmax) bounding boxes across the 24 tetrahedra's
/// 4 vertices.  Only the first `max_i` entries are populated.  Hoisted
/// out of the `f0` loop so the per-`f0` in-tetrahedron test reduces
/// from a 96-entry min/max scan to a pair of comparisons.
fn freq_vertices_bboxes(freq_vertices: &[[[f64; 4]; 24]; 3], max_i: usize) -> [(f64, f64); 3] {
    let mut out = [(0.0f64, 0.0f64); 3];
    for i in 0..max_i {
        let mut fmin = freq_vertices[i][0][0];
        let mut fmax = freq_vertices[i][0][0];
        for j in 0..24 {
            for k in 0..4 {
                let v = freq_vertices[i][j][k];
                if fmin > v {
                    fmin = v;
                }
                if fmax < v {
                    fmax = v;
                }
            }
        }
        out[i] = (fmin, fmax);
    }
    out
}

/// Compute g[0..max_i] and the iw_zero flag for one (f0, freq_vertices).
/// Mirrors `set_g`.  `bboxes[i]` must hold the (fmin, fmax) of
/// `freq_vertices[i]` for every `i in 0..max_i`.  Returns `(g, iw_zero)`
/// where `iw_zero == 1` means every populated `g[i]` is exactly zero.
fn compute_g(
    f0: f64,
    freq_vertices: &[[[f64; 4]; 24]; 3],
    bboxes: &[(f64, f64); 3],
    max_i: usize,
) -> ([f64; 3], i8) {
    let mut g = [0.0f64; 3];
    let mut iw_zero: i8 = 1;
    for i in 0..max_i {
        let (fmin, fmax) = bboxes[i];
        if fmin <= f0 && f0 <= fmax {
            g[i] = integration_weight(f0, &freq_vertices[i], WeightFunction::I);
            iw_zero = 0;
        } else {
            g[i] = 0.0;
        }
    }
    (g, iw_zero)
}

/// Pure helper: combine tetrahedron g-values into the per-channel output
/// slots for a single `(f0, freq_vertices)`.  The first
/// `tp_type.num_channels()` entries of the returned array are meaningful;
/// trailing entries are 0.0 padding.  No side effects.
fn compute_tetra_channels(
    f0: f64,
    freq_vertices: &[[[f64; 4]; 24]; 3],
    bboxes: &[(f64, f64); 3],
    tp_type: TpType,
) -> ([f64; 3], i8) {
    let max_i = max_tetra_channels(tp_type);
    let (g, iwz) = compute_g(f0, freq_vertices, bboxes, max_i);
    let ch = match tp_type {
        TpType::Type2 => [g[2], g[0] - g[1], 0.0],
        TpType::Type3 => [g[2], g[0] - g[1], g[0] + g[1] + g[2]],
        TpType::Type4 => [g[0], 0.0, 0.0],
    };
    (ch, iwz)
}

/// Pure helper: compute the per-channel Gaussian-smeared values and the
/// iw_zero flag for a single `(f0, f1, f2)`.  `cutoff <= 0` disables the
/// skip optimisation (matches the C convention).  Only the first
/// `tp_type.num_channels()` entries of the returned array are meaningful.
fn compute_sigma_channels(
    f0: f64,
    f1: f64,
    f2: f64,
    sigma: f64,
    cutoff: f64,
    tp_type: TpType,
) -> ([f64; 3], i8) {
    match tp_type {
        TpType::Type2 | TpType::Type3 => {
            if cutoff > 0.0
                && (f0 + f1 - f2).abs() > cutoff
                && (f0 - f1 + f2).abs() > cutoff
                && (f0 - f1 - f2).abs() > cutoff
            {
                return ([0.0, 0.0, 0.0], 1);
            }
            let g0 = gaussian(f0 + f1 - f2, sigma);
            let g1 = gaussian(f0 - f1 + f2, sigma);
            let g2 = gaussian(f0 - f1 - f2, sigma);
            let ch = match tp_type {
                TpType::Type2 => [g2, g0 - g1, 0.0],
                TpType::Type3 => [g2, g0 - g1, g0 + g1 + g2],
                TpType::Type4 => unreachable!(),
            };
            (ch, 0)
        }
        TpType::Type4 => {
            if cutoff > 0.0 && (f0 + f1 - f2).abs() > cutoff {
                ([0.0, 0.0, 0.0], 1)
            } else {
                ([gaussian(f0 + f1 - f2, sigma), 0.0, 0.0], 0)
            }
        }
    }
}

/// Mirrors `funcs_gaussian` from `c/funcs.c`.
fn gaussian(x: f64, sigma: f64) -> f64 {
    INV_SQRT_2PI / sigma * (-x * x / 2.0 / sigma / sigma).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_at_zero_is_peak() {
        let sigma = 0.1;
        let g0 = gaussian(0.0, sigma);
        let expected = INV_SQRT_2PI / sigma;
        assert!((g0 - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_one_sigma_drops_to_e_minus_half() {
        let sigma = 0.5;
        let g = gaussian(sigma, sigma);
        let expected = INV_SQRT_2PI / sigma * (-0.5_f64).exp();
        assert!((g - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_integration_weight_with_sigma_type4_no_cutoff() {
        let mut iw = vec![0.0f64; 2];
        let mut iw_zero = vec![0i8; 2];
        let frequency_points = [1.0, 2.0];
        let frequencies = [0.0, 1.0];
        // num_band = 1, num_band0 = 2, triplet = (_, 0, 1) so f1=0, f2=1.
        let mut chs: [&mut [f64]; 1] = [iw.as_mut_slice()];
        integration_weight_with_sigma_per_triplet(
            &mut chs,
            &mut iw_zero,
            0.5,
            -1.0, // cutoff disabled
            &frequency_points,
            2,
            [0, 0, 1],
            &frequencies,
            1,
            TpType::Type4,
        );
        // For j=0: f0=1, f1=0, f2=1, x=f0+f1-f2=0
        // For j=1: f0=2, f1=0, f2=1, x=1
        let expected0 = gaussian(0.0, 0.5);
        let expected1 = gaussian(1.0, 0.5);
        assert!((iw[0] - expected0).abs() < 1e-12);
        assert!((iw[1] - expected1).abs() < 1e-12);
        assert_eq!(iw_zero, vec![0, 0]);
    }

    #[test]
    fn gaussian_integration_weight_with_sigma_type4_cutoff_zeros() {
        let mut iw = vec![0.0f64; 1];
        let mut iw_zero = vec![0i8; 1];
        let mut chs: [&mut [f64]; 1] = [iw.as_mut_slice()];
        // cutoff small enough to discard everything (|x|=1 > 0.5)
        integration_weight_with_sigma_per_triplet(
            &mut chs,
            &mut iw_zero,
            0.5,
            0.5,
            &[2.0],
            1,
            [0, 0, 1],
            &[0.0, 1.0],
            1,
            TpType::Type4,
        );
        assert_eq!(iw, vec![0.0]);
        assert_eq!(iw_zero[0], 1);
    }
}
