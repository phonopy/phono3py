use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArray5, PyReadonlyArray6, PyReadwriteArray1,
    PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4, PyReadwriteArray5,
    PyReadwriteArray6, PyReadwriteArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::common::Cmplx;

mod bzgrid;
mod collision_matrix;
mod common;
mod dynmat;
mod fc3;
mod funcs;
mod grgrid;
mod imag_self_energy;
mod interaction;
mod isotope;
mod pp_collision;
mod real_self_energy;
mod real_to_reciprocal;
mod recip_rotations;
mod reciprocal_to_normal;
mod snf3x3;
mod tetrahedron_method;
mod transform_rotations;
mod triplet;
mod triplet_grid;
mod triplet_iw;

use bzgrid::{BzGridAddressesError, RotateBzGridError};
use recip_rotations::ReciprocalRotationsError;
use snf3x3::Snf3x3Error;
use transform_rotations::TransformRotationsError;
use tetrahedron_method::WeightFunction;
use triplet::RelativeGridAddress;
use triplet_grid::BzTripletsError;
use triplet_iw::{BzGridError, BzGridView, TpType};

// ---------------------------------------------------------------
// Boundary conversion helpers (numpy <-> fixed-size Rust arrays)
// ---------------------------------------------------------------

fn vec3_i(arr: &PyReadonlyArray1<i64>) -> PyResult<[i64; 3]> {
    let v = arr.as_array();
    if v.len() != 3 {
        return Err(PyValueError::new_err("expected shape (3,)"));
    }
    Ok([v[0], v[1], v[2]])
}

fn mat3_i(arr: &PyReadonlyArray2<i64>) -> PyResult<[[i64; 3]; 3]> {
    let v = arr.as_array();
    if v.shape() != [3, 3] {
        return Err(PyValueError::new_err("expected shape (3, 3)"));
    }
    Ok([
        [v[[0, 0]], v[[0, 1]], v[[0, 2]]],
        [v[[1, 0]], v[[1, 1]], v[[1, 2]]],
        [v[[2, 0]], v[[2, 1]], v[[2, 2]]],
    ])
}

fn mat3_f(arr: &PyReadonlyArray2<f64>) -> PyResult<[[f64; 3]; 3]> {
    let v = arr.as_array();
    if v.shape() != [3, 3] {
        return Err(PyValueError::new_err("expected shape (3, 3)"));
    }
    Ok([
        [v[[0, 0]], v[[0, 1]], v[[0, 2]]],
        [v[[1, 0]], v[[1, 1]], v[[1, 2]]],
        [v[[2, 0]], v[[2, 1]], v[[2, 2]]],
    ])
}

fn addresses_i(arr: &PyReadonlyArray2<i64>) -> PyResult<Vec<[i64; 3]>> {
    let v = arr.as_array();
    let s = v.shape();
    if s.len() != 2 || s[1] != 3 {
        return Err(PyValueError::new_err("expected shape (n, 3)"));
    }
    let n = s[0];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([v[[i, 0]], v[[i, 1]], v[[i, 2]]]);
    }
    Ok(out)
}

fn rots_i(arr: &PyReadonlyArray3<i64>) -> PyResult<Vec<[[i64; 3]; 3]>> {
    let v = arr.as_array();
    let s = v.shape();
    if s.len() != 3 || s[1] != 3 || s[2] != 3 {
        return Err(PyValueError::new_err("expected shape (n, 3, 3)"));
    }
    let n = s[0];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([
            [v[[i, 0, 0]], v[[i, 0, 1]], v[[i, 0, 2]]],
            [v[[i, 1, 0]], v[[i, 1, 1]], v[[i, 1, 2]]],
            [v[[i, 2, 0]], v[[i, 2, 1]], v[[i, 2, 2]]],
        ]);
    }
    Ok(out)
}

fn mat3_to_array<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    m: [[T; 3]; 3],
) -> Bound<'py, PyArray2<T>> {
    let flat: Vec<T> = m.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((3, 3), flat)
        .unwrap()
        .into_pyarray(py)
}

fn rots_to_array<'py>(py: Python<'py>, rots: Vec<[[i64; 3]; 3]>) -> Bound<'py, PyArray3<i64>> {
    let n = rots.len();
    let mut flat = Vec::with_capacity(n * 9);
    for r in rots {
        for row in r {
            flat.extend_from_slice(&row);
        }
    }
    Array3::from_shape_vec((n, 3, 3), flat)
        .unwrap()
        .into_pyarray(py)
}

fn addresses_to_array<'py>(
    py: Python<'py>,
    adrs: Vec<[i64; 3]>,
) -> Bound<'py, PyArray2<i64>> {
    let n = adrs.len();
    let mut flat = Vec::with_capacity(n * 3);
    for a in adrs {
        flat.extend_from_slice(&a);
    }
    Array2::from_shape_vec((n, 3), flat)
        .unwrap()
        .into_pyarray(py)
}

fn vec3_to_array<'py>(py: Python<'py>, v: [i64; 3]) -> Bound<'py, PyArray1<i64>> {
    Array1::from_vec(v.to_vec()).into_pyarray(py)
}

// ---------------------------------------------------------------
// Python-facing functions
// ---------------------------------------------------------------

/// Compute Smith Normal Form of a 3x3 integer matrix.
///
/// Returns ``(d_diag, p, q)`` such that ``p @ a @ q == diag(d_diag)``.
/// Raises ``ValueError`` for a singular input matrix and
/// ``RuntimeError`` if the iterative reduction does not converge.
#[pyfunction]
#[pyo3(name = "snf3x3")]
fn py_snf3x3<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let m = mat3_i(&a)?;
    match snf3x3::snf3x3(m) {
        Ok(r) => Ok((
            vec3_to_array(py, r.d_diag),
            mat3_to_array(py, r.p),
            mat3_to_array(py, r.q),
        )),
        Err(Snf3x3Error::Singular) => Err(PyValueError::new_err("snf3x3: singular input matrix")),
        Err(Snf3x3Error::NotConverged) => Err(PyRuntimeError::new_err("snf3x3 did not converge")),
    }
}

/// Return the GR grid-point index of a single grid address.
///
/// ``address`` components are reduced to ``[0, d_diag[i])`` before
/// the index is computed.
#[pyfunction]
#[pyo3(name = "grid_index_from_address")]
fn py_grid_index_from_address(
    address: PyReadonlyArray1<i64>,
    d_diag: PyReadonlyArray1<i64>,
) -> PyResult<i64> {
    Ok(grgrid::grid_index_from_address(
        vec3_i(&address)?,
        vec3_i(&d_diag)?,
    ))
}

/// Return all GR-grid addresses as a ``(prod(d_diag), 3)`` array.
#[pyfunction]
#[pyo3(name = "gr_grid_addresses")]
fn py_gr_grid_addresses<'py>(
    py: Python<'py>,
    d_diag: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let adrs = grgrid::all_grid_addresses(vec3_i(&d_diag)?);
    Ok(addresses_to_array(py, adrs))
}

/// Return ``(ir_grid_map, num_ir)`` for the given rotations.
#[pyfunction]
#[pyo3(name = "ir_grid_map")]
fn py_ir_grid_map<'py>(
    py: Python<'py>,
    rotations: PyReadonlyArray3<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    ps: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
    let rots = rots_i(&rotations)?;
    let d = vec3_i(&d_diag)?;
    let p = vec3_i(&ps)?;
    let r = grgrid::ir_grid_map(&rots, d, p);
    Ok((r.map.into_pyarray(py), r.num_ir))
}

/// Return the reciprocal-space point group as an ``(n, 3, 3)`` array.
/// Raises ``ValueError`` if the input contains too many rotations
/// (>48 unique) or too many for the time-reversal extension (>24).
#[pyfunction]
#[pyo3(name = "reciprocal_rotations")]
fn py_reciprocal_rotations<'py>(
    py: Python<'py>,
    rotations: PyReadonlyArray3<'py, i64>,
    is_time_reversal: bool,
) -> PyResult<Bound<'py, PyArray3<i64>>> {
    let rots = rots_i(&rotations)?;
    match recip_rotations::reciprocal_rotations(&rots, is_time_reversal) {
        Ok(v) => Ok(rots_to_array(py, v)),
        Err(ReciprocalRotationsError::TooManyRotations) => Err(PyValueError::new_err(
            "reciprocal_rotations: more than 48 unique rotations",
        )),
        Err(ReciprocalRotationsError::TooManyForInversion) => Err(PyValueError::new_err(
            "reciprocal_rotations: cannot add inversion (count exceeds 24)",
        )),
    }
}

/// Transform rotations by ``D * Q^-1 * R * Q * D^-1``.
///
/// Raises ``RuntimeError`` when the grid breaks the crystal symmetry,
/// or ``ValueError`` if ``Q`` is singular.
#[pyfunction]
#[pyo3(name = "transform_rotations")]
fn py_transform_rotations<'py>(
    py: Python<'py>,
    rotations: PyReadonlyArray3<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    q: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<i64>>> {
    let rots = rots_i(&rotations)?;
    let d = vec3_i(&d_diag)?;
    let qm = mat3_i(&q)?;
    match transform_rotations::transform_rotations(&rots, d, qm) {
        Ok(v) => Ok(rots_to_array(py, v)),
        Err(TransformRotationsError::SingularQ) => {
            Err(PyValueError::new_err("transform_rotations: singular Q"))
        }
        Err(TransformRotationsError::SymmetryBroken) => Err(PyRuntimeError::new_err(
            "Grid symmetry is broken. Use generalized regular grid.",
        )),
    }
}

/// Build BZ grid addresses.  Returns ``(addresses, bz_map, bzg2grg)``.
///
/// ``bz_grid_type`` is 1 for the sparse layout or 2 for the dense layout.
/// Raises ``ValueError`` for unsupported types or a non-unimodular ``Q``.
#[pyfunction]
#[pyo3(name = "bz_grid_addresses")]
fn py_bz_grid_addresses<'py>(
    py: Python<'py>,
    d_diag: PyReadonlyArray1<'py, i64>,
    q: PyReadonlyArray2<'py, i64>,
    ps: PyReadonlyArray1<'py, i64>,
    rec_lattice: PyReadonlyArray2<'py, f64>,
    bz_grid_type: i64,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let d = vec3_i(&d_diag)?;
    let qm = mat3_i(&q)?;
    let p = vec3_i(&ps)?;
    let rec = mat3_f(&rec_lattice)?;
    match bzgrid::bz_grid_addresses(d, qm, p, rec, bz_grid_type) {
        Ok(r) => Ok((
            addresses_to_array(py, r.addresses),
            r.bz_map.into_pyarray(py),
            r.bzg2grg.into_pyarray(py),
        )),
        Err(BzGridAddressesError::NotUnimodularQ) => {
            Err(PyValueError::new_err("bz_grid_addresses: Q is not unimodular"))
        }
        Err(BzGridAddressesError::BadGridType) => Err(PyValueError::new_err(
            "bz_grid_addresses: bz_grid_type must be 1 or 2",
        )),
    }
}

/// Rotate a BZ grid index and return the rotated point's BZ index.
///
/// ``bz_grid_type`` is 1 for the sparse layout or 2 for the dense layout.
/// Raises ``ValueError`` for other types.
#[pyfunction]
#[pyo3(name = "rotate_bz_grid_index")]
fn py_rotate_bz_grid_index<'py>(
    bz_grid_index: i64,
    rotation: PyReadonlyArray2<'py, i64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    ps: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
) -> PyResult<i64> {
    let rot = mat3_i(&rotation)?;
    let adrs = addresses_i(&bz_grid_addresses)?;
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be contiguous"))?;
    let d = vec3_i(&d_diag)?;
    let p = vec3_i(&ps)?;
    match bzgrid::rotate_bz_grid_index(
        bz_grid_index,
        rot,
        &adrs,
        bzmap_slice,
        d,
        p,
        bz_grid_type,
    ) {
        Ok(v) => Ok(v),
        Err(RotateBzGridError::BadGridType) => Err(PyValueError::new_err(
            "rotate_bz_grid_index: bz_grid_type must be 1 or 2",
        )),
    }
}

/// Search symmetry-reduced triplets at a fixed q-point.
///
/// Returns ``(map_triplets, map_q, num_ir)``.
#[pyfunction]
#[pyo3(name = "ir_triplets_at_q")]
fn py_ir_triplets_at_q<'py>(
    py: Python<'py>,
    grid_point: i64,
    d_diag: PyReadonlyArray1<'py, i64>,
    rec_rotations: PyReadonlyArray3<'py, i64>,
    is_time_reversal: bool,
    swappable: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    i64,
)> {
    let d = vec3_i(&d_diag)?;
    let rots = rots_i(&rec_rotations)?;
    match triplet_grid::ir_triplets_at_q(grid_point, d, &rots, is_time_reversal, swappable) {
        Ok(r) => Ok((
            r.map_triplets.into_pyarray(py),
            r.map_q.into_pyarray(py),
            r.num_ir,
        )),
        Err(ReciprocalRotationsError::TooManyRotations) => Err(PyValueError::new_err(
            "ir_triplets_at_q: more than 48 unique rotations",
        )),
        Err(ReciprocalRotationsError::TooManyForInversion) => Err(PyValueError::new_err(
            "ir_triplets_at_q: cannot add inversion (count exceeds 24)",
        )),
    }
}

/// Find symmetry-reduced BZ triplets at a fixed q-point.
///
/// Returns the ``(num_ir, 3)`` integer array of triplets.
#[pyfunction]
#[pyo3(name = "bz_triplets_at_q")]
fn py_bz_triplets_at_q<'py>(
    py: Python<'py>,
    grid_point: i64,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    map_triplets: PyReadonlyArray1<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    q: PyReadonlyArray2<'py, i64>,
    reciprocal_lattice: PyReadonlyArray2<'py, f64>,
    bz_grid_type: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let adrs = addresses_i(&bz_grid_addresses)?;
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be contiguous"))?;
    let mtv = map_triplets.as_array();
    let map_triplets_slice = mtv
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("map_triplets must be contiguous"))?;
    let d = vec3_i(&d_diag)?;
    let qm = mat3_i(&q)?;
    let rec = mat3_f(&reciprocal_lattice)?;
    match triplet_grid::bz_triplets_at_q(
        grid_point,
        &adrs,
        bzmap_slice,
        map_triplets_slice,
        d,
        qm,
        rec,
        bz_grid_type,
    ) {
        Ok(v) => Ok(addresses_to_array(py, v)),
        Err(BzTripletsError::BadGridType) => Err(PyValueError::new_err(
            "bz_triplets_at_q: bz_grid_type must be 1 or 2",
        )),
    }
}

/// Triplet tetrahedron-method integration weights.
///
/// Writes into ``iw`` (shape ``(num_channels, num_triplets, num_band0,
/// num_band1, num_band2)``, dtype ``float64``) and ``iw_zero`` (shape
/// ``(num_triplets, num_band0, num_band1, num_band2)``, dtype ``int8``).
/// ``tp_type`` is 2, 3, or 4.
#[pyfunction]
#[pyo3(name = "triplets_integration_weights")]
#[allow(clippy::too_many_arguments)]
fn py_triplets_integration_weights<'py>(
    py: Python<'py>,
    mut iw: PyReadwriteArray5<'py, f64>,
    mut iw_zero: PyReadwriteArray4<'py, i8>,
    frequency_points: PyReadonlyArray1<'py, f64>,
    relative_grid_address: PyReadonlyArray3<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    triplets: PyReadonlyArray2<'py, i64>,
    frequencies1: PyReadonlyArray2<'py, f64>,
    frequencies2: PyReadonlyArray2<'py, f64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
    tp_type: i64,
) -> PyResult<()> {
    let tp = TpType::try_from_i64(tp_type)
        .map_err(|_| PyValueError::new_err("tp_type must be 2, 3, or 4"))?;
    let rga = relative_grid_address_3d(&relative_grid_address)?;
    let trip = addresses_i(&triplets)?;
    let adrs = addresses_i(&bz_grid_addresses)?;
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be C-contiguous"))?;
    let d = vec3_i(&d_diag)?;

    let fp_view = frequency_points.as_array();
    let fp_slice = fp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequency_points must be C-contiguous"))?;
    let f1_view = frequencies1.as_array();
    let f1_slice = f1_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies1 must be C-contiguous"))?;
    let f2_view = frequencies2.as_array();
    let f2_slice = f2_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies2 must be C-contiguous"))?;
    let num_band1 = frequencies1.shape()[1] as i64;
    let num_band2 = frequencies2.shape()[1] as i64;

    let iw_slice = iw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("iw must be C-contiguous"))?;
    let iwz_slice = iw_zero
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("iw_zero must be C-contiguous"))?;

    py.allow_threads(|| {
        let bzgrid = BzGridView {
            d_diag: d,
            addresses: &adrs,
            gp_map: bzmap_slice,
            bz_grid_type,
        };
        triplet::integration_weight(
            iw_slice,
            iwz_slice,
            fp_slice,
            &rga,
            &trip,
            &bzgrid,
            f1_slice,
            num_band1,
            f2_slice,
            num_band2,
            tp,
        )
    })
    .map_err(|e| match e {
        BzGridError::BadGridType => {
            PyValueError::new_err("bz_grid_type must be 1 or 2")
        }
        BzGridError::BadTpType => PyValueError::new_err("tp_type must be 2, 3, or 4"),
    })
}

/// Triplet Gaussian-smeared integration weights.
///
/// ``iw`` is ``(num_channels, num_triplets, num_band0, num_band, num_band)``
/// and ``iw_zero`` is ``(num_triplets, num_band0, num_band, num_band)``.
/// ``tp_type`` is inferred from ``iw.shape[0]``: 2→Type2, 3→Type3.
/// Pass ``sigma_cutoff < 0`` to disable the cutoff-skip optimisation.
#[pyfunction]
#[pyo3(name = "triplets_integration_weights_with_sigma")]
fn py_triplets_integration_weights_with_sigma<'py>(
    py: Python<'py>,
    mut iw: PyReadwriteArray5<'py, f64>,
    mut iw_zero: PyReadwriteArray4<'py, i8>,
    frequency_points: PyReadonlyArray1<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    sigma: f64,
    sigma_cutoff: f64,
) -> PyResult<()> {
    let tp = TpType::try_from_i64(iw.shape()[0] as i64)
        .map_err(|_| PyValueError::new_err("iw.shape[0] must be 1, 2, or 3"))?;
    let trip = addresses_i(&triplets)?;
    let num_band = frequencies.shape()[1] as i64;

    let fp_view = frequency_points.as_array();
    let fp_slice = fp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequency_points must be C-contiguous"))?;
    let f_view = frequencies.as_array();
    let f_slice = f_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;

    let iw_slice = iw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("iw must be C-contiguous"))?;
    let iwz_slice = iw_zero
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("iw_zero must be C-contiguous"))?;

    py.allow_threads(|| {
        triplet::integration_weight_with_sigma(
            iw_slice,
            iwz_slice,
            sigma,
            sigma_cutoff,
            fp_slice,
            &trip,
            f_slice,
            num_band,
            tp,
        );
    });
    Ok(())
}

/// Reinterpret a `Complex64` slice as `[f64; 2]` slice (`Cmplx`).
///
/// SAFETY: `num_complex::Complex<f64>` is `repr(C)` with fields
/// `re: f64, im: f64`, giving the exact same layout and alignment as
/// `[f64; 2]`.  The lifetime of the returned slice is bounded by the
/// input, so no aliasing or escape is possible.
fn complex_as_cmplx_mut(s: &mut [Complex64]) -> &mut [Cmplx] {
    let len = s.len();
    let ptr = s.as_mut_ptr() as *mut Cmplx;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Reinterpret a flat slice as a slice of fixed-size arrays.
///
/// SAFETY: `[T; N]` has the same layout as `N` consecutive `T`s.
/// The returned slice borrows from the input, so no escape is
/// possible.  The input length must be a multiple of `N`.
fn group_as_array<T, const N: usize>(s: &[T]) -> &[[T; N]] {
    debug_assert_eq!(s.len() % N, 0);
    let n = s.len() / N;
    let ptr = s.as_ptr() as *const [T; N];
    unsafe { std::slice::from_raw_parts(ptr, n) }
}

fn group_as_array_2d<T, const N: usize, const M: usize>(s: &[T]) -> &[[[T; M]; N]] {
    debug_assert_eq!(s.len() % (N * M), 0);
    let n = s.len() / (N * M);
    let ptr = s.as_ptr() as *const [[T; M]; N];
    unsafe { std::slice::from_raw_parts(ptr, n) }
}

/// Build the dynamical matrix at a single q-point in place.
///
/// Writes into ``dynamical_matrix`` (complex128, shape
/// ``(num_patom*3, num_patom*3)``).  ``fc`` is float64 with shape
/// ``(n_rows, num_satom, 3, 3)``, where ``n_rows`` is ``num_satom``
/// for full force constants or ``num_patom`` for the compact layout.
/// ``charge_sum`` is the optional Wang-NAC contribution
/// (``(num_patom, num_patom, 3, 3)``); pass ``None`` for the no-NAC
/// or Gonze-Lee paths.
#[pyfunction]
#[pyo3(name = "dynamical_matrix_at_q")]
#[pyo3(signature = (dynamical_matrix, fc, q, svecs, multi, mass, s2p_map, p2s_map, charge_sum=None))]
#[allow(clippy::too_many_arguments)]
fn py_dynamical_matrix_at_q<'py>(
    py: Python<'py>,
    mut dynamical_matrix: PyReadwriteArray2<'py, Complex64>,
    fc: PyReadonlyArray4<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    svecs: PyReadonlyArray2<'py, f64>,
    multi: PyReadonlyArray3<'py, i64>,
    mass: PyReadonlyArray1<'py, f64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    charge_sum: Option<PyReadonlyArray4<'py, f64>>,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_band = num_patom * 3;

    if dynamical_matrix.shape() != [num_band, num_band] {
        return Err(PyValueError::new_err(
            "dynamical_matrix must have shape (num_patom*3, num_patom*3)",
        ));
    }
    let fc_shape = fc.shape();
    if fc_shape[1] != num_satom
        || fc_shape[2] != 3
        || fc_shape[3] != 3
        || (fc_shape[0] != num_patom && fc_shape[0] != num_satom)
    {
        return Err(PyValueError::new_err(
            "fc must have shape (num_patom or num_satom, num_satom, 3, 3)",
        ));
    }
    if q.shape() != [3] {
        return Err(PyValueError::new_err("q must have shape (3,)"));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multi.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multi must have shape (num_satom, num_patom, 2)",
        ));
    }
    if mass.shape() != [num_patom] {
        return Err(PyValueError::new_err("mass must have shape (num_patom,)"));
    }

    let q_view = q.as_array();
    let q3 = [q_view[0], q_view[1], q_view[2]];

    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);

    let multi_view = multi.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multi must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);

    let mass_view = mass.as_array();
    let mass_slice = mass_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mass must be C-contiguous"))?;

    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;

    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;

    let fc_view = fc.as_array();
    let fc_slice = fc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc must be C-contiguous"))?;

    let cs_view;
    let cs_flat;
    let cs: Option<&[[[f64; 3]; 3]]> = match charge_sum.as_ref() {
        None => None,
        Some(cs_arr) => {
            if cs_arr.shape() != [num_patom, num_patom, 3, 3] {
                return Err(PyValueError::new_err(
                    "charge_sum must have shape (num_patom, num_patom, 3, 3)",
                ));
            }
            cs_view = cs_arr.as_array();
            cs_flat = cs_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("charge_sum must be C-contiguous"))?;
            Some(group_as_array_2d::<f64, 3, 3>(cs_flat))
        }
    };

    let dm_slice = dynamical_matrix
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("dynamical_matrix must be C-contiguous"))?;
    let dm_cmplx = complex_as_cmplx_mut(dm_slice);

    py.allow_threads(|| {
        dynmat::get_dynamical_matrix_at_q(
            dm_cmplx,
            fc_slice,
            q3,
            svecs_slice,
            multi_slice,
            mass_slice,
            s2p_slice,
            p2s_slice,
            cs,
            num_patom,
            num_satom,
        );
    });

    Ok(())
}

/// Build the Wang-NAC charge sum.
///
/// Writes into ``charge_sum`` (float64, shape
/// ``(num_patom, num_patom, 3, 3)``).  ``born`` has shape
/// ``(num_patom, 3, 3)`` and ``q_cart`` has shape ``(3,)``.
/// ``factor`` is the prefactor assembled on the Python side
/// (``4*pi/V * unit_conversion / denominator``).
#[pyfunction]
#[pyo3(name = "charge_sum")]
fn py_charge_sum<'py>(
    mut charge_sum: PyReadwriteArray4<'py, f64>,
    factor: f64,
    q_cart: PyReadonlyArray1<'py, f64>,
    born: PyReadonlyArray3<'py, f64>,
) -> PyResult<()> {
    let num_patom = born.shape()[0];
    if born.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "born must have shape (num_patom, 3, 3)",
        ));
    }
    if charge_sum.shape() != [num_patom, num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "charge_sum must have shape (num_patom, num_patom, 3, 3)",
        ));
    }
    if q_cart.shape() != [3] {
        return Err(PyValueError::new_err("q_cart must have shape (3,)"));
    }

    let q_view = q_cart.as_array();
    let q3 = [q_view[0], q_view[1], q_view[2]];

    let born_view = born.as_array();
    let born_flat = born_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("born must be C-contiguous"))?;
    let born_slice: &[[[f64; 3]; 3]] = group_as_array_2d::<f64, 3, 3>(born_flat);

    let cs_slice = charge_sum
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("charge_sum must be C-contiguous"))?;
    let cs_blocks: &mut [[[f64; 3]; 3]] = {
        debug_assert_eq!(cs_slice.len() % 9, 0);
        let n = cs_slice.len() / 9;
        let ptr = cs_slice.as_mut_ptr() as *mut [[f64; 3]; 3];
        unsafe { std::slice::from_raw_parts_mut(ptr, n) }
    };

    dynmat::get_charge_sum(cs_blocks, factor, q3, born_slice);
    Ok(())
}

/// Build the q=0 reciprocal dipole-dipole correction.
///
/// Writes into ``dd_q0`` (complex128, shape ``(num_patom, 3, 3)``).
/// ``g_list`` is ``(num_G, 3)`` cartesian reciprocal vectors,
/// ``born`` is ``(num_patom, 3, 3)``, ``dielectric`` is ``(3, 3)``,
/// ``pos`` is ``(num_patom, 3)`` cartesian positions.  ``lambda``
/// is the Ewald parameter and ``tolerance`` is the |K| zero
/// threshold.
#[pyfunction]
#[pyo3(name = "recip_dipole_dipole_q0")]
#[allow(clippy::too_many_arguments)]
fn py_recip_dipole_dipole_q0<'py>(
    py: Python<'py>,
    mut dd_q0: PyReadwriteArray3<'py, Complex64>,
    g_list: PyReadonlyArray2<'py, f64>,
    born: PyReadonlyArray3<'py, f64>,
    dielectric: PyReadonlyArray2<'py, f64>,
    pos: PyReadonlyArray2<'py, f64>,
    lambda: f64,
    tolerance: f64,
) -> PyResult<()> {
    let num_patom = born.shape()[0];
    if born.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "born must have shape (num_patom, 3, 3)",
        ));
    }
    if dd_q0.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "dd_q0 must have shape (num_patom, 3, 3)",
        ));
    }
    if pos.shape() != [num_patom, 3] {
        return Err(PyValueError::new_err("pos must have shape (num_patom, 3)"));
    }
    let g_shape = g_list.shape();
    if g_shape.len() != 2 || g_shape[1] != 3 {
        return Err(PyValueError::new_err("g_list must have shape (num_G, 3)"));
    }

    let diel = mat3_f(&dielectric)?;

    let g_view = g_list.as_array();
    let g_flat = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_list must be C-contiguous"))?;
    let g_slice: &[[f64; 3]] = group_as_array(g_flat);

    let pos_view = pos.as_array();
    let pos_flat = pos_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("pos must be C-contiguous"))?;
    let pos_slice: &[[f64; 3]] = group_as_array(pos_flat);

    let born_view = born.as_array();
    let born_flat = born_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("born must be C-contiguous"))?;
    let born_slice: &[[[f64; 3]; 3]] = group_as_array_2d::<f64, 3, 3>(born_flat);

    let dd_slice = dd_q0
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("dd_q0 must be C-contiguous"))?;
    let dd_cmplx = complex_as_cmplx_mut(dd_slice);

    py.allow_threads(|| {
        dynmat::get_recip_dipole_dipole_q0(
            dd_cmplx,
            g_slice,
            num_patom,
            born_slice,
            &diel,
            pos_slice,
            lambda,
            tolerance,
        );
    });

    Ok(())
}

/// Build the Gonze-Lee reciprocal dipole-dipole correction at q.
///
/// Writes into ``dd`` (complex128, shape ``(num_patom, 3, num_patom, 3)``).
/// ``dd_q0`` (complex128, shape ``(num_patom, 3, 3)``) is the q=0
/// correction subtracted on the diagonal blocks.  ``q_direction_cart``
/// is a 3-vector or ``None`` (use ``None`` when q is non-zero).
/// ``factor`` is ``4*pi/V * unit_conversion``.
#[pyfunction]
#[pyo3(name = "recip_dipole_dipole")]
#[pyo3(signature = (dd, dd_q0, g_list, q_cart, born, dielectric, pos, factor, lambda_, tolerance, q_direction_cart=None))]
#[allow(clippy::too_many_arguments)]
fn py_recip_dipole_dipole<'py>(
    py: Python<'py>,
    mut dd: PyReadwriteArray4<'py, Complex64>,
    dd_q0: PyReadonlyArray3<'py, Complex64>,
    g_list: PyReadonlyArray2<'py, f64>,
    q_cart: PyReadonlyArray1<'py, f64>,
    born: PyReadonlyArray3<'py, f64>,
    dielectric: PyReadonlyArray2<'py, f64>,
    pos: PyReadonlyArray2<'py, f64>,
    factor: f64,
    lambda_: f64,
    tolerance: f64,
    q_direction_cart: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<()> {
    let num_patom = born.shape()[0];
    if born.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "born must have shape (num_patom, 3, 3)",
        ));
    }
    if dd.shape() != [num_patom, 3, num_patom, 3] {
        return Err(PyValueError::new_err(
            "dd must have shape (num_patom, 3, num_patom, 3)",
        ));
    }
    if dd_q0.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "dd_q0 must have shape (num_patom, 3, 3)",
        ));
    }
    if pos.shape() != [num_patom, 3] {
        return Err(PyValueError::new_err("pos must have shape (num_patom, 3)"));
    }
    let g_shape = g_list.shape();
    if g_shape.len() != 2 || g_shape[1] != 3 {
        return Err(PyValueError::new_err("g_list must have shape (num_G, 3)"));
    }
    if q_cart.shape() != [3] {
        return Err(PyValueError::new_err("q_cart must have shape (3,)"));
    }

    let diel = mat3_f(&dielectric)?;

    let q_view = q_cart.as_array();
    let q3 = [q_view[0], q_view[1], q_view[2]];

    let qd_view;
    let qd: Option<[f64; 3]> = match q_direction_cart.as_ref() {
        None => None,
        Some(arr) => {
            if arr.shape() != [3] {
                return Err(PyValueError::new_err(
                    "q_direction_cart must have shape (3,)",
                ));
            }
            qd_view = arr.as_array();
            Some([qd_view[0], qd_view[1], qd_view[2]])
        }
    };

    let g_view = g_list.as_array();
    let g_flat = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_list must be C-contiguous"))?;
    let g_slice: &[[f64; 3]] = group_as_array(g_flat);

    let pos_view = pos.as_array();
    let pos_flat = pos_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("pos must be C-contiguous"))?;
    let pos_slice: &[[f64; 3]] = group_as_array(pos_flat);

    let born_view = born.as_array();
    let born_flat = born_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("born must be C-contiguous"))?;
    let born_slice: &[[[f64; 3]; 3]] = group_as_array_2d::<f64, 3, 3>(born_flat);

    let dd_q0_view = dd_q0.as_array();
    let dd_q0_flat = dd_q0_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dd_q0 must be C-contiguous"))?;
    let dd_q0_cmplx: &[Cmplx] = {
        let n = dd_q0_flat.len();
        let ptr = dd_q0_flat.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let dd_slice = dd
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("dd must be C-contiguous"))?;
    let dd_cmplx = complex_as_cmplx_mut(dd_slice);

    py.allow_threads(|| {
        dynmat::get_recip_dipole_dipole(
            dd_cmplx,
            dd_q0_cmplx,
            g_slice,
            num_patom,
            q3,
            qd,
            born_slice,
            &diel,
            pos_slice,
            factor,
            lambda_,
            tolerance,
        );
    });

    Ok(())
}

/// Build dynamical matrices at the listed grid points (Wang- or
/// no-NAC path).
///
/// ``dynmats`` is complex128 with shape
/// ``(num_phonons, num_band, num_band)``; only blocks at
/// ``undone_grid_points`` are written.  ``undone_grid_points`` must
/// be strictly ascending (sorted and unique) with every entry in
/// ``0..num_phonons``.  Pass ``born=None`` for the no-NAC path;
/// otherwise ``dielectric`` and ``reciprocal_lattice`` must also be
/// provided.  ``q_direction`` selects the Gamma-limit direction
/// (fractional) and may be left ``None``.
#[pyfunction]
#[pyo3(name = "dynamical_matrices_at_gridpoints")]
#[pyo3(signature = (
    dynmats, undone_grid_points, grid_addresses, qd_inv,
    fc, svecs, multi, mass, p2s_map, s2p_map,
    born=None, dielectric=None, reciprocal_lattice=None,
    q_direction=None, nac_factor=0.0,
))]
#[allow(clippy::too_many_arguments)]
fn py_dynamical_matrices_at_gridpoints<'py>(
    py: Python<'py>,
    mut dynmats: PyReadwriteArray3<'py, Complex64>,
    undone_grid_points: PyReadonlyArray1<'py, i64>,
    grid_addresses: PyReadonlyArray2<'py, i64>,
    qd_inv: PyReadonlyArray2<'py, f64>,
    fc: PyReadonlyArray4<'py, f64>,
    svecs: PyReadonlyArray2<'py, f64>,
    multi: PyReadonlyArray3<'py, i64>,
    mass: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    born: Option<PyReadonlyArray3<'py, f64>>,
    dielectric: Option<PyReadonlyArray2<'py, f64>>,
    reciprocal_lattice: Option<PyReadonlyArray2<'py, f64>>,
    q_direction: Option<PyReadonlyArray1<'py, f64>>,
    nac_factor: f64,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_band = num_patom * 3;
    let num_phonons = grid_addresses.shape()[0];

    if dynmats.shape() != [num_phonons, num_band, num_band] {
        return Err(PyValueError::new_err(
            "dynmats must have shape (num_phonons, num_band, num_band)",
        ));
    }
    if grid_addresses.shape() != [num_phonons, 3] {
        return Err(PyValueError::new_err(
            "grid_addresses must have shape (num_phonons, 3)",
        ));
    }
    if qd_inv.shape() != [3, 3] {
        return Err(PyValueError::new_err("qd_inv must have shape (3, 3)"));
    }
    let fc_shape = fc.shape();
    if fc_shape[1] != num_satom
        || fc_shape[2] != 3
        || fc_shape[3] != 3
        || (fc_shape[0] != num_patom && fc_shape[0] != num_satom)
    {
        return Err(PyValueError::new_err(
            "fc must have shape (num_patom or num_satom, num_satom, 3, 3)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multi.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multi must have shape (num_satom, num_patom, 2)",
        ));
    }
    if mass.shape() != [num_patom] {
        return Err(PyValueError::new_err("mass must have shape (num_patom,)"));
    }

    let qd = mat3_f(&qd_inv)?;
    let ga = addresses_i(&grid_addresses)?;

    let ug_view = undone_grid_points.as_array();
    let ug_slice = ug_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("undone_grid_points must be C-contiguous"))?;

    let fc_view = fc.as_array();
    let fc_slice = fc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc must be C-contiguous"))?;

    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);

    let multi_view = multi.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multi must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);

    let mass_view = mass.as_array();
    let mass_slice = mass_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mass must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;

    let born_view;
    let wang = match born.as_ref() {
        None => None,
        Some(b) => {
            let diel_arr = dielectric.as_ref().ok_or_else(|| {
                PyValueError::new_err("dielectric must be provided when born is given")
            })?;
            let rec_arr = reciprocal_lattice.as_ref().ok_or_else(|| {
                PyValueError::new_err(
                    "reciprocal_lattice must be provided when born is given",
                )
            })?;
            if b.shape() != [num_patom, 3, 3] {
                return Err(PyValueError::new_err(
                    "born must have shape (num_patom, 3, 3)",
                ));
            }
            let diel = mat3_f(diel_arr)?;
            let rec = mat3_f(rec_arr)?;
            born_view = b.as_array();
            let born_flat = born_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("born must be C-contiguous"))?;
            let born_slice: &[[[f64; 3]; 3]] = group_as_array_2d::<f64, 3, 3>(born_flat);

            let qd_dir: Option<[f64; 3]> = match q_direction.as_ref() {
                None => None,
                Some(arr) => {
                    if arr.shape() != [3] {
                        return Err(PyValueError::new_err(
                            "q_direction must have shape (3,)",
                        ));
                    }
                    let v = arr.as_array();
                    Some([v[0], v[1], v[2]])
                }
            };
            Some(dynmat::WangNacParams {
                born: born_slice,
                dielectric: diel,
                reciprocal_lattice: rec,
                q_direction: qd_dir,
                nac_factor,
            })
        }
    };

    let dm_slice = dynmats
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("dynmats must be C-contiguous"))?;
    let dm_cmplx = complex_as_cmplx_mut(dm_slice);

    py.allow_threads(|| {
        dynmat::dynamical_matrices_at_gridpoints(
            dm_cmplx,
            ug_slice,
            &ga,
            &qd,
            fc_slice,
            svecs_slice,
            multi_slice,
            mass_slice,
            p2s_slice,
            s2p_slice,
            num_patom,
            num_satom,
            wang.as_ref(),
        );
    });

    Ok(())
}

/// Build dynamical matrices at the listed grid points (Gonze-Lee
/// NAC path).  Same output buffer layout as
/// ``dynamical_matrices_at_gridpoints``.  ``undone_grid_points``
/// must be strictly ascending (sorted and unique) with every entry
/// in ``0..num_phonons``.
#[pyfunction]
#[pyo3(name = "dynamical_matrices_at_gridpoints_gonze")]
#[pyo3(signature = (
    dynmats, undone_grid_points, grid_addresses, qd_inv,
    fc, svecs, multi, mass, p2s_map, s2p_map,
    born, dielectric, reciprocal_lattice,
    pos, dd_q0, g_list, nac_factor, lambda_,
    q_direction=None,
))]
#[allow(clippy::too_many_arguments)]
fn py_dynamical_matrices_at_gridpoints_gonze<'py>(
    py: Python<'py>,
    mut dynmats: PyReadwriteArray3<'py, Complex64>,
    undone_grid_points: PyReadonlyArray1<'py, i64>,
    grid_addresses: PyReadonlyArray2<'py, i64>,
    qd_inv: PyReadonlyArray2<'py, f64>,
    fc: PyReadonlyArray4<'py, f64>,
    svecs: PyReadonlyArray2<'py, f64>,
    multi: PyReadonlyArray3<'py, i64>,
    mass: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    born: PyReadonlyArray3<'py, f64>,
    dielectric: PyReadonlyArray2<'py, f64>,
    reciprocal_lattice: PyReadonlyArray2<'py, f64>,
    pos: PyReadonlyArray2<'py, f64>,
    dd_q0: PyReadonlyArray3<'py, Complex64>,
    g_list: PyReadonlyArray2<'py, f64>,
    nac_factor: f64,
    lambda_: f64,
    q_direction: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_band = num_patom * 3;
    let num_phonons = grid_addresses.shape()[0];

    if dynmats.shape() != [num_phonons, num_band, num_band] {
        return Err(PyValueError::new_err(
            "dynmats must have shape (num_phonons, num_band, num_band)",
        ));
    }
    if grid_addresses.shape() != [num_phonons, 3] {
        return Err(PyValueError::new_err(
            "grid_addresses must have shape (num_phonons, 3)",
        ));
    }
    if qd_inv.shape() != [3, 3] {
        return Err(PyValueError::new_err("qd_inv must have shape (3, 3)"));
    }
    let fc_shape = fc.shape();
    if fc_shape[1] != num_satom
        || fc_shape[2] != 3
        || fc_shape[3] != 3
        || (fc_shape[0] != num_patom && fc_shape[0] != num_satom)
    {
        return Err(PyValueError::new_err(
            "fc must have shape (num_patom or num_satom, num_satom, 3, 3)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multi.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multi must have shape (num_satom, num_patom, 2)",
        ));
    }
    if mass.shape() != [num_patom] {
        return Err(PyValueError::new_err("mass must have shape (num_patom,)"));
    }
    if born.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "born must have shape (num_patom, 3, 3)",
        ));
    }
    if pos.shape() != [num_patom, 3] {
        return Err(PyValueError::new_err("pos must have shape (num_patom, 3)"));
    }
    if dd_q0.shape() != [num_patom, 3, 3] {
        return Err(PyValueError::new_err(
            "dd_q0 must have shape (num_patom, 3, 3)",
        ));
    }
    let g_shape = g_list.shape();
    if g_shape.len() != 2 || g_shape[1] != 3 {
        return Err(PyValueError::new_err("g_list must have shape (num_G, 3)"));
    }

    let qd = mat3_f(&qd_inv)?;
    let diel = mat3_f(&dielectric)?;
    let rec = mat3_f(&reciprocal_lattice)?;
    let ga = addresses_i(&grid_addresses)?;

    let qd_dir: Option<[f64; 3]> = match q_direction.as_ref() {
        None => None,
        Some(arr) => {
            if arr.shape() != [3] {
                return Err(PyValueError::new_err("q_direction must have shape (3,)"));
            }
            let v = arr.as_array();
            Some([v[0], v[1], v[2]])
        }
    };

    let ug_view = undone_grid_points.as_array();
    let ug_slice = ug_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("undone_grid_points must be C-contiguous"))?;

    let fc_view = fc.as_array();
    let fc_slice = fc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc must be C-contiguous"))?;

    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);

    let multi_view = multi.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multi must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);

    let mass_view = mass.as_array();
    let mass_slice = mass_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mass must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;

    let born_view = born.as_array();
    let born_flat = born_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("born must be C-contiguous"))?;
    let born_slice: &[[[f64; 3]; 3]] = group_as_array_2d::<f64, 3, 3>(born_flat);

    let pos_view = pos.as_array();
    let pos_flat = pos_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("pos must be C-contiguous"))?;
    let pos_slice: &[[f64; 3]] = group_as_array(pos_flat);

    let g_view = g_list.as_array();
    let g_flat = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_list must be C-contiguous"))?;
    let g_slice: &[[f64; 3]] = group_as_array(g_flat);

    let dd_q0_view = dd_q0.as_array();
    let dd_q0_flat = dd_q0_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dd_q0 must be C-contiguous"))?;
    let dd_q0_cmplx: &[Cmplx] = {
        let n = dd_q0_flat.len();
        let ptr = dd_q0_flat.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let gonze = dynmat::GonzeNacParams {
        born: born_slice,
        dielectric: diel,
        reciprocal_lattice: rec,
        q_direction: qd_dir,
        nac_factor,
        pos: pos_slice,
        dd_q0: dd_q0_cmplx,
        g_list: g_slice,
        lambda: lambda_,
    };

    let dm_slice = dynmats
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("dynmats must be C-contiguous"))?;
    let dm_cmplx = complex_as_cmplx_mut(dm_slice);

    py.allow_threads(|| {
        dynmat::dynamical_matrices_at_gridpoints_gonze(
            dm_cmplx,
            ug_slice,
            &ga,
            &qd,
            fc_slice,
            svecs_slice,
            multi_slice,
            mass_slice,
            p2s_slice,
            s2p_slice,
            num_patom,
            num_satom,
            &gonze,
        );
    });

    Ok(())
}

/// Transform fc3 from real space to reciprocal space at a q-triplet.
///
/// Writes into ``fc3_reciprocal`` (complex128, shape
/// ``(num_patom, num_patom, num_patom, 3, 3, 3)``).  ``fc3`` is
/// float64 with shape ``(num_rows, num_satom, num_satom, 3, 3, 3)``
/// where ``num_rows`` is ``num_patom`` for compact fc3 or
/// ``num_satom`` for the full layout; ``is_compact_fc3`` selects the
/// variant.  ``q_vecs`` is the fractional q-triplet ``(3, 3)``.
/// ``all_shortest`` and ``nonzero_indices`` are int8 flag arrays.
///
/// This function handles a single q-triplet, so it always parallelises
/// over the ``(num_patom, num_patom, num_patom)`` atom triplet
/// (equivalent to ``num_triplets = 1 <= num_band``).
#[pyfunction]
#[pyo3(name = "real_to_reciprocal")]
#[allow(clippy::too_many_arguments)]
fn py_real_to_reciprocal<'py>(
    py: Python<'py>,
    mut fc3_reciprocal: PyReadwriteArray6<'py, Complex64>,
    q_vecs: PyReadonlyArray2<'py, f64>,
    fc3: PyReadonlyArray6<'py, f64>,
    is_compact_fc3: bool,
    svecs: PyReadonlyArray2<'py, f64>,
    multiplicity: PyReadonlyArray3<'py, i64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    make_r0_average: bool,
    all_shortest: PyReadonlyArray3<'py, i8>,
    nonzero_indices: PyReadonlyArray3<'py, i8>,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_rows = if is_compact_fc3 { num_patom } else { num_satom };

    if fc3_reciprocal.shape() != [num_patom, num_patom, num_patom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3_reciprocal must have shape (num_patom, num_patom, num_patom, 3, 3, 3)",
        ));
    }
    if fc3.shape() != [num_rows, num_satom, num_satom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_patom or num_satom, num_satom, num_satom, 3, 3, 3)",
        ));
    }
    if q_vecs.shape() != [3, 3] {
        return Err(PyValueError::new_err("q_vecs must have shape (3, 3)"));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multiplicity.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multiplicity must have shape (num_satom, num_patom, 2)",
        ));
    }
    if all_shortest.shape() != [num_patom, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "all_shortest must have shape (num_patom, num_satom, num_satom)",
        ));
    }
    if nonzero_indices.shape() != [num_rows, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "nonzero_indices must have shape matching fc3.shape[:3]",
        ));
    }

    let q_view = q_vecs.as_array();
    let q3 = [
        [q_view[[0, 0]], q_view[[0, 1]], q_view[[0, 2]]],
        [q_view[[1, 0]], q_view[[1, 1]], q_view[[1, 2]]],
        [q_view[[2, 0]], q_view[[2, 1]], q_view[[2, 2]]],
    ];

    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);

    let multi_view = multiplicity.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multiplicity must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);

    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;

    let all_short_view = all_shortest.as_array();
    let all_short_slice = all_short_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("all_shortest must be C-contiguous"))?;
    let nonzero_view = nonzero_indices.as_array();
    let nonzero_slice = nonzero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("nonzero_indices must be C-contiguous"))?;

    let fc3_view = fc3.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3 must be C-contiguous"))?;

    let rec_slice = fc3_reciprocal
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3_reciprocal must be C-contiguous"))?;
    let rec_cmplx = complex_as_cmplx_mut(rec_slice);

    py.allow_threads(|| {
        let atom_triplets = real_to_reciprocal::AtomTriplets {
            svecs: svecs_slice,
            num_satom,
            num_patom,
            multiplicity: multi_slice,
            p2s_map: p2s_slice,
            s2p_map: s2p_slice,
            make_r0_average,
            all_shortest: all_short_slice,
            nonzero_indices: nonzero_slice,
        };
        real_to_reciprocal::real_to_reciprocal(
            rec_cmplx,
            &q3,
            fc3_slice,
            is_compact_fc3,
            &atom_triplets,
            true,
        );
    });

    Ok(())
}

/// Compute `|<e0, fc3_reciprocal e1 e2>|^2 / (f0 * f1 * f2)` for each
/// `g_pos` entry.  See `reciprocal_to_normal::reciprocal_to_normal_squared`
/// for the full algorithm.
///
/// Shapes:
/// - ``fc3_normal_squared``: ``(n_out,)`` ``float64``.  Written at
///   ``g_pos[i, 3]`` positions.
/// - ``g_pos``: ``(num_g_pos, 4)`` ``int64``.
/// - ``fc3_reciprocal``: ``(num_patom, num_patom, num_patom, 3, 3, 3)``
///   ``complex128`` (atom-first layout matching
///   ``phono3py_rs.real_to_reciprocal``).
/// - ``freqs0``, ``freqs1``, ``freqs2``: ``(num_band,)`` ``float64``.
/// - ``eigvecs0``, ``eigvecs1``, ``eigvecs2``: ``(num_band, num_band)``
///   ``complex128`` row-major ``[component, band]`` (un-scaled).
/// - ``masses``: ``(num_patom,)`` ``float64``.
/// - ``band_indices``: ``(num_band0,)`` ``int64``.
///
/// This function handles a single q-triplet, so it always parallelises
/// internally (equivalent to ``num_triplets = 1 <= num_band``).
#[pyfunction]
#[pyo3(name = "reciprocal_to_normal_squared")]
#[allow(clippy::too_many_arguments)]
fn py_reciprocal_to_normal_squared<'py>(
    py: Python<'py>,
    mut fc3_normal_squared: PyReadwriteArray1<'py, f64>,
    g_pos: PyReadonlyArray2<'py, i64>,
    fc3_reciprocal: PyReadonlyArray6<'py, Complex64>,
    freqs0: PyReadonlyArray1<'py, f64>,
    freqs1: PyReadonlyArray1<'py, f64>,
    freqs2: PyReadonlyArray1<'py, f64>,
    eigvecs0: PyReadonlyArray2<'py, Complex64>,
    eigvecs1: PyReadonlyArray2<'py, Complex64>,
    eigvecs2: PyReadonlyArray2<'py, Complex64>,
    masses: PyReadonlyArray1<'py, f64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let num_patom = masses.shape()[0];
    let num_band = num_patom * 3;

    if fc3_reciprocal.shape() != [num_patom, num_patom, num_patom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3_reciprocal must have shape (num_patom, num_patom, num_patom, 3, 3, 3)",
        ));
    }
    if g_pos.shape().len() != 2 || g_pos.shape()[1] != 4 {
        return Err(PyValueError::new_err("g_pos must have shape (num_g_pos, 4)"));
    }
    for (name, f) in [("freqs0", &freqs0), ("freqs1", &freqs1), ("freqs2", &freqs2)] {
        if f.shape() != [num_band] {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (num_band,)"
            )));
        }
    }
    for (name, e) in [
        ("eigvecs0", &eigvecs0),
        ("eigvecs1", &eigvecs1),
        ("eigvecs2", &eigvecs2),
    ] {
        if e.shape() != [num_band, num_band] {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape (num_band, num_band)"
            )));
        }
    }

    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let g_pos_view = g_pos.as_array();
    let g_pos_flat = g_pos_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_pos must be C-contiguous"))?;
    let g_pos_slice: &[[i64; 4]] = group_as_array(g_pos_flat);

    let fc3_rec_view = fc3_reciprocal.as_array();
    let fc3_rec_flat = fc3_rec_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_reciprocal must be C-contiguous"))?;
    let fc3_rec_cmplx = complex_readonly_as_cmplx(fc3_rec_flat);

    let f0_view = freqs0.as_array();
    let f0_slice = f0_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("freqs0 must be C-contiguous"))?;
    let f1_view = freqs1.as_array();
    let f1_slice = f1_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("freqs1 must be C-contiguous"))?;
    let f2_view = freqs2.as_array();
    let f2_slice = f2_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("freqs2 must be C-contiguous"))?;

    let e0_view = eigvecs0.as_array();
    let e0_flat = e0_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigvecs0 must be C-contiguous"))?;
    let e0_cmplx = complex_readonly_as_cmplx(e0_flat);
    let e1_view = eigvecs1.as_array();
    let e1_flat = e1_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigvecs1 must be C-contiguous"))?;
    let e1_cmplx = complex_readonly_as_cmplx(e1_flat);
    let e2_view = eigvecs2.as_array();
    let e2_flat = e2_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigvecs2 must be C-contiguous"))?;
    let e2_cmplx = complex_readonly_as_cmplx(e2_flat);

    let masses_view = masses.as_array();
    let masses_slice = masses_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("masses must be C-contiguous"))?;

    let band_view = band_indices.as_array();
    let band_slice = band_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;

    let out_slice = fc3_normal_squared
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;

    py.allow_threads(|| {
        reciprocal_to_normal::reciprocal_to_normal_squared(
            out_slice,
            g_pos_slice,
            fc3_rec_cmplx,
            f0_slice,
            f1_slice,
            f2_slice,
            e0_cmplx,
            e1_cmplx,
            e2_cmplx,
            masses_slice,
            band_slice,
            num_patom,
            cutoff_frequency,
            true,
        );
    });

    Ok(())
}

fn relative_grid_address_3d(
    arr: &PyReadonlyArray3<i64>,
) -> PyResult<RelativeGridAddress> {
    let v = arr.as_array();
    if v.shape() != [24, 4, 3] {
        return Err(PyValueError::new_err(
            "relative_grid_address must have shape (24, 4, 3)",
        ));
    }
    let mut out: RelativeGridAddress = [[[0i64; 3]; 4]; 24];
    for i in 0..24 {
        for j in 0..4 {
            for k in 0..3 {
                out[i][j][k] = v[[i, j, k]];
            }
        }
    }
    Ok(out)
}

/// Imaginary-part of the bubble self-energy at a fixed temperature,
/// using pre-computed integration weights ``g``.  Mirrors
/// ``phono3py._phono3py.imag_self_energy_with_g``.
///
/// Shapes:
/// - ``imag_self_energy``: ``(num_band0,)`` ``float64``, write-only.
/// - ``fc3_normal_squared``: ``(num_triplets, num_band0, num_band, num_band)``
///   ``float64``.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``triplet_weights``: ``(num_triplets,)`` ``int64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``g``: ``(2, num_triplets, num_band0_or_freqpts, num_band, num_band)``
///   ``float64``; layer 0 is ``g1``, layer 1 is ``g2 - g3``.
/// - ``g_zero``: ``(num_triplets, num_band0_or_freqpts, num_band, num_band)``
///   ``byte``.
///
/// ``frequency_point_index`` is ``< 0`` for band-index mode; otherwise
/// the frequency point index to sample.
#[pyfunction]
#[pyo3(name = "imag_self_energy_with_g")]
#[allow(clippy::too_many_arguments)]
fn py_imag_self_energy_with_g<'py>(
    py: Python<'py>,
    mut imag_self_energy: PyReadwriteArray1<'py, f64>,
    fc3_normal_squared: PyReadonlyArray4<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplet_weights: PyReadonlyArray1<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    temperature_thz: f64,
    g: PyReadonlyArray5<'py, f64>,
    g_zero: PyReadonlyArray4<'py, i8>,
    cutoff_frequency: f64,
    frequency_point_index: i64,
) -> PyResult<()> {
    let fc3_shape = fc3_normal_squared.shape();
    if fc3_shape.len() != 4 {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let num_triplets = fc3_shape[0];
    let num_band0 = fc3_shape[1];
    let num_band = fc3_shape[2];
    if fc3_shape[3] != num_band {
        return Err(PyValueError::new_err(
            "fc3_normal_squared last two dims must be equal (num_band, num_band)",
        ));
    }

    if imag_self_energy.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "imag_self_energy must have shape (num_band0,)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    if triplet_weights.shape() != [num_triplets] {
        return Err(PyValueError::new_err(
            "triplet_weights must have shape (num_triplets,)",
        ));
    }
    if frequencies.shape().len() != 2 || frequencies.shape()[1] != num_band {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }

    let g_shape = g.shape();
    // Accept >=2 on axis 0: CollisionMatrix callers pass the (3, ...)
    // type-3 layout but the kernel only reads slabs 0 and 1.
    if g_shape.len() != 5 || g_shape[0] < 2 || g_shape[1] != num_triplets {
        return Err(PyValueError::new_err(
            "g must have shape (>=2, num_triplets, dim, num_band, num_band)",
        ));
    }
    let num_freq_points = g_shape[2];
    if g_shape[3] != num_band || g_shape[4] != num_band {
        return Err(PyValueError::new_err(
            "g last two dims must be (num_band, num_band)",
        ));
    }
    if g_zero.shape() != [num_triplets, num_freq_points, num_band, num_band] {
        return Err(PyValueError::new_err(
            "g_zero must have shape matching one layer of g",
        ));
    }
    if frequency_point_index < 0 && num_freq_points != num_band0 {
        return Err(PyValueError::new_err(
            "g third dim must equal num_band0 when frequency_point_index < 0",
        ));
    }
    if frequency_point_index >= 0 && frequency_point_index as usize >= num_freq_points {
        return Err(PyValueError::new_err(
            "frequency_point_index out of range",
        ));
    }

    let fc3_view = fc3_normal_squared.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;

    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);

    let weights_view = triplet_weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplet_weights must be C-contiguous"))?;

    let g_view = g.as_array();
    let g_slice = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be C-contiguous"))?;
    let g_zero_view = g_zero.as_array();
    let g_zero_slice = g_zero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_zero must be C-contiguous"))?;

    let ise_slice = imag_self_energy
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("imag_self_energy must be C-contiguous"))?;

    py.allow_threads(|| {
        imag_self_energy::get_imag_self_energy_with_g(
            ise_slice,
            fc3_slice,
            freqs_slice,
            triplets_slice,
            weights_slice,
            g_slice,
            g_zero_slice,
            temperature_thz,
            cutoff_frequency,
            num_freq_points as i64,
            frequency_point_index,
            num_band0,
            num_band,
        );
    });

    Ok(())
}

/// Ph-ph interaction strength in normal-mode coordinates, per triplet.
/// Mirrors ``phono3py._phono3py.interaction``.
///
/// Writes ``fc3_normal_squared`` of shape
/// ``(num_triplets, num_band0, num_band, num_band)`` ``float64``.
/// Input layouts follow the existing C binding:
///
/// - ``g_zero``: ``(num_triplets, num_band0, num_band, num_band)`` ``byte``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``eigenvectors``: ``(num_grid, num_band, num_band)`` ``complex128``
///   row-major ``[component, band]``.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``bz_grid_addresses``: ``(num_grid, 3)`` ``int64``.
/// - ``d_diag``: ``(3,)`` ``int64``.
/// - ``q_mat``: ``(3, 3)`` ``int64``.
/// - ``fc3``: ``(num_rows, num_satom, num_satom, 3, 3, 3)`` ``float64``
///   with ``num_rows = num_patom`` when ``is_compact_fc3`` else
///   ``num_satom``.
/// - ``fc3_nonzero_indices``: ``(num_rows, num_satom, num_satom)`` ``int8``.
/// - ``all_shortest``: ``(num_patom, num_satom, num_satom)`` ``int8``.
/// - ``svecs``: ``(n_svec, 3)`` ``float64``.
/// - ``multiplicity``: ``(num_satom, num_patom, 2)`` ``int64``.
/// - ``masses``: ``(num_patom,)`` ``float64``.
/// - ``p2s_map``: ``(num_patom,)`` ``int64``.
/// - ``s2p_map``: ``(num_satom,)`` ``int64``.
/// - ``band_indices``: ``(num_band0,)`` ``int64``.
#[pyfunction]
#[pyo3(name = "interaction")]
#[allow(clippy::too_many_arguments)]
fn py_interaction<'py>(
    py: Python<'py>,
    mut fc3_normal_squared: PyReadwriteArray4<'py, f64>,
    g_zero: PyReadonlyArray4<'py, i8>,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    triplets: PyReadonlyArray2<'py, i64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    q_mat: PyReadonlyArray2<'py, i64>,
    fc3: PyReadonlyArray6<'py, f64>,
    fc3_nonzero_indices: PyReadonlyArray3<'py, i8>,
    svecs: PyReadonlyArray2<'py, f64>,
    multiplicity: PyReadonlyArray3<'py, i64>,
    masses: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: PyReadonlyArray3<'py, i8>,
    cutoff_frequency: f64,
    is_compact_fc3: bool,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_rows = if is_compact_fc3 { num_patom } else { num_satom };
    let num_band = num_patom * 3;
    let num_band0 = band_indices.shape()[0];
    let num_triplets = triplets.shape()[0];
    let num_grid = bz_grid_addresses.shape()[0];

    if fc3_normal_squared.shape() != [num_triplets, num_band0, num_band, num_band] {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    if g_zero.shape() != [num_triplets, num_band0, num_band, num_band] {
        return Err(PyValueError::new_err(
            "g_zero must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    if frequencies.shape() != [num_grid, num_band] {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }
    if eigenvectors.shape() != [num_grid, num_band, num_band] {
        return Err(PyValueError::new_err(
            "eigenvectors must have shape (num_grid, num_band, num_band)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    if d_diag.shape() != [3] {
        return Err(PyValueError::new_err("d_diag must have shape (3,)"));
    }
    if q_mat.shape() != [3, 3] {
        return Err(PyValueError::new_err("q_mat must have shape (3, 3)"));
    }
    if fc3.shape() != [num_rows, num_satom, num_satom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_rows, num_satom, num_satom, 3, 3, 3)",
        ));
    }
    if fc3_nonzero_indices.shape() != [num_rows, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "fc3_nonzero_indices must match fc3.shape[:3]",
        ));
    }
    if all_shortest.shape() != [num_patom, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "all_shortest must have shape (num_patom, num_satom, num_satom)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multiplicity.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multiplicity must have shape (num_satom, num_patom, 2)",
        ));
    }
    if masses.shape() != [num_patom] {
        return Err(PyValueError::new_err(
            "masses must have shape (num_patom,)",
        ));
    }

    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let d_view = d_diag.as_array();
    let d3 = [d_view[0], d_view[1], d_view[2]];
    let q_view = q_mat.as_array();
    let q3 = [
        [q_view[[0, 0]], q_view[[0, 1]], q_view[[0, 2]]],
        [q_view[[1, 0]], q_view[[1, 1]], q_view[[1, 2]]],
        [q_view[[2, 0]], q_view[[2, 1]], q_view[[2, 2]]],
    ];

    let g_zero_view = g_zero.as_array();
    let g_zero_slice = g_zero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_zero must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let eig_cmplx = complex_readonly_as_cmplx(eig_flat);
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let addr_view = bz_grid_addresses.as_array();
    let addr_flat = addr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_grid_addresses must be C-contiguous"))?;
    let addr_slice: &[[i64; 3]] = group_as_array(addr_flat);
    let fc3_view = fc3.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3 must be C-contiguous"))?;
    let nonzero_view = fc3_nonzero_indices.as_array();
    let nonzero_slice = nonzero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_nonzero_indices must be C-contiguous"))?;
    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);
    let multi_view = multiplicity.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multiplicity must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);
    let masses_view = masses.as_array();
    let masses_slice = masses_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("masses must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;
    let band_view = band_indices.as_array();
    let band_slice = band_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let all_short_view = all_shortest.as_array();
    let all_short_slice = all_short_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("all_shortest must be C-contiguous"))?;

    let out_slice = fc3_normal_squared
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;

    py.allow_threads(|| {
        let atom_triplets = real_to_reciprocal::AtomTriplets {
            svecs: svecs_slice,
            num_satom,
            num_patom,
            multiplicity: multi_slice,
            p2s_map: p2s_slice,
            s2p_map: s2p_slice,
            make_r0_average,
            all_shortest: all_short_slice,
            nonzero_indices: nonzero_slice,
        };
        interaction::get_interaction(
            out_slice,
            g_zero_slice,
            freqs_slice,
            eig_cmplx,
            triplets_slice,
            addr_slice,
            d3,
            q3,
            fc3_slice,
            is_compact_fc3,
            &atom_triplets,
            masses_slice,
            band_slice,
            symmetrize_fc3_q,
            cutoff_frequency,
            num_band0,
            num_band,
        );
    });

    Ok(())
}

/// Detailed imaginary self-energy at a fixed temperature with Normal/
/// Umklapp splitting.  Mirrors
/// ``phono3py._phono3py.detailed_imag_self_energy_with_g``.
///
/// Shapes:
/// - ``detailed_imag_self_energy`` (out):
///   ``(num_triplets, num_band0, num_band, num_band)`` ``float64``.
/// - ``imag_self_energy_n`` / ``imag_self_energy_u`` (out):
///   ``(num_band0,)`` ``float64``.
/// - ``fc3_normal_squared``:
///   ``(num_triplets, num_band0, num_band, num_band)`` ``float64``.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``triplet_weights``: ``(num_triplets,)`` ``int64``.
/// - ``bz_grid_addresses``: ``(num_grid, 3)`` ``int64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``g``: ``(2, num_triplets, num_band0, num_band, num_band)`` ``float64``.
/// - ``g_zero``: ``(num_triplets, num_band0, num_band, num_band)`` ``byte``.
#[pyfunction]
#[pyo3(name = "detailed_imag_self_energy_with_g")]
#[allow(clippy::too_many_arguments)]
fn py_detailed_imag_self_energy_with_g<'py>(
    py: Python<'py>,
    mut detailed_imag_self_energy: PyReadwriteArray4<'py, f64>,
    mut imag_self_energy_n: PyReadwriteArray1<'py, f64>,
    mut imag_self_energy_u: PyReadwriteArray1<'py, f64>,
    fc3_normal_squared: PyReadonlyArray4<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplet_weights: PyReadonlyArray1<'py, i64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    temperature_thz: f64,
    g: PyReadonlyArray5<'py, f64>,
    g_zero: PyReadonlyArray4<'py, i8>,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let fc3_shape = fc3_normal_squared.shape();
    if fc3_shape.len() != 4 {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let num_triplets = fc3_shape[0];
    let num_band0 = fc3_shape[1];
    let num_band = fc3_shape[2];
    if fc3_shape[3] != num_band {
        return Err(PyValueError::new_err(
            "fc3_normal_squared last two dims must be equal (num_band, num_band)",
        ));
    }

    if detailed_imag_self_energy.shape() != [num_triplets, num_band0, num_band, num_band] {
        return Err(PyValueError::new_err(
            "detailed_imag_self_energy must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    if imag_self_energy_n.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "imag_self_energy_n must have shape (num_band0,)",
        ));
    }
    if imag_self_energy_u.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "imag_self_energy_u must have shape (num_band0,)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    if triplet_weights.shape() != [num_triplets] {
        return Err(PyValueError::new_err(
            "triplet_weights must have shape (num_triplets,)",
        ));
    }
    if bz_grid_addresses.shape().len() != 2 || bz_grid_addresses.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "bz_grid_addresses must have shape (num_grid, 3)",
        ));
    }
    if frequencies.shape().len() != 2 || frequencies.shape()[1] != num_band {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }
    let g_shape = g.shape();
    if g_shape.len() != 5
        || g_shape[0] != 2
        || g_shape[1] != num_triplets
        || g_shape[2] != num_band0
        || g_shape[3] != num_band
        || g_shape[4] != num_band
    {
        return Err(PyValueError::new_err(
            "g must have shape (2, num_triplets, num_band0, num_band, num_band)",
        ));
    }
    if g_zero.shape() != [num_triplets, num_band0, num_band, num_band] {
        return Err(PyValueError::new_err(
            "g_zero must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }

    let fc3_view = fc3_normal_squared.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let weights_view = triplet_weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplet_weights must be C-contiguous"))?;
    let addr_view = bz_grid_addresses.as_array();
    let addr_flat = addr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_grid_addresses must be C-contiguous"))?;
    let addr_slice: &[[i64; 3]] = group_as_array(addr_flat);
    let g_view = g.as_array();
    let g_slice = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be C-contiguous"))?;
    let g_zero_view = g_zero.as_array();
    let g_zero_slice = g_zero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g_zero must be C-contiguous"))?;

    let detailed_slice = detailed_imag_self_energy
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("detailed_imag_self_energy must be C-contiguous"))?;
    let ise_n_slice = imag_self_energy_n
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("imag_self_energy_n must be C-contiguous"))?;
    let ise_u_slice = imag_self_energy_u
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("imag_self_energy_u must be C-contiguous"))?;

    py.allow_threads(|| {
        imag_self_energy::get_detailed_imag_self_energy_with_g(
            detailed_slice,
            ise_n_slice,
            ise_u_slice,
            fc3_slice,
            freqs_slice,
            triplets_slice,
            weights_slice,
            addr_slice,
            g_slice,
            g_zero_slice,
            temperature_thz,
            cutoff_frequency,
            num_band0,
            num_band,
        );
    });

    Ok(())
}

/// Low-memory driver: tetrahedron-method gamma accumulation at a grid
/// point.  Mirrors ``phono3py._phono3py.pp_collision``.
///
/// Shapes:
/// - ``collisions`` (out): ``(num_temps, num_band0)`` ``float64`` when
///   ``is_n_u = False``, or ``(2, num_temps, num_band0)`` when ``True``
///   (layer 0 = Normal, layer 1 = Umklapp).
/// - ``relative_grid_address``: ``(24, 4, 3)`` ``int64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``eigenvectors``: ``(num_grid, num_band, num_band)`` ``complex128``.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``triplet_weights``: ``(num_triplets,)`` ``int64``.
/// - ``bz_grid_addresses``: ``(num_grid, 3)`` ``int64``.
/// - ``bz_map``: flat BZ map (``int64``); layout is selected by
///   ``bz_grid_type`` (1 = sparse, 2 = dense).
/// - ``d_diag``: ``(3,)`` ``int64``.
/// - ``q_mat``: ``(3, 3)`` ``int64``.
/// - ``fc3`` / ``fc3_nonzero_indices`` / ``svecs`` / ``multiplicity``
///   / ``masses`` / ``p2s_map`` / ``s2p_map`` / ``band_indices``
///   / ``all_shortest``: same layouts as in ``interaction``.
/// - ``temperatures_thz``: ``(num_temps,)`` ``float64``.
#[pyfunction]
#[pyo3(name = "pp_collision")]
#[allow(clippy::too_many_arguments)]
fn py_pp_collision<'py>(
    py: Python<'py>,
    mut collisions: PyReadwriteArrayDyn<'py, f64>,
    relative_grid_address: PyReadonlyArray3<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplet_weights: PyReadonlyArray1<'py, i64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
    d_diag: PyReadonlyArray1<'py, i64>,
    q_mat: PyReadonlyArray2<'py, i64>,
    fc3: PyReadonlyArray6<'py, f64>,
    fc3_nonzero_indices: PyReadonlyArray3<'py, i8>,
    svecs: PyReadonlyArray2<'py, f64>,
    multiplicity: PyReadonlyArray3<'py, i64>,
    masses: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    temperatures_thz: PyReadonlyArray1<'py, f64>,
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: PyReadonlyArray3<'py, i8>,
    cutoff_frequency: f64,
    is_compact_fc3: bool,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_rows = if is_compact_fc3 { num_patom } else { num_satom };
    let num_band = num_patom * 3;
    let num_band0 = band_indices.shape()[0];
    let num_temps = temperatures_thz.shape()[0];
    let expected_shape: &[usize] = if is_n_u {
        &[2, num_temps, num_band0]
    } else {
        &[num_temps, num_band0]
    };
    if collisions.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "collisions shape must be (num_temps, num_band0) or (2, num_temps, num_band0)",
        ));
    }
    let rga = relative_grid_address_3d(&relative_grid_address)?;
    let d3 = vec3_i(&d_diag)?;
    let q3 = mat3_i(&q_mat)?;

    if fc3.shape() != [num_rows, num_satom, num_satom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_rows, num_satom, num_satom, 3, 3, 3)",
        ));
    }
    if fc3_nonzero_indices.shape() != [num_rows, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "fc3_nonzero_indices must match fc3.shape[:3]",
        ));
    }
    if all_shortest.shape() != [num_patom, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "all_shortest must have shape (num_patom, num_satom, num_satom)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multiplicity.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multiplicity must have shape (num_satom, num_patom, 2)",
        ));
    }
    if masses.shape() != [num_patom] {
        return Err(PyValueError::new_err("masses must have shape (num_patom,)"));
    }

    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let eig_cmplx = complex_readonly_as_cmplx(eig_flat);
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let weights_view = triplet_weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplet_weights must be C-contiguous"))?;
    let addr_view = bz_grid_addresses.as_array();
    let addr_flat = addr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_grid_addresses must be C-contiguous"))?;
    let addr_slice: &[[i64; 3]] = group_as_array(addr_flat);
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be C-contiguous"))?;
    let fc3_view = fc3.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3 must be C-contiguous"))?;
    let nonzero_view = fc3_nonzero_indices.as_array();
    let nonzero_slice = nonzero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_nonzero_indices must be C-contiguous"))?;
    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);
    let multi_view = multiplicity.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multiplicity must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);
    let masses_view = masses.as_array();
    let masses_slice = masses_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("masses must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;
    let band_view = band_indices.as_array();
    let band_slice = band_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let all_short_view = all_shortest.as_array();
    let all_short_slice = all_short_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("all_shortest must be C-contiguous"))?;
    let temps_view = temperatures_thz.as_array();
    let temps_slice = temps_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("temperatures_thz must be C-contiguous"))?;

    let out_slice = collisions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collisions must be C-contiguous"))?;

    py.allow_threads(|| {
        let atom_triplets = real_to_reciprocal::AtomTriplets {
            svecs: svecs_slice,
            num_satom,
            num_patom,
            multiplicity: multi_slice,
            p2s_map: p2s_slice,
            s2p_map: s2p_slice,
            make_r0_average,
            all_shortest: all_short_slice,
            nonzero_indices: nonzero_slice,
        };
        let bzgrid = BzGridView {
            d_diag: d3,
            addresses: addr_slice,
            gp_map: bzmap_slice,
            bz_grid_type,
        };
        pp_collision::get_pp_collision(
            out_slice,
            &rga,
            freqs_slice,
            eig_cmplx,
            triplets_slice,
            weights_slice,
            &bzgrid,
            d3,
            q3,
            fc3_slice,
            is_compact_fc3,
            &atom_triplets,
            masses_slice,
            band_slice,
            temps_slice,
            is_n_u,
            symmetrize_fc3_q,
            cutoff_frequency,
            num_band0,
            num_band,
        )
    })
    .map_err(|e| match e {
        BzGridError::BadGridType => PyValueError::new_err("bz_grid_type must be 1 or 2"),
        BzGridError::BadTpType => PyValueError::new_err("tp_type must be 2, 3, or 4"),
    })?;
    Ok(())
}

/// Low-memory driver: Gaussian-smearing gamma accumulation at a grid
/// point.  Mirrors ``phono3py._phono3py.pp_collision_with_sigma``.
///
/// ``sigma_cutoff <= 0`` disables the cutoff-skip optimisation (matches
/// C semantics).  Shape conventions otherwise mirror ``pp_collision``.
#[pyfunction]
#[pyo3(name = "pp_collision_with_sigma")]
#[allow(clippy::too_many_arguments)]
fn py_pp_collision_with_sigma<'py>(
    py: Python<'py>,
    mut collisions: PyReadwriteArrayDyn<'py, f64>,
    sigma: f64,
    sigma_cutoff: f64,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplet_weights: PyReadonlyArray1<'py, i64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    q_mat: PyReadonlyArray2<'py, i64>,
    fc3: PyReadonlyArray6<'py, f64>,
    fc3_nonzero_indices: PyReadonlyArray3<'py, i8>,
    svecs: PyReadonlyArray2<'py, f64>,
    multiplicity: PyReadonlyArray3<'py, i64>,
    masses: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    temperatures_thz: PyReadonlyArray1<'py, f64>,
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: PyReadonlyArray3<'py, i8>,
    cutoff_frequency: f64,
    is_compact_fc3: bool,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_rows = if is_compact_fc3 { num_patom } else { num_satom };
    let num_band = num_patom * 3;
    let num_band0 = band_indices.shape()[0];
    let num_temps = temperatures_thz.shape()[0];
    let expected_shape: &[usize] = if is_n_u {
        &[2, num_temps, num_band0]
    } else {
        &[num_temps, num_band0]
    };
    if collisions.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "collisions shape must be (num_temps, num_band0) or (2, num_temps, num_band0)",
        ));
    }
    let d3 = vec3_i(&d_diag)?;
    let q3 = mat3_i(&q_mat)?;

    if fc3.shape() != [num_rows, num_satom, num_satom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_rows, num_satom, num_satom, 3, 3, 3)",
        ));
    }
    if fc3_nonzero_indices.shape() != [num_rows, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "fc3_nonzero_indices must match fc3.shape[:3]",
        ));
    }
    if all_shortest.shape() != [num_patom, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "all_shortest must have shape (num_patom, num_satom, num_satom)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multiplicity.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multiplicity must have shape (num_satom, num_patom, 2)",
        ));
    }
    if masses.shape() != [num_patom] {
        return Err(PyValueError::new_err("masses must have shape (num_patom,)"));
    }

    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let eig_cmplx = complex_readonly_as_cmplx(eig_flat);
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let weights_view = triplet_weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplet_weights must be C-contiguous"))?;
    let addr_view = bz_grid_addresses.as_array();
    let addr_flat = addr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_grid_addresses must be C-contiguous"))?;
    let addr_slice: &[[i64; 3]] = group_as_array(addr_flat);
    let fc3_view = fc3.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3 must be C-contiguous"))?;
    let nonzero_view = fc3_nonzero_indices.as_array();
    let nonzero_slice = nonzero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_nonzero_indices must be C-contiguous"))?;
    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);
    let multi_view = multiplicity.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multiplicity must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);
    let masses_view = masses.as_array();
    let masses_slice = masses_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("masses must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;
    let band_view = band_indices.as_array();
    let band_slice = band_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let all_short_view = all_shortest.as_array();
    let all_short_slice = all_short_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("all_shortest must be C-contiguous"))?;
    let temps_view = temperatures_thz.as_array();
    let temps_slice = temps_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("temperatures_thz must be C-contiguous"))?;

    let out_slice = collisions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collisions must be C-contiguous"))?;

    py.allow_threads(|| {
        let atom_triplets = real_to_reciprocal::AtomTriplets {
            svecs: svecs_slice,
            num_satom,
            num_patom,
            multiplicity: multi_slice,
            p2s_map: p2s_slice,
            s2p_map: s2p_slice,
            make_r0_average,
            all_shortest: all_short_slice,
            nonzero_indices: nonzero_slice,
        };
        pp_collision::get_pp_collision_with_sigma(
            out_slice,
            sigma,
            sigma_cutoff,
            freqs_slice,
            eig_cmplx,
            triplets_slice,
            weights_slice,
            addr_slice,
            d3,
            q3,
            fc3_slice,
            is_compact_fc3,
            &atom_triplets,
            masses_slice,
            band_slice,
            temps_slice,
            is_n_u,
            symmetrize_fc3_q,
            cutoff_frequency,
            num_band0,
            num_band,
        );
    });
    Ok(())
}

/// Low-memory gamma driver for one grid point with multiple sigmas.
///
/// Folds the Python-side per-gp flow (set_grid_point +
/// get_triplets_at_q + per-sigma pp_collision / pp_collision_with_sigma
/// loop) into a single call so that rayon workers stay warm across
/// sigma iterations and the GIL is released only once per gp.
///
/// Assumes phonons are pre-computed (``Interaction.phonon_all_done()``
/// is True) and no ``q_direction`` Gamma special handling is needed.
///
/// Shapes:
/// - ``collisions`` (out): ``(num_sigma, num_temps, num_band0)``
///   ``float64`` when ``is_n_u = False``, or
///   ``(num_sigma, 2, num_temps, num_band0)`` when ``True``.
/// - ``sigmas``: ``(num_sigma,)`` ``float64``.  NaN selects the
///   tetrahedron-method path for that slot; any finite value selects
///   Gaussian smearing with width ``sigmas[i]``.
/// - ``sigma_cutoffs``: ``(num_sigma,)`` ``float64``.  ``<= 0``
///   disables the cutoff-skip optimisation (matches C semantics).
///   Ignored for tetrahedron slots.
/// - ``grid_point``: BZ grid point index.
/// - ``bzg2grg``: ``(num_bz_gp,)`` ``int64``.  Maps a BZ grid point to
///   its generalized-regular grid point.
/// - ``reciprocal_rotations``: ``(num_rot, 3, 3)`` ``int64``.
/// - ``reciprocal_lattice``: ``(3, 3)`` ``float64``.  Use the reduced
///   reciprocal basis (caller-side ``get_reduced_bases_and_tmat_inv``).
/// - ``bz_triplets_q_mat``: ``(3, 3)`` ``int64``.  Use
///   ``tmat_inv_int @ bz_grid.Q`` (caller-side).
/// - ``q_mat``: ``(3, 3)`` ``int64``.  Use ``bz_grid.Q``.
/// - Remaining arrays mirror ``pp_collision`` layouts.
#[pyfunction]
#[pyo3(name = "collision_at_grid_point")]
#[allow(clippy::too_many_arguments)]
fn py_collision_at_grid_point<'py>(
    py: Python<'py>,
    mut collisions: PyReadwriteArrayDyn<'py, f64>,
    grid_point: i64,
    sigmas: PyReadonlyArray1<'py, f64>,
    sigma_cutoffs: PyReadonlyArray1<'py, f64>,
    relative_grid_address: PyReadonlyArray3<'py, i64>,
    bzg2grg: PyReadonlyArray1<'py, i64>,
    reciprocal_rotations: PyReadonlyArray3<'py, i64>,
    is_time_reversal: bool,
    swappable: bool,
    is_mesh_symmetry: bool,
    reciprocal_lattice: PyReadonlyArray2<'py, f64>,
    bz_triplets_q_mat: PyReadonlyArray2<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
    d_diag: PyReadonlyArray1<'py, i64>,
    q_mat: PyReadonlyArray2<'py, i64>,
    fc3: PyReadonlyArray6<'py, f64>,
    fc3_nonzero_indices: PyReadonlyArray3<'py, i8>,
    svecs: PyReadonlyArray2<'py, f64>,
    multiplicity: PyReadonlyArray3<'py, i64>,
    masses: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    temperatures_thz: PyReadonlyArray1<'py, f64>,
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: PyReadonlyArray3<'py, i8>,
    cutoff_frequency: f64,
    is_compact_fc3: bool,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_rows = if is_compact_fc3 { num_patom } else { num_satom };
    let num_band = num_patom * 3;
    let num_band0 = band_indices.shape()[0];
    let num_temps = temperatures_thz.shape()[0];
    let num_sigma = sigmas.shape()[0];
    if sigma_cutoffs.shape()[0] != num_sigma {
        return Err(PyValueError::new_err(
            "sigmas and sigma_cutoffs must have the same length",
        ));
    }
    let expected_shape: &[usize] = if is_n_u {
        &[num_sigma, 2, num_temps, num_band0]
    } else {
        &[num_sigma, num_temps, num_band0]
    };
    if collisions.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "collisions shape must be (num_sigma, num_temps, num_band0) or \
             (num_sigma, 2, num_temps, num_band0)",
        ));
    }
    let rga = relative_grid_address_3d(&relative_grid_address)?;
    let d3 = vec3_i(&d_diag)?;
    let q3 = mat3_i(&q_mat)?;
    let bz_trip_q = mat3_i(&bz_triplets_q_mat)?;
    let rec_lat = mat3_f(&reciprocal_lattice)?;

    if fc3.shape() != [num_rows, num_satom, num_satom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_rows, num_satom, num_satom, 3, 3, 3)",
        ));
    }
    if fc3_nonzero_indices.shape() != [num_rows, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "fc3_nonzero_indices must match fc3.shape[:3]",
        ));
    }
    if all_shortest.shape() != [num_patom, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "all_shortest must have shape (num_patom, num_satom, num_satom)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multiplicity.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multiplicity must have shape (num_satom, num_patom, 2)",
        ));
    }
    if masses.shape() != [num_patom] {
        return Err(PyValueError::new_err("masses must have shape (num_patom,)"));
    }

    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let eig_cmplx = complex_readonly_as_cmplx(eig_flat);
    let addr_view = bz_grid_addresses.as_array();
    let addr_flat = addr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_grid_addresses must be C-contiguous"))?;
    let addr_slice: &[[i64; 3]] = group_as_array(addr_flat);
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be C-contiguous"))?;
    let bzg2grg_view = bzg2grg.as_array();
    let bzg2grg_slice = bzg2grg_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bzg2grg must be C-contiguous"))?;
    let fc3_view = fc3.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3 must be C-contiguous"))?;
    let nonzero_view = fc3_nonzero_indices.as_array();
    let nonzero_slice = nonzero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_nonzero_indices must be C-contiguous"))?;
    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);
    let multi_view = multiplicity.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multiplicity must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);
    let masses_view = masses.as_array();
    let masses_slice = masses_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("masses must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;
    let band_view = band_indices.as_array();
    let band_slice = band_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let all_short_view = all_shortest.as_array();
    let all_short_slice = all_short_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("all_shortest must be C-contiguous"))?;
    let temps_view = temperatures_thz.as_array();
    let temps_slice = temps_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("temperatures_thz must be C-contiguous"))?;
    let sigmas_view = sigmas.as_array();
    let sigmas_slice = sigmas_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigmas must be C-contiguous"))?;
    let sigma_cutoffs_view = sigma_cutoffs.as_array();
    let sigma_cutoffs_slice = sigma_cutoffs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_cutoffs must be C-contiguous"))?;

    // Triplet enumeration: ir_triplets_at_q (when mesh symmetry is in
    // effect) + bz_triplets_at_q, replicating get_triplets_at_q and
    // get_nosym_triplets_at_q on the Python side.
    if grid_point < 0 || (grid_point as usize) >= bzg2grg_slice.len() {
        return Err(PyValueError::new_err(
            "grid_point out of range for bzg2grg",
        ));
    }
    let n_grg = (d3[0] as i64) * (d3[1] as i64) * (d3[2] as i64);
    if n_grg < 0 {
        return Err(PyValueError::new_err("d_diag must be positive"));
    }
    let n_grg_usize = n_grg as usize;

    let map_triplets: Vec<i64> = if is_mesh_symmetry {
        let rots = rots_i(&reciprocal_rotations)?;
        let gr_gp = bzg2grg_slice[grid_point as usize];
        match triplet_grid::ir_triplets_at_q(gr_gp, d3, &rots, is_time_reversal, swappable) {
            Ok(r) => r.map_triplets,
            Err(ReciprocalRotationsError::TooManyRotations) => {
                return Err(PyValueError::new_err(
                    "collision_at_grid_point: more than 48 unique rotations",
                ));
            }
            Err(ReciprocalRotationsError::TooManyForInversion) => {
                return Err(PyValueError::new_err(
                    "collision_at_grid_point: cannot add inversion (count exceeds 24)",
                ));
            }
        }
    } else {
        (0..n_grg).collect()
    };

    if map_triplets.len() != n_grg_usize {
        return Err(PyRuntimeError::new_err(
            "map_triplets length inconsistent with d_diag",
        ));
    }

    let mut weights_dense = vec![0i64; n_grg_usize];
    for &g in &map_triplets {
        if g < 0 || (g as usize) >= n_grg_usize {
            return Err(PyRuntimeError::new_err("map_triplets out of range"));
        }
        weights_dense[g as usize] += 1;
    }
    let ir_weights: Vec<i64> = weights_dense.into_iter().filter(|&w| w > 0).collect();

    let triplets_vec = match triplet_grid::bz_triplets_at_q(
        grid_point,
        addr_slice,
        bzmap_slice,
        &map_triplets,
        d3,
        bz_trip_q,
        rec_lat,
        bz_grid_type,
    ) {
        Ok(v) => v,
        Err(BzTripletsError::BadGridType) => {
            return Err(PyValueError::new_err("bz_grid_type must be 1 or 2"));
        }
    };
    if triplets_vec.len() != ir_weights.len() {
        return Err(PyRuntimeError::new_err(
            "triplets length does not match ir_weights length",
        ));
    }
    let triplets_slice: &[[i64; 3]] = &triplets_vec;
    let weights_slice: &[i64] = &ir_weights;

    // Output buffer split into per-sigma chunks.
    let out_slice = collisions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collisions must be C-contiguous"))?;
    let per_sigma_len = if is_n_u {
        2 * num_temps * num_band0
    } else {
        num_temps * num_band0
    };
    if out_slice.len() != num_sigma * per_sigma_len {
        return Err(PyRuntimeError::new_err(
            "collisions buffer length inconsistent with shape",
        ));
    }

    py.allow_threads(|| -> Result<(), BzGridError> {
        let atom_triplets = real_to_reciprocal::AtomTriplets {
            svecs: svecs_slice,
            num_satom,
            num_patom,
            multiplicity: multi_slice,
            p2s_map: p2s_slice,
            s2p_map: s2p_slice,
            make_r0_average,
            all_shortest: all_short_slice,
            nonzero_indices: nonzero_slice,
        };
        let bzgrid = BzGridView {
            d_diag: d3,
            addresses: addr_slice,
            gp_map: bzmap_slice,
            bz_grid_type,
        };
        for (i_sigma, chunk) in out_slice.chunks_mut(per_sigma_len).enumerate() {
            let sigma = sigmas_slice[i_sigma];
            if sigma.is_nan() {
                pp_collision::get_pp_collision(
                    chunk,
                    &rga,
                    freqs_slice,
                    eig_cmplx,
                    triplets_slice,
                    weights_slice,
                    &bzgrid,
                    d3,
                    q3,
                    fc3_slice,
                    is_compact_fc3,
                    &atom_triplets,
                    masses_slice,
                    band_slice,
                    temps_slice,
                    is_n_u,
                    symmetrize_fc3_q,
                    cutoff_frequency,
                    num_band0,
                    num_band,
                )?;
            } else {
                pp_collision::get_pp_collision_with_sigma(
                    chunk,
                    sigma,
                    sigma_cutoffs_slice[i_sigma],
                    freqs_slice,
                    eig_cmplx,
                    triplets_slice,
                    weights_slice,
                    addr_slice,
                    d3,
                    q3,
                    fc3_slice,
                    is_compact_fc3,
                    &atom_triplets,
                    masses_slice,
                    band_slice,
                    temps_slice,
                    is_n_u,
                    symmetrize_fc3_q,
                    cutoff_frequency,
                    num_band0,
                    num_band,
                );
            }
        }
        Ok(())
    })
    .map_err(|e| match e {
        BzGridError::BadGridType => PyValueError::new_err("bz_grid_type must be 1 or 2"),
        BzGridError::BadTpType => PyValueError::new_err("tp_type must be 2, 3, or 4"),
    })?;
    Ok(())
}

/// Batched low-memory gamma driver over a **list of grid points**.
///
/// Rationale: per-gp ``collision_at_grid_point`` has triplet counts
/// (~50-120 for NaMgF3 20-atom) that are smaller than the rayon
/// thread count (128).  The nested-par work-stealing that kept
/// threads busy still costs ~30% ``do_spin`` in single-gp runs.
/// Batching several gps into one flat ``(gp, triplet)`` rayon par
/// saturates the pool with ``inner_par=false``, eliminating the
/// nested coordination cost.
///
/// Shapes (additions vs ``collision_at_grid_point``):
/// - ``collisions``: ``(num_gp_batch, num_sigma, num_temps, num_band0)``
///   or ``(num_gp_batch, num_sigma, 2, num_temps, num_band0)``.
/// - ``grid_points``: ``(num_gp_batch,)`` ``int64``.  BZ grid point
///   indices, one per batch slot.
///
/// All other inputs match ``collision_at_grid_point``.
#[pyfunction]
#[pyo3(name = "collision_at_grid_points_batched")]
#[allow(clippy::too_many_arguments)]
fn py_collision_at_grid_points_batched<'py>(
    py: Python<'py>,
    mut collisions: PyReadwriteArrayDyn<'py, f64>,
    grid_points: PyReadonlyArray1<'py, i64>,
    sigmas: PyReadonlyArray1<'py, f64>,
    sigma_cutoffs: PyReadonlyArray1<'py, f64>,
    relative_grid_address: PyReadonlyArray3<'py, i64>,
    bzg2grg: PyReadonlyArray1<'py, i64>,
    reciprocal_rotations: PyReadonlyArray3<'py, i64>,
    is_time_reversal: bool,
    swappable: bool,
    is_mesh_symmetry: bool,
    reciprocal_lattice: PyReadonlyArray2<'py, f64>,
    bz_triplets_q_mat: PyReadonlyArray2<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
    d_diag: PyReadonlyArray1<'py, i64>,
    q_mat: PyReadonlyArray2<'py, i64>,
    fc3: PyReadonlyArray6<'py, f64>,
    fc3_nonzero_indices: PyReadonlyArray3<'py, i8>,
    svecs: PyReadonlyArray2<'py, f64>,
    multiplicity: PyReadonlyArray3<'py, i64>,
    masses: PyReadonlyArray1<'py, f64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    s2p_map: PyReadonlyArray1<'py, i64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    temperatures_thz: PyReadonlyArray1<'py, f64>,
    is_n_u: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: PyReadonlyArray3<'py, i8>,
    cutoff_frequency: f64,
    is_compact_fc3: bool,
) -> PyResult<()> {
    let num_patom = p2s_map.shape()[0];
    let num_satom = s2p_map.shape()[0];
    let num_rows = if is_compact_fc3 { num_patom } else { num_satom };
    let num_band = num_patom * 3;
    let num_band0 = band_indices.shape()[0];
    let num_temps = temperatures_thz.shape()[0];
    let num_sigma = sigmas.shape()[0];
    let num_gp_batch = grid_points.shape()[0];
    if sigma_cutoffs.shape()[0] != num_sigma {
        return Err(PyValueError::new_err(
            "sigmas and sigma_cutoffs must have the same length",
        ));
    }
    let expected_shape: &[usize] = if is_n_u {
        &[num_gp_batch, num_sigma, 2, num_temps, num_band0]
    } else {
        &[num_gp_batch, num_sigma, num_temps, num_band0]
    };
    if collisions.shape() != expected_shape {
        return Err(PyValueError::new_err(
            "collisions shape must be (num_gp_batch, num_sigma, num_temps, num_band0) \
             or (num_gp_batch, num_sigma, 2, num_temps, num_band0)",
        ));
    }
    let rga = relative_grid_address_3d(&relative_grid_address)?;
    let d3 = vec3_i(&d_diag)?;
    let q3 = mat3_i(&q_mat)?;
    let bz_trip_q = mat3_i(&bz_triplets_q_mat)?;
    let rec_lat = mat3_f(&reciprocal_lattice)?;

    if fc3.shape() != [num_rows, num_satom, num_satom, 3, 3, 3] {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_rows, num_satom, num_satom, 3, 3, 3)",
        ));
    }
    if fc3_nonzero_indices.shape() != [num_rows, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "fc3_nonzero_indices must match fc3.shape[:3]",
        ));
    }
    if all_shortest.shape() != [num_patom, num_satom, num_satom] {
        return Err(PyValueError::new_err(
            "all_shortest must have shape (num_patom, num_satom, num_satom)",
        ));
    }
    if svecs.shape().len() != 2 || svecs.shape()[1] != 3 {
        return Err(PyValueError::new_err("svecs must have shape (n, 3)"));
    }
    if multiplicity.shape() != [num_satom, num_patom, 2] {
        return Err(PyValueError::new_err(
            "multiplicity must have shape (num_satom, num_patom, 2)",
        ));
    }
    if masses.shape() != [num_patom] {
        return Err(PyValueError::new_err("masses must have shape (num_patom,)"));
    }

    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };

    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let eig_cmplx = complex_readonly_as_cmplx(eig_flat);
    let addr_view = bz_grid_addresses.as_array();
    let addr_flat = addr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_grid_addresses must be C-contiguous"))?;
    let addr_slice: &[[i64; 3]] = group_as_array(addr_flat);
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be C-contiguous"))?;
    let bzg2grg_view = bzg2grg.as_array();
    let bzg2grg_slice = bzg2grg_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bzg2grg must be C-contiguous"))?;
    let fc3_view = fc3.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3 must be C-contiguous"))?;
    let nonzero_view = fc3_nonzero_indices.as_array();
    let nonzero_slice = nonzero_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_nonzero_indices must be C-contiguous"))?;
    let svecs_view = svecs.as_array();
    let svecs_flat = svecs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("svecs must be C-contiguous"))?;
    let svecs_slice: &[[f64; 3]] = group_as_array(svecs_flat);
    let multi_view = multiplicity.as_array();
    let multi_flat = multi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("multiplicity must be C-contiguous"))?;
    let multi_slice: &[[i64; 2]] = group_as_array(multi_flat);
    let masses_view = masses.as_array();
    let masses_slice = masses_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("masses must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let s2p_view = s2p_map.as_array();
    let s2p_slice = s2p_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2p_map must be C-contiguous"))?;
    let band_view = band_indices.as_array();
    let band_slice = band_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let all_short_view = all_shortest.as_array();
    let all_short_slice = all_short_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("all_shortest must be C-contiguous"))?;
    let temps_view = temperatures_thz.as_array();
    let temps_slice = temps_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("temperatures_thz must be C-contiguous"))?;
    let sigmas_view = sigmas.as_array();
    let sigmas_slice = sigmas_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigmas must be C-contiguous"))?;
    let sigma_cutoffs_view = sigma_cutoffs.as_array();
    let sigma_cutoffs_slice = sigma_cutoffs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_cutoffs must be C-contiguous"))?;
    let grid_points_view = grid_points.as_array();
    let grid_points_slice = grid_points_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_points must be C-contiguous"))?;

    // Per-gp triplet enumeration: sequential, mirrors the single-gp path.
    let n_grg = (d3[0] as i64) * (d3[1] as i64) * (d3[2] as i64);
    if n_grg < 0 {
        return Err(PyValueError::new_err("d_diag must be positive"));
    }
    let n_grg_usize = n_grg as usize;
    let rots = if is_mesh_symmetry {
        Some(rots_i(&reciprocal_rotations)?)
    } else {
        None
    };

    let mut triplets_owned: Vec<Vec<[i64; 3]>> = Vec::with_capacity(num_gp_batch);
    let mut weights_owned: Vec<Vec<i64>> = Vec::with_capacity(num_gp_batch);
    for &gp in grid_points_slice {
        if gp < 0 || (gp as usize) >= bzg2grg_slice.len() {
            return Err(PyValueError::new_err("grid_point out of range for bzg2grg"));
        }
        let map_triplets: Vec<i64> = if let Some(rots) = rots.as_ref() {
            let gr_gp = bzg2grg_slice[gp as usize];
            match triplet_grid::ir_triplets_at_q(gr_gp, d3, rots, is_time_reversal, swappable) {
                Ok(r) => r.map_triplets,
                Err(ReciprocalRotationsError::TooManyRotations) => {
                    return Err(PyValueError::new_err(
                        "collision_at_grid_points_batched: more than 48 unique rotations",
                    ));
                }
                Err(ReciprocalRotationsError::TooManyForInversion) => {
                    return Err(PyValueError::new_err(
                        "collision_at_grid_points_batched: cannot add inversion (count exceeds 24)",
                    ));
                }
            }
        } else {
            (0..n_grg).collect()
        };
        if map_triplets.len() != n_grg_usize {
            return Err(PyRuntimeError::new_err(
                "map_triplets length inconsistent with d_diag",
            ));
        }
        let mut weights_dense = vec![0i64; n_grg_usize];
        for &g in &map_triplets {
            if g < 0 || (g as usize) >= n_grg_usize {
                return Err(PyRuntimeError::new_err("map_triplets out of range"));
            }
            weights_dense[g as usize] += 1;
        }
        let ir_weights: Vec<i64> = weights_dense.into_iter().filter(|&w| w > 0).collect();
        let triplets_vec = match triplet_grid::bz_triplets_at_q(
            gp,
            addr_slice,
            bzmap_slice,
            &map_triplets,
            d3,
            bz_trip_q,
            rec_lat,
            bz_grid_type,
        ) {
            Ok(v) => v,
            Err(BzTripletsError::BadGridType) => {
                return Err(PyValueError::new_err("bz_grid_type must be 1 or 2"));
            }
        };
        if triplets_vec.len() != ir_weights.len() {
            return Err(PyRuntimeError::new_err(
                "triplets length does not match ir_weights length",
            ));
        }
        triplets_owned.push(triplets_vec);
        weights_owned.push(ir_weights);
    }

    let triplets_per_gp: Vec<&[[i64; 3]]> =
        triplets_owned.iter().map(|v| v.as_slice()).collect();
    let weights_per_gp: Vec<&[i64]> = weights_owned.iter().map(|v| v.as_slice()).collect();

    let out_slice = collisions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collisions must be C-contiguous"))?;
    let per_sigma_len = if is_n_u {
        2 * num_temps * num_band0
    } else {
        num_temps * num_band0
    };
    let per_gp_len = num_sigma * per_sigma_len;
    if out_slice.len() != num_gp_batch * per_gp_len {
        return Err(PyRuntimeError::new_err(
            "collisions buffer length inconsistent with shape",
        ));
    }

    py.allow_threads(|| -> Result<(), BzGridError> {
        let atom_triplets = real_to_reciprocal::AtomTriplets {
            svecs: svecs_slice,
            num_satom,
            num_patom,
            multiplicity: multi_slice,
            p2s_map: p2s_slice,
            s2p_map: s2p_slice,
            make_r0_average,
            all_shortest: all_short_slice,
            nonzero_indices: nonzero_slice,
        };
        let bzgrid = BzGridView {
            d_diag: d3,
            addresses: addr_slice,
            gp_map: bzmap_slice,
            bz_grid_type,
        };
        // For each sigma, build per-gp mutable slots by slicing each
        // gp's per_gp_len chunk at the i_sigma offset.  The sigma_slots
        // Vec drops at end of iteration, so out_slice is available for
        // the next sigma pass.
        for i_sigma in 0..num_sigma {
            let sigma = sigmas_slice[i_sigma];
            let sigma_off = i_sigma * per_sigma_len;
            let mut sigma_slots: Vec<&mut [f64]> = out_slice
                .chunks_mut(per_gp_len)
                .map(|chunk| &mut chunk[sigma_off..sigma_off + per_sigma_len])
                .collect();
            if sigma.is_nan() {
                pp_collision::get_pp_collision_multi_gp(
                    sigma_slots.as_mut_slice(),
                    &triplets_per_gp,
                    &weights_per_gp,
                    &rga,
                    freqs_slice,
                    eig_cmplx,
                    &bzgrid,
                    d3,
                    q3,
                    fc3_slice,
                    is_compact_fc3,
                    &atom_triplets,
                    masses_slice,
                    band_slice,
                    temps_slice,
                    is_n_u,
                    symmetrize_fc3_q,
                    cutoff_frequency,
                    num_band0,
                    num_band,
                )?;
            } else {
                pp_collision::get_pp_collision_with_sigma_multi_gp(
                    sigma_slots.as_mut_slice(),
                    sigma,
                    sigma_cutoffs_slice[i_sigma],
                    &triplets_per_gp,
                    &weights_per_gp,
                    freqs_slice,
                    eig_cmplx,
                    addr_slice,
                    d3,
                    q3,
                    fc3_slice,
                    is_compact_fc3,
                    &atom_triplets,
                    masses_slice,
                    band_slice,
                    temps_slice,
                    is_n_u,
                    symmetrize_fc3_q,
                    cutoff_frequency,
                    num_band0,
                    num_band,
                );
            }
        }
        Ok(())
    })
    .map_err(|e| match e {
        BzGridError::BadGridType => PyValueError::new_err("bz_grid_type must be 1 or 2"),
        BzGridError::BadTpType => PyValueError::new_err("tp_type must be 2, 3, or 4"),
    })?;
    Ok(())
}

/// Gaussian-smearing isotope scattering strength.  Mirrors
/// ``phono3py._phono3py.isotope_strength``.
///
/// Shapes:
/// - ``gamma``: ``(num_band0,)`` ``float64``, write-only.
/// - ``ir_grid_points``: ``(num_grid_points,)`` ``int64``.
/// - ``weights``: ``(num_bz_gp,)`` ``float64``; indexed by BZ grid
///   point, so ``weights.len() >= max(ir_grid_points) + 1``.
/// - ``mass_variances``: ``(num_patom,)`` ``float64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``eigenvectors``: ``(num_grid, num_band, num_band)`` ``complex128``
///   row-major ``[component, band]``.
/// - ``band_indices``: ``(num_band0,)`` ``int64``.
#[pyfunction]
#[pyo3(name = "isotope_strength")]
#[allow(clippy::too_many_arguments)]
fn py_isotope_strength<'py>(
    py: Python<'py>,
    mut gamma: PyReadwriteArray1<'py, f64>,
    grid_point: i64,
    ir_grid_points: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    mass_variances: PyReadonlyArray1<'py, f64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    sigma: f64,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let freq_shape = frequencies.shape();
    if freq_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }
    let num_grid = freq_shape[0];
    let num_band = freq_shape[1];
    let num_band0 = band_indices.shape()[0];

    if gamma.shape() != [num_band0] {
        return Err(PyValueError::new_err("gamma must have shape (num_band0,)"));
    }
    if eigenvectors.shape() != [num_grid, num_band, num_band] {
        return Err(PyValueError::new_err(
            "eigenvectors must have shape (num_grid, num_band, num_band)",
        ));
    }
    if mass_variances.shape() != [num_band / 3] {
        return Err(PyValueError::new_err(
            "mass_variances must have shape (num_band / 3,)",
        ));
    }
    if grid_point < 0 || (grid_point as usize) >= num_grid {
        return Err(PyValueError::new_err("grid_point out of range"));
    }

    let ir_view = ir_grid_points.as_array();
    let ir_slice = ir_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ir_grid_points must be C-contiguous"))?;
    let weights_view = weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("weights must be C-contiguous"))?;
    let mv_view = mass_variances.as_array();
    let mv_slice = mv_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mass_variances must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };
    let eig_slice = complex_readonly_as_cmplx(eig_flat);
    let bi_view = band_indices.as_array();
    let bi_slice = bi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;

    let gamma_slice = gamma
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("gamma must be C-contiguous"))?;

    py.allow_threads(|| {
        isotope::isotope_strength(
            gamma_slice,
            grid_point,
            ir_slice,
            weights_slice,
            mv_slice,
            freqs_slice,
            eig_slice,
            bi_slice,
            num_band,
            sigma,
            cutoff_frequency,
        );
    });
    Ok(())
}

/// Tetrahedron-method isotope scattering strength.  Mirrors
/// ``phono3py._phono3py.thm_isotope_strength``.
///
/// Shapes: same as ``isotope_strength`` plus
/// - ``integration_weights``: ``(num_grid_points, num_band0, num_band)``
///   ``float64``.
#[pyfunction]
#[pyo3(name = "thm_isotope_strength")]
#[allow(clippy::too_many_arguments)]
fn py_thm_isotope_strength<'py>(
    py: Python<'py>,
    mut gamma: PyReadwriteArray1<'py, f64>,
    grid_point: i64,
    ir_grid_points: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    mass_variances: PyReadonlyArray1<'py, f64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    eigenvectors: PyReadonlyArray3<'py, Complex64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    integration_weights: PyReadonlyArray3<'py, f64>,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let freq_shape = frequencies.shape();
    if freq_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }
    let num_grid = freq_shape[0];
    let num_band = freq_shape[1];
    let num_band0 = band_indices.shape()[0];

    if gamma.shape() != [num_band0] {
        return Err(PyValueError::new_err("gamma must have shape (num_band0,)"));
    }
    if eigenvectors.shape() != [num_grid, num_band, num_band] {
        return Err(PyValueError::new_err(
            "eigenvectors must have shape (num_grid, num_band, num_band)",
        ));
    }
    if mass_variances.shape() != [num_band / 3] {
        return Err(PyValueError::new_err(
            "mass_variances must have shape (num_band / 3,)",
        ));
    }
    if integration_weights.shape() != [num_grid, num_band0, num_band] {
        return Err(PyValueError::new_err(
            "integration_weights must have shape (num_grid, num_band0, num_band)",
        ));
    }
    if grid_point < 0 || (grid_point as usize) >= num_grid {
        return Err(PyValueError::new_err("grid_point out of range"));
    }

    let ir_view = ir_grid_points.as_array();
    let ir_slice = ir_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ir_grid_points must be C-contiguous"))?;
    let weights_view = weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("weights must be C-contiguous"))?;
    let mv_view = mass_variances.as_array();
    let mv_slice = mv_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mass_variances must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let eig_view = eigenvectors.as_array();
    let eig_flat = eig_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("eigenvectors must be C-contiguous"))?;
    let complex_readonly_as_cmplx = |s: &[Complex64]| -> &[Cmplx] {
        let n = s.len();
        let ptr = s.as_ptr() as *const Cmplx;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    };
    let eig_slice = complex_readonly_as_cmplx(eig_flat);
    let bi_view = band_indices.as_array();
    let bi_slice = bi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let iw_view = integration_weights.as_array();
    let iw_slice = iw_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("integration_weights must be C-contiguous"))?;

    let gamma_slice = gamma
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("gamma must be C-contiguous"))?;

    py.allow_threads(|| {
        isotope::thm_isotope_strength(
            gamma_slice,
            grid_point,
            ir_slice,
            weights_slice,
            mv_slice,
            freqs_slice,
            eig_slice,
            bi_slice,
            num_band,
            iw_slice,
            cutoff_frequency,
        );
    });
    Ok(())
}

/// Real-part of the bubble-diagram self-energy at on-shell band
/// frequencies.  Mirrors ``phono3py._phono3py.real_self_energy_at_bands``.
///
/// Shapes:
/// - ``real_self_energy``: ``(num_band0,)`` ``float64``, write-only.
/// - ``fc3_normal_squared``: ``(num_triplets, num_band0, num_band,
///   num_band)`` ``float64``.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``triplet_weights``: ``(num_triplets,)`` ``int64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``band_indices``: ``(num_band0,)`` ``int64``.
#[pyfunction]
#[pyo3(name = "real_self_energy_at_bands")]
#[allow(clippy::too_many_arguments)]
fn py_real_self_energy_at_bands<'py>(
    py: Python<'py>,
    mut real_self_energy: PyReadwriteArray1<'py, f64>,
    fc3_normal_squared: PyReadonlyArray4<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplet_weights: PyReadonlyArray1<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    temperature_thz: f64,
    epsilon: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let fc3_shape = fc3_normal_squared.shape();
    if fc3_shape.len() != 4 {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let num_triplets = fc3_shape[0];
    let num_band0 = fc3_shape[1];
    let num_band = fc3_shape[2];
    if fc3_shape[3] != num_band {
        return Err(PyValueError::new_err(
            "fc3_normal_squared last two axes must be equal (num_band)",
        ));
    }
    if real_self_energy.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "real_self_energy must have shape (num_band0,)",
        ));
    }
    if band_indices.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "band_indices must have shape (num_band0,)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    if triplet_weights.shape() != [num_triplets] {
        return Err(PyValueError::new_err(
            "triplet_weights must have shape (num_triplets,)",
        ));
    }
    let freq_shape = frequencies.shape();
    if freq_shape.len() != 2 || freq_shape[1] != num_band {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }

    let fc3_view = fc3_normal_squared.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let weights_view = triplet_weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplet_weights must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let bi_view = band_indices.as_array();
    let bi_slice = bi_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("band_indices must be C-contiguous"))?;
    let rse_slice = real_self_energy
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("real_self_energy must be C-contiguous"))?;

    py.allow_threads(|| {
        real_self_energy::real_self_energy_at_bands(
            rse_slice,
            fc3_slice,
            num_triplets,
            num_band0,
            num_band,
            bi_slice,
            freqs_slice,
            triplets_slice,
            weights_slice,
            epsilon,
            temperature_thz,
            unit_conversion_factor,
            cutoff_frequency,
        );
    });
    Ok(())
}

/// Real-part of the bubble-diagram self-energy at a single external
/// frequency point.  Mirrors
/// ``phono3py._phono3py.real_self_energy_at_frequency_point``.
#[pyfunction]
#[pyo3(name = "real_self_energy_at_frequency_point")]
#[allow(clippy::too_many_arguments)]
fn py_real_self_energy_at_frequency_point<'py>(
    py: Python<'py>,
    mut real_self_energy: PyReadwriteArray1<'py, f64>,
    frequency_point: f64,
    fc3_normal_squared: PyReadonlyArray4<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplet_weights: PyReadonlyArray1<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    band_indices: PyReadonlyArray1<'py, i64>,
    temperature_thz: f64,
    epsilon: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let fc3_shape = fc3_normal_squared.shape();
    if fc3_shape.len() != 4 {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let num_triplets = fc3_shape[0];
    let num_band0 = fc3_shape[1];
    let num_band = fc3_shape[2];
    if fc3_shape[3] != num_band {
        return Err(PyValueError::new_err(
            "fc3_normal_squared last two axes must be equal (num_band)",
        ));
    }
    if real_self_energy.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "real_self_energy must have shape (num_band0,)",
        ));
    }
    if band_indices.shape() != [num_band0] {
        return Err(PyValueError::new_err(
            "band_indices must have shape (num_band0,)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    if triplet_weights.shape() != [num_triplets] {
        return Err(PyValueError::new_err(
            "triplet_weights must have shape (num_triplets,)",
        ));
    }
    let freq_shape = frequencies.shape();
    if freq_shape.len() != 2 || freq_shape[1] != num_band {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }

    let fc3_view = fc3_normal_squared.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let weights_view = triplet_weights.as_array();
    let weights_slice = weights_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplet_weights must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let rse_slice = real_self_energy
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("real_self_energy must be C-contiguous"))?;

    py.allow_threads(|| {
        real_self_energy::real_self_energy_at_frequency_point(
            rse_slice,
            frequency_point,
            fc3_slice,
            num_triplets,
            num_band0,
            num_band,
            freqs_slice,
            triplets_slice,
            weights_slice,
            epsilon,
            temperature_thz,
            unit_conversion_factor,
            cutoff_frequency,
        );
    });
    Ok(())
}

/// Collision matrix for direct LBTE solution with k-star reduction.
/// Mirrors ``phono3py._phono3py.collision_matrix``.
///
/// Shapes:
/// - ``collision_matrix``: ``(num_band0, 3, num_ir_gp, num_band, 3)``
///   ``float64``, accumulated.
/// - ``fc3_normal_squared``: ``(num_triplets, num_band0, num_band,
///   num_band)`` ``float64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``g``: ``(3, num_triplets, num_band0, num_band, num_band)``
///   ``float64``; only the type-3 slab ``g[2]`` is read.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``triplets_map``, ``map_q``: ``(num_gp,)`` ``int64``.
/// - ``rot_grid_points``: ``(num_ir_gp, num_rot)`` ``int64``.
/// - ``rotations_cartesian``: ``(num_rot, 3, 3)`` ``float64``.
#[pyfunction]
#[pyo3(name = "collision_matrix")]
#[allow(clippy::too_many_arguments)]
fn py_collision_matrix<'py>(
    py: Python<'py>,
    mut collision_matrix: PyReadwriteArray5<'py, f64>,
    fc3_normal_squared: PyReadonlyArray4<'py, f64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray5<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplets_map: PyReadonlyArray1<'py, i64>,
    map_q: PyReadonlyArray1<'py, i64>,
    rot_grid_points: PyReadonlyArray2<'py, i64>,
    rotations_cartesian: PyReadonlyArray3<'py, f64>,
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let fc3_shape = fc3_normal_squared.shape();
    if fc3_shape.len() != 4 {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let num_triplets = fc3_shape[0];
    let num_band0 = fc3_shape[1];
    let num_band = fc3_shape[2];
    if fc3_shape[3] != num_band {
        return Err(PyValueError::new_err(
            "fc3_normal_squared last two axes must be equal (num_band)",
        ));
    }
    let num_gp = triplets_map.shape()[0];
    if map_q.shape() != [num_gp] {
        return Err(PyValueError::new_err("map_q must have shape (num_gp,)"));
    }
    let rot_shape = rot_grid_points.shape();
    if rot_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "rot_grid_points must have shape (num_ir_gp, num_rot)",
        ));
    }
    let num_ir_gp = rot_shape[0];
    let num_rot = rot_shape[1];
    if rotations_cartesian.shape() != [num_rot, 3, 3] {
        return Err(PyValueError::new_err(
            "rotations_cartesian must have shape (num_rot, 3, 3)",
        ));
    }
    if collision_matrix.shape() != [num_band0, 3, num_ir_gp, num_band, 3] {
        return Err(PyValueError::new_err(
            "collision_matrix must have shape (num_band0, 3, num_ir_gp, num_band, 3)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    let g_shape = g.shape();
    if g_shape.len() != 5
        || g_shape[0] != 3
        || g_shape[1] != num_triplets
        || g_shape[2] != num_band0
        || g_shape[3] != num_band
        || g_shape[4] != num_band
    {
        return Err(PyValueError::new_err(
            "g must have shape (3, num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let freq_shape = frequencies.shape();
    if freq_shape.len() != 2 || freq_shape[1] != num_band {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }

    let fc3_view = fc3_normal_squared.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let g_view = g.as_array();
    let g_flat = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be C-contiguous"))?;
    let g_offset = 2 * num_triplets * num_band0 * num_band * num_band;
    let g_slab = &g_flat[g_offset..];
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let tm_view = triplets_map.as_array();
    let tm_slice = tm_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets_map must be C-contiguous"))?;
    let mq_view = map_q.as_array();
    let mq_slice = mq_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("map_q must be C-contiguous"))?;
    let rgp_view = rot_grid_points.as_array();
    let rgp_slice = rgp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rot_grid_points must be C-contiguous"))?;
    let rc_view = rotations_cartesian.as_array();
    let rc_slice = rc_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rotations_cartesian must be C-contiguous"))?;
    let cm_slice = collision_matrix
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collision_matrix must be C-contiguous"))?;

    py.allow_threads(|| {
        collision_matrix::collision_matrix(
            cm_slice,
            fc3_slice,
            num_band0,
            num_band,
            freqs_slice,
            triplets_slice,
            tm_slice,
            mq_slice,
            rgp_slice,
            num_ir_gp,
            num_rot,
            rc_slice,
            g_slab,
            temperature_thz,
            unit_conversion_factor,
            cutoff_frequency,
        );
    });
    Ok(())
}

/// Reducible collision matrix for direct LBTE solution.  Mirrors
/// ``phono3py._phono3py.reducible_collision_matrix``.
///
/// Shapes:
/// - ``collision_matrix``: ``(num_band0, num_gp, num_band)`` ``float64``,
///   accumulated.
/// - ``fc3_normal_squared``: ``(num_triplets, num_band0, num_band,
///   num_band)`` ``float64``.
/// - ``frequencies``: ``(num_grid, num_band)`` ``float64``.
/// - ``g``: ``(3, num_triplets, num_band0, num_band, num_band)``
///   ``float64``; only the type-3 slab ``g[2]`` is read.
/// - ``triplets``: ``(num_triplets, 3)`` ``int64``.
/// - ``triplets_map``, ``map_q``: ``(num_gp,)`` ``int64``.
#[pyfunction]
#[pyo3(name = "reducible_collision_matrix")]
#[allow(clippy::too_many_arguments)]
fn py_reducible_collision_matrix<'py>(
    py: Python<'py>,
    mut collision_matrix: PyReadwriteArray3<'py, f64>,
    fc3_normal_squared: PyReadonlyArray4<'py, f64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray5<'py, f64>,
    triplets: PyReadonlyArray2<'py, i64>,
    triplets_map: PyReadonlyArray1<'py, i64>,
    map_q: PyReadonlyArray1<'py, i64>,
    temperature_thz: f64,
    unit_conversion_factor: f64,
    cutoff_frequency: f64,
) -> PyResult<()> {
    let fc3_shape = fc3_normal_squared.shape();
    if fc3_shape.len() != 4 {
        return Err(PyValueError::new_err(
            "fc3_normal_squared must have shape (num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let num_triplets = fc3_shape[0];
    let num_band0 = fc3_shape[1];
    let num_band = fc3_shape[2];
    if fc3_shape[3] != num_band {
        return Err(PyValueError::new_err(
            "fc3_normal_squared last two axes must be equal (num_band)",
        ));
    }
    let num_gp = triplets_map.shape()[0];
    if map_q.shape() != [num_gp] {
        return Err(PyValueError::new_err("map_q must have shape (num_gp,)"));
    }
    if collision_matrix.shape() != [num_band0, num_gp, num_band] {
        return Err(PyValueError::new_err(
            "collision_matrix must have shape (num_band0, num_gp, num_band)",
        ));
    }
    if triplets.shape() != [num_triplets, 3] {
        return Err(PyValueError::new_err(
            "triplets must have shape (num_triplets, 3)",
        ));
    }
    let g_shape = g.shape();
    if g_shape.len() != 5
        || g_shape[0] != 3
        || g_shape[1] != num_triplets
        || g_shape[2] != num_band0
        || g_shape[3] != num_band
        || g_shape[4] != num_band
    {
        return Err(PyValueError::new_err(
            "g must have shape (3, num_triplets, num_band0, num_band, num_band)",
        ));
    }
    let freq_shape = frequencies.shape();
    if freq_shape.len() != 2 || freq_shape[1] != num_band {
        return Err(PyValueError::new_err(
            "frequencies must have shape (num_grid, num_band)",
        ));
    }

    let fc3_view = fc3_normal_squared.as_array();
    let fc3_slice = fc3_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fc3_normal_squared must be C-contiguous"))?;
    let freqs_view = frequencies.as_array();
    let freqs_slice = freqs_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let g_view = g.as_array();
    let g_flat = g_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be C-contiguous"))?;
    let g_offset = 2 * num_triplets * num_band0 * num_band * num_band;
    let g_slab = &g_flat[g_offset..];
    let triplets_view = triplets.as_array();
    let triplets_flat = triplets_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets must be C-contiguous"))?;
    let triplets_slice: &[[i64; 3]] = group_as_array(triplets_flat);
    let tm_view = triplets_map.as_array();
    let tm_slice = tm_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("triplets_map must be C-contiguous"))?;
    let mq_view = map_q.as_array();
    let mq_slice = mq_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("map_q must be C-contiguous"))?;
    let cm_slice = collision_matrix
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collision_matrix must be C-contiguous"))?;

    py.allow_threads(|| {
        collision_matrix::reducible_collision_matrix(
            cm_slice,
            fc3_slice,
            num_band0,
            num_band,
            freqs_slice,
            triplets_slice,
            tm_slice,
            mq_slice,
            g_slab,
            temperature_thz,
            unit_conversion_factor,
            cutoff_frequency,
        );
    });
    Ok(())
}

/// Symmetrize collision matrix in-place as ``(Omega + Omega^T) / 2``.
/// Mirrors ``phono3py._phono3py.symmetrize_collision_matrix``.
///
/// Accepts either a 6D array
/// ``(num_sigma, num_temp, num_grid_points, num_band, num_grid_points,
/// num_band)`` or an 8D array
/// ``(num_sigma, num_temp, num_grid_points, num_band, 3, num_grid_points,
/// num_band, 3)``.  In both cases each ``(sigma, temp)`` slice is
/// viewed as a square ``(num_column, num_column)`` matrix whose
/// ``num_column`` is inferred from ndim.
#[pyfunction]
#[pyo3(name = "symmetrize_collision_matrix")]
fn py_symmetrize_collision_matrix<'py>(
    py: Python<'py>,
    mut collision_matrix: PyReadwriteArrayDyn<'py, f64>,
) -> PyResult<()> {
    let shape = collision_matrix.shape().to_vec();
    let (num_sigma, num_temp, num_column) = match shape.len() {
        6 => {
            let num_grid_points = shape[2];
            let num_band = shape[3];
            if shape[4] != num_grid_points || shape[5] != num_band {
                return Err(PyValueError::new_err(
                    "6D collision_matrix must have shape (num_sigma, num_temp, \
                     num_grid_points, num_band, num_grid_points, num_band)",
                ));
            }
            (shape[0], shape[1], num_grid_points * num_band)
        }
        8 => {
            let num_grid_points = shape[2];
            let num_band = shape[3];
            if shape[4] != 3 || shape[5] != num_grid_points || shape[6] != num_band || shape[7] != 3
            {
                return Err(PyValueError::new_err(
                    "8D collision_matrix must have shape (num_sigma, num_temp, \
                     num_grid_points, num_band, 3, num_grid_points, num_band, 3)",
                ));
            }
            (shape[0], shape[1], num_grid_points * num_band * 3)
        }
        _ => {
            return Err(PyValueError::new_err(
                "collision_matrix must be 6D or 8D",
            ));
        }
    };

    let cm_slice = collision_matrix
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collision_matrix must be C-contiguous"))?;

    py.allow_threads(|| {
        collision_matrix::symmetrize_collision_matrix(cm_slice, num_column, num_temp, num_sigma);
    });
    Ok(())
}

/// Expand collision matrix to all grid points by k-star symmetry.
/// Mirrors ``phono3py._phono3py.expand_collision_matrix``.
///
/// Shapes:
/// - ``collision_matrix``: ``(num_sigma, num_temp, num_grid_points,
///   num_band, num_grid_points, num_band)`` ``float64``.
/// - ``ir_grid_points``: ``(num_ir_gp,)`` ``int64``.
/// - ``rot_grid_points``: ``(num_rot, num_grid_points)`` ``int64``.
#[pyfunction]
#[pyo3(name = "expand_collision_matrix")]
fn py_expand_collision_matrix<'py>(
    py: Python<'py>,
    mut collision_matrix: PyReadwriteArray6<'py, f64>,
    ir_grid_points: PyReadonlyArray1<'py, i64>,
    rot_grid_points: PyReadonlyArray2<'py, i64>,
) -> PyResult<()> {
    let cm_shape = collision_matrix.shape();
    let num_sigma = cm_shape[0];
    let num_temp = cm_shape[1];
    let num_grid_points = cm_shape[2];
    let num_band = cm_shape[3];
    if cm_shape[4] != num_grid_points || cm_shape[5] != num_band {
        return Err(PyValueError::new_err(
            "collision_matrix must have shape (num_sigma, num_temp, \
             num_grid_points, num_band, num_grid_points, num_band)",
        ));
    }
    let rot_shape = rot_grid_points.shape();
    if rot_shape.len() != 2 || rot_shape[1] != num_grid_points {
        return Err(PyValueError::new_err(
            "rot_grid_points must have shape (num_rot, num_grid_points)",
        ));
    }
    let num_rot = rot_shape[0];

    let ir_view = ir_grid_points.as_array();
    let ir_slice = ir_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ir_grid_points must be C-contiguous"))?;
    let rot_view = rot_grid_points.as_array();
    let rot_slice = rot_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rot_grid_points must be C-contiguous"))?;
    let cm_slice = collision_matrix
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("collision_matrix must be C-contiguous"))?;

    py.allow_threads(|| {
        collision_matrix::expand_collision_matrix(
            cm_slice,
            rot_slice,
            ir_slice,
            num_grid_points,
            num_rot,
            num_sigma,
            num_temp,
            num_band,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "distribute_fc3")]
fn py_distribute_fc3<'py>(
    py: Python<'py>,
    mut fc3: PyReadwriteArray6<'py, f64>,
    target: i64,
    source: i64,
    atom_mapping: PyReadonlyArray1<'py, i64>,
    rot_cart_inv: PyReadonlyArray2<'py, f64>,
) -> PyResult<()> {
    let fc3_shape = fc3.shape();
    let num_atom = fc3_shape[0];
    if fc3_shape[1] != num_atom
        || fc3_shape[2] != num_atom
        || fc3_shape[3] != 3
        || fc3_shape[4] != 3
        || fc3_shape[5] != 3
    {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_atom, num_atom, num_atom, 3, 3, 3)",
        ));
    }
    if atom_mapping.shape() != [num_atom] {
        return Err(PyValueError::new_err(
            "atom_mapping must have shape (num_atom,)",
        ));
    }
    let rot_shape = rot_cart_inv.shape();
    if rot_shape != [3, 3] {
        return Err(PyValueError::new_err(
            "rot_cart_inv must have shape (3, 3)",
        ));
    }
    if target < 0 || (target as usize) >= num_atom {
        return Err(PyValueError::new_err("target out of range"));
    }
    if source < 0 || (source as usize) >= num_atom {
        return Err(PyValueError::new_err("source out of range"));
    }

    let atom_view = atom_mapping.as_array();
    let atom_slice = atom_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("atom_mapping must be C-contiguous"))?;
    let rot_view = rot_cart_inv.as_array();
    let rot_slice = rot_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rot_cart_inv must be C-contiguous"))?;
    let fc3_slice = fc3
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3 must be C-contiguous"))?;

    py.allow_threads(|| {
        fc3::distribute_fc3(
            fc3_slice,
            target as usize,
            source as usize,
            atom_slice,
            num_atom,
            rot_slice,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "permutation_symmetry_fc3")]
fn py_permutation_symmetry_fc3<'py>(
    py: Python<'py>,
    mut fc3: PyReadwriteArray6<'py, f64>,
) -> PyResult<()> {
    let fc3_shape = fc3.shape();
    let num_atom = fc3_shape[0];
    if fc3_shape[1] != num_atom
        || fc3_shape[2] != num_atom
        || fc3_shape[3] != 3
        || fc3_shape[4] != 3
        || fc3_shape[5] != 3
    {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_atom, num_atom, num_atom, 3, 3, 3)",
        ));
    }
    let fc3_slice = fc3
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3 must be C-contiguous"))?;

    py.allow_threads(|| {
        fc3::set_permutation_symmetry_fc3(fc3_slice, num_atom);
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "transpose_compact_fc3")]
fn py_transpose_compact_fc3<'py>(
    py: Python<'py>,
    mut fc3: PyReadwriteArray6<'py, f64>,
    permutations: PyReadonlyArray2<'py, i64>,
    s2pp_map: PyReadonlyArray1<'py, i64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    nsym_list: PyReadonlyArray1<'py, i64>,
    t_type: i64,
) -> PyResult<()> {
    let fc3_shape = fc3.shape();
    let n_patom = fc3_shape[0];
    let n_satom = fc3_shape[1];
    if fc3_shape[2] != n_satom
        || fc3_shape[3] != 3
        || fc3_shape[4] != 3
        || fc3_shape[5] != 3
    {
        return Err(PyValueError::new_err(
            "fc3 must have shape (n_patom, n_satom, n_satom, 3, 3, 3)",
        ));
    }
    if p2s_map.shape() != [n_patom] {
        return Err(PyValueError::new_err("p2s_map must have shape (n_patom,)"));
    }
    if s2pp_map.shape() != [n_satom] || nsym_list.shape() != [n_satom] {
        return Err(PyValueError::new_err(
            "s2pp_map and nsym_list must have shape (n_satom,)",
        ));
    }
    let perms_shape = permutations.shape();
    if perms_shape.len() != 2 || perms_shape[1] != n_satom {
        return Err(PyValueError::new_err(
            "permutations must have shape (num_sym, n_satom)",
        ));
    }

    let perms_view = permutations.as_array();
    let perms_slice = perms_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("permutations must be C-contiguous"))?;
    let s2pp_view = s2pp_map.as_array();
    let s2pp_slice = s2pp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2pp_map must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let nsym_view = nsym_list.as_array();
    let nsym_slice = nsym_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("nsym_list must be C-contiguous"))?;
    let fc3_slice = fc3
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3 must be C-contiguous"))?;

    py.allow_threads(|| {
        fc3::transpose_compact_fc3(
            fc3_slice,
            p2s_slice,
            s2pp_slice,
            nsym_slice,
            perms_slice,
            n_satom,
            n_patom,
            t_type,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "permutation_symmetry_compact_fc3")]
fn py_permutation_symmetry_compact_fc3<'py>(
    py: Python<'py>,
    mut fc3: PyReadwriteArray6<'py, f64>,
    permutations: PyReadonlyArray2<'py, i64>,
    s2pp_map: PyReadonlyArray1<'py, i64>,
    p2s_map: PyReadonlyArray1<'py, i64>,
    nsym_list: PyReadonlyArray1<'py, i64>,
) -> PyResult<()> {
    let fc3_shape = fc3.shape();
    let n_patom = fc3_shape[0];
    let n_satom = fc3_shape[1];
    if fc3_shape[2] != n_satom
        || fc3_shape[3] != 3
        || fc3_shape[4] != 3
        || fc3_shape[5] != 3
    {
        return Err(PyValueError::new_err(
            "fc3 must have shape (n_patom, n_satom, n_satom, 3, 3, 3)",
        ));
    }
    if p2s_map.shape() != [n_patom] {
        return Err(PyValueError::new_err("p2s_map must have shape (n_patom,)"));
    }
    if s2pp_map.shape() != [n_satom] || nsym_list.shape() != [n_satom] {
        return Err(PyValueError::new_err(
            "s2pp_map and nsym_list must have shape (n_satom,)",
        ));
    }
    let perms_shape = permutations.shape();
    if perms_shape.len() != 2 || perms_shape[1] != n_satom {
        return Err(PyValueError::new_err(
            "permutations must have shape (num_sym, n_satom)",
        ));
    }

    let perms_view = permutations.as_array();
    let perms_slice = perms_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("permutations must be C-contiguous"))?;
    let s2pp_view = s2pp_map.as_array();
    let s2pp_slice = s2pp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("s2pp_map must be C-contiguous"))?;
    let p2s_view = p2s_map.as_array();
    let p2s_slice = p2s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("p2s_map must be C-contiguous"))?;
    let nsym_view = nsym_list.as_array();
    let nsym_slice = nsym_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("nsym_list must be C-contiguous"))?;
    let fc3_slice = fc3
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3 must be C-contiguous"))?;

    py.allow_threads(|| {
        fc3::set_permutation_symmetry_compact_fc3(
            fc3_slice,
            p2s_slice,
            s2pp_slice,
            nsym_slice,
            perms_slice,
            n_satom,
            n_patom,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "rotate_delta_fc2s")]
fn py_rotate_delta_fc2s<'py>(
    py: Python<'py>,
    mut fc3: PyReadwriteArray5<'py, f64>,
    delta_fc2s: PyReadonlyArray5<'py, f64>,
    inv_u: PyReadonlyArray2<'py, f64>,
    site_sym_cart: PyReadonlyArray3<'py, f64>,
    rot_map_syms: PyReadonlyArray2<'py, i64>,
) -> PyResult<()> {
    let fc3_shape = fc3.shape();
    let num_atom = fc3_shape[0];
    if fc3_shape[1] != num_atom
        || fc3_shape[2] != 3
        || fc3_shape[3] != 3
        || fc3_shape[4] != 3
    {
        return Err(PyValueError::new_err(
            "fc3 must have shape (num_atom, num_atom, 3, 3, 3)",
        ));
    }
    let df_shape = delta_fc2s.shape();
    let num_disp = df_shape[0];
    if df_shape[1] != num_atom
        || df_shape[2] != num_atom
        || df_shape[3] != 3
        || df_shape[4] != 3
    {
        return Err(PyValueError::new_err(
            "delta_fc2s must have shape (num_disp, num_atom, num_atom, 3, 3)",
        ));
    }
    let ss_shape = site_sym_cart.shape();
    let num_site_sym = ss_shape[0];
    if ss_shape[1] != 3 || ss_shape[2] != 3 {
        return Err(PyValueError::new_err(
            "site_sym_cart must have shape (num_site_sym, 3, 3)",
        ));
    }
    let inv_shape = inv_u.shape();
    if inv_shape[0] != 3 || inv_shape[1] != num_disp * num_site_sym {
        return Err(PyValueError::new_err(
            "inv_u must have shape (3, num_disp * num_site_sym)",
        ));
    }
    let rm_shape = rot_map_syms.shape();
    if rm_shape[0] != num_site_sym || rm_shape[1] != num_atom {
        return Err(PyValueError::new_err(
            "rot_map_syms must have shape (num_site_sym, num_atom)",
        ));
    }

    let df_view = delta_fc2s.as_array();
    let df_slice = df_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("delta_fc2s must be C-contiguous"))?;
    let inv_view = inv_u.as_array();
    let inv_slice = inv_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("inv_u must be C-contiguous"))?;
    let ss_view = site_sym_cart.as_array();
    let ss_slice = ss_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("site_sym_cart must be C-contiguous"))?;
    let rm_view = rot_map_syms.as_array();
    let rm_slice = rm_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("rot_map_syms must be C-contiguous"))?;
    let fc3_slice = fc3
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("fc3 must be C-contiguous"))?;

    py.allow_threads(|| {
        fc3::rotate_delta_fc2s(
            fc3_slice,
            df_slice,
            inv_slice,
            ss_slice,
            rm_slice,
            num_atom,
            num_site_sym,
            num_disp,
        );
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "neighboring_grid_points")]
fn py_neighboring_grid_points<'py>(
    py: Python<'py>,
    mut relative_grid_points: PyReadwriteArray1<'py, i64>,
    grid_points: PyReadonlyArray1<'py, i64>,
    relative_grid_address: PyReadonlyArray2<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
) -> PyResult<()> {
    let rga = addresses_i(&relative_grid_address)?;
    let adrs = addresses_i(&bz_grid_addresses)?;
    let d = vec3_i(&d_diag)?;
    let gp_view = grid_points.as_array();
    let gp_slice = gp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_points must be C-contiguous"))?;
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be C-contiguous"))?;
    let out = relative_grid_points
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("relative_grid_points must be C-contiguous"))?;
    if out.len() != gp_slice.len() * rga.len() {
        return Err(PyValueError::new_err(
            "relative_grid_points must have length num_grid_points * num_relative_grid_address",
        ));
    }

    py.allow_threads(|| {
        let bzgrid = BzGridView {
            d_diag: d,
            addresses: &adrs,
            gp_map: bzmap_slice,
            bz_grid_type,
        };
        triplet_iw::neighboring_grid_points_many(out, gp_slice, &rga, &bzgrid)
    })
    .map_err(|e| match e {
        BzGridError::BadGridType => PyValueError::new_err("bz_grid_type must be 1 or 2"),
        BzGridError::BadTpType => PyValueError::new_err("tp_type must be 2, 3, or 4"),
    })
}

#[pyfunction]
#[pyo3(name = "integration_weights_at_grid_points")]
fn py_integration_weights_at_grid_points<'py>(
    py: Python<'py>,
    mut iw: PyReadwriteArray3<'py, f64>,
    frequency_points: PyReadonlyArray1<'py, f64>,
    relative_grid_address: PyReadonlyArray3<'py, i64>,
    d_diag: PyReadonlyArray1<'py, i64>,
    grid_points: PyReadonlyArray1<'py, i64>,
    frequencies: PyReadonlyArray2<'py, f64>,
    bz_grid_addresses: PyReadonlyArray2<'py, i64>,
    bz_map: PyReadonlyArray1<'py, i64>,
    gp2irgp_map: PyReadonlyArray1<'py, i64>,
    bz_grid_type: i64,
    function: &str,
) -> PyResult<()> {
    let wf = match function {
        "I" => WeightFunction::I,
        "J" => WeightFunction::J,
        _ => return Err(PyValueError::new_err("function must be 'I' or 'J'")),
    };
    let rga = relative_grid_address_3d(&relative_grid_address)?;
    let adrs = addresses_i(&bz_grid_addresses)?;
    let d = vec3_i(&d_diag)?;

    let fp_view = frequency_points.as_array();
    let fp_slice = fp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequency_points must be C-contiguous"))?;
    let gp_view = grid_points.as_array();
    let gp_slice = gp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_points must be C-contiguous"))?;
    let fr_view = frequencies.as_array();
    let fr_slice = fr_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("frequencies must be C-contiguous"))?;
    let num_band = frequencies.shape()[1];
    let bzmap_view = bz_map.as_array();
    let bzmap_slice = bzmap_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bz_map must be C-contiguous"))?;
    let gp2ir_view = gp2irgp_map.as_array();
    let gp2ir_slice = gp2ir_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("gp2irgp_map must be C-contiguous"))?;

    let iw_shape = iw.shape();
    if iw_shape != [gp_slice.len(), fp_slice.len(), num_band] {
        return Err(PyValueError::new_err(
            "iw must have shape (num_grid_points, num_frequency_points, num_band)",
        ));
    }
    let iw_slice = iw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("iw must be C-contiguous"))?;

    py.allow_threads(|| {
        let bzgrid = BzGridView {
            d_diag: d,
            addresses: &adrs,
            gp_map: bzmap_slice,
            bz_grid_type,
        };
        triplet_iw::integration_weights_at_grid_points(
            iw_slice,
            fp_slice,
            &rga,
            gp_slice,
            fr_slice,
            num_band,
            &bzgrid,
            gp2ir_slice,
            wf,
        )
    })
    .map_err(|e| match e {
        BzGridError::BadGridType => PyValueError::new_err("bz_grid_type must be 1 or 2"),
        BzGridError::BadTpType => PyValueError::new_err("tp_type must be 2, 3, or 4"),
    })
}

#[pyfunction]
#[pyo3(name = "tetrahedra_relative_grid_address")]
fn py_tetrahedra_relative_grid_address<'py>(
    mut relative_grid_address: PyReadwriteArray3<'py, i64>,
    reciprocal_lattice: PyReadonlyArray2<'py, f64>,
) -> PyResult<()> {
    let rga_shape = relative_grid_address.shape();
    if rga_shape != [24, 4, 3] {
        return Err(PyValueError::new_err(
            "relative_grid_address must have shape (24, 4, 3)",
        ));
    }
    let rec = mat3_f(&reciprocal_lattice)?;
    let (table, _main_diag_index) = tetrahedron_method::relative_grid_address(rec);
    let out = relative_grid_address
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("relative_grid_address must be C-contiguous"))?;
    for i in 0..24 {
        for j in 0..4 {
            for k in 0..3 {
                out[(i * 4 + j) * 3 + k] = table[i][j][k];
            }
        }
    }
    Ok(())
}

#[pymodule]
fn phono3py_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_snf3x3, m)?)?;
    m.add_function(wrap_pyfunction!(py_grid_index_from_address, m)?)?;
    m.add_function(wrap_pyfunction!(py_gr_grid_addresses, m)?)?;
    m.add_function(wrap_pyfunction!(py_ir_grid_map, m)?)?;
    m.add_function(wrap_pyfunction!(py_reciprocal_rotations, m)?)?;
    m.add_function(wrap_pyfunction!(py_transform_rotations, m)?)?;
    m.add_function(wrap_pyfunction!(py_bz_grid_addresses, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotate_bz_grid_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_ir_triplets_at_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_bz_triplets_at_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_triplets_integration_weights, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_triplets_integration_weights_with_sigma,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_dynamical_matrix_at_q, m)?)?;
    m.add_function(wrap_pyfunction!(py_charge_sum, m)?)?;
    m.add_function(wrap_pyfunction!(py_recip_dipole_dipole_q0, m)?)?;
    m.add_function(wrap_pyfunction!(py_recip_dipole_dipole, m)?)?;
    m.add_function(wrap_pyfunction!(py_dynamical_matrices_at_gridpoints, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_dynamical_matrices_at_gridpoints_gonze,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_real_to_reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(py_reciprocal_to_normal_squared, m)?)?;
    m.add_function(wrap_pyfunction!(py_imag_self_energy_with_g, m)?)?;
    m.add_function(wrap_pyfunction!(py_detailed_imag_self_energy_with_g, m)?)?;
    m.add_function(wrap_pyfunction!(py_interaction, m)?)?;
    m.add_function(wrap_pyfunction!(py_pp_collision, m)?)?;
    m.add_function(wrap_pyfunction!(py_pp_collision_with_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(py_collision_at_grid_point, m)?)?;
    m.add_function(wrap_pyfunction!(py_collision_at_grid_points_batched, m)?)?;
    m.add_function(wrap_pyfunction!(py_isotope_strength, m)?)?;
    m.add_function(wrap_pyfunction!(py_thm_isotope_strength, m)?)?;
    m.add_function(wrap_pyfunction!(py_real_self_energy_at_bands, m)?)?;
    m.add_function(wrap_pyfunction!(py_real_self_energy_at_frequency_point, m)?)?;
    m.add_function(wrap_pyfunction!(py_collision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_reducible_collision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_symmetrize_collision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_expand_collision_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_distribute_fc3, m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_symmetry_fc3, m)?)?;
    m.add_function(wrap_pyfunction!(py_transpose_compact_fc3, m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_symmetry_compact_fc3, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotate_delta_fc2s, m)?)?;
    m.add_function(wrap_pyfunction!(py_tetrahedra_relative_grid_address, m)?)?;
    m.add_function(wrap_pyfunction!(py_neighboring_grid_points, m)?)?;
    m.add_function(wrap_pyfunction!(py_integration_weights_at_grid_points, m)?)?;
    Ok(())
}
