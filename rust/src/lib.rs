use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4,
    PyReadwriteArray5, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::common::Cmplx;

mod bzgrid;
mod common;
mod dynmat;
mod grgrid;
mod recip_rotations;
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
    Ok(())
}
