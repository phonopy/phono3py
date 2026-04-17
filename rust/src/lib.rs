use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadwriteArray4, PyReadwriteArray5, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod bzgrid;
mod common;
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
    Ok(())
}
