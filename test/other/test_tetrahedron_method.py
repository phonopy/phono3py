"""Tests for tetrahedron_method.py."""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.other.tetrahedron_method import (
    get_integration_weights,
    get_tetrahedra_relative_grid_address,
    get_unique_grid_points,
)
from phono3py.phonon.grid import BZGrid, get_ir_grid_points


def test_get_unique_grid_points(si_pbesol_111: Phono3py):
    """Test get_unique_grid_points returns unique tetrahedron-vertex grid points.

    Uses the first 3 irreducible grid points of a 4x4x4 mesh on Si.
    Verifies dtype, uniqueness, and regression values.

    """
    lat = si_pbesol_111.primitive.cell
    mesh = [4, 4, 4]
    bzgrid = BZGrid(mesh, lattice=lat, store_dense_gp_map=True)
    ir_grid_points, _, _ = get_ir_grid_points(bzgrid)
    input_gps = bzgrid.grg2bzg[ir_grid_points[:3]]
    unique_gps = get_unique_grid_points(input_gps, bzgrid)

    assert unique_gps.dtype == np.dtype("int64")
    np.testing.assert_array_equal(unique_gps, np.unique(unique_gps))

    ref = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        67,
        68,
        69,
        70,
        85,
        86,
        87,
        88,
    ]
    np.testing.assert_array_equal(unique_gps, ref)


def test_get_integration_weights(si_pbesol_111: Phono3py):
    """Test get_integration_weights for function="I" (derivative) and "J" (integral).

    Uses a 4x4x4 mesh on Si with sampling points in the phonon frequency range.
    Checks shape, dtype, and regression values for the first grid point.

    """
    si_pbesol_111.mesh_numbers = [4, 4, 4]
    si_pbesol_111.init_phph_interaction()
    si_pbesol_111.run_phonon_solver()
    frequencies, _, _ = si_pbesol_111.get_phonon_data()
    bzgrid = si_pbesol_111.grid
    assert bzgrid is not None
    grg_frequencies = frequencies[bzgrid.grg2bzg]
    sampling_points = np.linspace(3, 15, 5)

    iw_i = get_integration_weights(
        sampling_points,
        grg_frequencies,
        bzgrid,
        bzgp2irgp_map=bzgrid.bzg2grg,
        function="I",
    )
    iw_j = get_integration_weights(
        sampling_points,
        grg_frequencies,
        bzgrid,
        bzgp2irgp_map=bzgrid.bzg2grg,
        function="J",
    )

    assert iw_i.shape == (
        len(bzgrid.grg2bzg),
        len(sampling_points),
        grg_frequencies.shape[1],
    )
    assert iw_i.dtype == np.dtype("double")
    assert iw_j.shape == iw_i.shape

    ref_iw_i = [
        [
            0.00000000e00,
            0.00000000e00,
            2.41328264e-01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            4.80802638e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            6.76424668e-01,
            3.35855636e00,
            3.46723400e00,
        ],
    ]
    ref_iw_j = [
        [
            1.00000000e00,
            1.00000000e00,
            3.00480294e-01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            1.00000000e00,
            1.00000000e00,
            9.87488098e-01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            9.38379703e-01,
            3.83950270e-01,
            3.21357501e-01,
        ],
    ]
    np.testing.assert_allclose(iw_i[0], ref_iw_i, rtol=0, atol=1e-7)
    np.testing.assert_allclose(iw_j[0], ref_iw_j, rtol=0, atol=1e-7)


def test_get_unique_grid_points_rust_vs_c(si_pbesol_111: Phono3py):
    """Compare lang='Rust' and C paths of get_unique_grid_points.

    Exercises both bz_grid_type=2 (dense map) and type=1 (sparse map).

    """
    pytest.importorskip("phono3py_rs")

    lat = si_pbesol_111.primitive.cell
    mesh = [4, 4, 4]
    for store_dense in (True, False):
        bzgrid = BZGrid(mesh, lattice=lat, store_dense_gp_map=store_dense)
        ir_grid_points, _, _ = get_ir_grid_points(bzgrid)
        input_gps = bzgrid.grg2bzg[ir_grid_points[:3]]
        out_c = get_unique_grid_points(input_gps, bzgrid, lang="C")
        out_rust = get_unique_grid_points(input_gps, bzgrid, lang="Rust")
        np.testing.assert_array_equal(out_rust, out_c)


def test_get_integration_weights_rust_vs_c(si_pbesol_111: Phono3py):
    """Compare lang='Rust' and C paths of get_integration_weights.

    Runs both function='I' (derivative) and 'J' (integral).  Per-value
    FP rounding of one ULP is allowed.

    """
    pytest.importorskip("phono3py_rs")

    si_pbesol_111.mesh_numbers = [4, 4, 4]
    si_pbesol_111.init_phph_interaction()
    si_pbesol_111.run_phonon_solver()
    frequencies, _, _ = si_pbesol_111.get_phonon_data()
    bzgrid = si_pbesol_111.grid
    assert bzgrid is not None
    grg_frequencies = frequencies[bzgrid.grg2bzg]
    sampling_points = np.linspace(3, 15, 5)

    for function in ("I", "J"):
        iw_c = get_integration_weights(
            sampling_points,
            grg_frequencies,
            bzgrid,
            bzgp2irgp_map=bzgrid.bzg2grg,
            function=function,
            lang="C",
        )
        iw_rust = get_integration_weights(
            sampling_points,
            grg_frequencies,
            bzgrid,
            bzgp2irgp_map=bzgrid.bzg2grg,
            function=function,
            lang="Rust",
        )
        np.testing.assert_allclose(iw_rust, iw_c, rtol=0, atol=1e-14)


def test_get_tetrahedra_relative_grid_address_rust_vs_c():
    """Compare lang='Rust' and C paths of get_tetrahedra_relative_grid_address.

    Exercises the four main-diagonal branches by sweeping lattices whose
    shortest main diagonal differs.  The Rust core returns a pre-tabulated
    integer table, so output must be bit-equal to C.

    """
    pytest.importorskip("phono3py_rs")

    lattices = [
        np.eye(3) * 1.0,
        np.diag([1.0, 2.0, 3.0]),
        np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]]),
        np.array([[2.0, 0.5, 0.1], [0.1, 1.0, 0.3], [0.2, 0.1, 0.5]]),
    ]
    for lat in lattices:
        out_c = get_tetrahedra_relative_grid_address(lat, lang="C")
        out_rust = get_tetrahedra_relative_grid_address(lat, lang="Rust")
        np.testing.assert_array_equal(out_rust, out_c)
