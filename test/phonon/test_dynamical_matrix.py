"""Regression tests comparing the Rust dynamical-matrix builder.

The reference is phonopy's Python implementation.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix

phono3py_rs = pytest.importorskip("phono3py_rs")


def _extract_inputs(dm: DynamicalMatrix):
    prim = dm.primitive
    svecs, multi = prim.get_smallest_vectors()
    masses = np.asarray(prim.masses, dtype="double")
    fc = dm.force_constants
    p2s = np.asarray(prim.p2s_map, dtype="int64")
    s2p = np.asarray(prim.s2p_map, dtype="int64")
    if fc.shape[0] == fc.shape[1]:
        fc_p2s = p2s
        fc_s2p = s2p
    else:
        p2p = prim.p2p_map
        s2pp = np.array([p2p[s2p[i]] for i in range(len(s2p))], dtype="int64")
        fc_p2s = np.arange(len(p2s), dtype="int64")
        fc_s2p = s2pp
    return (
        np.ascontiguousarray(fc, dtype="double"),
        np.ascontiguousarray(svecs, dtype="double"),
        np.ascontiguousarray(multi, dtype="int64"),
        masses,
        fc_s2p,
        fc_p2s,
    )


def _rust_dm(dm: DynamicalMatrix, q: Sequence[float]) -> np.ndarray:
    fc, svecs, multi, masses, fc_s2p, fc_p2s = _extract_inputs(dm)
    num_band = len(fc_p2s) * 3
    out = np.zeros((num_band, num_band), dtype="complex128")
    phono3py_rs.dynamical_matrix_at_q(
        out,
        fc,
        np.ascontiguousarray(q, dtype="double"),
        svecs,
        multi,
        masses,
        fc_s2p,
        fc_p2s,
        None,
    )
    return out


def _ref_dm(dm: DynamicalMatrix, q: Sequence[float]) -> np.ndarray:
    dm.run(q)
    return np.asarray(dm.dynamical_matrix).copy()


@pytest.fixture(scope="module")
def si_dm(si_pbesol) -> DynamicalMatrix:
    """Build a no-NAC DynamicalMatrix for the Si fixture."""
    si_pbesol.mesh_numbers = [3, 3, 3]
    si_pbesol.init_phph_interaction()
    return si_pbesol.dynamical_matrix


Q_POINTS: list[list[float]] = [
    [0.0, 0.0, 0.0],
    [0.1, 0.2, 0.3],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [-0.25, 0.4, 0.1],
]


@pytest.mark.parametrize("q", Q_POINTS)
def test_dynamical_matrix_at_q_matches_phonopy(
    si_dm: DynamicalMatrix, q: list[float]
) -> None:
    """Rust and phonopy's Python dynamical matrix agree on a no-NAC Si fixture.

    The agreement holds to machine precision.

    """
    ref = _ref_dm(si_dm, q)
    out = _rust_dm(si_dm, q)
    np.testing.assert_allclose(out, ref, atol=1e-13, rtol=1e-13)


def test_dynamical_matrix_random_q_matches_phonopy(
    si_dm: DynamicalMatrix,
) -> None:
    """50 random q-points: Rust output equals phonopy to 1e-13."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-0.5, 0.5, size=3)
        ref = _ref_dm(si_dm, q)
        out = _rust_dm(si_dm, q)
        np.testing.assert_allclose(out, ref, atol=1e-13, rtol=1e-13)


def test_dynamical_matrix_is_hermitian(si_dm: DynamicalMatrix) -> None:
    """The Rust output is Hermitian by construction."""
    out = _rust_dm(si_dm, [0.13, -0.27, 0.41])
    np.testing.assert_allclose(out, out.conj().T, atol=1e-15)


def _ref_charge_sum(factor: float, q_cart: np.ndarray, born: np.ndarray) -> np.ndarray:
    """Compute the reference Wang-NAC charge sum.

    Matches phonopy's DynamicalMatrixWang._get_charge_sum (without the
    1/factor).

    """
    num_patom = born.shape[0]
    qb = np.einsum("k,ika->ia", q_cart, born)
    out = np.zeros((num_patom, num_patom, 3, 3), dtype="double")
    for i in range(num_patom):
        for j in range(num_patom):
            out[i, j] = np.outer(qb[i], qb[j]) * factor
    return out


@pytest.fixture(scope="module")
def nacl_dm_gl(nacl_pbe):
    """Build a Gonze-Lee NAC DynamicalMatrix on the NaCl fixture."""
    nacl_pbe.mesh_numbers = [3, 3, 3]
    nacl_pbe.init_phph_interaction()
    return nacl_pbe.dynamical_matrix


NACL_Q_POINTS: list[list[float]] = [
    [0.1, 0.2, 0.3],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [-0.25, 0.4, 0.1],
]


@pytest.mark.parametrize("q", NACL_Q_POINTS)
def test_recip_dipole_dipole_matches_phonopy(nacl_dm_gl, q: list[float]) -> None:
    """Rust dd at q equals phonopy's C-side dd to machine precision."""
    q_cart = np.ascontiguousarray(np.dot(q, nacl_dm_gl._rec_lat.T), dtype="double")
    pos = np.ascontiguousarray(nacl_dm_gl.primitive.positions, dtype="double")
    num_atom = len(pos)
    dd = np.zeros((num_atom, 3, num_atom, 3), dtype="cdouble", order="C")
    volume = nacl_dm_gl.primitive.volume
    factor = nacl_dm_gl._unit_conversion * 4.0 * np.pi / volume
    phono3py_rs.recip_dipole_dipole(
        dd,
        np.ascontiguousarray(nacl_dm_gl._dd_q0, dtype="cdouble"),
        np.ascontiguousarray(nacl_dm_gl._G_list, dtype="double"),
        q_cart,
        np.ascontiguousarray(nacl_dm_gl._born, dtype="double"),
        np.ascontiguousarray(nacl_dm_gl._dielectric, dtype="double"),
        pos,
        factor,
        nacl_dm_gl._Lambda,
        nacl_dm_gl.Q_DIRECTION_TOLERANCE,
        None,
    )
    ref = nacl_dm_gl._get_c_recip_dipole_dipole(q_cart, None)
    np.testing.assert_allclose(dd, ref, atol=1e-13, rtol=1e-13)


def test_recip_dipole_dipole_q0_matches_phonopy(nacl_dm_gl) -> None:
    """Rust dd_q0 equals phonopy's C-side dd_q0 to machine precision."""
    pos = np.ascontiguousarray(nacl_dm_gl.primitive.positions, dtype="double")
    dd_q0 = np.zeros((len(pos), 3, 3), dtype="cdouble", order="C")
    phono3py_rs.recip_dipole_dipole_q0(
        dd_q0,
        np.ascontiguousarray(nacl_dm_gl._G_list, dtype="double"),
        np.ascontiguousarray(nacl_dm_gl._born, dtype="double"),
        np.ascontiguousarray(nacl_dm_gl._dielectric, dtype="double"),
        pos,
        nacl_dm_gl._Lambda,
        nacl_dm_gl.Q_DIRECTION_TOLERANCE,
    )
    np.testing.assert_allclose(dd_q0, nacl_dm_gl._dd_q0, atol=1e-13, rtol=1e-13)


def _extract_grid_inputs(ph3):
    """Collect the non-NAC inputs for the grid-wide Rust builders.

    Gathers the fields needed from a Phono3py instance.

    """
    dm = ph3.dynamical_matrix
    prim = dm.primitive
    svecs, multi = prim.get_smallest_vectors()
    masses = np.asarray(prim.masses, dtype="double")
    fc = dm.force_constants
    p2s = np.asarray(prim.p2s_map, dtype="int64")
    s2p = np.asarray(prim.s2p_map, dtype="int64")
    if fc.shape[0] == fc.shape[1]:
        fc_p2s = p2s
        fc_s2p = s2p
    else:
        p2p = prim.p2p_map
        s2pp = np.array([p2p[s2p[i]] for i in range(len(s2p))], dtype="int64")
        fc_p2s = np.arange(len(p2s), dtype="int64")
        fc_s2p = s2pp
    return {
        "fc": np.ascontiguousarray(fc, dtype="double"),
        "svecs": np.ascontiguousarray(svecs, dtype="double"),
        "multi": np.ascontiguousarray(multi, dtype="int64"),
        "masses": masses,
        "fc_p2s": fc_p2s,
        "fc_s2p": fc_s2p,
        "grid_address": np.ascontiguousarray(ph3.grid.addresses, dtype="int64"),
        "QDinv": np.ascontiguousarray(ph3.grid.QDinv, dtype="double"),
    }


def _reference_dm_at_gridpoints(dm, grid_address, QDinv) -> np.ndarray:
    """Build the full dynamical-matrix buffer via phonopy.

    Calls phonopy's per-q-point DM builder at each grid point.

    """
    num_phonons = len(grid_address)
    num_band = len(dm.primitive) * 3
    out = np.zeros((num_phonons, num_band, num_band), dtype="cdouble")
    for gp in range(num_phonons):
        q = np.dot(grid_address[gp], QDinv.T)
        dm.run(q)
        out[gp] = dm.dynamical_matrix
    return out


def test_dynamical_matrices_at_gridpoints_no_nac_matches_phonopy(si_pbesol) -> None:
    """Rust grid-wide no-NAC DM matches phonopy's per-q-point DM.

    Checked at every grid point to machine precision.

    """
    si_pbesol.mesh_numbers = [3, 3, 3]
    si_pbesol.init_phph_interaction()
    dm = si_pbesol.dynamical_matrix
    inp = _extract_grid_inputs(si_pbesol)
    num_phonons = len(inp["grid_address"])
    num_band = len(dm.primitive) * 3

    dynmats = np.zeros((num_phonons, num_band, num_band), dtype="cdouble")
    undone = np.arange(num_phonons, dtype="int64")
    phono3py_rs.dynamical_matrices_at_gridpoints(
        dynmats,
        undone,
        inp["grid_address"],
        inp["QDinv"],
        inp["fc"],
        inp["svecs"],
        inp["multi"],
        inp["masses"],
        inp["fc_p2s"],
        inp["fc_s2p"],
    )
    ref = _reference_dm_at_gridpoints(dm, inp["grid_address"], inp["QDinv"])
    np.testing.assert_allclose(dynmats, ref, atol=1e-13, rtol=1e-13)


def test_dynamical_matrices_at_gridpoints_gonze_matches_phonopy(
    nacl_dm_gl, nacl_pbe
) -> None:
    """Rust grid-wide Gonze-Lee DM matches phonopy's per-q-point DM.

    Uses DynamicalMatrixGL output at every grid point.

    """
    dm = nacl_dm_gl
    inp = _extract_grid_inputs(nacl_pbe)
    num_phonons = len(inp["grid_address"])
    num_band = len(dm.primitive) * 3

    rec_lat = np.ascontiguousarray(np.linalg.inv(dm.primitive.cell), dtype="double")
    pos = np.ascontiguousarray(dm.primitive.positions, dtype="double")

    # Gonze uses fc with dipole-dipole contribution removed.
    gonze_fc = dm.short_range_force_constants
    if gonze_fc is None:
        dm.make_Gonze_nac_dataset()
        gonze_fc = dm.short_range_force_constants

    dynmats = np.zeros((num_phonons, num_band, num_band), dtype="cdouble")
    undone = np.arange(num_phonons, dtype="int64")
    phono3py_rs.dynamical_matrices_at_gridpoints_gonze(
        dynmats,
        undone,
        inp["grid_address"],
        inp["QDinv"],
        np.ascontiguousarray(gonze_fc, dtype="double"),
        inp["svecs"],
        inp["multi"],
        inp["masses"],
        inp["fc_p2s"],
        inp["fc_s2p"],
        np.ascontiguousarray(dm._born, dtype="double"),
        np.ascontiguousarray(dm._dielectric, dtype="double"),
        rec_lat,
        pos,
        np.ascontiguousarray(dm._dd_q0, dtype="cdouble"),
        np.ascontiguousarray(dm._G_list, dtype="double"),
        dm.nac_factor,
        dm._Lambda,
    )
    ref = _reference_dm_at_gridpoints(dm, inp["grid_address"], inp["QDinv"])
    np.testing.assert_allclose(dynmats, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_charge_sum_matches_reference(seed: int) -> None:
    """Rust charge_sum equals the explicit outer-product reference."""
    rng = np.random.default_rng(seed)
    num_patom = 3
    born = np.ascontiguousarray(rng.standard_normal((num_patom, 3, 3)))
    q_cart = np.ascontiguousarray(rng.standard_normal(3))
    factor = 0.7
    out = np.zeros((num_patom, num_patom, 3, 3), dtype="double")
    phono3py_rs.charge_sum(out, factor, q_cart, born)
    ref = _ref_charge_sum(factor, q_cart, born)
    np.testing.assert_allclose(out, ref, atol=1e-15, rtol=1e-15)
