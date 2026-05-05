"""Compare the Rust pp_collision backend against the C one.

Drives ``RTAScatteringSolver`` through its low-memory dispatch path so
that both the tetrahedron (sigma=None) and Gaussian-smearing branches
of ``phono3c.pp_collision`` / ``phono3c.pp_collision_with_sigma`` are
exercised against the Rust implementation.

"""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.conductivity.scattering_solvers import RTAScatteringSolver
from phono3py.phonon3.interaction import Interaction

pytest.importorskip("phonors")


def _build_interaction(ph3: Phono3py) -> Interaction:
    """Build an Interaction with default (C) phonon solver for parity tests."""
    ph3.mesh_numbers = [4, 4, 4]
    assert ph3.grid is not None
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        cutoff_frequency=1e-4,
    )
    itr.init_dynamical_matrix(ph3.fc2, ph3.phonon_supercell, ph3.phonon_primitive)
    itr.run_phonon_solver()
    return itr


def _run_rta_solver(
    ph3: Phono3py,
    *,
    sigmas,
    is_N_U: bool,
    lang: str,
    grid_point: int = 1,
):
    """Run one RTA solve at a grid point with the chosen backend."""
    itr = _build_interaction(ph3)
    solver = RTAScatteringSolver(
        itr,
        sigmas=sigmas,
        temperatures=np.array([300.0], dtype="double"),
        is_N_U=is_N_U,
        lang=lang,
    )
    result = solver.compute(grid_point)
    return result.gamma, solver.gamma_N, solver.gamma_U


def test_pp_collision_rust_vs_c_tetrahedron(si_pbesol: Phono3py):
    """Tetrahedron (sigma=None) low-memory path matches between backends."""
    gamma_c, _, _ = _run_rta_solver(si_pbesol, sigmas=[None], is_N_U=False, lang="C")
    gamma_rust, _, _ = _run_rta_solver(
        si_pbesol, sigmas=[None], is_N_U=False, lang="Rust"
    )
    np.testing.assert_allclose(gamma_rust, gamma_c, rtol=1e-10, atol=1e-14)


def test_pp_collision_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Smearing (sigma=0.1) low-memory path matches between backends."""
    gamma_c, _, _ = _run_rta_solver(si_pbesol, sigmas=[0.1], is_N_U=False, lang="C")
    gamma_rust, _, _ = _run_rta_solver(
        si_pbesol, sigmas=[0.1], is_N_U=False, lang="Rust"
    )
    np.testing.assert_allclose(gamma_rust, gamma_c, rtol=1e-10, atol=1e-14)


def test_pp_collision_rust_vs_c_tetrahedron_NU(si_pbesol: Phono3py):
    """Tetrahedron + N/U decomposition matches between backends."""
    gamma_c, gN_c, gU_c = _run_rta_solver(
        si_pbesol, sigmas=[None], is_N_U=True, lang="C"
    )
    gamma_rust, gN_rust, gU_rust = _run_rta_solver(
        si_pbesol, sigmas=[None], is_N_U=True, lang="Rust"
    )
    assert gN_c is not None and gU_c is not None
    assert gN_rust is not None and gU_rust is not None
    np.testing.assert_allclose(gamma_rust, gamma_c, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(gN_rust, gN_c, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(gU_rust, gU_c, rtol=1e-10, atol=1e-14)


def test_pp_collision_rust_vs_c_sigma_NU(si_pbesol: Phono3py):
    """Smearing + N/U decomposition matches between backends."""
    gamma_c, gN_c, gU_c = _run_rta_solver(
        si_pbesol, sigmas=[0.1], is_N_U=True, lang="C"
    )
    gamma_rust, gN_rust, gU_rust = _run_rta_solver(
        si_pbesol, sigmas=[0.1], is_N_U=True, lang="Rust"
    )
    assert gN_c is not None and gU_c is not None
    assert gN_rust is not None and gU_rust is not None
    np.testing.assert_allclose(gamma_rust, gamma_c, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(gN_rust, gN_c, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(gU_rust, gU_c, rtol=1e-10, atol=1e-14)


def _run_rta_solver_full_path(
    ph3: Phono3py,
    *,
    sigmas,
    lang: str,
    grid_point: int = 1,
):
    """Run one RTA solve forcing the full-gamma (ImagSelfEnergy) path."""
    itr = _build_interaction(ph3)
    solver = RTAScatteringSolver(
        itr,
        sigmas=sigmas,
        temperatures=np.array([300.0], dtype="double"),
        is_gamma_detail=True,
        lang=lang,
    )
    result = solver.compute(grid_point)
    return result.gamma, solver.gamma_detail_at_q


def test_full_gamma_path_rust_vs_c_tetrahedron(si_pbesol: Phono3py):
    """Full-gamma path (ImagSelfEnergy), tetrahedron: Rust matches C."""
    gamma_c, detail_c = _run_rta_solver_full_path(si_pbesol, sigmas=[None], lang="C")
    gamma_rust, detail_rust = _run_rta_solver_full_path(
        si_pbesol, sigmas=[None], lang="Rust"
    )
    assert detail_c is not None and detail_rust is not None
    np.testing.assert_allclose(gamma_rust, gamma_c, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(detail_rust, detail_c, rtol=1e-10, atol=1e-14)


def test_full_gamma_path_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Full-gamma path (ImagSelfEnergy), Gaussian smearing: Rust matches C."""
    gamma_c, detail_c = _run_rta_solver_full_path(si_pbesol, sigmas=[0.1], lang="C")
    gamma_rust, detail_rust = _run_rta_solver_full_path(
        si_pbesol, sigmas=[0.1], lang="Rust"
    )
    assert detail_c is not None and detail_rust is not None
    np.testing.assert_allclose(gamma_rust, gamma_c, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(detail_rust, detail_c, rtol=1e-10, atol=1e-14)
