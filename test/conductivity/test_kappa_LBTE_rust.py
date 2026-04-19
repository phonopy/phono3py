"""Compare the Rust LBTE backend against the C one.

Drives ``LBTECalculator`` through both the irreducible and reducible
collision-matrix paths so that the Rust ports of
``collision_matrix``, ``reducible_collision_matrix``,
``symmetrize_collision_matrix``, and ``expand_collision_matrix`` are
exercised end-to-end and compared against the C implementations.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.conductivity.calculators import LBTECalculator
from phono3py.conductivity.factory import conductivity_calculator
from phono3py.phonon3.interaction import Interaction

pytest.importorskip("phono3py_rs")


def _build_interaction(
    ph3: Phono3py, mesh: list[int], *, lang: str = "C"
) -> Interaction:
    """Build an Interaction with the given phonon-solver lang (default C)."""
    ph3.mesh_numbers = mesh
    assert ph3.grid is not None
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        cutoff_frequency=1e-4,
        lang=lang,
    )
    itr.init_dynamical_matrix(ph3.fc2, ph3.phonon_supercell, ph3.phonon_primitive)
    itr.run_phonon_solver()
    return itr


def _run_lbte(
    ph3: Phono3py,
    mesh: list[int],
    *,
    lang: str,
    interaction_lang: str = "C",
    is_reducible: bool = False,
    sigmas: Sequence[float | None] = (None,),
    is_isotope: bool = False,
) -> np.ndarray:
    """Run one LBTE solve and return the kappa tensor."""
    itr = _build_interaction(ph3, mesh, lang=interaction_lang)
    lbte = conductivity_calculator(
        itr,
        temperatures=np.array([300.0], dtype="double"),
        sigmas=list(sigmas),
        method="std-lbte",
        is_reducible_collision_matrix=is_reducible,
        is_isotope=is_isotope,
        pinv_solver=5,
        lang=lang,
    )
    assert isinstance(lbte, LBTECalculator)
    lbte.run()
    return lbte.kappa.copy()


def test_kappa_LBTE_rust_vs_c(si_pbesol: Phono3py):
    """Irreducible LBTE path: kappa from Rust matches C."""
    kappa_c = _run_lbte(si_pbesol, [5, 5, 5], lang="C")
    kappa_rust = _run_lbte(si_pbesol, [5, 5, 5], lang="Rust")
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_LBTE_reducible_rust_vs_c(si_pbesol: Phono3py):
    """Reducible LBTE path: kappa from Rust matches C."""
    kappa_c = _run_lbte(si_pbesol, [5, 5, 5], lang="C", is_reducible=True)
    kappa_rust = _run_lbte(si_pbesol, [5, 5, 5], lang="Rust", is_reducible=True)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_LBTE_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Irreducible LBTE with Gaussian smearing: Rust kappa matches C."""
    kappa_c = _run_lbte(si_pbesol, [5, 5, 5], lang="C", sigmas=[0.1])
    kappa_rust = _run_lbte(si_pbesol, [5, 5, 5], lang="Rust", sigmas=[0.1])
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_LBTE_rust_vs_c_isotope(si_pbesol: Phono3py):
    """Irreducible LBTE + isotope scattering: Rust kappa matches C."""
    kappa_c = _run_lbte(si_pbesol, [5, 5, 5], lang="C", is_isotope=True)
    kappa_rust = _run_lbte(si_pbesol, [5, 5, 5], lang="Rust", is_isotope=True)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_LBTE_rust_vs_c_full_rust(si_pbesol: Phono3py):
    """All-Rust (phonon solver + conductivity) vs all-C, dense mesh.

    The Rust and C phonon solvers can produce different eigenvectors
    within degenerate subspaces, which propagates into the interaction
    strength and kappa.  A denser mesh averages this down; the residual
    tolerance is looser than in the other tests.

    """
    kappa_c = _run_lbte(
        si_pbesol, [11, 11, 11], lang="C", interaction_lang="C"
    )
    kappa_rust = _run_lbte(
        si_pbesol, [11, 11, 11], lang="Rust", interaction_lang="Rust"
    )
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=5e-3, atol=1e-10)
