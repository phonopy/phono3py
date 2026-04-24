"""Compare the Rust RTA backend against the C one.

Drives ``RTACalculator`` through both the low-memory (fused
``pp_collision``) and full-gamma (separate kernels) paths so that the
Rust ports of ``pp_collision`` / ``imag_self_energy_with_g`` /
``isotope`` are exercised end-to-end and compared against the C
implementations.

"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.conductivity.calculators import RTACalculator
from phono3py.conductivity.factory import conductivity_calculator
from phono3py.conductivity.scattering_solvers import RTAScatteringSolver
from phono3py.phonon3.interaction import Interaction

pytest.importorskip("phono3py_rs")


def _build_interaction(
    ph3: Phono3py, mesh: Sequence[int], *, lang: str = "C"
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


def _run_rta(
    ph3: Phono3py,
    mesh: Sequence[int],
    *,
    lang: str,
    interaction_lang: str = "C",
    sigmas: Sequence[float | None] = (None,),
    is_isotope: bool = False,
    is_full_pp: bool = False,
    is_N_U: bool = False,
    is_gamma_detail: bool = False,
    rust_gp_batch_size: int | None = None,
) -> np.ndarray:
    """Run one RTA solve and return a copy of the kappa tensor."""
    itr = _build_interaction(ph3, mesh, lang=interaction_lang)
    rta = conductivity_calculator(
        itr,
        temperatures=np.array([300.0], dtype="double"),
        sigmas=list(sigmas),
        method="std-rta",
        is_isotope=is_isotope,
        is_full_pp=is_full_pp,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        lang=lang,
        rust_gp_batch_size=rust_gp_batch_size,
    )
    assert isinstance(rta, RTACalculator)
    rta.run()
    return rta.kappa.copy()


@pytest.mark.parametrize(
    "sigmas",
    [pytest.param([None], id="tetra"), pytest.param([0.1], id="sigma0.1")],
)
def test_kappa_RTA_rust_vs_c_lowmem(
    si_pbesol: Phono3py, sigmas: Sequence[float | None]
):
    """Low-memory (fused pp_collision) path: Rust kappa matches C."""
    kappa_c = _run_rta(si_pbesol, [5, 5, 5], lang="C", sigmas=sigmas)
    kappa_rust = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", sigmas=sigmas)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_vs_c_isotope(si_pbesol: Phono3py):
    """Low-memory + isotope scattering: Rust kappa matches C."""
    kappa_c = _run_rta(si_pbesol, [5, 5, 5], lang="C", is_isotope=True)
    kappa_rust = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", is_isotope=True)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_vs_c_N_U(si_pbesol: Phono3py):
    """Low-memory + N/U decomposition: Rust kappa matches C."""
    kappa_c = _run_rta(si_pbesol, [5, 5, 5], lang="C", is_N_U=True)
    kappa_rust = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", is_N_U=True)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_vs_c_full_pp(si_pbesol: Phono3py):
    """Full-gamma (separate kernels) path: Rust kappa matches C."""
    kappa_c = _run_rta(si_pbesol, [5, 5, 5], lang="C", is_full_pp=True)
    kappa_rust = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", is_full_pp=True)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_vs_c_compact_fc(si_pbesol_compact_fc: Phono3py):
    """Compact-fc fixture, low-memory path: Rust kappa matches C."""
    kappa_c = _run_rta(si_pbesol_compact_fc, [5, 5, 5], lang="C")
    kappa_rust = _run_rta(si_pbesol_compact_fc, [5, 5, 5], lang="Rust")
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_vs_c_gamma_detail(si_pbesol: Phono3py):
    """Full-gamma + detailed imag self energy: Rust kappa matches C."""
    kappa_c = _run_rta(si_pbesol, [5, 5, 5], lang="C", is_gamma_detail=True)
    kappa_rust = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", is_gamma_detail=True)
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_vs_c_nosym(si_pbesol: Phono3py, si_pbesol_nosym: Phono3py):
    """No-symmetry fixture, low-memory path: Rust kappa matches C."""
    si_pbesol_nosym.fc2 = si_pbesol.fc2
    si_pbesol_nosym.fc3 = si_pbesol.fc3
    kappa_c = _run_rta(si_pbesol_nosym, [4, 4, 4], lang="C")
    kappa_rust = _run_rta(si_pbesol_nosym, [4, 4, 4], lang="Rust")
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_kappa_RTA_rust_batched_matches_nonbatched(
    si_pbesol: Phono3py, batch_size: int
):
    """Rust batched path (rust_gp_batch_size > 0) matches non-batched path."""
    kappa_nonbatched = _run_rta(si_pbesol, [5, 5, 5], lang="Rust")
    kappa_batched = _run_rta(
        si_pbesol, [5, 5, 5], lang="Rust", rust_gp_batch_size=batch_size
    )
    np.testing.assert_allclose(kappa_batched, kappa_nonbatched, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_batched_N_U(si_pbesol: Phono3py):
    """Rust batched path with N/U decomposition matches non-batched path."""
    kappa_nonbatched = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", is_N_U=True)
    kappa_batched = _run_rta(
        si_pbesol, [5, 5, 5], lang="Rust", is_N_U=True, rust_gp_batch_size=4
    )
    np.testing.assert_allclose(kappa_batched, kappa_nonbatched, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_batched_env_var(
    si_pbesol: Phono3py, monkeypatch: pytest.MonkeyPatch
):
    """PHONO3PY_RUST_GP_BATCH_SIZE env var enables batched path when arg is None."""
    monkeypatch.setenv("PHONO3PY_RUST_GP_BATCH_SIZE", "4")
    kappa_via_env = _run_rta(si_pbesol, [5, 5, 5], lang="Rust")
    monkeypatch.delenv("PHONO3PY_RUST_GP_BATCH_SIZE")
    kappa_nonbatched = _run_rta(si_pbesol, [5, 5, 5], lang="Rust")
    np.testing.assert_allclose(kappa_via_env, kappa_nonbatched, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_arg_overrides_env_var(
    si_pbesol: Phono3py, monkeypatch: pytest.MonkeyPatch
):
    """rust_gp_batch_size arg takes precedence over env var (arg=0 disables)."""
    monkeypatch.setenv("PHONO3PY_RUST_GP_BATCH_SIZE", "4")
    kappa_arg_zero = _run_rta(si_pbesol, [5, 5, 5], lang="Rust", rust_gp_batch_size=0)
    monkeypatch.delenv("PHONO3PY_RUST_GP_BATCH_SIZE")
    kappa_nonbatched = _run_rta(si_pbesol, [5, 5, 5], lang="Rust")
    np.testing.assert_allclose(kappa_arg_zero, kappa_nonbatched, rtol=1e-10, atol=1e-10)


def test_kappa_RTA_rust_batched_path_taken_via_arg(si_pbesol: Phono3py):
    """When rust_gp_batch_size > 0, compute_batched is actually called."""
    original = RTAScatteringSolver.compute_batched
    with patch.object(
        RTAScatteringSolver,
        "compute_batched",
        autospec=True,
        side_effect=original,
    ) as spy:
        _run_rta(si_pbesol, [5, 5, 5], lang="Rust", rust_gp_batch_size=4)
    assert spy.call_count > 0
    for call in spy.call_args_list:
        # With autospec=True, call.args == (self, gp_list)
        gp_list = call.args[1]
        assert 0 < len(gp_list) <= 4


def test_kappa_RTA_rust_nonbatched_path_taken_when_disabled(si_pbesol: Phono3py):
    """When rust_gp_batch_size=0, compute_batched is not called."""
    original = RTAScatteringSolver.compute_batched
    with patch.object(
        RTAScatteringSolver,
        "compute_batched",
        autospec=True,
        side_effect=original,
    ) as spy:
        _run_rta(si_pbesol, [5, 5, 5], lang="Rust", rust_gp_batch_size=0)
    assert spy.call_count == 0


def test_kappa_RTA_rust_batched_path_taken_via_env_var(
    si_pbesol: Phono3py, monkeypatch: pytest.MonkeyPatch
):
    """Env var enables batched path when rust_gp_batch_size arg is None."""
    monkeypatch.setenv("PHONO3PY_RUST_GP_BATCH_SIZE", "3")
    original = RTAScatteringSolver.compute_batched
    with patch.object(
        RTAScatteringSolver,
        "compute_batched",
        autospec=True,
        side_effect=original,
    ) as spy:
        _run_rta(si_pbesol, [5, 5, 5], lang="Rust")
    assert spy.call_count > 0
    for call in spy.call_args_list:
        gp_list = call.args[1]
        assert 0 < len(gp_list) <= 3


def test_kappa_RTA_rust_vs_c_full_rust(si_pbesol: Phono3py):
    """All-Rust (phonon solver + conductivity) vs all-C, dense mesh.

    The Rust and C phonon solvers can produce different eigenvectors
    within degenerate subspaces, which propagates into the interaction
    strength and kappa.  A denser mesh averages this down; the residual
    tolerance is looser than in the other tests.

    """
    kappa_c = _run_rta(si_pbesol, [11, 11, 11], lang="C", interaction_lang="C")
    kappa_rust = _run_rta(si_pbesol, [11, 11, 11], lang="Rust", interaction_lang="Rust")
    np.testing.assert_allclose(kappa_rust, kappa_c, rtol=5e-3, atol=0.0)
