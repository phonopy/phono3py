"""Compare the Rust interaction backend against the C one."""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.interaction import Interaction, run_interaction_rust

pytest.importorskip("phono3py_rs")


def _make_interaction(
    ph3: Phono3py,
    *,
    symmetrize_fc3q: bool,
    make_r0_average: bool,
) -> Interaction:
    ph3.mesh_numbers = [4, 4, 4]
    assert ph3.grid is not None
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        cutoff_frequency=1e-4,
        symmetrize_fc3q=symmetrize_fc3q,
        make_r0_average=make_r0_average,
    )
    itr.init_dynamical_matrix(ph3.fc2, ph3.phonon_supercell, ph3.phonon_primitive)
    itr.run_phonon_solver()
    itr.set_grid_point(1)
    return itr


def _run_c_and_capture(itr: Interaction) -> np.ndarray:
    """Run C interaction and return a copy of the raw strength.

    Matches ``Interaction._run_c``: ``_g_zero`` is all zeros when
    symmetrize_fc3q is True or when g_zero is not supplied.  The unit
    conversion factor is applied in-place by ``_run_c`` after the C
    call; we divide it back out so the Rust comparison is against the
    pre-multiplied C kernel output.
    """
    itr.run()
    assert itr.interaction_strength is not None
    return np.array(itr.interaction_strength / itr.unit_conversion_factor)


def _run_rust(itr: Interaction) -> np.ndarray:
    num_band = len(itr.primitive) * 3
    triplets_at_q, *_ = itr.get_triplets_at_q()
    assert triplets_at_q is not None
    num_triplets = len(triplets_at_q)
    num_band0 = len(itr.band_indices)
    out = np.zeros(
        (num_triplets, num_band0, num_band, num_band),
        dtype="double",
        order="C",
    )
    g_zero = np.zeros(out.shape, dtype="byte", order="C")
    frequencies, eigenvectors, _ = itr.get_phonons()
    assert frequencies is not None and eigenvectors is not None
    svecs, multi = itr.primitive.get_smallest_vectors()
    run_interaction_rust(
        out,
        g_zero,
        frequencies,
        eigenvectors,
        triplets_at_q,
        itr.bz_grid.addresses,
        itr.bz_grid.D_diag,
        itr.bz_grid.Q,
        itr.fc3,
        itr.fc3_nonzero_indices,
        svecs,
        multi,
        np.asarray(itr.primitive.masses, dtype="double"),
        np.asarray(itr.primitive.p2s_map, dtype="int64"),
        np.asarray(itr.primitive.s2p_map, dtype="int64"),
        np.asarray(itr.band_indices, dtype="int64"),
        itr.symmetrize_fc3q,
        itr.make_r0_average,
        itr.all_shortest,
        itr.cutoff_frequency,
    )
    return out


def test_interaction_rust_vs_c_plain(si_pbesol: Phono3py):
    """Non-symmetrized, r0-average=True (default) matches C."""
    itr_c = _make_interaction(si_pbesol, symmetrize_fc3q=False, make_r0_average=True)
    c_out = _run_c_and_capture(itr_c)

    itr_rust = _make_interaction(si_pbesol, symmetrize_fc3q=False, make_r0_average=True)
    rust_out = _run_rust(itr_rust)
    np.testing.assert_allclose(rust_out, c_out, rtol=1e-10, atol=1e-18)


def test_interaction_rust_vs_c_symmetrized(si_pbesol: Phono3py):
    """symmetrize_fc3q=True path matches C."""
    itr_c = _make_interaction(si_pbesol, symmetrize_fc3q=True, make_r0_average=True)
    c_out = _run_c_and_capture(itr_c)

    itr_rust = _make_interaction(si_pbesol, symmetrize_fc3q=True, make_r0_average=True)
    rust_out = _run_rust(itr_rust)
    np.testing.assert_allclose(rust_out, c_out, rtol=1e-10, atol=1e-18)


def test_phonon_solver_rust_vs_c(si_pbesol: Phono3py):
    """Interaction(lang='Rust') uses the Rust phonon solver; must match C."""
    si_pbesol.mesh_numbers = [4, 4, 4]
    assert si_pbesol.grid is not None

    def _build(lang: str) -> Interaction:
        itr = Interaction(
            si_pbesol.primitive,
            si_pbesol.grid,
            si_pbesol.primitive_symmetry,
            fc3=si_pbesol.fc3,
            cutoff_frequency=1e-4,
            lang=lang,
        )
        itr.init_dynamical_matrix(
            si_pbesol.fc2, si_pbesol.phonon_supercell, si_pbesol.phonon_primitive
        )
        itr.run_phonon_solver()
        return itr

    itr_c = _build("C")
    itr_rust = _build("Rust")
    freq_c, _, _ = itr_c.get_phonons()
    freq_rust, _, _ = itr_rust.get_phonons()
    assert freq_c is not None and freq_rust is not None
    # Eigenvectors are not compared because degenerate eigenspaces can mix
    # individual eigenvectors even when the subspaces agree.
    np.testing.assert_allclose(freq_rust, freq_c, rtol=1e-10, atol=1e-12)
