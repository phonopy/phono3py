"""Test ReciprocalToNormal class."""

from __future__ import annotations

import numpy as np

from phono3py import Phono3py
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phono3py.phonon3.reciprocal_to_normal import ReciprocalToNormal


def _get_interaction(ph3: Phono3py, mesh: list[int]) -> Interaction:
    ph3.mesh_numbers = mesh
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


def _run_r2n(
    ph3: Phono3py,
    itr: Interaction,
    cutoff_frequency: float,
) -> tuple[ReciprocalToNormal, np.ndarray]:
    """Return (r2n instance, first grid_triplet) after calling r2n.run()."""
    triplets_at_q, *_ = itr.get_triplets_at_q()
    assert triplets_at_q is not None

    frequencies, eigenvectors, _ = itr.get_phonons()
    assert frequencies is not None
    assert eigenvectors is not None

    r2r = RealToReciprocal(ph3.fc3, ph3.primitive, itr.mesh_numbers)
    r2n = ReciprocalToNormal(
        ph3.primitive,
        frequencies,
        eigenvectors,
        itr.band_indices,
        cutoff_frequency=cutoff_frequency,
    )

    grid_triplet = triplets_at_q[0]
    r2r.run(itr.bz_grid.addresses[grid_triplet])
    fc3_reciprocal = r2r.get_fc3_reciprocal()
    assert fc3_reciprocal is not None

    r2n.run(fc3_reciprocal, grid_triplet)
    return r2n, grid_triplet


def test_reciprocal_to_normal_shape(si_pbesol: Phono3py):
    """Output shape is (len(band_indices), num_band, num_band)."""
    itr = _get_interaction(si_pbesol, [4, 4, 4])
    itr.set_grid_point(1)

    r2n, _ = _run_r2n(si_pbesol, itr, cutoff_frequency=1e-4)
    result = r2n.get_reciprocal_to_normal()
    assert result is not None

    num_band = len(si_pbesol.primitive) * 3
    assert result.shape == (len(itr.band_indices), num_band, num_band)
    assert result.dtype == np.dtype("cdouble")


def test_reciprocal_to_normal_vs_c(si_pbesol: Phono3py):
    """Python ReciprocalToNormal matches C interaction strength for first triplet."""
    itr = _get_interaction(si_pbesol, [4, 4, 4])
    itr.set_grid_point(1)
    itr.run(lang="C")

    c_interaction = itr.interaction_strength
    assert c_interaction is not None

    r2n, _ = _run_r2n(si_pbesol, itr, cutoff_frequency=itr.cutoff_frequency)
    fc3_normal = r2n.get_reciprocal_to_normal()
    assert fc3_normal is not None

    py_interaction = np.abs(fc3_normal) ** 2 * itr.unit_conversion_factor
    np.testing.assert_allclose(py_interaction, c_interaction[0], rtol=1e-10, atol=0)


def test_reciprocal_to_normal_regression(si_pbesol: Phono3py):
    """Regression: sum of |fc3_normal|^2 over j,k for each band of Si grid_point=1."""
    # grid_triplet[0] = [1, 0, 4] for mesh [4,4,4] at grid_point 1
    ref_sum_abs2 = [
        8.2805236177e-06,
        8.2805236177e-06,
        4.3999290205e-05,
        2.3867510837e-04,
        2.5396898909e-04,
        2.5396898909e-04,
    ]

    itr = _get_interaction(si_pbesol, [4, 4, 4])
    itr.set_grid_point(1)

    r2n, _ = _run_r2n(si_pbesol, itr, cutoff_frequency=1e-4)
    result = r2n.get_reciprocal_to_normal()
    assert result is not None

    sum_abs2 = (np.abs(result) ** 2).sum(axis=(1, 2))
    np.testing.assert_allclose(sum_abs2, ref_sum_abs2, rtol=1e-8, atol=0)


def test_reciprocal_to_normal_cutoff(si_pbesol: Phono3py):
    """Elements with any frequency below cutoff remain zero."""
    itr = _get_interaction(si_pbesol, [4, 4, 4])
    itr.set_grid_point(1)

    frequencies, _, _ = itr.get_phonons()
    assert frequencies is not None
    max_freq = float(np.max(frequencies)) + 1.0

    r2n, _ = _run_r2n(si_pbesol, itr, cutoff_frequency=max_freq)
    result = r2n.get_reciprocal_to_normal()
    assert result is not None
    np.testing.assert_array_equal(result, 0)
