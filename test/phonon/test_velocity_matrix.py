"""Tests for VelocityMatrix."""

import numpy as np

from phono3py import Phono3py
from phono3py.conductivity.utils import get_kappa_star_operations
from phono3py.phonon.velocity_matrix import VelocityMatrix


def test_velocity_matrices_are_not_projected(si_pbesol: Phono3py):
    """Test that run does not overwrite the diagonal of velocity matrices.

    Group velocities are projected by the little group of q, but velocity
    matrices are not. The q-point is on Gamma-L, where the projection changes
    the group velocities of some bands, as the test asserts. At a q-point where
    the projection does nothing, this test would pass with the overwritten
    diagonal.

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    pp = si_pbesol.phph_interaction
    reciprocal_operations, rotations_cartesian = get_kappa_star_operations(
        pp.bz_grid, True
    )
    velocity = VelocityMatrix(
        pp.dynamical_matrix,
        rotations_cartesian=rotations_cartesian,
        reciprocal_operations=reciprocal_operations,
        frequency_factor_to_THz=pp.frequency_factor_to_THz,
    )
    q_point = np.array([1, 1, 1], dtype="double") / 9

    velocity.run([q_point])

    expected = velocity._calculate_velocity_matrix_at_q(q_point)
    expected_diagonal = np.einsum("aii->ia", expected).real
    projected = expected_diagonal @ velocity._get_projector(q_point).T

    assert not np.allclose(expected_diagonal, projected)
    np.testing.assert_allclose(velocity.velocity_matrices[0], expected, atol=1e-10)
    np.testing.assert_allclose(velocity.group_velocities[0], projected, atol=1e-10)
