"""Tests for velocity solvers of conductivity calculations."""

from __future__ import annotations

import numpy as np

from phono3py import Phono3py
from phono3py.conductivity.velocity_solvers import (
    GroupVelocitySolver,
    VelocityMatrixSolver,
)


def test_GroupVelocitySolver_at_grid_points(aln_lda: Phono3py):
    """Test that grid points given at once and one by one agree."""
    aln_lda.mesh_numbers = [4, 4, 3]
    aln_lda.init_phph_interaction()
    pp = aln_lda.phph_interaction
    grid_points = pp.bz_grid.grg2bzg
    num_band0 = len(pp.band_indices)
    solver = GroupVelocitySolver(pp)

    result = solver.compute(grid_points)

    assert result.group_velocities.shape == (len(grid_points), num_band0, 3)
    assert result.gv_by_gv is not None
    assert result.gv_by_gv.shape == (len(grid_points), num_band0, 6)
    assert result.vm_by_vm is None
    # Group velocities depend on the grid point.
    assert np.abs(result.group_velocities - result.group_velocities[1]).max() > 1

    for i_gp, grid_point in enumerate(grid_points):
        result_at_gp = solver.compute(np.array([grid_point], dtype="int64"))
        np.testing.assert_allclose(
            result.group_velocities[i_gp], result_at_gp.group_velocities[0], atol=1e-8
        )
        assert result_at_gp.gv_by_gv is not None
        np.testing.assert_allclose(
            result.gv_by_gv[i_gp], result_at_gp.gv_by_gv[0], atol=1e-8
        )

    result_reversed = solver.compute(grid_points[::-1])
    assert result_reversed.gv_by_gv is not None
    np.testing.assert_allclose(
        result.group_velocities, result_reversed.group_velocities[::-1], atol=1e-8
    )
    np.testing.assert_allclose(
        result.gv_by_gv, result_reversed.gv_by_gv[::-1], atol=1e-8
    )


def test_VelocityMatrixSolver_at_grid_points(aln_lda: Phono3py):
    """Test that grid points given at once and one by one agree."""
    aln_lda.mesh_numbers = [4, 4, 3]
    aln_lda.init_phph_interaction()
    pp = aln_lda.phph_interaction
    grid_points = pp.bz_grid.grg2bzg
    num_band = len(pp.primitive) * 3
    solver = VelocityMatrixSolver(pp)

    result = solver.compute(grid_points)

    assert result.group_velocities.shape == (len(grid_points), num_band, 3)
    assert result.gv_by_gv is None
    assert result.vm_by_vm is not None
    assert result.vm_by_vm.shape == (len(grid_points), num_band, num_band, 6)
    # Velocity matrices depend on the grid point.
    assert np.abs(result.vm_by_vm - result.vm_by_vm[1]).max() > 1

    for i_gp, grid_point in enumerate(grid_points):
        result_at_gp = solver.compute(np.array([grid_point], dtype="int64"))
        np.testing.assert_allclose(
            result.group_velocities[i_gp], result_at_gp.group_velocities[0], atol=1e-8
        )
        assert result_at_gp.vm_by_vm is not None
        np.testing.assert_allclose(
            result.vm_by_vm[i_gp], result_at_gp.vm_by_vm[0], atol=1e-6
        )

    result_reversed = solver.compute(grid_points[::-1])
    assert result_reversed.vm_by_vm is not None
    np.testing.assert_allclose(
        result.group_velocities, result_reversed.group_velocities[::-1], atol=1e-8
    )
    np.testing.assert_allclose(
        result.vm_by_vm, result_reversed.vm_by_vm[::-1], atol=1e-6
    )


def test_velocity_solvers_without_grid_points(aln_lda: Phono3py):
    """Test that an empty list of grid points gives empty arrays."""
    aln_lda.mesh_numbers = [4, 4, 3]
    aln_lda.init_phph_interaction()
    pp = aln_lda.phph_interaction
    grid_points = np.array([], dtype="int64")
    num_band = len(pp.primitive) * 3

    result = GroupVelocitySolver(pp).compute(grid_points)
    assert result.group_velocities.shape == (0, len(pp.band_indices), 3)
    assert result.gv_by_gv is not None
    assert result.gv_by_gv.shape == (0, len(pp.band_indices), 6)

    result = VelocityMatrixSolver(pp).compute(grid_points)
    assert result.group_velocities.shape == (0, num_band, 3)
    assert result.vm_by_vm is not None
    assert result.vm_by_vm.shape == (0, num_band, num_band, 6)
