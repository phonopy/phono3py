"""Tests of collision matrix kernels."""

from __future__ import annotations

import dataclasses

import numpy as np

from phono3py import Phono3py
from phono3py.conductivity.build_components import build_rot_grid_points
from phono3py.conductivity.collision_matrix_kernel import (
    create_collision_matrix_kernel,
)


def test_mode_kappa_skips_lowest_modes_at_gamma(si_pbesol: Phono3py):
    """Test that mode kappa skips the three lowest modes at Gamma-point.

    The grid points are reversed so that Gamma-point is not the first.

    """
    si_pbesol.mesh_numbers = [4, 4, 4]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(is_LBTE=True, temperatures=[300])
    kappa_settings = si_pbesol.thermal_conductivity._kappa_settings
    grid_points = kappa_settings.grid_points[::-1].copy()
    assert grid_points[-1] == kappa_settings.bz_grid.gp_Gamma
    kappa_settings = dataclasses.replace(kappa_settings, grid_points=grid_points)

    assert si_pbesol.phph_interaction is not None
    phonons = si_pbesol.phph_interaction.phonons
    kernel = create_collision_matrix_kernel(
        kappa_settings,
        phonons.frequencies,
        phonons.degenerate_ids,
        rot_grid_points=build_rot_grid_points(kappa_settings),
    )
    num_gp = len(grid_points)
    num_band = phonons.frequencies.shape[1]
    X = np.ones((num_gp * num_band, 3), dtype="double")
    mode_kappa = np.zeros((1, 1, num_gp, num_band, 6), dtype="double")
    kernel._set_mode_kappa(mode_kappa, X, X, num_gp, 0, 0)

    assert (mode_kappa[0, 0, -1, :3] == 0).all()
    assert (mode_kappa[0, 0, -1, 3:, 0] > 0).all()
    assert (mode_kappa[0, 0, 0, :, 0] > 0).all()
