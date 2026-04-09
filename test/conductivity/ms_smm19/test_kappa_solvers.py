"""Unit tests for WignerRTAKappaSolver."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.grid_point_data import GridPointAggregates
from phono3py.conductivity.ms_smm19.kappa_solvers import WignerRTAKappaSolver


def _make_dummy_context(num_gp: int = 1) -> KappaSettings:
    """Create a minimal KappaSettings for unit tests."""
    bz_grid = MagicMock()
    return KappaSettings(
        grid_points=np.zeros(num_gp, dtype="int64"),
        grid_weights=np.ones(num_gp, dtype="int64"),
        bz_grid=bz_grid,
        mesh_numbers=np.array([1, 1, 1], dtype="int64"),
        is_kappa_star=False,
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[None],
        boundary_mfp=None,
        band_indices=np.arange(6, dtype="int64"),
        cutoff_frequency=0.0,
        conversion_factor=1.0,
        gv_delta_q=None,
        is_reducible_collision_matrix=False,
    )


def _make_aggregates(
    num_gp: int = 1,
    nat3: int = 6,
    num_sigma: int = 1,
    num_temp: int = 1,
    *,
    with_velocity_operator: bool = True,
) -> GridPointAggregates:
    """Create a GridPointAggregates with minimal valid data.

    Parameters
    ----------
    num_gp : int
        Number of grid points.
    nat3 : int
        Number of bands (num_band0 == num_band for Wigner).
    num_sigma : int
        Number of broadening widths.
    num_temp : int
        Number of temperatures.
    with_velocity_operator : bool
        If True, include a random velocity_operator in extra.

    """
    extra: dict = {}
    if with_velocity_operator:
        extra["velocity_operator"] = (
            np.random.default_rng(42)
            .random((num_gp, nat3, nat3, 3))
            .astype("complex128")
        )
    return GridPointAggregates(
        group_velocities=np.zeros((num_gp, nat3, 3), dtype="double"),
        mode_heat_capacities=np.ones((num_temp, num_gp, nat3), dtype="double"),
        gv_by_gv=np.zeros((num_gp, nat3, 6), dtype="double"),
        gamma=np.ones((num_sigma, num_temp, num_gp, nat3), dtype="double") * 0.1,
        vm_by_vm=np.zeros((num_gp, nat3, nat3, 6), dtype="complex128"),
        extra=extra,
    )


def test_get_extra_grid_point_output_stores_velocity_operator():
    """Velocity operator is stored per grid point and returned."""
    num_gp = 2
    ctx = _make_dummy_context(num_gp=num_gp)
    acc = WignerRTAKappaSolver(
        kappa_settings=ctx,
        frequencies=np.ones((1, 6), dtype="double"),
        volume=1.0,
    )

    aggregates = _make_aggregates(num_gp=num_gp, with_velocity_operator=True)
    acc.finalize(aggregates)

    extra = acc.get_extra_grid_point_output()
    assert extra is not None
    np.testing.assert_array_equal(
        extra["velocity_operator"],
        aggregates.extra["velocity_operator"],
    )


def test_get_extra_grid_point_output_none_without_velocity_operator():
    """Velocity operator is None when not provided in aggregates.extra."""
    ctx = _make_dummy_context()
    acc = WignerRTAKappaSolver(
        kappa_settings=ctx,
        frequencies=np.ones((1, 6), dtype="double"),
        volume=1.0,
    )

    aggregates = _make_aggregates(with_velocity_operator=False)
    acc.finalize(aggregates)

    extra = acc.get_extra_grid_point_output()
    assert extra is not None
    assert extra["velocity_operator"] is None
