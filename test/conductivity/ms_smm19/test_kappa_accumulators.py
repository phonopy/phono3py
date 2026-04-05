"""Unit tests for WignerRTAKappaAccumulator."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.grid_point_data import GridPointAggregates
from phono3py.conductivity.ms_smm19.kappa_accumulators import WignerRTAKappaAccumulator


def _make_dummy_context(num_gp: int = 1) -> ConductivityContext:
    """Create a minimal ConductivityContext for unit tests."""
    bz_grid = MagicMock()
    return ConductivityContext(
        grid_points=np.zeros(num_gp, dtype="int64"),
        ir_grid_points=np.zeros(num_gp, dtype="int64"),
        grid_weights=np.ones(num_gp, dtype="int64"),
        bz_grid=bz_grid,
        mesh_numbers=np.array([1, 1, 1], dtype="int64"),
        frequencies=np.ones((1, 6), dtype="double"),
        eigenvectors=np.eye(6, dtype="complex128").reshape(1, 6, 6),
        point_operations=np.eye(3, dtype="int64").reshape(1, 3, 3),
        rotations_cartesian=np.eye(3, dtype="double").reshape(1, 3, 3),
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[None],
        sigma_cutoff_width=None,
        boundary_mfp=None,
        band_indices=np.arange(6, dtype="int64"),
        cutoff_frequency=0.0,
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
        num_sampling_grid_points=num_gp,
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
    acc = WignerRTAKappaAccumulator(context=ctx, volume=1.0)

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
    acc = WignerRTAKappaAccumulator(context=ctx, volume=1.0)

    aggregates = _make_aggregates(with_velocity_operator=False)
    acc.finalize(aggregates)

    extra = acc.get_extra_grid_point_output()
    assert extra is not None
    assert extra["velocity_operator"] is None
