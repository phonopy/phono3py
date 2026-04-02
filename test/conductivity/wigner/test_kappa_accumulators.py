"""Unit tests for WignerRTAKappaAccumulator."""

from __future__ import annotations

import numpy as np

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.conductivity.wigner.kappa_accumulators import WignerRTAKappaAccumulator


def _make_result_with_velocity_operator(
    nat3: int = 6,
    num_sigma: int = 1,
    num_temp: int = 2,
) -> GridPointResult:
    """Create a GridPointResult with velocity_operator in extra.

    Wigner requires num_band0 == num_band (all phonon branches), so both
    are set to ``nat3``.

    """
    gp_input = GridPointInput(
        grid_point=0,
        q_point=np.zeros(3, dtype="double"),
        frequencies=np.ones(nat3, dtype="double"),
        eigenvectors=np.eye(nat3, dtype="complex128"),
        grid_weight=1,
        band_indices=np.arange(nat3, dtype="int64"),
    )
    result = GridPointResult(input=gp_input)
    result.group_velocities = np.zeros((nat3, 3), dtype="double")
    result.velocity_product = np.zeros((nat3, nat3, 6), dtype="complex128")
    result.heat_capacities = np.ones((num_temp, nat3), dtype="double")
    result.gamma = np.ones((num_sigma, num_temp, nat3), dtype="double") * 0.1
    result.extra["velocity_operator"] = (
        np.random.default_rng(42).random((nat3, nat3, 3)).astype("complex128")
    )
    return result


def test_get_extra_grid_point_output_stores_velocity_operator():
    """Velocity operator is stored per grid point and returned."""
    acc = WignerRTAKappaAccumulator(cutoff_frequency=0.0, conversion_factor_WTE=1.0)
    acc.prepare(num_sigma=1, num_temp=2, num_gp=2, num_band0=6)

    result0 = _make_result_with_velocity_operator()
    result1 = _make_result_with_velocity_operator()

    acc.accumulate(0, result0)
    acc.accumulate(1, result1)

    extra0 = acc.get_extra_grid_point_output(0)
    extra1 = acc.get_extra_grid_point_output(1)

    assert extra0 is not None
    assert extra1 is not None
    np.testing.assert_array_equal(
        extra0["velocity_operator"],
        result0.extra["velocity_operator"],
    )
    np.testing.assert_array_equal(
        extra1["velocity_operator"],
        result1.extra["velocity_operator"],
    )


def test_get_extra_grid_point_output_returns_none_without_velocity_operator():
    """Returns None when no velocity operator was stored."""
    acc = WignerRTAKappaAccumulator(cutoff_frequency=0.0, conversion_factor_WTE=1.0)
    acc.prepare(num_sigma=1, num_temp=2, num_gp=1, num_band0=6)

    # Result without velocity_operator in extra
    nat3 = 6
    gp_input = GridPointInput(
        grid_point=0,
        q_point=np.zeros(3, dtype="double"),
        frequencies=np.ones(nat3, dtype="double"),
        eigenvectors=np.eye(nat3, dtype="complex128"),
        grid_weight=1,
        band_indices=np.arange(nat3, dtype="int64"),
    )
    result = GridPointResult(input=gp_input)
    result.group_velocities = np.zeros((nat3, 3), dtype="double")
    result.velocity_product = np.zeros((nat3, nat3, 6), dtype="complex128")
    result.heat_capacities = np.ones((2, nat3), dtype="double")
    result.gamma = np.ones((1, 2, nat3), dtype="double") * 0.1

    acc.accumulate(0, result)

    assert acc.get_extra_grid_point_output(0) is None
