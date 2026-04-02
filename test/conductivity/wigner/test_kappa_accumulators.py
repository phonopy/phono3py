"""Unit tests for WignerRTAKappaAccumulator."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.conductivity.wigner.kappa_accumulators import WignerRTAKappaAccumulator


def _make_result_with_velocity_operator(
    num_band0: int = 3,
    nat3: int = 6,
    num_sigma: int = 1,
    num_temp: int = 2,
) -> GridPointResult:
    """Create a GridPointResult with velocity_operator in extra."""
    gp_input = GridPointInput(
        grid_point=0,
        q_point=np.zeros(3, dtype="double"),
        frequencies=np.ones(nat3, dtype="double"),
        eigenvectors=np.eye(nat3, dtype="complex128"),
        grid_weight=1,
        band_indices=np.arange(num_band0, dtype="int64"),
    )
    result = GridPointResult(input=gp_input)
    result.group_velocities = np.zeros((num_band0, 3), dtype="double")
    result.velocity_product = np.zeros((num_band0, nat3, 6), dtype="complex128")
    result.heat_capacities = np.ones((num_temp, num_band0), dtype="double")
    result.gamma = np.ones((num_sigma, num_temp, num_band0), dtype="double") * 0.1
    result.extra["velocity_operator"] = (
        np.random.default_rng(42).random((num_band0, nat3, 3)).astype("complex128")
    )
    return result


def test_get_extra_grid_point_output_stores_velocity_operator():
    """Velocity operator is stored per grid point and returned."""
    formula = MagicMock()
    # formula.compute returns mode_kappa_P shaped (num_sigma, num_temp, num_band0, 6)
    formula.compute.return_value = np.zeros((1, 2, 3, 6), dtype="double")

    acc = WignerRTAKappaAccumulator(formula)
    acc.prepare(num_sigma=1, num_temp=2, num_gp=2, num_band0=3)

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
    formula = MagicMock()
    formula.compute.return_value = np.zeros((1, 2, 3, 6), dtype="double")

    acc = WignerRTAKappaAccumulator(formula)
    acc.prepare(num_sigma=1, num_temp=2, num_gp=1, num_band0=3)

    # Result without velocity_operator in extra
    gp_input = GridPointInput(
        grid_point=0,
        q_point=np.zeros(3, dtype="double"),
        frequencies=np.ones(6, dtype="double"),
        eigenvectors=np.eye(6, dtype="complex128"),
        grid_weight=1,
        band_indices=np.arange(3, dtype="int64"),
    )
    result = GridPointResult(input=gp_input)
    result.group_velocities = np.zeros((3, 3), dtype="double")
    result.velocity_product = np.zeros((3, 6, 6), dtype="complex128")
    result.heat_capacities = np.ones((2, 3), dtype="double")
    result.gamma = np.ones((1, 2, 3), dtype="double") * 0.1

    acc.accumulate(0, result)

    assert acc.get_extra_grid_point_output(0) is None
