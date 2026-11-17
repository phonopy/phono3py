"""Tests for grid_point_data module (Phase 1 scaffolding)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    HeatCapacityResult,
    ScatteringResult,
    VelocityResult,
)
from phono3py.conductivity.protocols import (
    HeatCapacitySolver,
    ScatteringSolver,
    VelocitySolver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_BAND = 6
NUM_BAND0 = 6
NUM_TEMP = 3


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


class _DummyVelocityProvider:
    """Minimal implementation satisfying VelocitySolver protocol."""

    produces_gv_by_gv: bool = True
    produces_vm_by_vm: bool = False

    def compute(self, grid_points: np.ndarray) -> VelocityResult:
        return VelocityResult(
            group_velocities=np.zeros((len(grid_points), NUM_BAND0, 3), dtype="double"),
            gv_by_gv=np.zeros((len(grid_points), NUM_BAND0, 6), dtype="double"),
        )


class _DummyHeatCapacityProvider:
    produces_heat_capacity_matrix: bool = False

    def compute(
        self,
        frequencies: np.ndarray,
        grid_points: np.ndarray,
        temperatures: np.ndarray,
        band_indices: np.ndarray,
        cutoff_frequency: float,
    ) -> HeatCapacityResult:
        return HeatCapacityResult(
            heat_capacities=np.zeros(
                (len(temperatures), len(grid_points), NUM_BAND0), dtype="double"
            ),
        )


class _DummyScatteringProvider:
    def compute(self, grid_point: int) -> ScatteringResult:
        return ScatteringResult(
            gamma=np.zeros((1, NUM_TEMP, NUM_BAND0), dtype="double"),
        )


def test_velocity_provider_protocol_satisfied():
    """A class with the right signature is accepted as VelocitySolver."""
    provider: VelocitySolver = _DummyVelocityProvider()
    result = provider.compute(np.array([10, 2], dtype="int64"))
    assert result.group_velocities.shape == (2, NUM_BAND0, 3)


def test_heat_capacity_provider_protocol_satisfied():
    """Test HeatCapacitySolver protocol with a dummy implementation."""
    provider: HeatCapacitySolver = _DummyHeatCapacityProvider()
    freqs = np.ones((100, NUM_BAND), dtype="double")
    gps = np.array([0, 1, 2], dtype="int64")
    temps = np.array([100.0, 200.0, 300.0])
    bands = np.arange(NUM_BAND0, dtype="int64")
    result = provider.compute(freqs, gps, temps, bands, 0.0)
    assert result.heat_capacities is not None
    assert result.heat_capacities.shape == (3, 3, NUM_BAND0)


def test_scattering_provider_protocol_satisfied():
    """Test ScatteringSolver protocol with a dummy implementation."""
    provider: ScatteringSolver = _DummyScatteringProvider()
    result = provider.compute(10)
    assert result.gamma is not None
    assert result.gamma.shape == (1, NUM_TEMP, NUM_BAND0)


# ---------------------------------------------------------------------------
# Deprecated extra
# ---------------------------------------------------------------------------


def _make_result(cls: type, **kwargs: Any) -> Any:
    array = np.zeros((1, NUM_BAND0, 3), dtype="double")
    if cls is VelocityResult:
        return cls(group_velocities=array, **kwargs)
    if cls is HeatCapacityResult:
        return cls(heat_capacities=array, **kwargs)
    if cls is ScatteringResult:
        return cls(gamma=array, **kwargs)
    return cls(group_velocities=array, mode_heat_capacities=array, **kwargs)


_RESULT_TYPES = [
    VelocityResult,
    HeatCapacityResult,
    ScatteringResult,
    GridPointAggregates,
]


@pytest.mark.parametrize("cls", _RESULT_TYPES)
def test_no_warning_without_extra(cls: type):
    """Test that no warning is emitted when extra is not used."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make_result(cls)


# REMOVE-AT-V5
@pytest.mark.parametrize("cls", _RESULT_TYPES)
def test_extra_is_deprecated(cls: type):
    """Test that extra warns at construction, read and write, and keeps data."""
    data = {"key": 1}
    with pytest.warns(DeprecationWarning, match="extra is deprecated") as record:
        result = _make_result(cls, extra=data)
    assert record[0].filename == __file__

    with pytest.warns(DeprecationWarning, match="extra is deprecated") as record:
        assert result.extra is data
    assert record[0].filename == __file__

    new_data = {"key": 2}
    with pytest.warns(DeprecationWarning, match="extra is deprecated") as record:
        result.extra = new_data
    assert record[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        assert result.extra is new_data
