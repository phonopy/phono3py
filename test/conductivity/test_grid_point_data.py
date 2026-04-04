"""Tests for grid_point_data module (Phase 1 scaffolding)."""

from __future__ import annotations

import numpy as np

from phono3py.conductivity.grid_point_data import (
    GridPointInput,
    HeatCapacityResult,
    ScatteringResult,
    VelocityResult,
)
from phono3py.conductivity.protocols import (
    HeatCapacityProvider,
    ScatteringProvider,
    VelocityProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_BAND = 6
NUM_BAND0 = 6
NUM_TEMP = 3


def make_gp_input() -> GridPointInput:
    """Return a minimal GridPointInput for testing."""
    return GridPointInput(
        grid_point=10,
        q_point=np.array([0.1, 0.2, 0.3]),
        frequencies=np.ones(NUM_BAND, dtype="double"),
        eigenvectors=np.eye(NUM_BAND, dtype="complex128"),
        grid_weight=2,
        band_indices=np.arange(NUM_BAND0, dtype="int64"),
    )


# ---------------------------------------------------------------------------
# GridPointInput
# ---------------------------------------------------------------------------


def test_grid_point_input_fields():
    """Test GridPointInput field shapes and values."""
    gp = make_gp_input()
    assert gp.grid_point == 10
    assert gp.q_point.shape == (3,)
    assert gp.frequencies.shape == (NUM_BAND,)
    assert gp.eigenvectors.shape == (NUM_BAND, NUM_BAND)
    assert gp.grid_weight == 2
    assert gp.band_indices.shape == (NUM_BAND0,)


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


class _DummyVelocityProvider:
    """Minimal implementation satisfying VelocityProvider protocol."""

    def compute(self, gp: GridPointInput) -> VelocityResult:
        return VelocityResult(
            group_velocities=np.zeros((NUM_BAND0, 3), dtype="double"),
            gv_by_gv=np.zeros((NUM_BAND0, 6), dtype="double"),
            num_sampling_grid_points=gp.grid_weight,
        )


class _DummyHeatCapacityProvider:
    def compute(
        self,
        gp: GridPointInput,
        temperatures: np.ndarray,
    ) -> HeatCapacityResult:
        return HeatCapacityResult(
            heat_capacities=np.zeros((len(temperatures), NUM_BAND0), dtype="double"),
        )


class _DummyScatteringProvider:
    def compute(self, gp: GridPointInput) -> ScatteringResult:
        return ScatteringResult(
            gamma=np.zeros((1, NUM_TEMP, NUM_BAND0), dtype="double"),
        )


def test_velocity_provider_protocol_satisfied():
    """A class with the right signature is accepted as VelocityProvider."""
    provider: VelocityProvider = _DummyVelocityProvider()
    gp = make_gp_input()
    result = provider.compute(gp)
    assert result.group_velocities is not None
    assert result.num_sampling_grid_points == gp.grid_weight


def test_heat_capacity_provider_protocol_satisfied():
    """Test HeatCapacityProvider protocol with a dummy implementation."""
    provider: HeatCapacityProvider = _DummyHeatCapacityProvider()
    gp = make_gp_input()
    temps = np.array([100.0, 200.0, 300.0])
    result = provider.compute(gp, temps)
    assert result.heat_capacities is not None
    assert result.heat_capacities.shape == (3, NUM_BAND0)


def test_scattering_provider_protocol_satisfied():
    """Test ScatteringProvider protocol with a dummy implementation."""
    provider: ScatteringProvider = _DummyScatteringProvider()
    gp = make_gp_input()
    result = provider.compute(gp)
    assert result.gamma is not None
    assert result.gamma.shape == (1, NUM_TEMP, NUM_BAND0)
