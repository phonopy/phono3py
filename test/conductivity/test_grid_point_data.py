"""Tests for grid_point_data module (Phase 1 scaffolding)."""

from __future__ import annotations

import numpy as np

from phono3py.conductivity.grid_point_data import (
    GridPointInput,
    GridPointResult,
    HeatCapacityProvider,
    KappaFormula,
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
# GridPointResult
# ---------------------------------------------------------------------------


def test_grid_point_result_defaults():
    """Test GridPointResult default field values are None / 0."""
    gp = make_gp_input()
    result = GridPointResult(input=gp)
    assert result.group_velocities is None
    assert result.velocity_product is None
    assert result.heat_capacities is None
    assert result.heat_capacity_matrix is None
    assert result.gamma is None
    assert result.num_sampling_grid_points == 0


def test_grid_point_result_can_set_fields():
    """Test that GridPointResult fields accept standard BTE-shaped arrays."""
    gp = make_gp_input()
    result = GridPointResult(input=gp)
    result.group_velocities = np.zeros((NUM_BAND0, 3), dtype="double")
    result.velocity_product = np.zeros((NUM_BAND0, 6), dtype="double")
    result.heat_capacities = np.zeros((NUM_TEMP, NUM_BAND0), dtype="double")
    result.gamma = np.zeros((1, NUM_TEMP, NUM_BAND0), dtype="double")
    result.num_sampling_grid_points = 4
    assert result.group_velocities.shape == (NUM_BAND0, 3)
    assert result.velocity_product.shape == (NUM_BAND0, 6)
    assert result.heat_capacities.shape == (NUM_TEMP, NUM_BAND0)
    assert result.gamma.shape == (1, NUM_TEMP, NUM_BAND0)
    assert result.num_sampling_grid_points == 4


def test_grid_point_result_wigner_velocity_product_shape():
    """Wigner variant: velocity_product is complex (num_band0, num_band, 6)."""
    gp = make_gp_input()
    result = GridPointResult(input=gp)
    result.velocity_product = np.zeros((NUM_BAND0, NUM_BAND, 6), dtype="complex128")
    assert result.velocity_product.shape == (NUM_BAND0, NUM_BAND, 6)
    assert result.velocity_product.dtype == np.dtype("complex128")


def test_grid_point_result_kubo_heat_capacity_matrix_shape():
    """Kubo variant: heat_capacity_matrix is (num_temp, num_band0, num_band)."""
    gp = make_gp_input()
    result = GridPointResult(input=gp)
    result.heat_capacity_matrix = np.zeros(
        (NUM_TEMP, NUM_BAND0, NUM_BAND), dtype="double"
    )
    assert result.heat_capacity_matrix.shape == (NUM_TEMP, NUM_BAND0, NUM_BAND)


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


class _DummyVelocityProvider:
    """Minimal implementation satisfying VelocityProvider protocol."""

    def compute(self, gp: GridPointInput) -> GridPointResult:
        result = GridPointResult(input=gp)
        result.group_velocities = np.zeros((NUM_BAND0, 3), dtype="double")
        result.velocity_product = np.zeros((NUM_BAND0, 6), dtype="double")
        result.num_sampling_grid_points = gp.grid_weight
        return result


class _DummyHeatCapacityProvider:
    def compute(
        self,
        gp: GridPointInput,
        temperatures: np.ndarray,
    ) -> GridPointResult:
        result = GridPointResult(input=gp)
        result.heat_capacities = np.zeros(
            (len(temperatures), NUM_BAND0), dtype="double"
        )
        return result


class _DummyScatteringProvider:
    def compute_gamma(self, gp: GridPointInput) -> GridPointResult:
        result = GridPointResult(input=gp)
        result.gamma = np.zeros((1, NUM_TEMP, NUM_BAND0), dtype="double")
        return result


class _DummyKappaFormula:
    def compute(self, result: GridPointResult) -> np.ndarray:
        return np.zeros((NUM_TEMP, 6), dtype="double")


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
    result = provider.compute_gamma(gp)
    assert result.gamma is not None
    assert result.gamma.shape == (1, NUM_TEMP, NUM_BAND0)


def test_kappa_formula_protocol_satisfied():
    """Test KappaFormula protocol with a dummy implementation."""
    formula: KappaFormula = _DummyKappaFormula()
    gp = make_gp_input()
    result = GridPointResult(input=gp)
    kappa = formula.compute(result)
    assert kappa.shape == (NUM_TEMP, 6)
