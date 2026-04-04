"""Tests for the register_calculator plugin API."""

from __future__ import annotations

import pytest

from phono3py.conductivity import (
    GridPointInput,
    HeatCapacityProvider,
    HeatCapacityResult,
    ScatteringProvider,
    ScatteringResult,
    VelocityProvider,
    VelocityResult,
    register_calculator,
)
from phono3py.conductivity.factory import _BUILTIN_METHODS, _REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyCalculator:
    """Minimal stand-in returned by a custom factory."""

    def __init__(self, interaction, **kwargs):
        self.interaction = interaction
        self.kwargs = kwargs


def _dummy_factory(interaction, **kwargs):
    return _DummyCalculator(interaction, **kwargs)


# ---------------------------------------------------------------------------
# registry tests
# ---------------------------------------------------------------------------


def test_register_calculator_stores_factory():
    """Registered factory is stored under the given method name."""
    try:
        register_calculator("test-method-store", _dummy_factory)
        assert "test-method-store" in _REGISTRY
        assert callable(_REGISTRY["test-method-store"])
    finally:
        _REGISTRY.pop("test-method-store", None)


def test_register_calculator_raises_on_builtin_override():
    """Registering a built-in method name raises ValueError."""
    for builtin in _BUILTIN_METHODS:
        with pytest.raises(ValueError, match="built-in"):
            register_calculator(builtin, _dummy_factory)


def test_make_conductivity_calculator_calls_registered_factory():
    """make_conductivity_calculator dispatches to a registered factory."""
    from types import SimpleNamespace

    from phono3py.conductivity.factory import make_conductivity_calculator

    fake_interaction = SimpleNamespace()
    received: list[tuple] = []

    def capturing_factory(interaction, **kwargs):
        received.append((interaction, kwargs))
        return _DummyCalculator(interaction, **kwargs)

    try:
        register_calculator("test-plugin", capturing_factory)
        result = make_conductivity_calculator(
            fake_interaction, method="test-plugin", log_level=0
        )
        assert isinstance(result, _DummyCalculator)
        assert len(received) == 1
        assert received[0][0] is fake_interaction
        assert received[0][1]["log_level"] == 0
    finally:
        _REGISTRY.pop("test-plugin", None)


def test_make_conductivity_calculator_unknown_method_raises():
    """Unregistered unknown method raises NotImplementedError."""
    from types import SimpleNamespace

    from phono3py.conductivity.factory import make_conductivity_calculator

    with pytest.raises(NotImplementedError, match="not implemented"):
        make_conductivity_calculator(SimpleNamespace(), method="no-such-method")


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------


def test_public_api_exports_register_calculator():
    """register_calculator is importable from phono3py.conductivity."""
    from phono3py.conductivity import register_calculator as rc

    assert callable(rc)


def test_public_api_exports_protocols():
    """Protocol classes are importable from phono3py.conductivity."""
    for cls in (
        VelocityProvider,
        HeatCapacityProvider,
        ScatteringProvider,
    ):
        assert hasattr(cls, "__protocol_attrs__") or hasattr(cls, "_is_protocol")


def test_public_api_exports_data_containers():
    """Data containers are importable from phono3py.conductivity."""
    assert GridPointInput is not None
    assert VelocityResult is not None
    assert HeatCapacityResult is not None
    assert ScatteringResult is not None
