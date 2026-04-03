"""Conductivity module public plugin API.

Plugin authors import from this package:

    from phono3py.conductivity import (
        register_variant,
        register_calculator,
        VariantBuildContext,
        GridPointAggregates,
        GridPointInput,
        GridPointResult,
        VelocityProvider,
        HeatCapacityProvider,
        ScatteringProvider,
    )
"""

from phono3py.conductivity.calculator_factory import VariantBuildContext
from phono3py.conductivity.factory import register_calculator, register_variant
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    GridPointInput,
    GridPointResult,
)
from phono3py.conductivity.protocols import (
    HeatCapacityProvider,
    ScatteringProvider,
    VelocityProvider,
)

__all__ = [
    "register_variant",
    "register_calculator",
    "VariantBuildContext",
    "GridPointAggregates",
    "GridPointInput",
    "GridPointResult",
    "VelocityProvider",
    "HeatCapacityProvider",
    "ScatteringProvider",
]
