"""Conductivity module public plugin API.

Plugin authors import from this package:

    from phono3py.conductivity import (
        register_variant,
        VariantBuildContext,
        GridPointAggregates,
        GridPointInput,
        VelocityResult,
        HeatCapacityResult,
        ScatteringResult,
        VelocityProvider,
        HeatCapacityProvider,
        ScatteringProvider,
    )
"""

from phono3py.conductivity.build_components import VariantBuildContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
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

__all__ = [
    "register_variant",
    "VariantBuildContext",
    "GridPointAggregates",
    "GridPointInput",
    "VelocityResult",
    "HeatCapacityResult",
    "ScatteringResult",
    "VelocityProvider",
    "HeatCapacityProvider",
    "ScatteringProvider",
]
