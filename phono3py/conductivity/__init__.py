"""Conductivity module public plugin API.

Plugin authors import from this package:

    from phono3py.conductivity import (
        register_variant,
        VariantContext,
        GridPointAggregates,
        VelocityResult,
        HeatCapacityResult,
        ScatteringResult,
        VelocitySolver,
        HeatCapacitySolver,
        ScatteringSolver,
    )
"""

from phono3py.conductivity.build_components import VariantContext
from phono3py.conductivity.factory import register_variant
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

__all__ = [
    "register_variant",
    "VariantContext",
    "GridPointAggregates",
    "VelocityResult",
    "HeatCapacityResult",
    "ScatteringResult",
    "VelocitySolver",
    "HeatCapacitySolver",
    "ScatteringSolver",
]
