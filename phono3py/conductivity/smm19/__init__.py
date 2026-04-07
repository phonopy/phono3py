"""SMM19 plugin for phono3py conductivity.

Importing this package registers the ``"SMM19-rta"`` and ``"SMM19-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("SMM19-rta", ...)`` and
``make_conductivity_calculator("SMM19-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.build_components import VariantBuildContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.smm19.kappa_accumulators import (
    SMM19RTAKappaAccumulator,
)
from phono3py.conductivity.velocity_providers import VelocityMatrixProvider


def _make_velocity_provider(ctx: VariantBuildContext) -> VelocityMatrixProvider:
    return VelocityMatrixProvider(
        ctx.interaction,
        reciprocal_operations=ctx.point_operations,
        rotations_cartesian=ctx.rotations_cartesian,
        grid_points=ctx.context.grid_points,
        grid_weights=ctx.context.grid_weights,
        is_kappa_star=ctx.is_kappa_star,
        gv_delta_q=ctx.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_cv_provider(ctx: VariantBuildContext) -> ModeHeatCapacityProvider:
    return ModeHeatCapacityProvider(ctx.interaction, ctx.context.temperatures)


def _make_rta_accumulator(ctx: VariantBuildContext) -> SMM19RTAKappaAccumulator:
    return SMM19RTAKappaAccumulator(
        context=ctx.context,
        conversion_factor=ctx.conversion_factor,
        log_level=ctx.log_level,
    )


register_variant(
    "SMM19",
    make_velocity_provider=_make_velocity_provider,
    make_cv_provider=_make_cv_provider,
    make_rta_accumulator=_make_rta_accumulator,
)
