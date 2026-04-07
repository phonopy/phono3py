"""Green-Kubo plugin for phono3py conductivity.

Importing this package registers the ``"NJC23-rta"`` and ``"NJC23-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("NJC23-rta", ...)`` and
``make_conductivity_calculator("NJC23-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.build_components import VariantBuildContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.heat_capacity_providers import (
    HeatCapacityMatrixProvider,
)
from phono3py.conductivity.njc23.kappa_accumulators import (
    KuboLBTEKappaAccumulator,
    KuboRTAKappaAccumulator,
)
from phono3py.conductivity.velocity_providers import VelocityMatrixProvider


def _make_velocity_provider(ctx: VariantBuildContext) -> VelocityMatrixProvider:
    return VelocityMatrixProvider(
        ctx.interaction,
        is_kappa_star=ctx.is_kappa_star,
        gv_delta_q=ctx.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_cv_provider(ctx: VariantBuildContext) -> HeatCapacityMatrixProvider:
    return HeatCapacityMatrixProvider(ctx.interaction, ctx.context.temperatures)


def _make_rta_accumulator(ctx: VariantBuildContext) -> KuboRTAKappaAccumulator:
    return KuboRTAKappaAccumulator(
        context=ctx.context,
        conversion_factor=ctx.conversion_factor,
        log_level=ctx.log_level,
    )


def _make_lbte_accumulator(ctx: VariantBuildContext) -> KuboLBTEKappaAccumulator:
    return KuboLBTEKappaAccumulator(
        solver=ctx.solver,
        context=ctx.context,
        conversion_factor=ctx.conversion_factor,
        log_level=ctx.log_level,
    )


register_variant(
    "NJC23",
    make_velocity_provider=_make_velocity_provider,
    make_cv_provider=_make_cv_provider,
    make_rta_accumulator=_make_rta_accumulator,
    make_lbte_accumulator=_make_lbte_accumulator,
)
