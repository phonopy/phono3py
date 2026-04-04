"""Green-Kubo plugin for phono3py conductivity.

Importing this package registers the ``"kubo-rta"`` and ``"kubo-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("kubo-rta", ...)`` and
``make_conductivity_calculator("kubo-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.kubo.heat_capacity_providers import (
    HeatCapacityMatrixProvider,
)
from phono3py.conductivity.kubo.kappa_accumulators import (
    KuboLBTEKappaAccumulator,
    KuboRTAKappaAccumulator,
)
from phono3py.conductivity.kubo.velocity_providers import VelocityMatrixProvider


def _make_velocity_provider(ctx):
    return VelocityMatrixProvider(
        ctx.interaction,
        point_operations=ctx.point_operations,
        is_kappa_star=ctx.is_kappa_star,
        gv_delta_q=ctx.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_cv_provider(ctx):
    return HeatCapacityMatrixProvider(ctx.interaction)


def _make_rta_accumulator(ctx):
    return KuboRTAKappaAccumulator(
        context=ctx.context,
        conversion_factor=ctx.conversion_factor,
        log_level=ctx.log_level,
    )


def _make_lbte_accumulator(ctx):
    return KuboLBTEKappaAccumulator(
        solver=ctx.solver,
        context=ctx.context,
        conversion_factor=ctx.conversion_factor,
        log_level=ctx.log_level,
    )


register_variant(
    "kubo",
    make_velocity_provider=_make_velocity_provider,
    make_cv_provider=_make_cv_provider,
    make_rta_accumulator=_make_rta_accumulator,
    make_lbte_accumulator=_make_lbte_accumulator,
)
