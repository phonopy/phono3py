"""Wigner transport equation plugin for phono3py conductivity.

Importing this package registers the ``"MS-SMM19-rta"`` and ``"MS-SMM19-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("MS-SMM19-rta", ...)`` works out of the box.

This package can also be installed as a standalone ``phono3py-wigner`` package
via namespace packages, in which case the factory auto-discovery still works
because ``factory.py`` does ``try: import phono3py.conductivity.ms_smm19``.

"""

from phono3py.conductivity.build_components import VariantBuildContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.ms_smm19.kappa_accumulators import (
    WignerLBTEKappaAccumulator,
    WignerRTAKappaAccumulator,
)
from phono3py.conductivity.ms_smm19.velocity_providers import VelocityOperatorProvider


def _make_velocity_provider(ctx: VariantBuildContext) -> VelocityOperatorProvider:
    return VelocityOperatorProvider(
        ctx.interaction,
        is_kappa_star=ctx.is_kappa_star,
        gv_delta_q=ctx.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_rta_accumulator(ctx: VariantBuildContext) -> WignerRTAKappaAccumulator:
    return WignerRTAKappaAccumulator(
        context=ctx.context,
        volume=ctx.interaction.primitive.volume,
        log_level=ctx.log_level,
    )


def _make_lbte_accumulator(ctx: VariantBuildContext) -> WignerLBTEKappaAccumulator:
    return WignerLBTEKappaAccumulator(
        solver=ctx.solver,
        context=ctx.context,
        volume=ctx.interaction.primitive.volume,
        is_reducible_collision_matrix=ctx.is_reducible_collision_matrix,
        log_level=ctx.log_level,
    )


register_variant(
    "MS-SMM19",
    make_velocity_provider=_make_velocity_provider,
    make_rta_accumulator=_make_rta_accumulator,
    make_lbte_accumulator=_make_lbte_accumulator,
)
