"""SMM19 plugin for phono3py conductivity.

Importing this package registers the ``"SMM19-rta"`` and ``"SMM19-lbte"``
methods with the conductivity factory so that
``conductivity_calculator("SMM19-rta", ...)`` and
``conductivity_calculator("SMM19-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.build_components import VariantContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.heat_capacity_solvers import ModeHeatCapacitySolver
from phono3py.conductivity.smm19.kappa_solvers import (
    SMM19RTAKappaSolver,
)
from phono3py.conductivity.velocity_solvers import VelocityMatrixSolver


def _make_velocity_solver(ctx: VariantContext) -> VelocityMatrixSolver:
    return VelocityMatrixSolver(
        ctx.interaction,
        is_kappa_star=ctx.kappa_settings.is_kappa_star,
        gv_delta_q=ctx.kappa_settings.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_cv_solver(ctx: VariantContext) -> ModeHeatCapacitySolver:
    return ModeHeatCapacitySolver(ctx.interaction, ctx.kappa_settings.temperatures)


def _make_rta_kappa_solver(ctx: VariantContext) -> SMM19RTAKappaSolver:
    frequencies, _, _ = ctx.interaction.get_phonons()
    return SMM19RTAKappaSolver(
        kappa_settings=ctx.kappa_settings,
        frequencies=frequencies,
        log_level=ctx.log_level,
    )


register_variant(
    "SMM19",
    make_velocity_solver=_make_velocity_solver,
    make_cv_solver=_make_cv_solver,
    make_rta_kappa_solver=_make_rta_kappa_solver,
)
