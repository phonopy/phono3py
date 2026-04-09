"""NJC23 (Green-Kubo) plugin for phono3py conductivity.

Importing this package registers the ``"NJC23-rta"`` and ``"NJC23-lbte"``
methods with the conductivity factory so that
``conductivity_calculator("NJC23-rta", ...)`` and
``conductivity_calculator("NJC23-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.build_components import VariantContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.heat_capacity_solvers import (
    HeatCapacityMatrixSolver,
)
from phono3py.conductivity.interband_kappa_solvers import (
    InterBandLBTEKappaSolver,
    InterBandRTAKappaSolver,
)
from phono3py.conductivity.njc23.kappa_solvers import compute_njc23_mode_kappa
from phono3py.conductivity.velocity_solvers import VelocityMatrixSolver


def _make_velocity_solver(ctx: VariantContext) -> VelocityMatrixSolver:
    return VelocityMatrixSolver(
        ctx.interaction,
        is_kappa_star=ctx.kappa_settings.is_kappa_star,
        gv_delta_q=ctx.kappa_settings.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_cv_solver(ctx: VariantContext) -> HeatCapacityMatrixSolver:
    return HeatCapacityMatrixSolver(ctx.interaction, ctx.kappa_settings.temperatures)


def _make_rta_kappa_solver(ctx: VariantContext) -> InterBandRTAKappaSolver:
    frequencies, _, _ = ctx.interaction.get_phonons()
    return InterBandRTAKappaSolver(
        kappa_settings=ctx.kappa_settings,
        frequencies=frequencies,
        compute_mode_kappa=compute_njc23_mode_kappa,
        log_level=ctx.log_level,
    )


def _make_lbte_kappa_solver(ctx: VariantContext) -> InterBandLBTEKappaSolver:
    frequencies, _, _ = ctx.interaction.get_phonons()
    return InterBandLBTEKappaSolver(
        solver=ctx.collision_matrix_kernel,
        kappa_settings=ctx.kappa_settings,
        frequencies=frequencies,
        compute_mode_kappa=compute_njc23_mode_kappa,
        log_level=ctx.log_level,
    )


register_variant(
    "NJC23",
    make_velocity_solver=_make_velocity_solver,
    make_cv_solver=_make_cv_solver,
    make_rta_kappa_solver=_make_rta_kappa_solver,
    make_lbte_kappa_solver=_make_lbte_kappa_solver,
)
