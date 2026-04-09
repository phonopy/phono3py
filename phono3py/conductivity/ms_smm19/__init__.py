"""Wigner transport equation plugin for phono3py conductivity.

Importing this package registers the ``"MS-SMM19-rta"`` and ``"MS-SMM19-lbte"``
methods with the conductivity factory so that
``conductivity_calculator("MS-SMM19-rta", ...)`` works out of the box.

This package can also be installed as a standalone ``phono3py-wigner`` package
via namespace packages, in which case the factory auto-discovery still works
because ``factory.py`` does ``try: import phono3py.conductivity.ms_smm19``.

"""

from phono3py.conductivity.build_components import VariantContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.ms_smm19.kappa_solvers import (
    WignerLBTEKappaSolver,
    WignerRTAKappaSolver,
)
from phono3py.conductivity.ms_smm19.velocity_solvers import VelocityOperatorSolver


def _make_velocity_solver(ctx: VariantContext) -> VelocityOperatorSolver:
    return VelocityOperatorSolver(
        ctx.interaction,
        is_kappa_star=ctx.kappa_settings.is_kappa_star,
        gv_delta_q=ctx.kappa_settings.gv_delta_q,
        log_level=ctx.log_level,
    )


def _make_rta_kappa_solver(ctx: VariantContext) -> WignerRTAKappaSolver:
    frequencies, _, _ = ctx.interaction.get_phonons()
    return WignerRTAKappaSolver(
        kappa_settings=ctx.kappa_settings,
        frequencies=frequencies,
        volume=ctx.interaction.primitive.volume,
        log_level=ctx.log_level,
    )


def _make_lbte_kappa_solver(ctx: VariantContext) -> WignerLBTEKappaSolver:
    frequencies, _, _ = ctx.interaction.get_phonons()
    return WignerLBTEKappaSolver(
        solver=ctx.collision_matrix_kernel,
        kappa_settings=ctx.kappa_settings,
        frequencies=frequencies,
        volume=ctx.interaction.primitive.volume,
        is_reducible_collision_matrix=(
            ctx.kappa_settings.is_reducible_collision_matrix
        ),
        log_level=ctx.log_level,
    )


register_variant(
    "MS-SMM19",
    make_velocity_solver=_make_velocity_solver,
    make_rta_kappa_solver=_make_rta_kappa_solver,
    make_lbte_kappa_solver=_make_lbte_kappa_solver,
)
