"""Registration helper for inter-band (coherence) transport variants.

The inter-band variants NJC23, IBDB19, and SMM19 share the same component
construction: the velocity matrix, the heat capacity matrix solver, and the
generic inter-band mode-kappa kernel. They differ only in the mode heat
capacity matrix function (e.g. ``mode_cv_matrix_njc23``,
``mode_cv_matrix_ibdb19``, ``mode_cv_matrix_smm19``).

``register_interband_variant`` captures this shared boilerplate so that each
variant module only needs to supply its name and heat capacity matrix
function.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.build_components import VariantContext
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.heat_capacity_solvers import (
    CvMatrixFunc,
    HeatCapacityMatrixSolver,
)
from phono3py.conductivity.interband_kappa_formula import (
    compute_interband_mode_kappa,
)
from phono3py.conductivity.interband_kappa_solvers import (
    InterBandLBTEKappaSolver,
    InterBandRTAKappaSolver,
)
from phono3py.conductivity.velocity_solvers import VelocityMatrixSolver


def register_interband_variant(name: str, cv_matrix_func: CvMatrixFunc) -> None:
    """Register an inter-band variant with RTA and LBTE methods.

    Parameters
    ----------
    name : str
        Variant name (e.g. ``"NJC23"``). Becomes the prefix of the
        registered method names ``"{name}-rta"`` and ``"{name}-lbte"``.
    cv_matrix_func : callable
        Mode heat capacity matrix function selecting the variant,
        ``(temps, freqs_ev) -> (num_temp, num_band, num_band)``.

    """

    def _make_velocity_solver(ctx: VariantContext) -> VelocityMatrixSolver:
        return VelocityMatrixSolver(
            ctx.interaction,
            is_kappa_star=ctx.kappa_settings.is_kappa_star,
            gv_delta_q=ctx.kappa_settings.gv_delta_q,
            log_level=ctx.log_level,
        )

    def _make_cv_solver(ctx: VariantContext) -> HeatCapacityMatrixSolver:
        return HeatCapacityMatrixSolver(
            ctx.interaction,
            ctx.kappa_settings.temperatures,
            cv_matrix_func=cv_matrix_func,
        )

    def _make_rta_kappa_solver(ctx: VariantContext) -> InterBandRTAKappaSolver:
        frequencies: NDArray[np.double] = ctx.interaction.get_phonons()[0]
        return InterBandRTAKappaSolver(
            kappa_settings=ctx.kappa_settings,
            frequencies=frequencies,
            compute_mode_kappa=compute_interband_mode_kappa,
            log_level=ctx.log_level,
        )

    def _make_lbte_kappa_solver(ctx: VariantContext) -> InterBandLBTEKappaSolver:
        frequencies: NDArray[np.double] = ctx.interaction.get_phonons()[0]
        return InterBandLBTEKappaSolver(
            solver=ctx.collision_matrix_kernel,
            kappa_settings=ctx.kappa_settings,
            frequencies=frequencies,
            compute_mode_kappa=compute_interband_mode_kappa,
            log_level=ctx.log_level,
        )

    register_variant(
        name,
        make_velocity_solver=_make_velocity_solver,
        make_cv_solver=_make_cv_solver,
        make_rta_kappa_solver=_make_rta_kappa_solver,
        make_lbte_kappa_solver=_make_lbte_kappa_solver,
    )
