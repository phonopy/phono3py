"""Factory for creating conductivity calculator instances.

Built-in methods
----------------
"std-rta"
    Standard BTE in the relaxation time approximation.
"std-lbte"
    Standard BTE via direct LBTE solution.
"MS-SMM19-rta"
    MS-SMM19 transport equation in RTA.
"MS-SMM19-lbte"
    MS-SMM19 transport equation via direct solution.
"NJC23-rta"
    Green-Kubo formula in RTA.
"NJC23-lbte"
    Green-Kubo formula via direct solution.

External plugins
----------------
Register a variant via register_variant():

    from phono3py.conductivity import register_variant

    register_variant(
        "my-variant", make_velocity_solver=..., make_rta_kappa_solver=...,
    )

The variant provides component factories that receive a VariantContext. The
framework handles base-component construction and Calculator assembly.

"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.build_components import (
    CalculatorConfig,
    VariantContext,
    build_lbte_kappa_settings,
    build_rot_grid_points,
    build_rta_kappa_settings,
)
from phono3py.conductivity.collision_matrix_kernel import CollisionMatrixKernel
from phono3py.conductivity.heat_capacity_solvers import ModeHeatCapacitySolver
from phono3py.conductivity.kappa_solvers import LBTEKappaSolver, RTAKappaSolver
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.lbte_collision_solver import LBTECollisionSolver
from phono3py.conductivity.rta_calculator import RTACalculator
from phono3py.conductivity.scattering_solvers import RTAScatteringSolver
from phono3py.conductivity.velocity_solvers import GroupVelocitySolver
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction

# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

# Internal registry: all entries are (interaction, config) -> Calculator.
_InternalFactory: TypeAlias = Callable[["Interaction", "CalculatorConfig"], Any]

# Populated at module load with all built-in factories.
# External plugins are added via register_variant().
_REGISTRY: dict[str, _InternalFactory] = {}


def _run_phonon_solver(interaction: Interaction) -> None:
    """Ensure phonons are solved (idempotent)."""
    interaction.nac_q_direction = None
    interaction.run_phonon_solver_at_gamma()
    if not interaction.phonon_all_done:
        interaction.run_phonon_solver()


def _build_rta_calculator(
    interaction: Interaction,
    config: CalculatorConfig,
    make_velocity_solver: Callable[[VariantContext], Any],
    make_cv_solver: Callable[[VariantContext], Any] | None,
    make_rta_kappa_solver: Callable[[VariantContext], Any],
) -> RTACalculator:
    """Build an RTACalculator from variant component factories."""
    _run_phonon_solver(interaction)
    kappa_settings = build_rta_kappa_settings(interaction, config)
    frequencies, _, _ = interaction.get_phonons()
    scattering_solver = RTAScatteringSolver(
        interaction,
        sigmas=config.sigmas,
        temperatures=config.temperatures,
        sigma_cutoff=config.sigma_cutoff,
        is_full_pp=config.is_full_pp,
        use_ave_pp=config.use_ave_pp,
        read_pp=config.read_pp,
        store_pp=config.store_pp,
        pp_filename=config.pp_filename,
        is_N_U=config.is_N_U,
        is_gamma_detail=config.is_gamma_detail,
        log_level=config.log_level,
    )
    ctx = VariantContext(
        interaction=interaction,
        kappa_settings=kappa_settings,
        log_level=config.log_level,
    )
    velocity_solver = make_velocity_solver(ctx)
    cv_solver = (
        make_cv_solver(ctx)
        if make_cv_solver is not None
        else ModeHeatCapacitySolver(interaction, kappa_settings.temperatures)
    )
    kappa_solver = make_rta_kappa_solver(ctx)

    return RTACalculator(
        interaction,
        velocity_solver=velocity_solver,
        cv_solver=cv_solver,
        scattering_solver=scattering_solver,
        kappa_solver=kappa_solver,
        kappa_settings=kappa_settings,
        frequencies=frequencies,
        is_isotope=config.is_isotope,
        mass_variances=config.mass_variances,
        is_N_U=config.is_N_U,
        is_gamma_detail=config.is_gamma_detail,
        sigma_cutoff_width=config.sigma_cutoff,
        log_level=config.log_level,
    )


def _build_lbte_calculator(
    interaction: Interaction,
    config: CalculatorConfig,
    make_velocity_solver: Callable[[VariantContext], Any],
    make_cv_solver: Callable[[VariantContext], Any] | None,
    make_lbte_kappa_solver: Callable[[VariantContext], Any],
) -> LBTECalculator:
    """Build an LBTECalculator from variant component factories."""
    _run_phonon_solver(interaction)
    kappa_settings = build_lbte_kappa_settings(interaction, config)
    frequencies, _, _ = interaction.get_phonons()
    rot_grid_points = build_rot_grid_points(kappa_settings)
    collision = CollisionMatrix(
        interaction,
        rot_grid_points=rot_grid_points,
        is_kappa_star=config.is_kappa_star,
        log_level=config.log_level,
        lang=config.lang,
    )
    collision_solver = LBTECollisionSolver(
        interaction,
        collision,
        sigmas=config.sigmas,
        sigma_cutoff=config.sigma_cutoff,
        temperatures=config.temperatures,
        is_full_pp=config.is_full_pp,
        read_pp=config.read_pp,
        pp_filename=config.pp_filename,
        log_level=config.log_level,
    )
    colmat_kernel = CollisionMatrixKernel(
        kappa_settings=kappa_settings,
        frequencies=frequencies,
        rot_grid_points=rot_grid_points,
        solve_collective_phonon=config.solve_collective_phonon,
        pinv_cutoff=config.pinv_cutoff,
        pinv_solver=config.pinv_solver,
        pinv_method=config.pinv_method,
        lang=config.lang,
        log_level=config.log_level,
    )
    ctx = VariantContext(
        interaction=interaction,
        kappa_settings=kappa_settings,
        log_level=config.log_level,
        collision_matrix_kernel=colmat_kernel,
    )
    velocity_solver = make_velocity_solver(ctx)
    cv_solver = (
        make_cv_solver(ctx)
        if make_cv_solver is not None
        else ModeHeatCapacitySolver(interaction, kappa_settings.temperatures)
    )
    kappa_solver = make_lbte_kappa_solver(ctx)

    return LBTECalculator(
        interaction,
        velocity_solver=velocity_solver,
        cv_solver=cv_solver,
        collision_solver=collision_solver,
        kappa_solver=kappa_solver,
        kappa_settings=kappa_settings,
        frequencies=frequencies,
        is_isotope=config.is_isotope,
        mass_variances=config.mass_variances,
        is_full_pp=config.is_full_pp,
        sigma_cutoff_width=config.sigma_cutoff,
        log_level=config.log_level,
    )


def register_variant(
    name: str,
    *,
    make_velocity_solver: Callable[[VariantContext], Any],
    make_rta_kappa_solver: Callable[[VariantContext], Any],
    make_cv_solver: Callable[[VariantContext], Any] | None = None,
    make_lbte_kappa_solver: Callable[[VariantContext], Any] | None = None,
) -> None:
    """Register a conductivity variant with RTA and optionally LBTE methods.

    Plugin authors provide small
    component factories that receive a ``VariantContext``.  The
    framework handles base-component construction, Calculator assembly, and
    all keyword argument routing.

    Two method names are registered automatically: ``"{name}-rta"`` and
    (if ``make_lbte_kappa_solver`` is provided) ``"{name}-lbte"``.

    Parameters
    ----------
    name : str
        Variant name (e.g. ``"std"``, ``"MS-SMM19"``, ``"NJC23"``).  Becomes
        the prefix of the registered method names.
    make_velocity_solver : callable
        ``(ctx: VariantContext) -> VelocitySolver``.
    make_rta_kappa_solver : callable
        ``(ctx: VariantContext) -> kappa_solver`` for RTA.
    make_cv_solver : callable or None, optional
        ``(ctx: VariantContext) -> HeatCapacitySolver``.
        Defaults to ``ModeHeatCapacitySolver(ctx.interaction)``.
    make_lbte_kappa_solver : callable or None, optional
        ``(ctx: VariantContext) -> kappa_solver`` for LBTE.
        When None, only the RTA method is registered.

    Examples
    --------
    ::

        from phono3py.conductivity import register_variant

        register_variant(
            "my-variant",
            make_velocity_solver=lambda ctx: MyVelocitySolver(
                ctx.interaction,
                is_kappa_star=ctx.kappa_settings.is_kappa_star,
                log_level=ctx.log_level,
            ),
            make_rta_kappa_solver=lambda ctx: MyRTAKappaSolver(
                kappa_settings=ctx.kappa_settings,
                log_level=ctx.log_level,
            ),
        )

    """
    _REGISTRY[f"{name}-rta".lower()] = lambda interaction, config: (
        _build_rta_calculator(
            interaction,
            config,
            make_velocity_solver,
            make_cv_solver,
            make_rta_kappa_solver,
        )
    )

    if make_lbte_kappa_solver is not None:
        _REGISTRY[f"{name}-lbte".lower()] = lambda interaction, config: (
            _build_lbte_calculator(
                interaction,
                config,
                make_velocity_solver,
                make_cv_solver,
                make_lbte_kappa_solver,
            )
        )


def conductivity_calculator(
    interaction: Interaction,
    temperatures: NDArray[np.double],
    sigmas: Sequence[float | None],
    method: str = "std-rta",
    grid_points: NDArray[np.int64] | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: NDArray[np.double] | None = None,
    boundary_mfp: float | None = None,
    use_ave_pp: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    read_pp: bool = False,
    store_pp: bool = False,
    pp_filename: str | os.PathLike | None = None,
    is_N_U: bool = False,
    is_gamma_detail: bool = False,
    is_reducible_collision_matrix: bool = False,
    solve_collective_phonon: bool = False,
    pinv_cutoff: float = 1.0e-8,
    pinv_solver: int = 0,
    pinv_method: int = 0,
    lang: Literal["C", "Python"] = "C",
    log_level: int = 0,
) -> RTACalculator | LBTECalculator:
    """Create a conductivity calculator with the appropriate building blocks.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance.  init_dynamical_matrix must have been called.
    temperatures : ndarray of double
        Temperatures in Kelvin, shape (num_temp,).
    sigmas : sequence of (float or None)
        Smearing widths.  A None entry selects the tetrahedron method.
    method : str, optional
        Calculation method.  Built-in: "std-rta", "std-lbte", "MS-SMM19-rta",
        "MS-SMM19-lbte", "NJC23-rta", "NJC23-lbte".  Default "std-rta".
    grid_points : array-like or None, optional
        BZ grid point indices.  None uses irreducible grid points.  Default None.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.  Default None.
    is_isotope : bool, optional
        Include isotope scattering.  Default False.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.  Default None.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.  Default None.
    use_ave_pp : bool, optional
        Use pre-averaged ph-ph interaction (RTA only).  Default False.
    is_kappa_star : bool, optional
        Use k-star symmetry.  Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for group velocity (NAC).  Default None.
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.  Default False.
    read_pp : bool, optional
        Read ph-ph interaction from file.  Default False.
    store_pp : bool, optional
        Store ph-ph interaction to file (RTA only).  Default False.
    pp_filename : str or path or None, optional
        Filename for ph-ph interaction I/O.  Default None.
    is_N_U : bool, optional
        Decompose gamma into Normal and Umklapp (RTA only).  Default False.
    is_gamma_detail : bool, optional
        Store per-triplet gamma (RTA only).  Default False.
    is_reducible_collision_matrix : bool, optional
        Use full reducible collision matrix (LBTE only).  Default False.
    solve_collective_phonon : bool, optional
        Use Chaput collective-phonon method (LBTE only).  Default False.
    pinv_cutoff : float, optional
        Eigenvalue cutoff for pseudo-inversion (LBTE only).  Default 1e-8.
    pinv_solver : int, optional
        Solver selection index (LBTE only).  Default 0.
    pinv_method : int, optional
        Pseudo-inverse criterion (LBTE only).  Default 0.
    lang : {"C", "Python"}, optional
        Backend for C-extension operations (LBTE only).  Default "C".
    log_level : int, optional
        Verbosity.  Default 0.

    Returns
    -------
    RTACalculator or LBTECalculator
        For plugin-registered methods the return type is determined by the
        registered factory.

    """
    if method.lower() not in _REGISTRY:
        raise NotImplementedError(
            f"method='{method}' is not implemented. "
            f"Registered methods: {sorted(_REGISTRY)}. "
            "Use register_variant() to add custom methods."
        )

    config = CalculatorConfig(
        temperatures=temperatures,
        sigmas=sigmas,
        grid_points=grid_points,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        store_pp=store_pp,
        pp_filename=pp_filename,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        solve_collective_phonon=solve_collective_phonon,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        lang=lang,
        log_level=log_level,
    )

    return _REGISTRY[method.lower()](interaction, config)


# ---------------------------------------------------------------------------
# Built-in variant: standard BTE ("std-rta" / "std-lbte")
# ---------------------------------------------------------------------------


def _std_make_velocity_solver(ctx: VariantContext) -> GroupVelocitySolver:
    return GroupVelocitySolver(
        ctx.interaction,
        is_kappa_star=ctx.kappa_settings.is_kappa_star,
        gv_delta_q=ctx.kappa_settings.gv_delta_q,
        log_level=ctx.log_level,
    )


def _std_make_rta_kappa_solver(ctx: VariantContext) -> RTAKappaSolver:
    frequencies, _, _ = ctx.interaction.get_phonons()
    return RTAKappaSolver(
        kappa_settings=ctx.kappa_settings,
        frequencies=frequencies,
        log_level=ctx.log_level,
    )


def _std_make_lbte_kappa_solver(ctx: VariantContext) -> LBTEKappaSolver:
    return LBTEKappaSolver(
        ctx.collision_matrix_kernel,
        kappa_settings=ctx.kappa_settings,
        log_level=ctx.log_level,
    )


register_variant(
    "std",
    make_velocity_solver=_std_make_velocity_solver,
    make_rta_kappa_solver=_std_make_rta_kappa_solver,
    make_lbte_kappa_solver=_std_make_lbte_kappa_solver,
)

try:
    import phono3py.conductivity.njc23  # noqa: F401, E402
except ImportError:
    pass

try:
    import phono3py.conductivity.smm19  # noqa: F401, E402
except ImportError:
    pass

try:
    import phono3py.conductivity.ms_smm19  # noqa: F401, E402
except ImportError:
    pass
