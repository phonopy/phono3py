"""Factory for creating conductivity calculator instances.

Built-in methods
----------------
"rta"
    Standard BTE in the relaxation time approximation.
"lbte"
    Standard BTE via direct LBTE solution.
"wigner-rta"
    Wigner transport equation in RTA (registered plugin, can be overridden).
"wigner-lbte"
    Wigner transport equation via LBTE (registered plugin, can be overridden).
"kubo-rta"
    Green-Kubo formula in RTA (registered plugin, can be overridden).

External plugins
----------------
Register a factory callable via register_calculator():

    from phono3py.conductivity import register_calculator

    def my_factory(interaction, **kwargs):
        ...
        return MyCalculator(...)

    register_calculator("my-method", my_factory)

The factory receives the Interaction object as the first positional argument
and all keyword arguments that were passed to make_conductivity_calculator().
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.kappa_accumulators import StandardKappaAccumulator
from phono3py.conductivity.kappa_formulas import BTEKappaFormula
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.lbte_collision_provider import LBTECollisionProvider
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator
from phono3py.conductivity.rta_calculator import ConductivityCalculator
from phono3py.conductivity.scattering_providers import RTAScatteringProvider
from phono3py.conductivity.utils import get_unit_to_WmK
from phono3py.conductivity.velocity_providers import GroupVelocityProvider
from phono3py.phonon.grid import get_grid_points_by_rotations, get_ir_grid_points
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction

# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

CalculatorFactory: TypeAlias = Callable[..., Any]

# "rta" and "lbte" are implemented directly in make_conductivity_calculator
# and cannot be overridden.  All other built-in methods are registered in
# _REGISTRY below and can be overridden by external plugins.
_BUILTIN_METHODS: frozenset[str] = frozenset({"rta", "lbte"})

# Populated at module load with built-in wigner/kubo factories.
# External plugins are added via register_calculator().
_REGISTRY: dict[str, CalculatorFactory] = {}


def register_calculator(method: str, factory: CalculatorFactory) -> None:
    """Register an external calculator factory for a new method name.

    The factory is called as::

        calculator = factory(interaction, **kwargs)

    where ``kwargs`` are all keyword arguments passed to
    ``make_conductivity_calculator()`` except ``method`` itself.

    Parameters
    ----------
    method : str
        Method name used as the ``method`` argument of
        ``make_conductivity_calculator()``.  Must not conflict with
        built-in method names.
    factory : callable
        Factory function.  Signature: ``(interaction, **kwargs) -> Any``.
        May accept only the kwargs it needs and ignore the rest.

    Raises
    ------
    ValueError
        If ``method`` conflicts with a built-in method name.

    Examples
    --------
    Register a custom RTA-based calculator::

        from phono3py.conductivity import register_calculator

        def my_factory(interaction, *, temperatures=None, sigmas=None, **kwargs):
            return MyCalculator(interaction, temperatures=temperatures, ...)

        register_calculator("my-rta", my_factory)
    """
    if method in _BUILTIN_METHODS:
        raise ValueError(
            f"'{method}' is a built-in method and cannot be overridden. "
            f"Built-in methods: {sorted(_BUILTIN_METHODS)}."
        )
    _REGISTRY[method] = factory


def make_conductivity_calculator(
    interaction: Interaction,
    method: str = "rta",
    grid_points: Sequence[int] | NDArray[np.int64] | None = None,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
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
) -> ConductivityCalculator | LBTECalculator:
    """Create a conductivity calculator with the appropriate building blocks.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance.  init_dynamical_matrix must have been called.
    method : str, optional
        Calculation method.  Built-in: "rta", "lbte", "wigner-rta",
        "wigner-lbte", "kubo-rta".  Default "rta".
    grid_points : array-like or None, optional
        BZ grid point indices.  None uses irreducible grid points.  Default None.
    temperatures : array-like or None, optional
        Temperatures in Kelvin.  Default None.
    sigmas : sequence or None, optional
        Smearing widths.  None selects the tetrahedron method.
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
    ConductivityCalculator or LBTECalculator
        For plugin-registered methods the return type is determined by the
        registered factory.
    """
    # Collect all kwargs to forward uniformly to registered factories.
    _all_kwargs = dict(
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=sigmas,
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

    if method in _REGISTRY:
        return _REGISTRY[method](interaction, **_all_kwargs)

    if method not in _BUILTIN_METHODS:
        raise NotImplementedError(
            f"method='{method}' is not implemented. "
            f"Built-in methods: {sorted(_BUILTIN_METHODS)}. "
            "Registered methods: "
            f"{sorted(_REGISTRY)}. "
            "Use register_calculator() to add custom methods."
        )

    _sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
    _temperatures: NDArray[np.double] | None = (
        np.asarray(temperatures, dtype="double") if temperatures is not None else None
    )

    # Point operations (needed by velocity providers).
    if is_kappa_star:
        point_ops = interaction.bz_grid.reciprocal_operations
        rot_cart: NDArray[np.double] = interaction.bz_grid.rotations_cartesian
    else:
        point_ops = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)
        rot_cart = np.eye(3, dtype="double", order="C").reshape(1, 3, 3)

    # ------------------------------------------------------------------
    # "lbte" branch
    # ------------------------------------------------------------------
    if method == "lbte":
        return _make_lbte_calculator(
            interaction,
            sigmas=_sigmas,
            sigma_cutoff=sigma_cutoff,
            temperatures=_temperatures,
            rot_cart=rot_cart,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            is_kappa_star=is_kappa_star,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            solve_collective_phonon=solve_collective_phonon,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            pp_filename=pp_filename,
            pinv_cutoff=pinv_cutoff,
            pinv_solver=pinv_solver,
            pinv_method=pinv_method,
            lang=lang,
            gv_delta_q=gv_delta_q,
            log_level=log_level,
        )

    # ------------------------------------------------------------------
    # "rta" branch
    # ------------------------------------------------------------------
    cv_provider = ModeHeatCapacityProvider(interaction)

    scattering_provider = RTAScatteringProvider(
        interaction,
        sigmas=_sigmas,
        temperatures=(
            _temperatures
            if _temperatures is not None
            else np.arange(0, 1001, 10, dtype="double")
        ),
        sigma_cutoff=sigma_cutoff,
        is_full_pp=is_full_pp,
        use_ave_pp=use_ave_pp,
        read_pp=read_pp,
        store_pp=store_pp,
        pp_filename=pp_filename,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume
    velocity_provider = GroupVelocityProvider(
        interaction,
        point_operations=point_ops,
        rotations_cartesian=rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )
    kappa_formula = BTEKappaFormula(
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor=conversion_factor,
    )
    accumulator = StandardKappaAccumulator(kappa_formula)

    return ConductivityCalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        scattering_provider=scattering_provider,
        accumulator=accumulator,
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=_sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        is_kappa_star=is_kappa_star,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )


def _make_lbte_calculator(
    interaction: Interaction,
    *,
    sigmas: list[float | None],
    sigma_cutoff: float | None,
    temperatures: NDArray[np.double] | None,
    rot_cart: NDArray[np.double],
    is_isotope: bool,
    mass_variances: Sequence[float] | NDArray[np.double] | None,
    boundary_mfp: float | None,
    is_kappa_star: bool,
    is_reducible_collision_matrix: bool,
    solve_collective_phonon: bool,
    is_full_pp: bool,
    read_pp: bool,
    pp_filename: str | os.PathLike | None,
    pinv_cutoff: float,
    pinv_solver: int,
    pinv_method: int,
    lang: Literal["C", "Python"],
    gv_delta_q: float | None,
    log_level: int,
) -> LBTECalculator:
    """Build and return an LBTECalculator."""
    if temperatures is None:
        _temps: NDArray[np.double] = np.array([300.0], dtype="double")
    else:
        _temps = temperatures

    # Ensure phonons at gamma are solved (idempotent).
    interaction.nac_q_direction = None
    interaction.run_phonon_solver_at_gamma()
    if not interaction.phonon_all_done:
        interaction.run_phonon_solver()
    frequencies, _, _ = interaction.get_phonons()

    # IR grid points in BZ-grid format.
    ir_grg, _, _ = get_ir_grid_points(interaction.bz_grid)
    ir_gps_bzg = np.array(interaction.bz_grid.grg2bzg[ir_grg], dtype="int64")

    # rot_grid_points in BZ-grid format: shape (num_ir_gp, num_ops).
    rot_grid_points: NDArray[np.int64] | None
    if is_reducible_collision_matrix:
        rot_grid_points = None
    else:
        if is_kappa_star:
            reciprocal_rotations = interaction.bz_grid.rotations
        else:
            reciprocal_rotations = np.eye(3, dtype="int64").reshape(1, 3, 3)
        num_ir_gp = len(ir_gps_bzg)
        num_ops = len(reciprocal_rotations)
        rot_grid_points = np.zeros((num_ir_gp, num_ops), dtype="int64")
        for i, ir_gp in enumerate(ir_gps_bzg):
            rot_grid_points[i] = get_grid_points_by_rotations(
                ir_gp,
                interaction.bz_grid,
                reciprocal_rotations=reciprocal_rotations,
            )

    # Build CollisionMatrix (pre-initialized with global grid structure).
    if is_reducible_collision_matrix:
        collision_matrix_obj = CollisionMatrix(
            interaction,
            is_reducible_collision_matrix=True,
            log_level=log_level,
            lang=lang,
        )
    else:
        assert rot_grid_points is not None
        collision_matrix_obj = CollisionMatrix(
            interaction,
            rotations_cartesian=rot_cart,
            num_ir_grid_points=len(ir_gps_bzg),
            rot_grid_points=rot_grid_points,
            log_level=log_level,
            lang=lang,
        )

    collision_provider = LBTECollisionProvider(
        interaction,
        collision_matrix_obj,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        temperatures=_temps,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        pp_filename=pp_filename,
        log_level=log_level,
    )

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume

    accumulator = LBTEKappaAccumulator(
        interaction,
        ir_grid_points=ir_gps_bzg,
        rot_grid_points=rot_grid_points,
        rotations_cartesian=rot_cart,
        frequencies=frequencies,
        sigmas=sigmas,
        temperatures=_temps,
        conversion_factor=conversion_factor,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        solve_collective_phonon=solve_collective_phonon,
        boundary_mfp=boundary_mfp,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        lang=lang,
        log_level=log_level,
    )

    point_ops = (
        interaction.bz_grid.reciprocal_operations
        if is_kappa_star
        else np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)
    )

    velocity_provider = GroupVelocityProvider(
        interaction,
        point_operations=point_ops,
        rotations_cartesian=rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    cv_provider = ModeHeatCapacityProvider(interaction)

    return LBTECalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        collision_provider=collision_provider,
        accumulator=accumulator,
        temperatures=_temps,
        sigmas=sigmas,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_full_pp=is_full_pp,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# Register built-in wigner and kubo factories.
# These entries can be overridden by external plugins via register_calculator().
# ---------------------------------------------------------------------------

from phono3py.conductivity.kubo_factory import make_kubo_rta_calculator  # noqa: E402

_REGISTRY["kubo-rta"] = make_kubo_rta_calculator

# Wigner is auto-registered by its own __init__.py.  The try/except allows
# phono3py-wigner to be installed as a standalone package via namespace packages.
try:
    import phono3py.conductivity.wigner  # noqa: F401, E402
except ImportError:
    pass
