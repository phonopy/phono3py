"""Factory functions for Wigner transport equation calculators.

These functions are registered as built-in entries in factory._REGISTRY and
serve as reference implementations for the plugin API.  External code can
override them by calling register_calculator() with the same method name.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.lbte_collision_provider import LBTECollisionProvider
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator
from phono3py.conductivity.rta_calculator import ConductivityCalculator
from phono3py.conductivity.scattering_providers import RTAScatteringProvider
from phono3py.conductivity.utils import get_unit_to_WmK
from phono3py.conductivity.wigner.accumulators import WignerKappaAccumulator
from phono3py.conductivity.wigner.formulas import (
    WignerKappaFormula,
    get_conversion_factor_WTE,
)
from phono3py.conductivity.wigner.lbte_calculator import WignerLBTECalculator
from phono3py.conductivity.wigner.providers import VelocityOperatorProvider
from phono3py.phonon.grid import get_grid_points_by_rotations, get_ir_grid_points
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction


def make_wigner_rta_calculator(
    interaction: Interaction,
    *,
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
    log_level: int = 0,
    **_ignored: Any,
) -> ConductivityCalculator:
    """Build a ConductivityCalculator for the Wigner-RTA method.

    Implements the Wigner transport equation in the RTA, adding the
    off-diagonal coherence contribution (C-term) to the standard
    population contribution (P-term).

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    grid_points : array-like or None, optional
        BZ grid point indices. None uses all irreducible grid points.
    temperatures : array-like or None, optional
        Temperatures in Kelvin.
    sigmas : sequence or None, optional
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.
    is_isotope : bool, optional
        Include isotope scattering.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.
    use_ave_pp : bool, optional
        Use pre-averaged ph-ph interaction.
    is_kappa_star : bool, optional
        Use k-star symmetry.
    gv_delta_q : float or None, optional
        Finite-difference step for group velocity (NAC).
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.
    read_pp : bool, optional
        Read ph-ph interaction from file.
    store_pp : bool, optional
        Store ph-ph interaction to file.
    pp_filename : str or path or None, optional
        Filename for ph-ph interaction I/O.
    is_N_U : bool, optional
        Decompose gamma into Normal and Umklapp.
    is_gamma_detail : bool, optional
        Store per-triplet gamma.
    log_level : int, optional
        Verbosity.
    **_ignored
        Absorbs LBTE-only keyword arguments passed by the dispatcher.

    Returns
    -------
    ConductivityCalculator
    """
    _sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
    _temperatures: NDArray[np.double] | None = (
        np.asarray(temperatures, dtype="double") if temperatures is not None else None
    )

    if is_kappa_star:
        point_ops = interaction.bz_grid.reciprocal_operations
        rot_cart: NDArray[np.double] = interaction.bz_grid.rotations_cartesian
    else:
        point_ops = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)
        rot_cart = np.eye(3, dtype="double", order="C").reshape(1, 3, 3)

    velocity_provider = VelocityOperatorProvider(
        interaction,
        point_operations=point_ops,
        rotations_cartesian=rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

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

    kappa_formula = WignerKappaFormula(
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor_WTE=get_conversion_factor_WTE(interaction.primitive.volume),
    )
    accumulator = WignerKappaAccumulator(kappa_formula)

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


def make_wigner_lbte_calculator(
    interaction: Interaction,
    *,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
    boundary_mfp: float | None = None,
    is_kappa_star: bool = True,
    is_reducible_collision_matrix: bool = False,
    solve_collective_phonon: bool = False,
    is_full_pp: bool = False,
    read_pp: bool = False,
    pp_filename: str | os.PathLike | None = None,
    pinv_cutoff: float = 1.0e-8,
    pinv_solver: int = 0,
    pinv_method: int = 0,
    lang: Literal["C", "Python"] = "C",
    gv_delta_q: float | None = None,
    log_level: int = 0,
    **_ignored: Any,
) -> WignerLBTECalculator:
    """Build a WignerLBTECalculator.

    Implements the Wigner transport equation via direct LBTE solution,
    adding the coherence (C) term on top of the LBTE population (P) terms.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    sigmas : sequence or None, optional
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.
    temperatures : array-like or None, optional
        Temperatures in Kelvin. Default [300.0].
    is_isotope : bool, optional
        Include isotope scattering.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.
    is_kappa_star : bool, optional
        Use k-star symmetry.
    is_reducible_collision_matrix : bool, optional
        Use the full reducible collision matrix.
    solve_collective_phonon : bool, optional
        Use Chaput collective-phonon method.
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.
    read_pp : bool, optional
        Read ph-ph interaction from file.
    pp_filename : str or path or None, optional
        Filename for ph-ph interaction I/O.
    pinv_cutoff : float, optional
        Eigenvalue cutoff for pseudo-inversion.
    pinv_solver : int, optional
        Solver selection index.
    pinv_method : int, optional
        Pseudo-inverse criterion.
    lang : {"C", "Python"}, optional
        Backend for C-extension operations.
    gv_delta_q : float or None, optional
        Finite-difference step for group velocity (NAC).
    log_level : int, optional
        Verbosity.
    **_ignored
        Absorbs RTA-only keyword arguments passed by the dispatcher.

    Returns
    -------
    WignerLBTECalculator
    """
    _sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
    _temps: NDArray[np.double] = (
        np.array([300.0], dtype="double")
        if temperatures is None
        else np.asarray(temperatures, dtype="double")
    )

    if is_kappa_star:
        rot_cart: NDArray[np.double] = interaction.bz_grid.rotations_cartesian
    else:
        rot_cart = np.eye(3, dtype="double", order="C").reshape(1, 3, 3)

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
        sigmas=_sigmas,
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
        sigmas=_sigmas,
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

    velocity_provider = VelocityOperatorProvider(
        interaction,
        point_operations=point_ops,
        rotations_cartesian=rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    cv_provider = ModeHeatCapacityProvider(interaction)

    return WignerLBTECalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        collision_provider=collision_provider,
        accumulator=accumulator,
        temperatures=_temps,
        sigmas=_sigmas,
        conversion_factor_WTE=get_conversion_factor_WTE(interaction.primitive.volume),
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_full_pp=is_full_pp,
        log_level=log_level,
    )
