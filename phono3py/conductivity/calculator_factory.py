"""Factory functions for the built-in RTA and LBTE calculators.

These functions are registered as non-overridable entries in factory._REGISTRY
for the ``"rta"`` and ``"lbte"`` methods.

``build_rta_base_components`` is a shared helper used by ``make_rta_calculator``
and the Wigner-RTA / Kubo-RTA factories in their respective subpackages.

``build_lbte_base_components`` is a shared helper used by both
``make_lbte_calculator`` and the Wigner-LBTE factory in
``conductivity/wigner/calculator_factory.py``.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.collision_matrix_solver import CollisionMatrixSolver
from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.lbte_collision_provider import LBTECollisionProvider
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator
from phono3py.conductivity.rta_calculator import RTACalculator
from phono3py.conductivity.rta_kappa_accumulator import RTAKappaAccumulator
from phono3py.conductivity.scattering_providers import RTAScatteringProvider
from phono3py.conductivity.utils import get_unit_to_WmK
from phono3py.conductivity.velocity_providers import GroupVelocityProvider
from phono3py.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
)
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction


@dataclass
class LBTEBaseComponents:
    """Shared LBTE infrastructure built by build_lbte_base_components().

    Contains the pre-assembled building blocks that are common to the
    standard LBTE, Wigner-LBTE, and Kubo-LBTE calculators.  Each caller
    wraps the solver in its own accumulator and substitutes its own
    velocity provider; everything else is reused as-is.

    Attributes
    ----------
    sigmas : list of float or None
        Smearing widths (empty list selects tetrahedron method).
    temperatures : ndarray of double, shape (num_temp,)
        Temperatures in Kelvin.
    rot_cart : ndarray of double, shape (num_ops, 3, 3)
        Cartesian rotation matrices.
    point_ops : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations.
    frequencies : ndarray of double, shape (num_gp, num_band)
        Phonon frequencies at all BZ grid points.
    ir_grid_points : ndarray of int64, shape (num_ir_gp,)
        Irreducible BZ grid point indices.
    rot_grid_points : ndarray of int64, shape (num_ir_gp, num_ops), or None
        Rotated grid point indices (None for reducible collision matrix).
    collision_provider : LBTECollisionProvider
        Pre-configured collision matrix provider.
    solver : CollisionMatrixSolver
        Pre-configured collision matrix solver.

    """

    sigmas: list[float | None]
    temperatures: NDArray[np.double]
    rot_cart: NDArray[np.double]
    point_ops: NDArray[np.int64]
    frequencies: NDArray[np.double]
    ir_grid_points: NDArray[np.int64]
    rot_grid_points: NDArray[np.int64] | None
    collision_provider: LBTECollisionProvider
    solver: CollisionMatrixSolver
    context: ConductivityContext


def build_lbte_base_components(
    interaction: Interaction,
    *,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    temperatures: NDArray[np.double] | None = None,
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
    boundary_mfp: float | None = None,
    log_level: int = 0,
) -> LBTEBaseComponents:
    """Build LBTE infrastructure shared by the standard and Wigner LBTE calculators.

    Runs the phonon solver, builds the collision matrix, and assembles the
    ``LBTECollisionProvider`` and ``LBTEKappaAccumulator``.  The caller is
    responsible for constructing the velocity provider (standard or Wigner) and
    the final ``LBTECalculator``.

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
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.
    log_level : int, optional
        Verbosity.

    Returns
    -------
    LBTEBaseComponents

    """
    _sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
    _temps: NDArray[np.double] = (
        np.array([300.0], dtype="double")
        if temperatures is None
        else np.asarray(temperatures, dtype="double")
    )

    if is_kappa_star:
        rot_cart: NDArray[np.double] = interaction.bz_grid.rotations_cartesian
        reciprocal_rotations: NDArray[np.int64] = interaction.bz_grid.rotations
        point_ops: NDArray[np.int64] = interaction.bz_grid.reciprocal_operations
    else:
        rot_cart = np.eye(3, dtype="double", order="C").reshape(1, 3, 3)
        reciprocal_rotations = np.eye(3, dtype="int64").reshape(1, 3, 3)
        point_ops = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)

    # Ensure phonons at gamma are solved (idempotent).
    interaction.nac_q_direction = None
    interaction.run_phonon_solver_at_gamma()
    if not interaction.phonon_all_done:
        interaction.run_phonon_solver()
    frequencies, eigenvectors, _ = interaction.get_phonons()

    # IR grid points in BZ-grid format.
    ir_grg, ir_weights, _ = get_ir_grid_points(interaction.bz_grid)
    ir_gps_bzg = np.array(interaction.bz_grid.grg2bzg[ir_grg], dtype="int64")

    # rot_grid_points: shape (num_ir_gp, num_ops), or None for reducible matrix.
    rot_grid_points: NDArray[np.int64] | None
    if is_reducible_collision_matrix:
        rot_grid_points = None
    else:
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

    context = ConductivityContext(
        grid_points=ir_gps_bzg,
        ir_grid_points=ir_gps_bzg,
        grid_weights=np.array(ir_weights, dtype="int64"),
        bz_grid=interaction.bz_grid,
        mesh_numbers=interaction.mesh_numbers,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        point_operations=point_ops,
        rotations_cartesian=rot_cart,
        temperatures=_temps,
        sigmas=_sigmas,
        sigma_cutoff_width=sigma_cutoff,
        boundary_mfp=boundary_mfp,
        band_indices=np.asarray(interaction.band_indices, dtype="int64"),
        cutoff_frequency=interaction.cutoff_frequency,
        rot_grid_points=rot_grid_points,
    )

    solver = CollisionMatrixSolver(
        context,
        conversion_factor=conversion_factor,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        solve_collective_phonon=solve_collective_phonon,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        lang=lang,
        log_level=log_level,
    )

    return LBTEBaseComponents(
        sigmas=_sigmas,
        temperatures=_temps,
        rot_cart=rot_cart,
        point_ops=point_ops,
        frequencies=frequencies,
        ir_grid_points=ir_gps_bzg,
        rot_grid_points=rot_grid_points,
        collision_provider=collision_provider,
        solver=solver,
        context=context,
    )


@dataclass
class RTABaseComponents:
    """Shared RTA infrastructure built by build_rta_base_components().

    Contains the pre-assembled building blocks that are common to standard
    BTE-RTA, Wigner-RTA, and Kubo-RTA calculators.  Each caller substitutes
    its own velocity provider, heat-capacity provider, and accumulator;
    everything else is reused as-is.

    Attributes
    ----------
    sigmas : list of float or None
        Smearing widths (empty list selects tetrahedron method).
    temperatures : ndarray of double or None
        Temperatures in Kelvin, or None when not provided.
    point_ops : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations.
    rot_cart : ndarray of double, shape (num_ops, 3, 3)
        Cartesian rotation matrices.
    scattering_provider : RTAScatteringProvider
        Pre-configured scattering provider.

    """

    sigmas: list[float | None]
    temperatures: NDArray[np.double] | None
    point_ops: NDArray[np.int64]
    rot_cart: NDArray[np.double]
    scattering_provider: RTAScatteringProvider
    context: ConductivityContext


def _resolve_rta_grid_points(
    bz_grid: BZGrid,
    grid_points: NDArray[np.int64] | None,
    is_kappa_star: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Resolve grid points, ir grid points, and weights for RTA.

    Returns
    -------
    grid_points : NDArray[np.int64]
        BZ grid point indices to iterate over.
    ir_grid_points : NDArray[np.int64]
        Irreducible BZ grid point indices.
    grid_weights : NDArray[np.int64]
        Symmetry weights for ``grid_points``.

    """
    ir_gps, ir_weights, ir_map = get_ir_grid_points(bz_grid)
    ir_gps_bzg = np.array(bz_grid.grg2bzg[ir_gps], dtype="int64")

    if grid_points is not None:
        # User-specified grid points: compute weights from IR map.
        weights = np.zeros_like(ir_map)
        for gp in ir_map:
            weights[gp] += 1
        gp_weights = np.array(
            weights[ir_map[bz_grid.bzg2grg[grid_points]]], dtype="int64"
        )
        return np.array(grid_points, dtype="int64"), ir_gps_bzg, gp_weights

    if not is_kappa_star:
        all_gps = np.array(bz_grid.grg2bzg, dtype="int64")
        return all_gps, all_gps, np.ones(len(all_gps), dtype="int64")

    return ir_gps_bzg, ir_gps_bzg, np.array(ir_weights, dtype="int64")


def build_rta_base_components(
    interaction: Interaction,
    *,
    grid_points: NDArray[np.int64] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    temperatures: NDArray[np.double] | None = None,
    boundary_mfp: float | None = None,
    is_kappa_star: bool = True,
    is_full_pp: bool = False,
    use_ave_pp: bool = False,
    read_pp: bool = False,
    store_pp: bool = False,
    pp_filename: str | os.PathLike | None = None,
    is_N_U: bool = False,
    is_gamma_detail: bool = False,
    log_level: int = 0,
) -> RTABaseComponents:
    """Build RTA infrastructure shared by standard, Wigner, and Kubo RTA.

    Normalises sigmas/temperatures, extracts symmetry operations, solves
    phonons, resolves grid points, constructs the ``RTAScatteringProvider``,
    and builds the ``ConductivityContext``.  The caller is responsible
    for constructing velocity/heat-capacity providers and the accumulator.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    grid_points : array-like or None, optional
        BZ grid point indices. None uses irreducible (or all, if not
        is_kappa_star) grid points.
    sigmas : sequence or None, optional
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.
    temperatures : array-like or None, optional
        Temperatures in Kelvin.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.
    is_kappa_star : bool, optional
        Use k-star symmetry.
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.
    use_ave_pp : bool, optional
        Use pre-averaged ph-ph interaction.
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

    Returns
    -------
    RTABaseComponents

    """
    _sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
    _temperatures: NDArray[np.double] | None = (
        np.asarray(temperatures, dtype="double") if temperatures is not None else None
    )

    if is_kappa_star:
        point_ops: NDArray[np.int64] = interaction.bz_grid.reciprocal_operations
        rot_cart: NDArray[np.double] = interaction.bz_grid.rotations_cartesian
    else:
        point_ops = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)
        rot_cart = np.eye(3, dtype="double", order="C").reshape(1, 3, 3)

    # Ensure phonons are solved (idempotent).
    interaction.nac_q_direction = None
    interaction.run_phonon_solver_at_gamma()
    if not interaction.phonon_all_done:
        interaction.run_phonon_solver()
    frequencies, eigenvectors, _ = interaction.get_phonons()

    # Resolve grid points.
    resolved_gps, ir_gps, gp_weights = _resolve_rta_grid_points(
        interaction.bz_grid, grid_points, is_kappa_star
    )

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

    context = ConductivityContext(
        grid_points=resolved_gps,
        ir_grid_points=ir_gps,
        grid_weights=gp_weights,
        bz_grid=interaction.bz_grid,
        mesh_numbers=interaction.mesh_numbers,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        point_operations=point_ops,
        rotations_cartesian=rot_cart,
        temperatures=_temperatures,
        sigmas=_sigmas,
        sigma_cutoff_width=sigma_cutoff,
        boundary_mfp=boundary_mfp,
        band_indices=np.asarray(interaction.band_indices, dtype="int64"),
        cutoff_frequency=interaction.cutoff_frequency,
    )

    return RTABaseComponents(
        sigmas=_sigmas,
        temperatures=_temperatures,
        point_ops=point_ops,
        rot_cart=rot_cart,
        scattering_provider=scattering_provider,
        context=context,
    )


def make_rta_calculator(
    interaction: Interaction,
    *,
    grid_points: NDArray[np.int64] | None = None,
    temperatures: NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
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
    log_level: int = 0,
    **_ignored: Any,
) -> RTACalculator:
    """Build a RTACalculator for the standard BTE-RTA method.

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
    RTACalculator

    """
    base = build_rta_base_components(
        interaction,
        grid_points=grid_points,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        temperatures=temperatures,
        boundary_mfp=boundary_mfp,
        is_kappa_star=is_kappa_star,
        is_full_pp=is_full_pp,
        use_ave_pp=use_ave_pp,
        read_pp=read_pp,
        store_pp=store_pp,
        pp_filename=pp_filename,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )

    cv_provider = ModeHeatCapacityProvider(interaction)

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume
    velocity_provider = GroupVelocityProvider(
        interaction,
        point_operations=base.point_ops,
        rotations_cartesian=base.rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )
    accumulator = RTAKappaAccumulator(
        context=base.context,
        conversion_factor=conversion_factor,
        log_level=log_level,
    )

    return RTACalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        scattering_provider=base.scattering_provider,
        accumulator=accumulator,
        context=base.context,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        log_level=log_level,
    )


def make_lbte_calculator(
    interaction: Interaction,
    *,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    temperatures: NDArray[np.double] | None = None,
    is_isotope: bool = False,
    mass_variances: NDArray[np.double] | None = None,
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
) -> LBTECalculator:
    """Build an LBTECalculator for the standard LBTE direct-solution method.

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
    LBTECalculator

    """
    base = build_lbte_base_components(
        interaction,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        temperatures=temperatures,
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
        boundary_mfp=boundary_mfp,
        log_level=log_level,
    )

    velocity_provider = GroupVelocityProvider(
        interaction,
        point_operations=base.point_ops,
        rotations_cartesian=base.rot_cart,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        log_level=log_level,
    )

    cv_provider = ModeHeatCapacityProvider(interaction)

    accumulator = LBTEKappaAccumulator(base.solver)

    return LBTECalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        collision_provider=base.collision_provider,
        accumulator=accumulator,
        context=base.context,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        is_full_pp=is_full_pp,
        log_level=log_level,
    )
