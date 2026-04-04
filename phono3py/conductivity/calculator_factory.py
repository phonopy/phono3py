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
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.collision_matrix_solver import CollisionMatrixSolver
from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.kappa_accumulators import (
    LBTEKappaAccumulator,
    RTAKappaAccumulator,
)
from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.lbte_collision_provider import LBTECollisionProvider
from phono3py.conductivity.rta_calculator import RTACalculator
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
class VariantBuildContext:
    """Context provided to variant component factories during construction.

    An instance is created by the framework and passed to each component
    factory callable registered via ``register_variant()``.  Plugin authors
    use the fields to construct their velocity providers, CV providers, and
    kappa accumulators.

    Attributes
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, settings).
    point_operations : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations.
    rotations_cartesian : ndarray of double, shape (num_ops, 3, 3)
        Cartesian rotation matrices.
    conversion_factor : float
        Standard unit conversion factor (get_unit_to_WmK() / volume).
        Plugins may compute their own factor from ``interaction``.
    is_kappa_star : bool
        Whether k-star symmetry is used.
    gv_delta_q : float or None
        Finite-difference step for group velocity (NAC).
    is_reducible_collision_matrix : bool
        Whether the full reducible collision matrix is used (LBTE only).
    log_level : int
        Verbosity level.
    solver : CollisionMatrixSolver or None
        Pre-configured collision matrix solver (LBTE only; None for RTA).

    """

    interaction: Interaction
    context: ConductivityContext
    point_operations: NDArray[np.int64]
    rotations_cartesian: NDArray[np.double]
    conversion_factor: float
    is_kappa_star: bool
    gv_delta_q: float | None
    is_reducible_collision_matrix: bool
    log_level: int
    solver: CollisionMatrixSolver | None = None


@dataclass
class CalculatorConfig:
    """All settings for conductivity calculator construction.

    Created by ``make_conductivity_calculator()`` from its keyword arguments
    and passed through the factory chain.  Plugin developers using
    ``register_variant()`` never touch this directly -- it is consumed by
    the framework's internal factory closures.

    Attributes
    ----------
    grid_points : ndarray of int64 or None
        BZ grid point indices. None uses irreducible grid points.
    temperatures : ndarray of double or None
        Temperatures in Kelvin.
    sigmas : sequence of (float or None) or None
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None
        Smearing cutoff in units of sigma.
    is_isotope : bool
        Include isotope scattering.
    mass_variances : ndarray of double or None
        Mass variances for isotope scattering.
    boundary_mfp : float or None
        Boundary mean free path in micrometres.
    use_ave_pp : bool
        Use pre-averaged ph-ph interaction (RTA only).
    is_kappa_star : bool
        Use k-star symmetry.
    gv_delta_q : float or None
        Finite-difference step for group velocity (NAC).
    is_full_pp : bool
        Compute full ph-ph interaction matrix.
    read_pp : bool
        Read ph-ph interaction from file.
    store_pp : bool
        Store ph-ph interaction to file (RTA only).
    pp_filename : str, path, or None
        Filename for ph-ph interaction I/O.
    is_N_U : bool
        Decompose gamma into Normal and Umklapp (RTA only).
    is_gamma_detail : bool
        Store per-triplet gamma (RTA only).
    is_reducible_collision_matrix : bool
        Use full reducible collision matrix (LBTE only).
    solve_collective_phonon : bool
        Use Chaput collective-phonon method (LBTE only).
    pinv_cutoff : float
        Eigenvalue cutoff for pseudo-inversion (LBTE only).
    pinv_solver : int
        Solver selection index (LBTE only).
    pinv_method : int
        Pseudo-inverse criterion (LBTE only).
    lang : {"C", "Python"}
        Backend for C-extension operations (LBTE only).
    log_level : int
        Verbosity.

    """

    grid_points: NDArray[np.int64] | None = None
    temperatures: NDArray[np.double] | None = None
    sigmas: Sequence[float | None] | None = None
    sigma_cutoff: float | None = None
    is_isotope: bool = False
    mass_variances: NDArray[np.double] | None = None
    boundary_mfp: float | None = None
    use_ave_pp: bool = False
    is_kappa_star: bool = True
    gv_delta_q: float | None = None
    is_full_pp: bool = False
    read_pp: bool = False
    store_pp: bool = False
    pp_filename: str | os.PathLike | None = None
    is_N_U: bool = False
    is_gamma_detail: bool = False
    is_reducible_collision_matrix: bool = False
    solve_collective_phonon: bool = False
    pinv_cutoff: float = 1.0e-8
    pinv_solver: int = 0
    pinv_method: int = 0
    lang: Literal["C", "Python"] = "C"
    log_level: int = 0


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
    config: CalculatorConfig,
) -> LBTEBaseComponents:
    """Build LBTE infrastructure shared by the standard and Wigner LBTE calculators.

    Runs the phonon solver, builds the collision matrix, and assembles the
    ``LBTECollisionProvider`` and ``CollisionMatrixSolver``.  The caller is
    responsible for constructing the velocity provider (standard or Wigner) and
    the final ``LBTECalculator``.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    config : CalculatorConfig
        Calculator settings.

    Returns
    -------
    LBTEBaseComponents

    """
    _sigmas: list[float | None] = [] if config.sigmas is None else list(config.sigmas)
    _temps: NDArray[np.double] = (
        np.array([300.0], dtype="double")
        if config.temperatures is None
        else np.asarray(config.temperatures, dtype="double")
    )

    if config.is_kappa_star:
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
    if config.is_reducible_collision_matrix:
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
    if config.is_reducible_collision_matrix:
        collision_matrix_obj = CollisionMatrix(
            interaction,
            is_reducible_collision_matrix=True,
            log_level=config.log_level,
            lang=config.lang,
        )
    else:
        assert rot_grid_points is not None
        collision_matrix_obj = CollisionMatrix(
            interaction,
            rotations_cartesian=rot_cart,
            num_ir_grid_points=len(ir_gps_bzg),
            rot_grid_points=rot_grid_points,
            log_level=config.log_level,
            lang=config.lang,
        )

    collision_provider = LBTECollisionProvider(
        interaction,
        collision_matrix_obj,
        sigmas=_sigmas,
        sigma_cutoff=config.sigma_cutoff,
        temperatures=_temps,
        is_full_pp=config.is_full_pp,
        read_pp=config.read_pp,
        pp_filename=config.pp_filename,
        log_level=config.log_level,
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
        sigma_cutoff_width=config.sigma_cutoff,
        boundary_mfp=config.boundary_mfp,
        band_indices=np.asarray(interaction.band_indices, dtype="int64"),
        cutoff_frequency=interaction.cutoff_frequency,
        rot_grid_points=rot_grid_points,
    )

    solver = CollisionMatrixSolver(
        context,
        conversion_factor=conversion_factor,
        is_reducible_collision_matrix=config.is_reducible_collision_matrix,
        is_kappa_star=config.is_kappa_star,
        solve_collective_phonon=config.solve_collective_phonon,
        pinv_cutoff=config.pinv_cutoff,
        pinv_solver=config.pinv_solver,
        pinv_method=config.pinv_method,
        lang=config.lang,
        log_level=config.log_level,
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
    config: CalculatorConfig,
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
    config : CalculatorConfig
        Calculator settings.

    Returns
    -------
    RTABaseComponents

    """
    _sigmas: list[float | None] = [] if config.sigmas is None else list(config.sigmas)
    _temperatures: NDArray[np.double] | None = (
        np.asarray(config.temperatures, dtype="double")
        if config.temperatures is not None
        else None
    )

    if config.is_kappa_star:
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
        interaction.bz_grid, config.grid_points, config.is_kappa_star
    )

    scattering_provider = RTAScatteringProvider(
        interaction,
        sigmas=_sigmas,
        temperatures=(
            _temperatures
            if _temperatures is not None
            else np.arange(0, 1001, 10, dtype="double")
        ),
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
        sigma_cutoff_width=config.sigma_cutoff,
        boundary_mfp=config.boundary_mfp,
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
    config: CalculatorConfig,
) -> RTACalculator:
    """Build a RTACalculator for the standard BTE-RTA method.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    config : CalculatorConfig
        Calculator settings.

    Returns
    -------
    RTACalculator

    """
    base = build_rta_base_components(interaction, config)

    cv_provider = ModeHeatCapacityProvider(interaction)

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume
    velocity_provider = GroupVelocityProvider(
        interaction,
        point_operations=base.point_ops,
        rotations_cartesian=base.rot_cart,
        is_kappa_star=config.is_kappa_star,
        gv_delta_q=config.gv_delta_q,
        log_level=config.log_level,
    )
    accumulator = RTAKappaAccumulator(
        context=base.context,
        conversion_factor=conversion_factor,
        log_level=config.log_level,
    )

    return RTACalculator(
        interaction,
        velocity_provider=velocity_provider,
        cv_provider=cv_provider,
        scattering_provider=base.scattering_provider,
        accumulator=accumulator,
        context=base.context,
        is_isotope=config.is_isotope,
        mass_variances=config.mass_variances,
        is_N_U=config.is_N_U,
        is_gamma_detail=config.is_gamma_detail,
        log_level=config.log_level,
    )


def make_lbte_calculator(
    interaction: Interaction,
    config: CalculatorConfig,
) -> LBTECalculator:
    """Build an LBTECalculator for the standard LBTE direct-solution method.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    config : CalculatorConfig
        Calculator settings.

    Returns
    -------
    LBTECalculator

    """
    base = build_lbte_base_components(interaction, config)

    velocity_provider = GroupVelocityProvider(
        interaction,
        point_operations=base.point_ops,
        rotations_cartesian=base.rot_cart,
        is_kappa_star=config.is_kappa_star,
        gv_delta_q=config.gv_delta_q,
        log_level=config.log_level,
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
        is_isotope=config.is_isotope,
        mass_variances=config.mass_variances,
        is_full_pp=config.is_full_pp,
        log_level=config.log_level,
    )
