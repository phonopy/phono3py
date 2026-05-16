"""Shared building blocks for conductivity calculator construction.

``build_rta_kappa_settings`` and ``build_lbte_kappa_settings`` resolve
grid points and phonons, then return a ``KappaSettings``.  Solver
construction is handled by the factory functions in ``factory.py``.

``VariantContext`` and ``CalculatorConfig`` are data classes consumed
by the framework and plugin component factories.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
)

from phono3py.conductivity.utils import get_unit_to_WmK
from phono3py.phonon3.interaction import Interaction

if TYPE_CHECKING:
    from phono3py.conductivity.collision_matrix_kernel import CollisionMatrixKernel


@dataclass(frozen=True)
class KappaSettings:
    """Computation metadata shared by Calculator and KappaSolver.

    This dataclass consolidates grid, symmetry, and configuration
    metadata that was previously scattered across Calculator and KappaSolver
    constructors.  Immutable after construction -- use
    ``dataclasses.replace()`` to create a modified copy.

    """

    # -- Grid metadata --

    grid_points: NDArray[np.int64]
    """BZ grid point indices that are iterated over.

    For RTA this may be user-specified grid points or the irreducible set.
    For LBTE this is always the irreducible set.

    """

    grid_weights: NDArray[np.int64]
    """Symmetry weights for ``grid_points``."""

    bz_grid: BZGrid
    """BZ grid object."""

    mesh_numbers: NDArray[np.int64]
    """BZ mesh numbers, shape (3,)."""

    # -- Configuration --

    is_kappa_star: bool
    """Whether k-star symmetry is used."""

    temperatures: NDArray[np.double]
    """Temperatures in Kelvin, shape (num_temp,)."""

    sigmas: Sequence[float | None]
    """Smearing widths.  None entry selects the tetrahedron method."""

    boundary_mfp: float | None
    """Boundary mean free path in micrometres."""

    band_indices: NDArray[np.int64]
    """Band indices used for the calculation."""

    cutoff_frequency: float
    """Modes below this frequency (THz) are excluded."""

    conversion_factor: float
    """Unit conversion factor (get_unit_to_WmK() / volume)."""

    gv_delta_q: float | None
    """Finite-difference step for group velocity (NAC)."""

    is_reducible_collision_matrix: bool
    """Whether the full reducible collision matrix is used (LBTE only)."""


@dataclass
class VariantContext:
    """Context provided to variant component factories during construction.

    An instance is created by the framework and passed to each component
    factory callable registered via ``register_variant()``.  Plugin authors
    use the fields to construct their velocity solvers, CV providers, and
    kappa solvers.

    Attributes
    ----------
    interaction : Interaction
        Interaction instance with dynamical matrix initialised.
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, settings).
    log_level : int
        Verbosity level.
    collision_matrix_kernel : CollisionMatrixKernel or None
        Pre-configured collision matrix solver (LBTE only; None for RTA).

    """

    interaction: Interaction
    kappa_settings: KappaSettings
    log_level: int
    collision_matrix_kernel: CollisionMatrixKernel | None = None


@dataclass
class CalculatorConfig:
    """All settings for conductivity calculator construction.

    Created by ``conductivity_calculator()`` from its keyword arguments
    and passed through the factory chain.  Plugin developers using
    ``register_variant()`` never touch this directly -- it is consumed by
    the framework's internal factory closures.

    Attributes
    ----------
    temperatures : ndarray of double
        Temperatures in Kelvin.
    sigmas : sequence of (float or None)
        Smearing widths. None entry selects the tetrahedron method.
    grid_points : ndarray of int64 or None
        BZ grid point indices. None uses irreducible grid points.
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
    lang : {"C", "Python", "Rust"}
        Backend for scattering-kernel operations.  ``"Rust"`` is
        currently only wired through the RTA low-memory collision path;
        LBTE and other paths fall back to the C backend.
    log_level : int
        Verbosity.
    rust_gp_batch_size : int or None
        Batch size for the Rust batched grid-point path (RTA only).
        ``None`` defers to the ``PHONO3PY_RUST_GP_BATCH_SIZE`` env var
        (default 0 = batching disabled).  ``0`` forces the non-batched
        per-gp path; a positive integer enables batched
        ``compute_batched`` calls of that size.

    """

    temperatures: NDArray[np.double]
    sigmas: Sequence[float | None]
    grid_points: NDArray[np.int64] | None = None
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
    lang: Literal["C", "Python", "Rust"] = "Rust"
    log_level: int = 0
    rust_gp_batch_size: int | None = None


def build_lbte_kappa_settings(
    interaction: Interaction,
    config: CalculatorConfig,
) -> KappaSettings:
    """Resolve LBTE grid points and build KappaSettings.

    The caller must ensure that phonons have been solved before calling
    this function.

    """
    # Grid points in BZ-grid format.
    if config.is_kappa_star:
        ir_grg, ir_weights, _ = get_ir_grid_points(interaction.bz_grid)
        ir_gps_bzg = np.array(interaction.bz_grid.grg2bzg[ir_grg], dtype="int64")
    else:
        ir_gps_bzg = np.array(interaction.bz_grid.grg2bzg, dtype="int64")
        ir_weights = np.ones(len(ir_gps_bzg), dtype="int64")

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume

    return KappaSettings(
        grid_points=ir_gps_bzg,
        grid_weights=ir_weights,
        bz_grid=interaction.bz_grid,
        mesh_numbers=interaction.mesh_numbers,
        is_kappa_star=config.is_kappa_star,
        temperatures=config.temperatures,
        sigmas=config.sigmas,
        boundary_mfp=config.boundary_mfp,
        band_indices=np.asarray(interaction.band_indices, dtype="int64"),
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor=conversion_factor,
        gv_delta_q=config.gv_delta_q,
        is_reducible_collision_matrix=config.is_reducible_collision_matrix,
    )


def build_rot_grid_points(
    kappa_settings: KappaSettings,
) -> NDArray[np.int64] | None:
    """Compute rot_grid_points for LBTE collision matrix.

    Returns None when ``is_reducible_collision_matrix`` is True.

    """
    if kappa_settings.is_reducible_collision_matrix:
        return None

    if kappa_settings.is_kappa_star:
        rotations = kappa_settings.bz_grid.rotations
    else:
        rotations = np.eye(3, dtype="int64").reshape(1, 3, 3)

    gps = kappa_settings.grid_points
    num_ops = len(rotations)
    rot_gps = np.zeros((len(gps), num_ops), dtype="int64")
    for i, gp in enumerate(gps):
        rot_gps[i] = get_grid_points_by_rotations(
            gp,
            kappa_settings.bz_grid,
            reciprocal_rotations=rotations,
        )
    return rot_gps


def _resolve_rta_grid_points(
    bz_grid: BZGrid,
    grid_points: NDArray[np.int64] | None,
    is_kappa_star: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Resolve grid points and weights for RTA.

    Returns
    -------
    grid_points : NDArray[np.int64]
        BZ grid point indices to iterate over.
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
        return np.array(grid_points, dtype="int64"), gp_weights

    if not is_kappa_star:
        all_gps = bz_grid.grg2bzg
        return all_gps, np.ones(len(all_gps), dtype="int64")

    return ir_gps_bzg, ir_weights


def build_rta_kappa_settings(
    interaction: Interaction,
    config: CalculatorConfig,
) -> KappaSettings:
    """Resolve RTA grid points and build KappaSettings.

    The caller must ensure that phonons have been solved before calling
    this function.

    """
    # Resolve grid points.
    grid_points, gp_weights = _resolve_rta_grid_points(
        interaction.bz_grid, config.grid_points, config.is_kappa_star
    )

    conversion_factor = get_unit_to_WmK() / interaction.primitive.volume

    return KappaSettings(
        grid_points=grid_points,
        grid_weights=gp_weights,
        bz_grid=interaction.bz_grid,
        mesh_numbers=interaction.mesh_numbers,
        is_kappa_star=config.is_kappa_star,
        temperatures=config.temperatures,
        sigmas=config.sigmas,
        boundary_mfp=config.boundary_mfp,
        band_indices=np.asarray(interaction.band_indices, dtype="int64"),
        cutoff_frequency=interaction.cutoff_frequency,
        conversion_factor=conversion_factor,
        gv_delta_q=config.gv_delta_q,
        is_reducible_collision_matrix=False,
    )
