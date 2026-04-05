"""Per-grid-point data containers.

This module defines GridPointInput, provider result types
(VelocityResult, HeatCapacityResult, ScatteringResult), and
GridPointAggregates used in the conductivity calculation.

Protocol interfaces (VelocityProvider, HeatCapacityProvider,
ScatteringProvider) are defined in ``phono3py.conductivity.protocols``
and re-exported here for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.phonon.grid import BZGrid, get_qpoints_from_bz_grid_points

# ---------------------------------------------------------------------------
# Per-grid-point data containers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider result types
# ---------------------------------------------------------------------------


@dataclass
class VelocityResult:
    """Result from a velocity provider at a single grid point.

    Parameters
    ----------
    group_velocities : ndarray of double, shape (num_band0, 3)
        Diagonal (standard) group velocities.
    gv_by_gv : ndarray of double, shape (num_band0, 6)
        Symmetrised outer product v x v in Voigt notation.
    vm_by_vm : ndarray of cdouble, shape (num_band0, num_band, 6), optional
        Off-diagonal velocity operator/matrix outer product.
        Only set by Wigner and Kubo velocity providers.
    num_sampling_grid_points : int
        k-star order (number of arms) for this irreducible point.
    extra : dict
        Plugin-specific data (e.g. velocity_operator for Wigner HDF5 output).

    """

    group_velocities: NDArray[np.double]
    gv_by_gv: NDArray[np.double] | None = None
    vm_by_vm: NDArray[np.cdouble] | None = None
    num_sampling_grid_points: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeatCapacityResult:
    """Result from a heat capacity provider at a single grid point.

    Parameters
    ----------
    heat_capacities : ndarray of double, shape (num_temp, num_band0)
        Mode heat capacities (scalar Cv per mode).
    heat_capacity_matrix : ndarray of double, optional
        Shape (num_temp, num_band0, num_band).
        Only set by HeatCapacityMatrixProvider (Kubo).
    extra : dict
        Plugin-specific data.

    """

    heat_capacities: NDArray[np.double]
    heat_capacity_matrix: NDArray[np.double] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScatteringResult:
    """Result from a scattering provider at a single grid point.

    Parameters
    ----------
    gamma : ndarray of double, shape (num_sigma, num_temp, num_band0)
        Ph-ph linewidth (imaginary part of self-energy).
    averaged_pp_interaction : ndarray of double, shape (num_band0,), optional
        Averaged ph-ph interaction strength.
    extra : dict
        Plugin-specific data.

    """

    gamma: NDArray[np.double]
    averaged_pp_interaction: NDArray[np.double] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Grid-point input / result containers
# ---------------------------------------------------------------------------


@dataclass
class GridPointInput:
    """Phonon quantities at a single irreducible BZ grid point.

    All fields are filled before being passed to building blocks.

    Parameters
    ----------
    grid_point : int
        BZ grid point index.
    q_point : ndarray, shape (3,)
        q-point coordinates in reduced reciprocal coordinates.
    frequencies : ndarray, shape (num_band,)
        Phonon frequencies in THz for *all* bands at this grid point.
    eigenvectors : ndarray, shape (num_band, num_band), complex
        Phonon eigenvectors at this grid point.
    grid_weight : int
        Symmetry weight for BZ summation (number of arms of the k-star).
    band_indices : ndarray of int64, shape (num_band0,)
        Selected band indices. num_band0 <= num_band.

    """

    grid_point: int
    q_point: NDArray[np.double]
    frequencies: NDArray[np.double]
    eigenvectors: NDArray[np.cdouble]
    grid_weight: int
    band_indices: NDArray[np.int64]


# ---------------------------------------------------------------------------
# Aggregated per-grid-point data (Calculator -> Accumulator.finalize())
# ---------------------------------------------------------------------------


@dataclass
class GridPointAggregates:
    """Aggregated per-grid-point data passed from Calculator to Accumulator.

    Built by the Calculator after the grid-point loop and passed to
    ``accumulator.finalize()``.  Replaces the former untyped
    ``grid_point_data: dict[str, Any]``.

    Always present
    ~~~~~~~~~~~~~~
    num_sampling_grid_points : int
        Total number of BZ grid points represented by the sampled
        irreducible grid points.
    group_velocities : (num_gp, num_band0, 3), real
        Group velocities at each irreducible grid point.
    mode_heat_capacities : (num_temp, num_gp, num_band0), real
        Mode heat capacities.

    Common optional
    ~~~~~~~~~~~~~~~
    gv_by_gv : (num_gp, num_band0, 6), real
        Symmetrised outer product v x v in Voigt notation.
    gamma : (num_sigma, num_temp, num_gp, num_band0), real
        Ph-ph linewidths.
    gamma_isotope : (num_sigma, num_gp, num_band0), real
        Isotope scattering linewidths.
    gamma_boundary : (num_gp, num_band0), real
        Boundary scattering linewidths.
    gamma_elph : (num_sigma, num_temp, num_gp, num_band0), real
        Electron-phonon scattering linewidths.

    Plugin-specific
    ~~~~~~~~~~~~~~~
    vm_by_vm : (num_gp, num_band0, num_band, 6), complex
        Off-diagonal velocity operator outer product (Wigner/Kubo).
    heat_capacity_matrix : (num_temp, num_gp, num_band0, num_band), real
        Heat-capacity matrix (Kubo).
    extra : dict
        Plugin-specific data (e.g. velocity_operator for Wigner).

    """

    num_sampling_grid_points: int
    group_velocities: NDArray[np.double]
    mode_heat_capacities: NDArray[np.double]
    gv_by_gv: NDArray[np.double] | None = None
    gamma: NDArray[np.double] | None = None
    gamma_isotope: NDArray[np.double] | None = None
    gamma_boundary: NDArray[np.double] | None = None
    gamma_elph: NDArray[np.double] | None = None
    vm_by_vm: NDArray[np.cdouble] | None = None
    heat_capacity_matrix: NDArray[np.double] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_effective_gamma(
    aggregates: GridPointAggregates,
) -> NDArray[np.double]:
    """Return effective linewidth from aggregated scattering arrays.

    Sums ph-ph, isotope, boundary, and electron-phonon linewidths
    with proper broadcasting over the
    (num_sigma, num_temp, num_gp, num_band0) shape.

    Parameters
    ----------
    aggregates : GridPointAggregates
        Must have ``gamma`` set.

    Returns
    -------
    ndarray of double, shape (num_sigma, num_temp, num_gp, num_band0)
        Effective linewidth.

    """
    assert aggregates.gamma is not None
    out = aggregates.gamma.copy()
    if aggregates.gamma_isotope is not None:
        out += aggregates.gamma_isotope[:, np.newaxis, :, :]
    if aggregates.gamma_boundary is not None:
        out += aggregates.gamma_boundary[np.newaxis, np.newaxis, :, :]
    if aggregates.gamma_elph is not None:
        out += aggregates.gamma_elph
    return out


def make_grid_point_input(
    grid_point: int,
    grid_weight: int,
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    bz_grid: BZGrid,
    band_indices: NDArray[np.int64],
) -> GridPointInput:
    """Create a GridPointInput for a single BZ grid point.

    Parameters
    ----------
    grid_point : int
        BZ grid point index.
    grid_weight : int
        Symmetry weight for BZ summation.
    frequencies : ndarray of double, shape (num_bz_gp, num_band)
        Phonon frequencies array indexed by BZ grid point.
    eigenvectors : ndarray of cdouble, shape (num_bz_gp, num_band, num_band)
        Phonon eigenvectors array indexed by BZ grid point.
    bz_grid : BZGrid
        Brillouin zone grid object.
    band_indices : ndarray of int64, shape (num_band0,)
        Selected band indices.

    Returns
    -------
    GridPointInput

    """
    return GridPointInput(
        grid_point=grid_point,
        q_point=np.array(
            get_qpoints_from_bz_grid_points(grid_point, bz_grid),
            dtype="double",
        ),
        frequencies=frequencies[grid_point],
        eigenvectors=eigenvectors[grid_point],
        grid_weight=grid_weight,
        band_indices=band_indices,
    )


# ---------------------------------------------------------------------------
# Building-block Protocol interfaces
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Re-export Protocol interfaces for backward compatibility.
# Canonical definitions live in phono3py.conductivity.protocols.
# ---------------------------------------------------------------------------

from phono3py.conductivity.protocols import (  # noqa: E402
    HeatCapacityProvider,
    ScatteringProvider,
    VelocityProvider,
)

__all__ = [
    "GridPointAggregates",
    "GridPointInput",
    "HeatCapacityResult",
    "ScatteringResult",
    "VelocityResult",
    "compute_effective_gamma",
    "make_grid_point_input",
    "VelocityProvider",
    "HeatCapacityProvider",
    "ScatteringProvider",
]
