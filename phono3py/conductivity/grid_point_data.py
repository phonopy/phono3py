"""Per-grid-point data containers.

This module defines GridPointInput and GridPointResult, the core data
structures passed between building blocks in the conductivity calculation.

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


@dataclass
class GridPointResult:
    """Computed quantities at a single irreducible BZ grid point.

    Fields are filled incrementally by the building blocks (velocity
    provider, heat-capacity provider, scattering provider) and finally
    by the kappa formula.

    Shape conventions
    -----------------
    num_band0  : number of selected bands (len(band_indices))
    num_band   : total number of bands
    num_temp   : number of temperatures
    num_sigma  : number of broadening widths (smearing parameters)

    Velocity fields
    ~~~~~~~~~~~~~~~
    group_velocities : (num_band0, 3), real
        Diagonal (standard) group velocities; filled by all velocity providers.
    gv_by_gv : (num_band0, 6), real
        Symmetrised outer product v x v (Voigt notation).
        Filled by all velocity providers.
    vm_by_vm : (num_band0, num_band, 6), complex, optional
        Off-diagonal velocity operator/matrix outer product.
        Only set by Wigner and Kubo velocity providers.

    Heat-capacity fields
    ~~~~~~~~~~~~~~~~~~~~
    heat_capacities : (num_temp, num_band0), real
        Mode heat capacities (scalar Cv per mode).
    heat_capacity_matrix : (num_temp, num_band0, num_band), real, optional
        Heat-capacity matrix; only set by HeatCapacityMatrixProvider (Kubo).

    Scattering fields
    ~~~~~~~~~~~~~~~~~
    All scattering fields contain ph-ph, isotope, boundary, and elph
    contributions separately.  The effective linewidth used in the kappa
    formula is their sum.

    gamma : (num_sigma, num_temp, num_band0), real
        ph-ph linewidth (imaginary part of self-energy).
    gamma_isotope : (num_sigma, num_band0), real, optional
        Isotope scattering linewidth (temperature-independent).
    gamma_boundary : (num_band0,), real, optional
        Boundary scattering linewidth (sigma- and temperature-independent).
    gamma_elph : (num_sigma, num_temp, num_band0), real, optional
        Electron-phonon scattering linewidth.

    Note: isotope and boundary scattering are diagonal-only contributions
    that enter both the RTA and the LBTE collision matrix diagonal in the
    same way.

    Plugin-specific output fields
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    extra : dict[str, Any]
        Arbitrary plugin-specific data set by kappa-formula implementations.
        Keys are plugin-defined (e.g. ``"wigner_mode_kappa_C"``).
        Default is an empty dict; standard formulas do not populate this field.

    Summation helper
    ~~~~~~~~~~~~~~~~
    num_sampling_grid_points : int
        Cumulative count of BZ grid points represented by this irreducible
        point (i.e. k-star order). Accumulated by the velocity provider.
    """

    input: GridPointInput

    # --- velocity ---
    group_velocities: NDArray[np.double] | None = field(default=None)
    gv_by_gv: NDArray[np.double] | None = field(default=None)
    vm_by_vm: NDArray[np.cdouble] | None = field(default=None)

    # --- heat capacity ---
    heat_capacities: NDArray[np.double] | None = field(default=None)
    heat_capacity_matrix: NDArray[np.double] | None = field(default=None)

    # --- scattering ---
    gamma: NDArray[np.double] | None = field(default=None)
    gamma_isotope: NDArray[np.double] | None = field(default=None)
    gamma_boundary: NDArray[np.double] | None = field(default=None)
    gamma_elph: NDArray[np.double] | None = field(default=None)

    # --- auxiliary output ---
    averaged_pp_interaction: NDArray[np.double] | None = field(default=None)
    # Plugin-specific data; keys are plugin-defined strings.
    extra: dict[str, Any] = field(default_factory=dict)

    # --- BZ summation helper ---
    num_sampling_grid_points: int = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_effective_gamma(result: GridPointResult) -> NDArray[np.double]:
    """Return effective linewidth combining all diagonal scattering contributions.

    Sums ph-ph, isotope, boundary, and electron-phonon linewidths:

        gamma_eff = gamma + gamma_isotope + gamma_boundary + gamma_elph

    Only ``gamma`` (ph-ph) is required; the others are added when present.

    Parameters
    ----------
    result : GridPointResult
        Must have ``gamma`` (num_sigma, num_temp, num_band0) set.
        Optional contributions: ``gamma_isotope`` (num_sigma, num_band0),
        ``gamma_boundary`` (num_band0,), ``gamma_elph``
        (num_sigma, num_temp, num_band0).

    Returns
    -------
    ndarray of double, shape (num_sigma, num_temp, num_band0)
        Effective linewidth.
    """
    assert result.gamma is not None
    gamma = result.gamma.copy()
    if result.gamma_isotope is not None:
        gamma += result.gamma_isotope[:, np.newaxis, :]
    if result.gamma_boundary is not None:
        gamma += result.gamma_boundary[np.newaxis, np.newaxis, :]
    if result.gamma_elph is not None:
        gamma += result.gamma_elph
    return gamma


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
    "GridPointInput",
    "GridPointResult",
    "compute_effective_gamma",
    "make_grid_point_input",
    "VelocityProvider",
    "HeatCapacityProvider",
    "ScatteringProvider",
]
