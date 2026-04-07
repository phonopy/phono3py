"""Heat capacity provider building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import HeatCapacityResult
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix
from phono3py.phonon3.interaction import Interaction

# ---------------------------------------------------------------------------
# Bulk computation functions
# ---------------------------------------------------------------------------


def compute_bulk_mode_cv(
    frequencies: NDArray[np.double],
    grid_points: NDArray[np.int64],
    temperatures: NDArray[np.double],
    band_indices: NDArray[np.int64],
    cutoff_frequency: float,
) -> NDArray[np.double]:
    """Compute mode heat capacities for all grid points at once.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_bz_gp, num_band)
        Phonon frequencies in THz for all BZ grid points.
    grid_points : ndarray of int64, shape (num_gp,)
        BZ grid point indices.
    temperatures : ndarray of double, shape (num_temp,)
        Temperatures in Kelvin.
    band_indices : ndarray of int64, shape (num_band0,)
        Selected band indices.
    cutoff_frequency : float
        Cutoff frequency in THz.

    Returns
    -------
    ndarray of double, shape (num_temp, num_gp, num_band0)
        Mode heat capacities.

    """
    thz_to_ev = get_physical_units().THzToEv
    freqs_ev = frequencies[grid_points] * thz_to_ev  # (num_gp, num_band)
    cutoff_ev = cutoff_frequency * thz_to_ev
    num_gp, num_band = freqs_ev.shape
    num_temp = len(temperatures)

    flat_freqs = freqs_ev.ravel()  # (num_gp * num_band,)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cv_flat = mode_cv(temperatures, flat_freqs)  # (num_temp, num_gp * num_band)
    cv = cv_flat.reshape(num_temp, num_gp, num_band)

    condition = freqs_ev > cutoff_ev  # (num_gp, num_band)
    cv = np.where(np.isfinite(cv) & condition[None, :, :], cv, 0.0)
    return cv[:, :, band_indices]


def compute_bulk_cv_matrix(
    frequencies: NDArray[np.double],
    grid_points: NDArray[np.int64],
    temperatures: NDArray[np.double],
    band_indices: NDArray[np.int64],
    cutoff_frequency: float,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Compute cv and cv_matrix for all grid points.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_bz_gp, num_band)
        Phonon frequencies in THz for all BZ grid points.
    grid_points : ndarray of int64, shape (num_gp,)
        BZ grid point indices.
    temperatures : ndarray of double, shape (num_temp,)
        Temperatures in Kelvin.
    band_indices : ndarray of int64, shape (num_band0,)
        Selected band indices.
    cutoff_frequency : float
        Cutoff frequency in THz.

    Returns
    -------
    cv : ndarray of double, shape (num_temp, num_gp, num_band0)
        Mode heat capacities (diagonal).
    cv_mat : ndarray of double, shape (num_temp, num_gp, num_band0, num_band)
        Heat capacity matrix.

    """
    thz_to_ev = get_physical_units().THzToEv
    num_gp = len(grid_points)
    num_band = frequencies.shape[1]
    num_temp = len(temperatures)

    cv = compute_bulk_mode_cv(
        frequencies, grid_points, temperatures, band_indices, cutoff_frequency
    )

    cutoff_ev = cutoff_frequency * thz_to_ev
    cv_mat = np.zeros(
        (num_temp, num_gp, len(band_indices), num_band), dtype="double", order="C"
    )
    for i_gp, gp in enumerate(grid_points):
        freqs_ev = frequencies[gp] * thz_to_ev
        condition = freqs_ev > cutoff_ev
        condition_2d = np.logical_and.outer(condition, condition)
        # cv at this GP for fallback on diagonal
        cv_at_gp = compute_bulk_mode_cv(
            frequencies,
            np.array([gp], dtype="int64"),
            temperatures,
            np.arange(num_band, dtype="int64"),
            cutoff_frequency,
        )[:, 0, :]  # (num_temp, num_band)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cvm = mode_cv_matrix(
                temperatures, freqs_ev
            )  # (num_temp, num_band, num_band)
            cvm = np.where(
                condition_2d[None, :, :],
                np.where(np.isfinite(cvm), cvm, cv_at_gp[:, None, :]),
                0.0,
            )
        cv_mat[:, i_gp, :, :] = cvm[:, band_indices, :]

    return cv, cv_mat


class ModeHeatCapacityProvider:
    """Compute scalar mode heat capacities for all grid points at once.

    This provider implements the ``HeatCapacityProvider`` protocol.  It
    computes the mode heat capacity Cv (per mode, per unit cell) at the
    requested temperatures using the standard Einstein/harmonic-oscillator
    formula via ``compute_bulk_mode_cv``.

    The returned ``HeatCapacityResult`` contains ``heat_capacities``
    with shape ``(num_temp, num_gp, num_band0)``.

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  Phonon solver must have been run
        (``phonon_all_done == True``) before calling ``compute``.

    """

    produces_heat_capacity_matrix: bool = False

    def __init__(self, pp: Interaction, temperatures: NDArray[np.double]):
        """Init method."""
        self._pp = pp
        self._temperatures = temperatures

    def compute(
        self,
        grid_points: NDArray[np.int64],
    ) -> HeatCapacityResult:
        """Compute mode heat capacities for all grid points.

        Parameters
        ----------
        grid_points : ndarray of int64, shape (num_gp,)
            BZ grid point indices.

        Returns
        -------
        HeatCapacityResult
            ``heat_capacities`` (num_temp, num_gp, num_band0) is set.

        """
        frequencies = self._pp.get_phonons()[0]
        cv = compute_bulk_mode_cv(
            frequencies,
            grid_points,
            self._temperatures,
            self._pp.band_indices,
            self._pp.cutoff_frequency,
        )
        return HeatCapacityResult(heat_capacities=cv)


class HeatCapacityMatrixProvider:
    """Compute heat capacity matrix for all grid points (Kubo formula).

    This provider implements the ``HeatCapacityProvider`` protocol for the
    Green-Kubo formula.  It computes the off-diagonal heat capacity matrix
    ``C_{qjj'}`` using ``mode_cv_matrix`` from
    ``phono3py.phonon.heat_capacity_matrix``.

    The returned ``HeatCapacityResult`` contains:
    - ``heat_capacities`` (num_temp, num_gp, num_band0): diagonal (standard)
      mode heat capacities.
    - ``heat_capacity_matrix`` (num_temp, num_gp, num_band0, num_band): full
      heat capacity matrix for selected bands (rows) vs all bands (columns).

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  Phonon solver must have been run before calling
        ``compute``.

    """

    produces_heat_capacity_matrix: bool = True

    def __init__(self, pp: Interaction, temperatures: NDArray[np.double]):
        """Init method."""
        self._pp = pp
        self._temperatures = temperatures

    def compute(
        self,
        grid_points: NDArray[np.int64],
    ) -> HeatCapacityResult:
        """Compute heat capacity matrix for all grid points.

        Parameters
        ----------
        grid_points : ndarray of int64, shape (num_gp,)
            BZ grid point indices.

        Returns
        -------
        HeatCapacityResult
            ``heat_capacities`` (num_temp, num_gp, num_band0) and
            ``heat_capacity_matrix`` (num_temp, num_gp, num_band0, num_band)
            are set.

        """
        frequencies = self._pp.get_phonons()[0]
        cv, cv_mat = compute_bulk_cv_matrix(
            frequencies,
            grid_points,
            self._temperatures,
            self._pp.band_indices,
            self._pp.cutoff_frequency,
        )
        return HeatCapacityResult(
            heat_capacities=cv,
            heat_capacity_matrix=cv_mat,
        )
