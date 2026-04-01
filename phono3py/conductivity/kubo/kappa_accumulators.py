"""Kappa accumulator for the Green-Kubo method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointResult
from phono3py.conductivity.kubo.kappa_formulas import KuboKappaFormula


class KuboKappaAccumulator:
    """Kappa accumulator for the Green-Kubo formula.

    Accumulates the full band-pair kappa matrix ``mode_kappa_mat`` returned by
    ``KuboKappaFormula``.  The ``kappa`` property sums over all band pairs.

    Parameters
    ----------
    formula : KuboKappaFormula
        Formula instance used to compute mode kappa at each grid point.

    """

    def __init__(self, formula: KuboKappaFormula) -> None:
        """Init method."""
        self._formula = formula
        self._kappa: NDArray[np.double]
        # Full band-pair matrix; allocated lazily on first accumulate().
        self._mode_kappa_mat: NDArray[np.double] | None = None

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate kappa array; mode_kappa_mat is allocated lazily."""
        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._num_gp = num_gp

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and accumulate Kubo mode-kappa at grid point ``i_gp``."""
        mkm = self._formula.compute(result)
        # mkm: (num_sigma, num_temp, num_band0, num_band, 6)
        if self._mode_kappa_mat is None:
            self._mode_kappa_mat = np.zeros(
                (self._num_gp,) + mkm.shape, dtype="double", order="C"
            )
        self._mode_kappa_mat[i_gp] = mkm
        # Sum over both band indices to accumulate kappa.
        self._kappa += mkm.sum(axis=(2, 3))

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise accumulated kappa by the total number of sampling points."""
        if num_sampling_grid_points > 0:
            self._kappa /= num_sampling_grid_points

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa summed over j_band.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        if self._mode_kappa_mat is None:
            raise RuntimeError("mode_kappa_mat has not been computed yet.")
        # Sum over j_band axis (axis=-2 of the per-gp array, axis=4 in stored array).
        return self._mode_kappa_mat.sum(axis=4)

    @property
    def mode_kappa_mat(self) -> NDArray[np.double] | None:
        """Return full band-pair kappa matrix.

        Shape: (num_gp, num_sigma, num_temp, num_band0, num_band, 6).

        """
        return self._mode_kappa_mat
