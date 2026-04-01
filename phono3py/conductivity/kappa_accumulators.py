"""Kappa accumulator building blocks for conductivity calculations.

An accumulator owns the kappa formula and the BZ-summation arrays.
``ConductivityCalculator`` calls ``accumulate()`` once per grid point and
``finalize()`` at the end; it delegates output properties to the accumulator
via ``__getattr__``.

Adding a new transport variant requires only a new accumulator class registered
via ``register_calculator()`` — ``ConductivityCalculator`` and the core factory
do not need to be modified.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointResult
from phono3py.conductivity.kappa_formulas import BTEKappaFormula


@runtime_checkable
class KappaAccumulator(Protocol):
    """Protocol for BZ-summation of kappa contributions.

    Implementations
    ---------------
    StandardKappaAccumulator
        Standard BTE diagonal formula; exposes ``kappa``, ``mode_kappa``.

    """

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate output arrays before the grid-point loop starts.

        Called by ``ConductivityCalculator._allocate_values()``.

        """
        ...

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and store the kappa contribution for grid point ``i_gp``.

        The formula is called inside this method; ``result`` must already
        have ``velocity_product``, ``heat_capacities``, ``gamma`` (and
        optional ``gamma_isotope``, ``gamma_boundary``, ``gamma_elph``)
        populated.

        """
        ...

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise accumulated kappa by the total number of sampling points."""
        ...

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa tensor, shape (num_sigma, num_temp, 6)."""
        ...

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode-resolved kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).
        """
        ...


class StandardKappaAccumulator:
    """Kappa accumulator for the standard BTE diagonal formula.

    Parameters
    ----------
    formula : BTEKappaFormula
        Formula instance used to compute mode kappa at each grid point.

    """

    def __init__(self, formula: BTEKappaFormula) -> None:
        """Init method."""
        self._formula = formula
        self._kappa: NDArray[np.double]
        self._mode_kappa: NDArray[np.double]

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate kappa and mode_kappa arrays."""
        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and accumulate mode kappa at grid point ``i_gp``."""
        mode_kappa = self._formula.compute(result)
        self._mode_kappa[:, :, i_gp, :, :] = mode_kappa
        # gv_by_gv already encodes the k-star order (sum over rotations
        # divided by site multiplicity). Do NOT multiply by grid_weight;
        # the normalization is done once via num_sampling_grid_points.
        self._kappa += np.sum(mode_kappa, axis=2)

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise by the total number of sampling grid points."""
        if num_sampling_grid_points > 0:
            self._kappa /= num_sampling_grid_points

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._mode_kappa
