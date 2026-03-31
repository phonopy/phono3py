"""Public Protocol interfaces for the conductivity building blocks.

This module defines the stable interfaces that plugin authors implement to
extend phono3py with new conductivity calculation methods.

A plugin registers a factory function via
``phono3py.conductivity.register_calculator()`` and implements one or more of
the Protocols defined here.

Protocols
---------
VelocityProvider
    Computes velocity-related quantities at a single BZ grid point.
HeatCapacityProvider
    Computes heat capacity at a single BZ grid point.
ScatteringProvider
    Computes phonon linewidths at a single BZ grid point.
KappaFormula
    Converts per-grid-point results into a kappa tensor contribution.

Data containers
---------------
GridPointInput and GridPointResult are defined in
``phono3py.conductivity.grid_point_data`` and re-exported here for
convenience.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult

__all__ = [
    "VelocityProvider",
    "HeatCapacityProvider",
    "ScatteringProvider",
    "KappaFormula",
    "GridPointInput",
    "GridPointResult",
]


class VelocityProvider(Protocol):
    """Protocol for computing velocity-related quantities at a grid point.

    Built-in implementations
    ------------------------
    GroupVelocityProvider
        Standard group velocity and symmetrised v x v product (BTE, LBTE).
    VelocityOperatorProvider
        Full velocity operator and its outer product (Wigner).
    VelocityMatrixProvider
        Off-diagonal velocity matrix and its outer product (Kubo).
    """

    def compute(self, gp: GridPointInput) -> GridPointResult:
        """Compute velocity quantities and return a partially filled result.

        The returned GridPointResult must have at minimum
        ``group_velocities``, ``velocity_product``, and
        ``num_sampling_grid_points`` set.
        """
        ...


class HeatCapacityProvider(Protocol):
    """Protocol for computing heat capacity at a grid point.

    Built-in implementations
    ------------------------
    ModeHeatCapacityProvider
        Scalar mode heat capacity Cv (all variants).
    HeatCapacityMatrixProvider
        Heat-capacity matrix Cv_mat (Kubo).
    """

    def compute(
        self,
        gp: GridPointInput,
        temperatures: NDArray[np.double],
    ) -> GridPointResult:
        """Compute heat-capacity quantities and return a partially filled result.

        The returned GridPointResult must have at minimum
        ``heat_capacities`` set (and optionally ``heat_capacity_matrix``).
        """
        ...


class ScatteringProvider(Protocol):
    """Protocol for computing phonon linewidths at a grid point.

    Built-in implementations
    ------------------------
    RTAScatteringProvider
        Relaxation-time approximation (diagonal collision matrix).

    Notes
    -----
    Isotope and boundary scattering are separate diagonal-only contributions
    handled by IsotopeScatteringProvider and BoundaryScatteringProvider.
    They set ``gamma_isotope`` and ``gamma_boundary`` in GridPointResult.
    """

    def compute_gamma(
        self,
        gp: GridPointInput,
    ) -> GridPointResult:
        """Compute phonon linewidths and return a partially filled result.

        The returned GridPointResult must have at minimum
        ``gamma`` of shape (num_sigma, num_temp, num_band0) set.
        """
        ...


class KappaFormula(Protocol):
    """Protocol for computing the kappa contribution at a single grid point.

    The formula combines velocity, heat-capacity, and scattering data
    from a GridPointResult into a kappa tensor contribution.

    Built-in implementations
    ------------------------
    BTEKappaFormula
        Standard diagonal BTE formula: kappa = sum Cv * (v x v) * tau.
    WignerKappaFormula
        Wigner transport equation including off-diagonal coherence terms.
    KuboKappaFormula
        Green-Kubo formula using velocity matrix and heat-capacity matrix.
    """

    def compute(self, result: GridPointResult) -> NDArray[np.double]:
        """Return the kappa contribution for this grid point.

        Returns
        -------
        kappa : ndarray, shape (num_sigma, num_temp, num_band0, 6)
            Six independent components of the symmetric kappa tensor:
            xx, yy, zz, yz, xz, xy.
        """
        ...
