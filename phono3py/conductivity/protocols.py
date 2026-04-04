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

Data containers
---------------
GridPointInput and provider result types (VelocityResult,
HeatCapacityResult, ScatteringResult) are defined in
``phono3py.conductivity.grid_point_data``.

"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
    GridPointInput,
    HeatCapacityResult,
    ScatteringResult,
    VelocityResult,
)

__all__ = [
    "VelocityProvider",
    "HeatCapacityProvider",
    "ScatteringProvider",
    "GridPointInput",
    "VelocityResult",
    "HeatCapacityResult",
    "ScatteringResult",
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

    def compute(self, gp: GridPointInput) -> VelocityResult:
        """Compute velocity quantities at a grid point.

        The returned VelocityResult must have at minimum
        ``group_velocities``, ``gv_by_gv``, and
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
    ) -> HeatCapacityResult:
        """Compute heat-capacity quantities at a grid point.

        The returned HeatCapacityResult must have at minimum
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

    """

    def compute(
        self,
        gp: GridPointInput,
    ) -> ScatteringResult:
        """Compute phonon linewidths at a grid point.

        The returned ScatteringResult must have at minimum
        ``gamma`` of shape (num_sigma, num_temp, num_band0) set.

        """
        ...
