"""Public Protocol interfaces for the conductivity building blocks.

This module defines the stable interfaces that plugin authors implement to
extend phono3py with new conductivity calculation methods.

A plugin registers a factory function via
``phono3py.conductivity.register_variant()`` and implements one or more of
the Protocols defined here.

Protocols
---------
VelocitySolver
    Computes velocity-related quantities at a single BZ grid point.
HeatCapacitySolver
    Computes heat capacity for all grid points at once (bulk).
ScatteringSolver
    Computes phonon linewidths at a single BZ grid point.

Data containers
---------------
Solver result types (VelocityResult, HeatCapacityResult,
ScatteringResult) are defined in ``phono3py.conductivity.grid_point_data``.

"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
    HeatCapacityResult,
    ScatteringResult,
    VelocityResult,
)

__all__ = [
    "VelocitySolver",
    "HeatCapacitySolver",
    "ScatteringSolver",
    "VelocityResult",
    "HeatCapacityResult",
    "ScatteringResult",
]


class VelocitySolver(Protocol):
    """Protocol for computing velocity-related quantities at a grid point.

    Built-in implementations
    ------------------------
    GroupVelocitySolver
        Standard group velocity and symmetrised v x v product (BTE, LBTE).
    VelocityMatrixSolver
        Off-diagonal velocity matrix and its outer product (SMM19, NJC23).

    Class attributes
    ----------------
    produces_gv_by_gv : bool
        True when compute() sets VelocityResult.gv_by_gv.
    produces_vm_by_vm : bool
        True when compute() sets VelocityResult.vm_by_vm.

    These flags tell the calculator which arrays to pre-allocate
    before the grid-point loop.

    """

    produces_gv_by_gv: bool
    produces_vm_by_vm: bool

    def compute(self, grid_point: int) -> VelocityResult:
        """Compute velocity quantities at a grid point."""
        ...


class HeatCapacitySolver(Protocol):
    """Protocol for computing heat capacity for all grid points at once.

    Built-in implementations
    ------------------------
    ModeHeatCapacitySolver
        Scalar mode heat capacity Cv (all variants).
    HeatCapacityMatrixSolver
        Heat-capacity matrix Cv_mat (Kubo).

    Class attributes
    ----------------
    produces_heat_capacity_matrix : bool
        True when compute() sets HeatCapacityResult.heat_capacity_matrix.

    These flags tell the calculator which arrays to pre-allocate
    before the grid-point loop.

    """

    produces_heat_capacity_matrix: bool

    def compute(
        self,
        grid_points: NDArray[np.int64],
    ) -> HeatCapacityResult:
        """Compute heat-capacity quantities for all grid points at once."""
        ...


class ScatteringSolver(Protocol):
    """Protocol for computing phonon linewidths at a grid point.

    Built-in implementations
    ------------------------
    RTAScatteringSolver
        Relaxation-time approximation (diagonal collision matrix).

    Notes
    -----
    Isotope and boundary scattering are separate diagonal-only contributions
    handled by IsotopeScatteringSolver and compute_bulk_boundary_scattering.

    """

    def compute(
        self,
        grid_point: int,
    ) -> ScatteringResult:
        """Compute phonon linewidths at a grid point.

        The returned ScatteringResult must have at minimum
        ``gamma`` of shape (num_sigma, num_temp, num_band0) set.

        """
        ...
