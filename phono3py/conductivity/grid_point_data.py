"""Per-grid-point data containers.

This module defines solver result types (VelocityResult,
HeatCapacityResult, ScatteringResult) and GridPointAggregates used in
the conductivity calculation.

Protocol interfaces (VelocitySolver, HeatCapacitySolver,
ScatteringSolver) are defined in ``phono3py.conductivity.protocols``
and re-exported here for backward compatibility.
"""

from __future__ import annotations

import warnings
from dataclasses import InitVar, dataclass, field
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Deprecated extra
# ---------------------------------------------------------------------------

# REMOVE-AT-V5
_T = TypeVar("_T")


# REMOVE-AT-V5
def _warn_extra_deprecated(name: str, stacklevel: int) -> None:
    warnings.warn(
        f"{name}.extra is deprecated and will be removed in v5.0. "
        "Keep plugin-specific data in the plugin instead.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


# REMOVE-AT-V5
def _deprecate_extra(cls: type[_T]) -> type[_T]:
    """Add the deprecated extra property to a dataclass.

    The dataclass has the init-only variable extra and the private field
    _extra.  extra cannot be a field and a property at the same time, so the
    property is added after the dataclass is created.

    """
    name = cls.__name__

    def _get(self: Any) -> dict[str, Any]:
        _warn_extra_deprecated(name, 2)
        return self._extra

    def _set(self: Any, value: dict[str, Any]) -> None:
        _warn_extra_deprecated(name, 2)
        self._extra = value

    target: Any = cls
    target.extra = property(_get, _set, doc="Deprecated and will be removed in v5.0.")
    return cls


# ---------------------------------------------------------------------------
# Per-grid-point data containers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Solver result types
# ---------------------------------------------------------------------------


# REMOVE-AT-V5
@_deprecate_extra
@dataclass
class VelocityResult:
    """Result from a velocity solver at grid points.

    Parameters
    ----------
    group_velocities : ndarray of double, shape (num_gp, num_band0, 3)
        Diagonal (standard) group velocities.
    gv_by_gv : ndarray of double, shape (num_gp, num_band0, 6)
        Symmetrised outer product v x v in Voigt notation.
    vm_by_vm : ndarray of cdouble, shape (num_gp, num_band0, num_band, 6), optional
        Off-diagonal velocity operator/matrix outer product.
        Only set by off-diagonal velocity solvers (e.g. SMM19, NJC23).
    extra : dict
        Plugin-specific data.  Deprecated and will be removed in v5.0.

    """

    group_velocities: NDArray[np.double]
    gv_by_gv: NDArray[np.double] | None = None
    vm_by_vm: NDArray[np.cdouble] | None = None
    # REMOVE-AT-V5
    extra: InitVar[dict[str, Any] | None] = None
    _extra: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self, extra: dict[str, Any] | None) -> None:
        """Keep extra given at construction with a deprecation warning."""
        if extra is not None:
            _warn_extra_deprecated(type(self).__name__, 3)
            self._extra = extra


# REMOVE-AT-V5
@_deprecate_extra
@dataclass
class HeatCapacityResult:
    """Result from a heat capacity solver (bulk computation).

    Parameters
    ----------
    heat_capacities : ndarray of double, shape (num_temp, num_gp, num_band0)
        Mode heat capacities (scalar Cv per mode) for all grid points.
    heat_capacity_matrix : ndarray of double, optional
        Shape (num_temp, num_gp, num_band0, num_band).
        Only set by HeatCapacityMatrixSolver (Kubo).
    extra : dict
        Plugin-specific data.  Deprecated and will be removed in v5.0.

    """

    heat_capacities: NDArray[np.double]
    heat_capacity_matrix: NDArray[np.double] | None = None
    # REMOVE-AT-V5
    extra: InitVar[dict[str, Any] | None] = None
    _extra: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self, extra: dict[str, Any] | None) -> None:
        """Keep extra given at construction with a deprecation warning."""
        if extra is not None:
            _warn_extra_deprecated(type(self).__name__, 3)
            self._extra = extra


# REMOVE-AT-V5
@_deprecate_extra
@dataclass
class ScatteringResult:
    """Result from a scattering solver at a single grid point.

    Parameters
    ----------
    gamma : ndarray of double, shape (num_sigma, num_temp, num_band0)
        Ph-ph linewidth (imaginary part of self-energy).
    averaged_pp_interaction : ndarray of double, shape (num_band0,), optional
        Averaged ph-ph interaction strength.
    extra : dict
        Plugin-specific data.  Deprecated and will be removed in v5.0.

    """

    gamma: NDArray[np.double]
    averaged_pp_interaction: NDArray[np.double] | None = None
    # REMOVE-AT-V5
    extra: InitVar[dict[str, Any] | None] = None
    _extra: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self, extra: dict[str, Any] | None) -> None:
        """Keep extra given at construction with a deprecation warning."""
        if extra is not None:
            _warn_extra_deprecated(type(self).__name__, 3)
            self._extra = extra


# ---------------------------------------------------------------------------
# Aggregated per-grid-point data (Calculator -> KappaSolver.finalize())
# ---------------------------------------------------------------------------


# REMOVE-AT-V5
@_deprecate_extra
@dataclass
class GridPointAggregates:
    """Aggregated per-grid-point data passed from Calculator to KappaSolver.

    Built by the Calculator after the grid-point loop and passed to
    ``kappa_solver.finalize()``.  Replaces the former untyped
    ``grid_point_data: dict[str, Any]``.

    Always present
    ~~~~~~~~~~~~~~
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
        Off-diagonal velocity operator outer product (SMM19/NJC23).
    heat_capacity_matrix : (num_temp, num_gp, num_band0, num_band), real
        Heat-capacity matrix (Kubo).
    extra : dict
        Plugin-specific data.  Deprecated and will be removed in v5.0.

    """

    group_velocities: NDArray[np.double]
    mode_heat_capacities: NDArray[np.double]
    gv_by_gv: NDArray[np.double] | None = None
    gamma: NDArray[np.double] | None = None
    gamma_isotope: NDArray[np.double] | None = None
    gamma_boundary: NDArray[np.double] | None = None
    gamma_elph: NDArray[np.double] | None = None
    vm_by_vm: NDArray[np.cdouble] | None = None
    heat_capacity_matrix: NDArray[np.double] | None = None
    # REMOVE-AT-V5
    extra: InitVar[dict[str, Any] | None] = None
    _extra: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self, extra: dict[str, Any] | None) -> None:
        """Keep extra given at construction with a deprecation warning."""
        if extra is not None:
            _warn_extra_deprecated(type(self).__name__, 3)
            self._extra = extra


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


# ---------------------------------------------------------------------------
# Re-export Protocol interfaces for backward compatibility.
# Canonical definitions live in phono3py.conductivity.protocols.
# ---------------------------------------------------------------------------

from phono3py.conductivity.protocols import (  # noqa: E402
    HeatCapacitySolver,
    ScatteringSolver,
    VelocitySolver,
)

__all__ = [
    "GridPointAggregates",
    "HeatCapacityResult",
    "ScatteringResult",
    "VelocityResult",
    "compute_effective_gamma",
    "VelocitySolver",
    "HeatCapacitySolver",
    "ScatteringSolver",
]
