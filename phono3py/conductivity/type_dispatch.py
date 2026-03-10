"""Dispatch helpers for conductivity type selection.

This module centralizes the mapping from user-facing conductivity type strings
to implementation classes. The goal is to keep selection logic in one place and
reduce repeated conditionals in initialization entry points.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.direct_solution import ConductivityLBTE
from phono3py.conductivity.direct_solution_base import ConductivityLBTEBase
from phono3py.conductivity.kubo_rta import ConductivityKuboRTA
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.wigner_direct_solution import ConductivityWignerLBTE
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA

ConductivityType: TypeAlias = Literal["wigner", "kubo"] | None
ApproximationType: TypeAlias = Literal["rta", "lbte"]
RTAProgressMode: TypeAlias = Literal["default", "wigner"]

RTAConductivityClass: TypeAlias = type[ConductivityRTABase]
LBTEConductivityClass: TypeAlias = type[ConductivityLBTEBase]

DispatchEntryDict: TypeAlias = dict[str, Any]
ConductivityDispatchMatrix: TypeAlias = dict[
    ApproximationType, dict[ConductivityType, DispatchEntryDict]
]
ConductivityClassMatrix: TypeAlias = dict[
    ApproximationType,
    dict[ConductivityType, RTAConductivityClass | LBTEConductivityClass],
]

__all__ = [
    # Public types.
    "ApproximationType",
    "ConductivityType",
    "RTAProgressMode",
    "RTAConductivityClass",
    "LBTEConductivityClass",
    "ConductivityDispatchMatrix",
    "ConductivityClassMatrix",
    "RTAWriterGridData",
    "RTAWriterKappaData",
    "LBTEWriterKappaData",
    # Public matrix/class selectors.
    "get_conductivity_dispatch_matrix",
    "get_conductivity_class_matrix",
    "get_conductivity_class",
    "get_rta_conductivity_class",
    "get_lbte_conductivity_class",
    "get_rta_progress_mode",
    # Public writer helpers.
    "get_rta_writer_grid_data",
    "get_rta_writer_kappa_data",
    "get_lbte_writer_kappa_data",
]


class RTAWriterGridData(TypedDict):
    """Per-grid named data used by `ConductivityRTAWriter.write_gamma`."""

    group_velocities_i: NDArray[np.double] | None
    gv_by_gv_i: NDArray[np.double] | None
    velocity_operator_i: NDArray[np.double] | None
    mode_heat_capacities: NDArray[np.double] | None


class RTAWriterKappaData(TypedDict):
    """Kappa named data used by `ConductivityRTAWriter.write_kappa`."""

    kappa: NDArray[np.double] | None
    mode_kappa: NDArray[np.double] | None
    group_velocities: NDArray[np.double] | None
    gv_by_gv: NDArray[np.double] | None
    kappa_TOT_RTA: NDArray[np.double] | None
    kappa_P_RTA: NDArray[np.double] | None
    kappa_C: NDArray[np.double] | None
    mode_kappa_P_RTA: NDArray[np.double] | None
    mode_kappa_C: NDArray[np.double] | None
    mode_heat_capacities: NDArray[np.double] | None


class LBTEWriterKappaData(TypedDict):
    """Kappa named data used by `ConductivityLBTEWriter.write_kappa`."""

    kappa: NDArray[np.double] | None
    mode_kappa: NDArray[np.double] | None
    kappa_RTA: NDArray[np.double] | None
    mode_kappa_RTA: NDArray[np.double] | None
    group_velocities: NDArray[np.double] | None
    gv_by_gv: NDArray[np.double] | None
    kappa_P_exact: NDArray[np.double] | None
    kappa_P_RTA: NDArray[np.double] | None
    kappa_C: NDArray[np.double] | None
    mode_kappa_P_exact: NDArray[np.double] | None
    mode_kappa_P_RTA: NDArray[np.double] | None
    mode_kappa_C: NDArray[np.double] | None


@dataclass(frozen=True)
class DispatchEntry:
    """Metadata-rich entry for one approximation/type dispatch slot."""

    conductivity_class: RTAConductivityClass | LBTEConductivityClass
    approximation: ApproximationType
    conductivity_type: ConductivityType
    has_dedicated_class: bool
    writer_attrs: tuple[str, ...]
    progress_mode: RTAProgressMode = "default"


_RTA_BASE_WRITER_ATTRS: tuple[str, ...] = (
    "group_velocities",
    "gv_by_gv",
    "mode_heat_capacities",
)

_RTA_KAPPA_WRITER_ATTRS: tuple[str, ...] = (
    "kappa",
    "mode_kappa",
)

_RTA_WIGNER_WRITER_ATTRS: tuple[str, ...] = (
    "velocity_operator",
    "kappa_TOT_RTA",
    "kappa_P_RTA",
    "kappa_C",
    "mode_kappa_P_RTA",
    "mode_kappa_C",
)

_LBTE_BASE_WRITER_ATTRS: tuple[str, ...] = (
    "group_velocities",
    "gv_by_gv",
    "kappa",
    "mode_kappa",
    "kappa_RTA",
    "mode_kappa_RTA",
)

_LBTE_WIGNER_WRITER_ATTRS: tuple[str, ...] = (
    "kappa_P_exact",
    "kappa_P_RTA",
    "kappa_C",
    "mode_kappa_P_exact",
    "mode_kappa_P_RTA",
    "mode_kappa_C",
)

_APPROXIMATION_TYPES: tuple[ApproximationType, ApproximationType] = ("rta", "lbte")


_DISPATCH_REGISTRY: dict[ApproximationType, dict[ConductivityType, DispatchEntry]] = {
    "rta": {
        None: DispatchEntry(
            conductivity_class=ConductivityRTA,
            approximation="rta",
            conductivity_type=None,
            has_dedicated_class=True,
            writer_attrs=_RTA_BASE_WRITER_ATTRS + _RTA_KAPPA_WRITER_ATTRS,
            progress_mode="default",
        ),
        "kubo": DispatchEntry(
            conductivity_class=ConductivityKuboRTA,
            approximation="rta",
            conductivity_type="kubo",
            has_dedicated_class=True,
            writer_attrs=(),
            progress_mode="default",
        ),
        "wigner": DispatchEntry(
            conductivity_class=ConductivityWignerRTA,
            approximation="rta",
            conductivity_type="wigner",
            has_dedicated_class=True,
            writer_attrs=_RTA_BASE_WRITER_ATTRS + _RTA_WIGNER_WRITER_ATTRS,
            progress_mode="wigner",
        ),
    },
    "lbte": {
        None: DispatchEntry(
            conductivity_class=ConductivityLBTE,
            approximation="lbte",
            conductivity_type=None,
            has_dedicated_class=True,
            writer_attrs=_LBTE_BASE_WRITER_ATTRS,
        ),
        "kubo": DispatchEntry(
            conductivity_class=ConductivityLBTE,
            approximation="lbte",
            conductivity_type="kubo",
            has_dedicated_class=False,
            writer_attrs=_LBTE_BASE_WRITER_ATTRS,
        ),
        "wigner": DispatchEntry(
            conductivity_class=ConductivityWignerLBTE,
            approximation="lbte",
            conductivity_type="wigner",
            has_dedicated_class=True,
            writer_attrs=_LBTE_WIGNER_WRITER_ATTRS,
        ),
    },
}


# Private helpers: dispatch registry access and matrix/class builders.
def _get_dispatch_entries(
    approximation: ApproximationType,
) -> dict[ConductivityType, DispatchEntry]:
    """Return dispatch entries for one approximation axis."""
    return _DISPATCH_REGISTRY[approximation]


def _build_rta_class_registry() -> dict[ConductivityType, RTAConductivityClass]:
    """Build typed RTA class lookup map."""
    registry: dict[ConductivityType, RTAConductivityClass] = {}
    for ctype, entry in _get_dispatch_entries("rta").items():
        conductivity_class = entry.conductivity_class
        if not issubclass(conductivity_class, ConductivityRTABase):
            raise TypeError(f"Invalid RTA conductivity class: {conductivity_class!r}")
        registry[ctype] = conductivity_class
    return registry


def _build_lbte_class_registry() -> dict[ConductivityType, LBTEConductivityClass]:
    """Build typed LBTE class lookup map."""
    registry: dict[ConductivityType, LBTEConductivityClass] = {}
    for ctype, entry in _get_dispatch_entries("lbte").items():
        conductivity_class = entry.conductivity_class
        if not issubclass(conductivity_class, ConductivityLBTEBase):
            raise TypeError(f"Invalid LBTE conductivity class: {conductivity_class!r}")
        registry[ctype] = conductivity_class
    return registry


_RTA_CLASS_REGISTRY: dict[ConductivityType, RTAConductivityClass] = (
    _build_rta_class_registry()
)

_LBTE_CLASS_REGISTRY: dict[ConductivityType, LBTEConductivityClass] = (
    _build_lbte_class_registry()
)


def _build_dispatch_class_registry(
    approximation: ApproximationType,
) -> dict[type[object], DispatchEntry]:
    """Build exact-class to dispatch-entry lookup map."""
    class_registry: dict[type[object], DispatchEntry] = {}
    for entry in _DISPATCH_REGISTRY[approximation].values():
        conductivity_class = entry.conductivity_class
        class_registry.setdefault(conductivity_class, entry)
    return class_registry


_DISPATCH_CLASS_REGISTRY: dict[ApproximationType, dict[type[object], DispatchEntry]] = {
    "rta": _build_dispatch_class_registry("rta"),
    "lbte": _build_dispatch_class_registry("lbte"),
}


def _get_dispatch_entry(
    approximation: ApproximationType,
    conductivity_type: ConductivityType,
) -> DispatchEntry:
    """Return one dispatch entry for approximation and conductivity type."""
    return _get_dispatch_entries(approximation)[conductivity_type]


def _build_dispatch_metadata_matrix() -> ConductivityDispatchMatrix:
    """Build metadata matrix from dispatch registry entries."""
    matrix: ConductivityDispatchMatrix = {}
    for approximation in _APPROXIMATION_TYPES:
        entries = _get_dispatch_entries(approximation)
        matrix[approximation] = {
            ctype: asdict(entry) for ctype, entry in entries.items()
        }
    return matrix


def _build_conductivity_class_matrix() -> ConductivityClassMatrix:
    """Build class matrix from dispatch registry entries."""
    matrix: ConductivityClassMatrix = {}
    for approximation in _APPROXIMATION_TYPES:
        entries = _get_dispatch_entries(approximation)
        matrix[approximation] = {
            ctype: entry.conductivity_class for ctype, entry in entries.items()
        }
    return matrix


# Public API: class and progress selectors.
def get_conductivity_class(
    approximation: ApproximationType,
    conductivity_type: ConductivityType,
) -> RTAConductivityClass | LBTEConductivityClass:
    """Return conductivity class for approximation and conductivity_type.

    This function explicitly represents the two-axis selection:
    approximation (RTA/LBTE) x conductivity type (None/wigner/kubo).

    """
    return _get_dispatch_entry(approximation, conductivity_type).conductivity_class


# Public API: dispatch/class matrix views.
def get_conductivity_dispatch_matrix() -> ConductivityDispatchMatrix:
    """Return metadata-rich dispatch registry for readability/testing."""
    return _build_dispatch_metadata_matrix()


def get_conductivity_class_matrix() -> ConductivityClassMatrix:
    """Return a copy of class mapping matrix for readability/testing."""
    return _build_conductivity_class_matrix()


def get_rta_progress_mode(conductivity_type: ConductivityType) -> RTAProgressMode:
    """Return progress display mode for selected RTA conductivity type."""
    return _get_dispatch_entry("rta", conductivity_type).progress_mode


# Private helpers: dispatch entry resolution.
def _resolve_dispatch_entry(
    approximation: ApproximationType,
    conductivity: ConductivityRTABase | ConductivityLBTEBase,
) -> DispatchEntry:
    """Resolve dispatch entry by exact class, then by isinstance fallback."""
    class_registry = _DISPATCH_CLASS_REGISTRY[approximation]
    entry = class_registry.get(type(conductivity))
    if entry is not None:
        return entry

    fallback_entry = _find_dispatch_entry_by_isinstance(approximation, conductivity)
    if fallback_entry is not None:
        return fallback_entry

    raise TypeError(
        "Unsupported conductivity instance for dispatch: "
        f"approximation={approximation!r}, class={type(conductivity).__name__!r}"
    )


def _find_dispatch_entry_by_isinstance(
    approximation: ApproximationType,
    conductivity: ConductivityRTABase | ConductivityLBTEBase,
) -> DispatchEntry | None:
    """Fallback dispatch entry resolution for subclass/test-double cases."""
    for entry in _get_dispatch_entries(approximation).values():
        conductivity_class = entry.conductivity_class
        if isinstance(conductivity, conductivity_class):
            return entry
    return None


def _build_fallback_dispatch_entry(
    approximation: ApproximationType,
    conductivity: object,
) -> DispatchEntry:
    """Build fallback dispatch entry for attribute-based test doubles.

    Detects conductivity type by checking for wigner-specific writer attributes.
    """
    if approximation == "rta":
        conductivity_type: ConductivityType = (
            "wigner"
            if any(
                hasattr(conductivity, attr_name)
                for attr_name in _RTA_WIGNER_WRITER_ATTRS
            )
            else None
        )
        writer_attrs = (
            _RTA_BASE_WRITER_ATTRS
            + _RTA_KAPPA_WRITER_ATTRS
            + (_RTA_WIGNER_WRITER_ATTRS if conductivity_type == "wigner" else ())
        )
        return DispatchEntry(
            conductivity_class=ConductivityRTABase,
            approximation="rta",
            conductivity_type=conductivity_type,
            has_dedicated_class=False,
            writer_attrs=writer_attrs,
            progress_mode="wigner" if conductivity_type == "wigner" else "default",
        )

    # LBTE fallback
    conductivity_type = (
        "wigner"
        if any(
            hasattr(conductivity, attr_name) for attr_name in _LBTE_WIGNER_WRITER_ATTRS
        )
        else None
    )
    writer_attrs = _LBTE_BASE_WRITER_ATTRS + (
        _LBTE_WIGNER_WRITER_ATTRS if conductivity_type == "wigner" else ()
    )
    return DispatchEntry(
        conductivity_class=ConductivityLBTEBase,
        approximation="lbte",
        conductivity_type=conductivity_type,
        has_dedicated_class=False,
        writer_attrs=writer_attrs,
    )


def _resolve_writer_dispatch_entry(
    approximation: ApproximationType,
    conductivity: object,
) -> DispatchEntry:
    """Resolve dispatch entry for writer helpers.

    Writer helpers also support attribute-based test doubles that are not
    conductivity-class instances.

    """
    try:
        return _resolve_dispatch_entry(
            approximation,
            conductivity,  # type: ignore[arg-type]
        )
    except TypeError:
        return _build_fallback_dispatch_entry(approximation, conductivity)


# Private helpers: data extraction and tuple shaping.
def _get_wigner_velocity_operator_i(
    conductivity: object, i: int
) -> NDArray[np.double] | None:
    """Return velocity-operator slice at grid index, if available."""
    velocity_operator = _get_attr_or_none(conductivity, "velocity_operator")
    if velocity_operator is None:
        return None
    return velocity_operator[i]


def _get_attr_or_none(obj: Any, name: str) -> NDArray[np.double] | None:
    """Return ndarray attribute value or None when attribute is unavailable."""
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    return _ensure_writer_value(value, context=f"attribute {name!r}")


def _ensure_writer_value(value: object, context: str) -> NDArray[np.double] | None:
    """Validate and normalize writer value type."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(
        f"Expected numpy.ndarray or None for {context}, got {type(value).__name__}."
    )


def _get_data_attr(
    obj: Any,
    entry: DispatchEntry,
    attr_name: str,
) -> NDArray[np.double] | None:
    """Return data attribute when permitted by dispatch capability."""
    if attr_name not in entry.writer_attrs:
        return None
    return _get_attr_or_none(obj, attr_name)


def _get_data_attr_map(
    obj: Any,
    entry: DispatchEntry,
    attr_names: tuple[str, ...],
) -> dict[str, NDArray[np.double] | None]:
    """Return dict of data attributes for requested names."""
    return {name: _get_data_attr(obj, entry, name) for name in attr_names}


# Public API: writer data helpers.
def get_rta_conductivity_class(
    conductivity_type: ConductivityType,
) -> RTAConductivityClass:
    """Return RTA conductivity class selected by conductivity_type."""
    return _RTA_CLASS_REGISTRY[conductivity_type]


def get_lbte_conductivity_class(
    conductivity_type: ConductivityType,
) -> LBTEConductivityClass:
    """Return direct-solution conductivity class selected by conductivity_type.

    "kubo" falls back to the normal LBTE implementation because a dedicated
    Kubo direct-solution class is not implemented.
    """
    return _LBTE_CLASS_REGISTRY[conductivity_type]


def get_rta_writer_grid_data(
    conductivity: ConductivityRTABase,
    i: int,
) -> RTAWriterGridData:
    """Return named per-grid-point data used by RTA writer."""
    dispatch_entry = _resolve_writer_dispatch_entry("rta", conductivity)

    group_velocities_i = None
    gv_by_gv_i = None
    group_velocities = _get_data_attr(conductivity, dispatch_entry, "group_velocities")
    gv_by_gv = _get_data_attr(conductivity, dispatch_entry, "gv_by_gv")
    mode_heat_capacities = _get_data_attr(
        conductivity, dispatch_entry, "mode_heat_capacities"
    )

    if group_velocities is not None and gv_by_gv is not None:
        group_velocities_i = group_velocities[i]
        gv_by_gv_i = gv_by_gv[i]

    velocity_operator_i = None
    if "velocity_operator" in dispatch_entry.writer_attrs:
        velocity_operator_i = _get_wigner_velocity_operator_i(conductivity, i)

    return RTAWriterGridData(
        group_velocities_i=group_velocities_i,
        gv_by_gv_i=gv_by_gv_i,
        velocity_operator_i=velocity_operator_i,
        mode_heat_capacities=mode_heat_capacities,
    )


def get_rta_writer_kappa_data(
    conductivity: ConductivityRTABase,
) -> RTAWriterKappaData:
    """Return named conductivity data used by RTA kappa writer."""
    dispatch_entry = _resolve_writer_dispatch_entry("rta", conductivity)
    return RTAWriterKappaData(
        **_get_data_attr_map(
            conductivity,
            dispatch_entry,
            tuple(RTAWriterKappaData.__annotations__),
        )
    )


def get_lbte_writer_kappa_data(
    conductivity: ConductivityLBTEBase,
) -> LBTEWriterKappaData:
    """Return named conductivity data used by LBTE kappa writer."""
    dispatch_entry = _resolve_writer_dispatch_entry("lbte", conductivity)
    return LBTEWriterKappaData(
        **_get_data_attr_map(
            conductivity,
            dispatch_entry,
            tuple(LBTEWriterKappaData.__annotations__),
        )
    )
