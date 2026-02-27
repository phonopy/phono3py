"""Dispatch helpers for conductivity type selection.

This module centralizes the mapping from user-facing conductivity type strings
to implementation classes. The goal is to keep selection logic in one place and
reduce repeated conditionals in initialization entry points.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias, TypedDict, cast

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

RTAWriterGridData: TypeAlias = tuple[Any | None, Any | None, Any | None, Any | None]
RTAWriterKappaData: TypeAlias = tuple[
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
]
LBTEWriterKappaData: TypeAlias = tuple[
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
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
    "RTAWriterGridPayload",
    "RTAWriterKappaPayload",
    "LBTEWriterKappaPayload",
    # Public matrix/class selectors.
    "get_conductivity_dispatch_matrix",
    "get_conductivity_class_matrix",
    "get_conductivity_class",
    "get_rta_conductivity_class",
    "get_lbte_conductivity_class",
    "get_rta_progress_mode",
    # Public writer helpers.
    "get_rta_writer_grid_data",
    "get_rta_writer_grid_payload",
    "get_rta_writer_kappa_data",
    "get_rta_writer_kappa_payload",
    "get_lbte_writer_kappa_data",
    "get_lbte_writer_kappa_payload",
]


class RTAWriterGridPayload(TypedDict):
    """Per-grid payload used by `ConductivityRTAWriter.write_gamma`."""

    group_velocities_i: Any | None
    gv_by_gv_i: Any | None
    velocity_operator_i: Any | None
    mode_heat_capacities: Any | None


class RTAWriterKappaPayload(TypedDict):
    """Kappa payload used by `ConductivityRTAWriter.write_kappa`."""

    kappa: Any | None
    mode_kappa: Any | None
    group_velocities: Any | None
    gv_by_gv: Any | None
    kappa_TOT_RTA: Any | None
    kappa_P_RTA: Any | None
    kappa_C: Any | None
    mode_kappa_P_RTA: Any | None
    mode_kappa_C: Any | None
    mode_heat_capacities: Any | None


class LBTEWriterKappaPayload(TypedDict):
    """Kappa payload used by `ConductivityLBTEWriter.write_kappa`."""

    kappa: Any | None
    mode_kappa: Any | None
    kappa_RTA: Any | None
    mode_kappa_RTA: Any | None
    group_velocities: Any | None
    gv_by_gv: Any | None
    kappa_P_exact: Any | None
    kappa_P_RTA: Any | None
    kappa_C: Any | None
    mode_kappa_P_exact: Any | None
    mode_kappa_P_RTA: Any | None
    mode_kappa_C: Any | None


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

_RTA_GRID_ATTR_KEYS: tuple[str, ...] = (
    "group_velocities",
    "gv_by_gv",
    "mode_heat_capacities",
)

_RTA_GRID_DATA_KEYS: tuple[str, ...] = (
    "group_velocities_i",
    "gv_by_gv_i",
    "velocity_operator_i",
    "mode_heat_capacities",
)

_RTA_KAPPA_KEYS: tuple[str, ...] = (
    "kappa",
    "mode_kappa",
    "group_velocities",
    "gv_by_gv",
    "kappa_TOT_RTA",
    "kappa_P_RTA",
    "kappa_C",
    "mode_kappa_P_RTA",
    "mode_kappa_C",
    "mode_heat_capacities",
)

_LBTE_KAPPA_KEYS: tuple[str, ...] = (
    "kappa",
    "mode_kappa",
    "kappa_RTA",
    "mode_kappa_RTA",
    "group_velocities",
    "gv_by_gv",
    "kappa_P_exact",
    "kappa_P_RTA",
    "kappa_C",
    "mode_kappa_P_exact",
    "mode_kappa_P_RTA",
    "mode_kappa_C",
)


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


def _build_conductivity_class_registry(
    approximation: ApproximationType,
) -> dict[ConductivityType, RTAConductivityClass | LBTEConductivityClass]:
    """Build conductivity-class lookup map for one approximation axis."""
    return {
        ctype: entry.conductivity_class
        for ctype, entry in _get_dispatch_entries(approximation).items()
    }


_RTA_CLASS_REGISTRY: dict[ConductivityType, RTAConductivityClass] = {
    ctype: cast(RTAConductivityClass, conductivity_class)
    for ctype, conductivity_class in _build_conductivity_class_registry("rta").items()
}

_LBTE_CLASS_REGISTRY: dict[ConductivityType, LBTEConductivityClass] = {
    ctype: cast(LBTEConductivityClass, conductivity_class)
    for ctype, conductivity_class in _build_conductivity_class_registry("lbte").items()
}


def _build_dispatch_class_registry(
    approximation: ApproximationType,
) -> dict[type[object], DispatchEntry]:
    """Build exact-class to dispatch-entry lookup map."""
    class_registry: dict[type[object], DispatchEntry] = {}
    for entry in _DISPATCH_REGISTRY[approximation].values():
        conductivity_class = cast(type[object], entry.conductivity_class)
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
    return cast(
        ConductivityDispatchMatrix,
        {
            approximation: {ctype: asdict(entry) for ctype, entry in entries.items()}
            for approximation, entries in (
                ("rta", _get_dispatch_entries("rta")),
                ("lbte", _get_dispatch_entries("lbte")),
            )
        },
    )


def _build_conductivity_class_matrix() -> ConductivityClassMatrix:
    """Build class matrix from dispatch registry entries."""
    return cast(
        ConductivityClassMatrix,
        {
            approximation: {
                ctype: entry.conductivity_class for ctype, entry in entries.items()
            }
            for approximation, entries in (
                ("rta", _get_dispatch_entries("rta")),
                ("lbte", _get_dispatch_entries("lbte")),
            )
        },
    )


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
) -> DispatchEntry | None:
    """Resolve dispatch entry by exact class, then by isinstance fallback."""
    class_registry = _DISPATCH_CLASS_REGISTRY[approximation]
    entry = class_registry.get(type(conductivity))
    if entry is not None:
        return entry

    return _find_dispatch_entry_by_isinstance(approximation, conductivity)


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


# Private helpers: payload extraction and tuple shaping.
def _get_wigner_velocity_operator_i(br: Any, i: int) -> Any | None:
    """Return velocity-operator slice at grid index, if available."""
    velocity_operator = getattr(br, "velocity_operator", None)
    if velocity_operator is None:
        conductivity_components = getattr(br, "_conductivity_components", None)
        velocity_operator = getattr(conductivity_components, "velocity_operator", None)
    return None if velocity_operator is None else cast(Any, velocity_operator)[i]


def _has_all_attrs(obj: Any, *names: str) -> bool:
    """Return True when object has all requested attributes."""
    return all(hasattr(obj, name) for name in names)


def _get_attr_or_none(obj: Any, name: str) -> Any | None:
    """Return attribute value or None when attribute is missing."""
    if not hasattr(obj, name):
        return None
    return getattr(obj, name)


def _get_payload_attr(
    obj: Any,
    entry: DispatchEntry | None,
    attr_name: str,
) -> Any | None:
    """Return payload attribute when permitted by dispatch capability."""
    if entry is not None and attr_name not in entry.writer_attrs:
        return None
    return _get_attr_or_none(obj, attr_name)


def _get_payload_attrs(
    obj: Any,
    entry: DispatchEntry | None,
    attr_names: tuple[str, ...],
) -> tuple[Any | None, ...]:
    """Return tuple of payload attributes for requested names."""
    return tuple(_get_payload_attr(obj, entry, attr_name) for attr_name in attr_names)


def _should_pick_wigner_velocity_operator(
    br: Any,
    dispatch_entry: DispatchEntry | None,
) -> bool:
    """Return True when wigner velocity-operator payload should be read."""
    velocity_operator = _get_payload_attr(br, dispatch_entry, "velocity_operator")
    return velocity_operator is not None or _has_all_attrs(
        br, "_conductivity_components"
    )


def _payload_values(
    payload: Mapping[str, Any | None],
    keys: tuple[str, ...],
) -> tuple[Any | None, ...]:
    """Return payload values as a tuple in the requested key order."""
    return tuple(payload[key] for key in keys)


# Public API: writer payload and compatibility wrappers.
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
    br: ConductivityRTABase,
    i: int,
) -> RTAWriterGridData:
    """Return optional per-grid-point arrays used by RTA writer."""
    payload = get_rta_writer_grid_payload(br, i)
    return cast(RTAWriterGridData, _payload_values(payload, _RTA_GRID_DATA_KEYS))


def get_rta_writer_grid_payload(
    br: ConductivityRTABase,
    i: int,
) -> RTAWriterGridPayload:
    """Return named per-grid-point payload used by RTA writer."""
    dispatch_entry = _resolve_dispatch_entry("rta", br)

    group_velocities_i = None
    gv_by_gv_i = None
    group_velocities, gv_by_gv, mode_heat_capacities = _get_payload_attrs(
        br, dispatch_entry, _RTA_GRID_ATTR_KEYS
    )
    if group_velocities is not None and gv_by_gv is not None:
        group_velocities_i = cast(Any, group_velocities)[i]
        gv_by_gv_i = cast(Any, gv_by_gv)[i]

    velocity_operator_i = None
    if _should_pick_wigner_velocity_operator(br, dispatch_entry):
        velocity_operator_i = _get_wigner_velocity_operator_i(br, i)

    return {
        "group_velocities_i": group_velocities_i,
        "gv_by_gv_i": gv_by_gv_i,
        "velocity_operator_i": velocity_operator_i,
        "mode_heat_capacities": mode_heat_capacities,
    }


def get_rta_writer_kappa_data(
    br: ConductivityRTABase,
) -> RTAWriterKappaData:
    """Return optional conductivity arrays used by RTA kappa writer."""
    payload = get_rta_writer_kappa_payload(br)
    return cast(RTAWriterKappaData, _payload_values(payload, _RTA_KAPPA_KEYS))


def get_rta_writer_kappa_payload(
    br: ConductivityRTABase,
) -> RTAWriterKappaPayload:
    """Return named conductivity payload used by RTA kappa writer."""
    dispatch_entry = _resolve_dispatch_entry("rta", br)

    (
        kappa,
        mode_kappa,
        gv,
        gv_by_gv,
        kappa_TOT_RTA,
        kappa_P_RTA,
        kappa_C,
        mode_kappa_P_RTA,
        mode_kappa_C,
        mode_cv,
    ) = _get_payload_attrs(
        br,
        dispatch_entry,
        _RTA_KAPPA_KEYS,
    )

    return {
        "kappa": kappa,
        "mode_kappa": mode_kappa,
        "group_velocities": gv,
        "gv_by_gv": gv_by_gv,
        "kappa_TOT_RTA": kappa_TOT_RTA,
        "kappa_P_RTA": kappa_P_RTA,
        "kappa_C": kappa_C,
        "mode_kappa_P_RTA": mode_kappa_P_RTA,
        "mode_kappa_C": mode_kappa_C,
        "mode_heat_capacities": mode_cv,
    }


def get_lbte_writer_kappa_data(lbte: ConductivityLBTEBase) -> LBTEWriterKappaData:
    """Return optional conductivity arrays used by LBTE kappa writer."""
    payload = get_lbte_writer_kappa_payload(lbte)
    return cast(LBTEWriterKappaData, _payload_values(payload, _LBTE_KAPPA_KEYS))


def get_lbte_writer_kappa_payload(
    lbte: ConductivityLBTEBase,
) -> LBTEWriterKappaPayload:
    """Return named conductivity payload used by LBTE kappa writer."""
    dispatch_entry = _resolve_dispatch_entry("lbte", lbte)

    (
        kappa,
        mode_kappa,
        kappa_RTA,
        mode_kappa_RTA,
        gv,
        gv_by_gv,
        kappa_P_exact,
        kappa_P_RTA,
        kappa_C,
        mode_kappa_P_exact,
        mode_kappa_P_RTA,
        mode_kappa_C,
    ) = _get_payload_attrs(
        lbte,
        dispatch_entry,
        _LBTE_KAPPA_KEYS,
    )

    return {
        "kappa": kappa,
        "mode_kappa": mode_kappa,
        "kappa_RTA": kappa_RTA,
        "mode_kappa_RTA": mode_kappa_RTA,
        "group_velocities": gv,
        "gv_by_gv": gv_by_gv,
        "kappa_P_exact": kappa_P_exact,
        "kappa_P_RTA": kappa_P_RTA,
        "kappa_C": kappa_C,
        "mode_kappa_P_exact": mode_kappa_P_exact,
        "mode_kappa_P_RTA": mode_kappa_P_RTA,
        "mode_kappa_C": mode_kappa_C,
    }
