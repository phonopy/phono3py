"""Dispatch helpers for conductivity type selection.

This module centralizes the mapping from user-facing conductivity type strings
to implementation classes. The goal is to keep selection logic in one place and
reduce repeated conditionals in initialization entry points.

"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

from phono3py.conductivity.direct_solution import ConductivityLBTE
from phono3py.conductivity.kubo_rta import ConductivityKuboRTA
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.wigner_direct_solution import ConductivityWignerLBTE
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA

ConductivityType: TypeAlias = Literal["wigner", "kubo"] | None


def get_rta_conductivity_class(
    conductivity_type: ConductivityType,
) -> type[ConductivityRTA] | type[ConductivityKuboRTA] | type[ConductivityWignerRTA]:
    """Return RTA conductivity class selected by conductivity_type."""
    class_registry = {
        None: ConductivityRTA,
        "kubo": ConductivityKuboRTA,
        "wigner": ConductivityWignerRTA,
    }
    return class_registry[conductivity_type]


def get_lbte_conductivity_class(
    conductivity_type: ConductivityType,
) -> type[ConductivityLBTE] | type[ConductivityWignerLBTE]:
    """Return direct-solution conductivity class selected by conductivity_type.

    "kubo" falls back to the normal LBTE implementation because a dedicated
    Kubo direct-solution class is not implemented.
    """
    class_registry = {
        None: ConductivityLBTE,
        "kubo": ConductivityLBTE,
        "wigner": ConductivityWignerLBTE,
    }
    return class_registry[conductivity_type]


def get_rta_writer_grid_data(
    br: ConductivityRTA | ConductivityKuboRTA | ConductivityWignerRTA,
    i: int,
):
    """Return optional per-grid-point arrays used by RTA writer."""
    if isinstance(br, ConductivityRTA):
        group_velocities_i = br.group_velocities[i]
        gv_by_gv_i = br.gv_by_gv[i]
    else:
        group_velocities_i = None
        gv_by_gv_i = None

    if isinstance(br, ConductivityWignerRTA):
        velocity_operator_i = cast(Any, br).velocity_operator[i]
    else:
        velocity_operator_i = None

    if isinstance(br, (ConductivityRTA, ConductivityWignerRTA)):
        mode_heat_capacities = br.mode_heat_capacities
    else:
        mode_heat_capacities = None

    return group_velocities_i, gv_by_gv_i, velocity_operator_i, mode_heat_capacities


def get_rta_writer_kappa_data(
    br: ConductivityRTA | ConductivityKuboRTA | ConductivityWignerRTA,
):
    """Return optional conductivity arrays used by RTA kappa writer."""
    if isinstance(br, ConductivityRTA):
        kappa = br.kappa
        mode_kappa = br.mode_kappa
        gv = br.group_velocities
        gv_by_gv = br.gv_by_gv
    else:
        kappa = None
        mode_kappa = None
        gv = None
        gv_by_gv = None

    if isinstance(br, ConductivityWignerRTA):
        kappa_TOT_RTA = br.kappa_TOT_RTA
        kappa_P_RTA = br.kappa_P_RTA
        kappa_C = br.kappa_C
        mode_kappa_P_RTA = br.mode_kappa_P_RTA
        mode_kappa_C = br.mode_kappa_C
    else:
        kappa_TOT_RTA = None
        kappa_P_RTA = None
        kappa_C = None
        mode_kappa_P_RTA = None
        mode_kappa_C = None

    if isinstance(br, (ConductivityRTA, ConductivityWignerRTA)):
        mode_cv = br.mode_heat_capacities
    else:
        mode_cv = None

    return (
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
    )


def get_lbte_writer_kappa_data(lbte: ConductivityLBTE | ConductivityWignerLBTE):
    """Return optional conductivity arrays used by LBTE kappa writer."""
    if isinstance(lbte, ConductivityLBTE):
        kappa = lbte.kappa
        mode_kappa = lbte.mode_kappa
        kappa_RTA = lbte.kappa_RTA
        mode_kappa_RTA = lbte.mode_kappa_RTA
        gv = lbte.group_velocities
        gv_by_gv = lbte.gv_by_gv
    else:
        kappa = None
        mode_kappa = None
        kappa_RTA = None
        mode_kappa_RTA = None
        gv = None
        gv_by_gv = None

    if isinstance(lbte, ConductivityWignerLBTE):
        kappa_P_exact = lbte.kappa_P_exact
        kappa_P_RTA = lbte.kappa_P_RTA
        kappa_C = lbte.kappa_C
        mode_kappa_P_exact = lbte.mode_kappa_P_exact
        mode_kappa_P_RTA = lbte.mode_kappa_P_RTA
        mode_kappa_C = lbte.mode_kappa_C
    else:
        kappa_P_exact = None
        kappa_P_RTA = None
        kappa_C = None
        mode_kappa_P_exact = None
        mode_kappa_P_RTA = None
        mode_kappa_C = None

    return (
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
    )
