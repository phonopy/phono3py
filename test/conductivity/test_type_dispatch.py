"""Unit tests for conductivity type dispatch mappings."""

from types import SimpleNamespace

import numpy as np

from phono3py.conductivity.direct_solution import ConductivityLBTE
from phono3py.conductivity.kubo_base import (
    ConductivityKuboComponents,
)
from phono3py.conductivity.kubo_rta import ConductivityKuboRTA
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.type_dispatch import (
    get_conductivity_class,
    get_conductivity_class_matrix,
    get_conductivity_dispatch_matrix,
    get_lbte_conductivity_class,
    get_lbte_writer_kappa_data,
    get_lbte_writer_kappa_data_map,
    get_rta_conductivity_class,
    get_rta_progress_mode,
    get_rta_writer_grid_data,
    get_rta_writer_grid_data_map,
    get_rta_writer_kappa_data,
    get_rta_writer_kappa_data_map,
)
from phono3py.conductivity.wigner_direct_solution import ConductivityWignerLBTE
from phono3py.conductivity.wigner_rta import ConductivityWignerRTA


def test_kubo_rta_inheritance_is_single_base_regression():
    """Kubo RTA no longer uses multiple inheritance."""
    assert issubclass(ConductivityKuboRTA, ConductivityRTABase)
    assert ConductivityKuboRTA.__bases__ == (ConductivityRTABase,)


def test_kubo_components_class_exists_regression():
    """Kubo components class remains available for composition."""
    assert isinstance(ConductivityKuboComponents, type)


def test_get_conductivity_class_rta_mapping():
    """RTA axis mapping is explicit and stable."""
    assert get_conductivity_class("rta", None) is ConductivityRTA
    assert get_conductivity_class("rta", "kubo") is ConductivityKuboRTA
    assert get_conductivity_class("rta", "wigner") is ConductivityWignerRTA


def test_get_conductivity_class_lbte_mapping():
    """LBTE axis mapping is explicit and stable."""
    assert get_conductivity_class("lbte", None) is ConductivityLBTE
    assert get_conductivity_class("lbte", "kubo") is ConductivityLBTE
    assert get_conductivity_class("lbte", "wigner") is ConductivityWignerLBTE


def test_legacy_dispatch_wrappers_match_general_dispatch():
    """Compatibility wrappers keep returning the same classes."""
    for conductivity_type in (None, "kubo", "wigner"):
        assert get_rta_conductivity_class(conductivity_type) is get_conductivity_class(
            "rta", conductivity_type
        )
        assert get_lbte_conductivity_class(conductivity_type) is get_conductivity_class(
            "lbte", conductivity_type
        )


def test_get_conductivity_class_matrix_content():
    """Class matrix exposes all six user-facing combinations."""
    matrix = get_conductivity_class_matrix()
    assert matrix["rta"][None] is ConductivityRTA
    assert matrix["rta"]["kubo"] is ConductivityKuboRTA
    assert matrix["rta"]["wigner"] is ConductivityWignerRTA
    assert matrix["lbte"][None] is ConductivityLBTE
    assert matrix["lbte"]["kubo"] is ConductivityLBTE
    assert matrix["lbte"]["wigner"] is ConductivityWignerLBTE


def test_get_conductivity_dispatch_matrix_metadata():
    """Dispatch matrix contains class and metadata for each mode combination."""
    matrix = get_conductivity_dispatch_matrix()

    assert matrix["rta"][None]["conductivity_class"] is ConductivityRTA
    assert matrix["rta"]["kubo"]["conductivity_class"] is ConductivityKuboRTA
    assert matrix["rta"]["wigner"]["conductivity_class"] is ConductivityWignerRTA
    assert matrix["lbte"][None]["conductivity_class"] is ConductivityLBTE
    assert matrix["lbte"]["kubo"]["conductivity_class"] is ConductivityLBTE
    assert matrix["lbte"]["wigner"]["conductivity_class"] is ConductivityWignerLBTE

    assert matrix["lbte"]["kubo"]["has_dedicated_class"] is False
    assert matrix["rta"]["kubo"]["has_dedicated_class"] is True


def test_get_rta_progress_mode():
    """RTA progress mode is derived from dispatch metadata/class mapping."""
    assert get_rta_progress_mode(None) == "default"
    assert get_rta_progress_mode("kubo") == "default"
    assert get_rta_progress_mode("wigner") == "wigner"


def test_get_rta_writer_grid_data_from_attr_capabilities():
    """Grid data is extracted by available attributes, not class identity."""
    velocity_operator = np.array([[[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]]])
    br_like = SimpleNamespace(
        group_velocities=np.array([[[1.0, 2.0, 3.0]]]),
        gv_by_gv=np.array([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]]),
        mode_heat_capacities=np.array([[[9.0]]]),
        velocity_operator=velocity_operator,
    )

    gv_i, gv_by_gv_i, velocity_op_i, mode_cv = get_rta_writer_grid_data(br_like, 0)
    np.testing.assert_allclose(gv_i, br_like.group_velocities[0])
    np.testing.assert_allclose(gv_by_gv_i, br_like.gv_by_gv[0])
    np.testing.assert_allclose(velocity_op_i, velocity_operator[0])
    np.testing.assert_allclose(mode_cv, br_like.mode_heat_capacities)


def test_get_rta_writer_grid_data_map_from_attr_capabilities():
    """Named RTA grid data keeps stable key-based access."""
    velocity_operator = np.array([[[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]]])
    br_like = SimpleNamespace(
        group_velocities=np.array([[[1.0, 2.0, 3.0]]]),
        gv_by_gv=np.array([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]]),
        mode_heat_capacities=np.array([[[9.0]]]),
        velocity_operator=velocity_operator,
    )

    data_map = get_rta_writer_grid_data_map(br_like, 0)
    np.testing.assert_allclose(
        data_map["group_velocities_i"], br_like.group_velocities[0]
    )
    np.testing.assert_allclose(data_map["gv_by_gv_i"], br_like.gv_by_gv[0])
    np.testing.assert_allclose(data_map["velocity_operator_i"], velocity_operator[0])
    np.testing.assert_allclose(
        data_map["mode_heat_capacities"], br_like.mode_heat_capacities
    )


def test_get_rta_writer_kappa_data_from_attr_capabilities():
    """Kappa data fields are picked directly from available attributes."""
    br_like = SimpleNamespace(
        kappa=np.array([[[1.0] * 6]]),
        mode_kappa=np.array([[[[[1.0] * 6]]]]),
        group_velocities=np.array([[[1.0, 0.0, 0.0]]]),
        gv_by_gv=np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        kappa_TOT_RTA=np.array([[[2.0] * 6]]),
        kappa_P_RTA=np.array([[[1.5] * 6]]),
        kappa_C=np.array([[[0.5] * 6]]),
        mode_kappa_P_RTA=np.array([[[[[1.5] * 6]]]]),
        mode_kappa_C=np.array([[[[[0.5] * 6]]]]),
        mode_heat_capacities=np.array([[[10.0]]]),
    )

    data = get_rta_writer_kappa_data(br_like)
    assert data[0] is br_like.kappa
    assert data[1] is br_like.mode_kappa
    assert data[2] is br_like.group_velocities
    assert data[3] is br_like.gv_by_gv
    assert data[4] is br_like.kappa_TOT_RTA
    assert data[5] is br_like.kappa_P_RTA
    assert data[6] is br_like.kappa_C
    assert data[7] is br_like.mode_kappa_P_RTA
    assert data[8] is br_like.mode_kappa_C
    assert data[9] is br_like.mode_heat_capacities


def test_get_rta_writer_kappa_data_map_from_attr_capabilities():
    """Named RTA kappa data keeps stable key-based access."""
    br_like = SimpleNamespace(
        kappa=np.array([[[1.0] * 6]]),
        mode_kappa=np.array([[[[[1.0] * 6]]]]),
        group_velocities=np.array([[[1.0, 0.0, 0.0]]]),
        gv_by_gv=np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        kappa_TOT_RTA=np.array([[[2.0] * 6]]),
        kappa_P_RTA=np.array([[[1.5] * 6]]),
        kappa_C=np.array([[[0.5] * 6]]),
        mode_kappa_P_RTA=np.array([[[[[1.5] * 6]]]]),
        mode_kappa_C=np.array([[[[[0.5] * 6]]]]),
        mode_heat_capacities=np.array([[[10.0]]]),
    )

    data_map = get_rta_writer_kappa_data_map(br_like)
    assert data_map["kappa"] is br_like.kappa
    assert data_map["mode_kappa"] is br_like.mode_kappa
    assert data_map["group_velocities"] is br_like.group_velocities
    assert data_map["gv_by_gv"] is br_like.gv_by_gv
    assert data_map["kappa_TOT_RTA"] is br_like.kappa_TOT_RTA
    assert data_map["kappa_P_RTA"] is br_like.kappa_P_RTA
    assert data_map["kappa_C"] is br_like.kappa_C
    assert data_map["mode_kappa_P_RTA"] is br_like.mode_kappa_P_RTA
    assert data_map["mode_kappa_C"] is br_like.mode_kappa_C
    assert data_map["mode_heat_capacities"] is br_like.mode_heat_capacities


def test_get_lbte_writer_kappa_data_from_attr_capabilities():
    """LBTE data fields are picked directly from available attributes."""
    lbte_like = SimpleNamespace(
        kappa=np.array([[[1.0] * 6]]),
        mode_kappa=np.array([[[[[1.0] * 6]]]]),
        kappa_RTA=np.array([[[0.8] * 6]]),
        mode_kappa_RTA=np.array([[[[[0.8] * 6]]]]),
        group_velocities=np.array([[[1.0, 0.0, 0.0]]]),
        gv_by_gv=np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        kappa_P_exact=np.array([[[0.9] * 6]]),
        kappa_P_RTA=np.array([[[0.7] * 6]]),
        kappa_C=np.array([[[0.2] * 6]]),
        mode_kappa_P_exact=np.array([[[[[0.9] * 6]]]]),
        mode_kappa_P_RTA=np.array([[[[[0.7] * 6]]]]),
        mode_kappa_C=np.array([[[[[0.2] * 6]]]]),
    )

    data = get_lbte_writer_kappa_data(lbte_like)
    assert data[0] is lbte_like.kappa
    assert data[1] is lbte_like.mode_kappa
    assert data[2] is lbte_like.kappa_RTA
    assert data[3] is lbte_like.mode_kappa_RTA
    assert data[4] is lbte_like.group_velocities
    assert data[5] is lbte_like.gv_by_gv
    assert data[6] is lbte_like.kappa_P_exact
    assert data[7] is lbte_like.kappa_P_RTA
    assert data[8] is lbte_like.kappa_C
    assert data[9] is lbte_like.mode_kappa_P_exact
    assert data[10] is lbte_like.mode_kappa_P_RTA
    assert data[11] is lbte_like.mode_kappa_C


def test_get_lbte_writer_kappa_data_map_from_attr_capabilities():
    """Named LBTE kappa data keeps stable key-based access."""
    lbte_like = SimpleNamespace(
        kappa=np.array([[[1.0] * 6]]),
        mode_kappa=np.array([[[[[1.0] * 6]]]]),
        kappa_RTA=np.array([[[0.8] * 6]]),
        mode_kappa_RTA=np.array([[[[[0.8] * 6]]]]),
        group_velocities=np.array([[[1.0, 0.0, 0.0]]]),
        gv_by_gv=np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        kappa_P_exact=np.array([[[0.9] * 6]]),
        kappa_P_RTA=np.array([[[0.7] * 6]]),
        kappa_C=np.array([[[0.2] * 6]]),
        mode_kappa_P_exact=np.array([[[[[0.9] * 6]]]]),
        mode_kappa_P_RTA=np.array([[[[[0.7] * 6]]]]),
        mode_kappa_C=np.array([[[[[0.2] * 6]]]]),
    )

    data_map = get_lbte_writer_kappa_data_map(lbte_like)
    assert data_map["kappa"] is lbte_like.kappa
    assert data_map["mode_kappa"] is lbte_like.mode_kappa
    assert data_map["kappa_RTA"] is lbte_like.kappa_RTA
    assert data_map["mode_kappa_RTA"] is lbte_like.mode_kappa_RTA
    assert data_map["group_velocities"] is lbte_like.group_velocities
    assert data_map["gv_by_gv"] is lbte_like.gv_by_gv
    assert data_map["kappa_P_exact"] is lbte_like.kappa_P_exact
    assert data_map["kappa_P_RTA"] is lbte_like.kappa_P_RTA
    assert data_map["kappa_C"] is lbte_like.kappa_C
    assert data_map["mode_kappa_P_exact"] is lbte_like.mode_kappa_P_exact
    assert data_map["mode_kappa_P_RTA"] is lbte_like.mode_kappa_P_RTA
    assert data_map["mode_kappa_C"] is lbte_like.mode_kappa_C
