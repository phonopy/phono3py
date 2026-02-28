"""Unit tests for ConductivityRTAWriter."""

from types import SimpleNamespace

import numpy as np

from phono3py.conductivity import rta_output
from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.rta_output import ConductivityRTAWriter


def test_write_kappa_calls_hdf5_writer_per_sigma(monkeypatch):
    """`write_kappa` forwards per-sigma payload to hdf5 writer."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    kappa = np.arange(12, dtype="double").reshape(2, 1, 6)
    mode_kappa = np.arange(12, dtype="double").reshape(2, 1, 1, 1, 6)
    gv = np.ones((1, 1, 3), dtype="double")
    gv_by_gv = np.ones((1, 1, 6), dtype="double")
    mode_cv = np.ones((1, 1, 1), dtype="double")

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.get_rta_writer_kappa_payload",
        lambda _br: {
            "kappa": kappa,
            "mode_kappa": mode_kappa,
            "group_velocities": gv,
            "gv_by_gv": gv_by_gv,
            "kappa_TOT_RTA": None,
            "kappa_P_RTA": None,
            "kappa_C": None,
            "mode_kappa_P_RTA": None,
            "mode_kappa_C": None,
            "mode_heat_capacities": mode_cv,
        },
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )

    br = SimpleNamespace(
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[None, 0.1],
        sigma_cutoff_width=None,
        gamma=np.zeros((2, 1, 1, 1), dtype="double"),
        gamma_isotope=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        frequencies=np.array([[5.0]], dtype="double"),
        averaged_pp_interaction=None,
        qpoints=np.array([[0.0, 0.0, 0.0]], dtype="double"),
        grid_points=np.array([0], dtype="int64"),
        grid_weights=np.array([1], dtype="int64"),
        boundary_mfp=None,
        get_gamma_N_U=lambda: (None, None),
    )

    ConductivityRTAWriter.write_kappa(
        br,
        volume=2.0,
        compression="gzip",
        filename="dummy.hdf5",
        log_level=1,
    )

    assert len(calls) == 2
    for i, (_args, kwargs) in enumerate(calls):
        np.testing.assert_allclose(kwargs["kappa"], kappa[i])
        assert kwargs["sigma"] == br.sigmas[i]
        assert kwargs["compression"] == "gzip"
        assert kwargs["filename"] == "dummy.hdf5"
        assert kwargs["verbose"] == 1
        np.testing.assert_allclose(
            kwargs["kappa_unit_conversion"], get_unit_to_WmK() / 2.0
        )


def test_write_gamma_all_bands(monkeypatch):
    """`write_gamma` writes one payload per sigma in all-band mode."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: True
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.get_rta_writer_grid_payload",
        lambda _br, _i: {
            "group_velocities_i": np.array([[1.0, 2.0, 3.0]], dtype="double"),
            "gv_by_gv_i": np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype="double"),
            "velocity_operator_i": np.array([[[1.0, 0.0, 0.0]]], dtype="complex128"),
            "mode_heat_capacities": np.array([[[9.0]]], dtype="double"),
        },
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )

    interaction = SimpleNamespace(
        primitive=SimpleNamespace(volume=4.0),
        band_indices=np.array([0], dtype="int64"),
        bz_grid=SimpleNamespace(),
        get_phonons=lambda: (np.array([[4.0]], dtype="double"), None, None),
    )
    br = SimpleNamespace(
        grid_points=np.array([0], dtype="int64"),
        averaged_pp_interaction=np.array([[2.0]], dtype="double"),
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        temperatures=np.array([300.0], dtype="double"),
        gamma=np.array([[[1.0]], [[2.0]]], dtype="double"),
        gamma_isotope=np.array([[[0.1]], [[0.2]]], dtype="double"),
        sigmas=[None, 0.1],
        sigma_cutoff_width=None,
        get_gamma_N_U=lambda: (None, None),
    )

    ConductivityRTAWriter.write_gamma(br, interaction, i=0)

    assert len(calls) == 2
    for i, (_args, kwargs) in enumerate(calls):
        assert "band_index" not in kwargs
        assert kwargs["grid_point"] == 0
        assert kwargs["sigma"] == br.sigmas[i]
        np.testing.assert_allclose(kwargs["heat_capacity"], np.array([[9.0]]))


def test_write_gamma_band_resolved(monkeypatch):
    """`write_gamma` writes one payload per (sigma, band) when not all bands exist."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: False
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.get_rta_writer_grid_payload",
        lambda _br, _i: {
            "group_velocities_i": np.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="double"
            ),
            "gv_by_gv_i": np.array(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype="double",
            ),
            "velocity_operator_i": np.array(
                [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype="complex128"
            ),
            "mode_heat_capacities": np.array([[[3.0, 4.0]]], dtype="double"),
        },
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )

    interaction = SimpleNamespace(
        primitive=SimpleNamespace(volume=2.0),
        band_indices=np.array([2, 4], dtype="int64"),
        bz_grid=SimpleNamespace(),
        get_phonons=lambda: (
            np.array([[0.0, 0.0, 12.0, 0.0, 20.0]], dtype="double"),
            None,
            None,
        ),
    )
    br = SimpleNamespace(
        grid_points=np.array([0], dtype="int64"),
        averaged_pp_interaction=np.array([[5.0, 6.0]], dtype="double"),
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        temperatures=np.array([300.0], dtype="double"),
        gamma=np.array([[[[1.0, 2.0]]]], dtype="double"),
        gamma_isotope=np.array([[[0.1, 0.2]]], dtype="double"),
        sigmas=[0.05],
        sigma_cutoff_width=3.0,
        get_gamma_N_U=lambda: (None, None),
    )

    ConductivityRTAWriter.write_gamma(br, interaction, i=0)

    assert len(calls) == 2
    assert {calls[0][1]["band_index"], calls[1][1]["band_index"]} == {2, 4}
    assert all(call[1]["sigma"] == 0.05 for call in calls)


def test_write_gamma_detail_all_bands(monkeypatch):
    """`write_gamma_detail` writes one payload per sigma in all-band mode."""
    calls = []

    def _fake_write_gamma_detail_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: True
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.get_all_triplets",
        lambda _gp, _bz_grid: np.array([[0, 1, 2]], dtype="int64"),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_gamma_detail_to_hdf5",
        _fake_write_gamma_detail_to_hdf5,
    )

    interaction = SimpleNamespace(
        bz_grid=SimpleNamespace(),
        get_triplets_at_q=lambda: (
            np.array([[0, 1, 2]], dtype="int64"),
            np.array([1], dtype="int64"),
            None,
            None,
        ),
    )
    gamma_detail = np.ones((2, 1, 1, 1, 1), dtype="double")
    br = SimpleNamespace(
        get_gamma_detail_at_q=lambda: gamma_detail,
        temperatures=np.array([300.0], dtype="double"),
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        grid_points=np.array([11], dtype="int64"),
        sigmas=[None, 0.1],
        sigma_cutoff_width=None,
    )

    ConductivityRTAWriter.write_gamma_detail(br, interaction, i=0)

    assert len(calls) == 2
    for i, (_args, kwargs) in enumerate(calls):
        assert kwargs["grid_point"] == 11
        assert kwargs["sigma"] == br.sigmas[i]
        assert "band_index" not in kwargs
        assert kwargs["gamma_detail"] is gamma_detail


def test_write_gamma_detail_band_resolved(monkeypatch):
    """`write_gamma_detail` writes one payload per (sigma, band) in band mode."""
    calls = []

    def _fake_write_gamma_detail_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: False
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.get_all_triplets",
        lambda _gp, _bz_grid: np.array([[0, 1, 2]], dtype="int64"),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_gamma_detail_to_hdf5",
        _fake_write_gamma_detail_to_hdf5,
    )

    interaction = SimpleNamespace(
        bz_grid=SimpleNamespace(),
        get_triplets_at_q=lambda: (
            np.array([[0, 1, 2]], dtype="int64"),
            np.array([1], dtype="int64"),
            None,
            None,
        ),
        band_indices=np.array([3, 7], dtype="int64"),
    )
    gamma_detail = np.arange(2 * 1 * 2 * 1 * 1, dtype="double").reshape(2, 1, 2, 1, 1)
    br = SimpleNamespace(
        get_gamma_detail_at_q=lambda: gamma_detail,
        temperatures=np.array([300.0], dtype="double"),
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        grid_points=np.array([9], dtype="int64"),
        sigmas=[0.2],
        sigma_cutoff_width=3.0,
    )

    ConductivityRTAWriter.write_gamma_detail(br, interaction, i=0)

    assert len(calls) == 2
    assert {calls[0][1]["band_index"], calls[1][1]["band_index"]} == {3, 7}
    assert all(call[1]["sigma"] == 0.2 for call in calls)


def test_show_rta_progress_dispatch_wigner(monkeypatch):
    """`show_rta_progress` dispatches to wigner handler for "wigner"."""
    called = []
    monkeypatch.setattr(
        rta_output,
        "_show_rta_progress_wigner",
        lambda _br, _log_level: called.append("wigner"),
    )
    monkeypatch.setattr(
        rta_output,
        "_RTA_PROGRESS_HANDLERS",
        {
            "default": rta_output._show_rta_progress_default,
            "wigner": rta_output._show_rta_progress_wigner,
        },
    )
    monkeypatch.setattr(
        rta_output,
        "get_rta_progress_mode",
        lambda _conductivity_type: "wigner",
    )

    rta_output.show_rta_progress(SimpleNamespace(), "wigner", 1)
    assert called == ["wigner"]


def test_show_rta_progress_dispatch_default(monkeypatch):
    """`show_rta_progress` dispatches to default handler for None and kubo."""
    called = []
    monkeypatch.setattr(
        rta_output,
        "_show_rta_progress_default",
        lambda _br, _log_level: called.append("default"),
    )
    monkeypatch.setattr(
        rta_output,
        "_RTA_PROGRESS_HANDLERS",
        {
            "default": rta_output._show_rta_progress_default,
            "wigner": rta_output._show_rta_progress_wigner,
        },
    )
    monkeypatch.setattr(
        rta_output,
        "get_rta_progress_mode",
        lambda _conductivity_type: "default",
    )

    rta_output.show_rta_progress(SimpleNamespace(), None, 1)
    rta_output.show_rta_progress(SimpleNamespace(), "kubo", 1)
    assert called == ["default", "default"]
