"""Unit tests for ConductivityRTAWriter."""

from types import SimpleNamespace

import numpy as np

from phono3py.conductivity.rta_output import ConductivityRTAWriter


def test_write_gamma_detail_all_bands(monkeypatch):
    """`write_gamma_detail` writes one data block per sigma in all-band mode."""
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
    """`write_gamma_detail` writes one data block per (sigma, band) in band mode."""
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
