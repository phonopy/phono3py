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


def test_write_gamma_passes_extra_grid_point_output(monkeypatch):
    """`write_gamma` passes extra_datasets from get_extra_grid_point_output."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: True
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )

    # Shape: (num_gp=1, num_band0=3, nat3=6, 3)
    vel_op = np.ones((1, 3, 6, 3), dtype="complex128")
    extra_data = {"velocity_operator": vel_op}

    interaction = SimpleNamespace(
        primitive=SimpleNamespace(volume=1.0),
        get_phonons=lambda: (np.ones((20, 3), dtype="double"), None, None),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _: True
    )

    br = SimpleNamespace(
        grid_points=np.array([0], dtype="int64"),
        group_velocities=np.zeros((1, 3, 3), dtype="double"),
        gv_by_gv=np.zeros((1, 3, 6), dtype="double"),
        get_extra_grid_point_output=lambda: extra_data,
        mode_heat_capacities=np.zeros((2, 1, 3), dtype="double"),
        averaged_pp_interaction=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        temperatures=np.array([300.0], dtype="double"),
        gamma=np.zeros((1, 2, 1, 3), dtype="double"),
        gamma_isotope=None,
        sigmas=[None],
        sigma_cutoff_width=None,
        get_gamma_N_U=lambda: (None, None),
        frequencies=np.ones((1, 3), dtype="double"),
    )

    ConductivityRTAWriter.write_gamma(br, interaction, i=0)

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0]["extra_datasets"]["velocity_operator"], vel_op[0]
    )


def test_write_gamma_no_extra_grid_point_output(monkeypatch):
    """`write_gamma` passes None extra_datasets when no extra output."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: True
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )

    interaction = SimpleNamespace(
        primitive=SimpleNamespace(volume=1.0),
        get_phonons=lambda: (np.ones((20, 3), dtype="double"), None, None),
    )

    br = SimpleNamespace(
        grid_points=np.array([0], dtype="int64"),
        group_velocities=np.zeros((1, 3, 3), dtype="double"),
        gv_by_gv=np.zeros((1, 3, 6), dtype="double"),
        get_extra_grid_point_output=lambda: None,
        mode_heat_capacities=np.zeros((2, 1, 3), dtype="double"),
        averaged_pp_interaction=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        temperatures=np.array([300.0], dtype="double"),
        gamma=np.zeros((1, 2, 1, 3), dtype="double"),
        gamma_isotope=None,
        sigmas=[None],
        sigma_cutoff_width=None,
        get_gamma_N_U=lambda: (None, None),
        frequencies=np.ones((1, 3), dtype="double"),
    )

    ConductivityRTAWriter.write_gamma(br, interaction, i=0)

    assert len(calls) == 1
    assert calls[0]["extra_datasets"] is None


def test_write_gamma_band_resolved_slices_extra_data(monkeypatch):
    """`write_gamma` slices extra_datasets per band in band-resolved mode."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.all_bands_exist", lambda _interaction: False
    )
    monkeypatch.setattr(
        "phono3py.conductivity.rta_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )

    # velocity_operator shape: (num_gp=1, num_band0=2, nat3=6, 3)
    vel_op = np.arange(2 * 6 * 3, dtype="complex128").reshape(1, 2, 6, 3)
    extra_data = {"velocity_operator": vel_op}

    interaction = SimpleNamespace(
        primitive=SimpleNamespace(volume=1.0),
        get_phonons=lambda: (np.ones((20, 6), dtype="double"), None, None),
        band_indices=np.array([1, 3], dtype="int64"),
    )

    br = SimpleNamespace(
        grid_points=np.array([0], dtype="int64"),
        group_velocities=np.zeros((1, 2, 3), dtype="double"),
        gv_by_gv=np.zeros((1, 2, 6), dtype="double"),
        get_extra_grid_point_output=lambda: extra_data,
        mode_heat_capacities=np.zeros((1, 1, 2), dtype="double"),
        averaged_pp_interaction=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        temperatures=np.array([300.0], dtype="double"),
        gamma=np.zeros((1, 1, 1, 2), dtype="double"),
        gamma_isotope=None,
        sigmas=[None],
        sigma_cutoff_width=None,
        get_gamma_N_U=lambda: (None, None),
        frequencies=np.ones((1, 2), dtype="double"),
    )

    ConductivityRTAWriter.write_gamma(br, interaction, i=0)

    assert len(calls) == 2
    # First band (k=0): velocity_operator[gp=0][band=0]
    np.testing.assert_array_equal(
        calls[0]["extra_datasets"]["velocity_operator"], vel_op[0, 0]
    )
    # Second band (k=1): velocity_operator[gp=0][band=1]
    np.testing.assert_array_equal(
        calls[1]["extra_datasets"]["velocity_operator"], vel_op[0, 1]
    )
