"""Unit tests for ConductivityLBTEWriter."""

from types import SimpleNamespace

import numpy as np

from phono3py.conductivity.output import ConductivityLBTEWriter


def test_write_collision_all_bands(monkeypatch):
    """`write_collision` writes once per sigma in all-bands mode."""
    calls = []

    def _fake_write_collision_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.output.all_bands_exist",
        lambda _interaction: True,
    )
    monkeypatch.setattr(
        "phono3py.conductivity.output.write_collision_to_hdf5",
        _fake_write_collision_to_hdf5,
    )

    lbte = SimpleNamespace(
        grid_points=np.array([10], dtype="int64"),
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[None, 0.1],
        sigma_cutoff_width=None,
        gamma=np.array([[[1.0]], [[2.0]]], dtype="double"),
        gamma_isotope=np.array([[[0.1]], [[0.2]]], dtype="double"),
        collision_matrix=np.array([[[[7.0]]], [[[8.0]]]], dtype="double"),
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
    )
    interaction = SimpleNamespace(
        band_indices=np.array([0], dtype="int64"),
        bz_grid=SimpleNamespace(bzg2grg=np.array([0], dtype="int64")),
    )

    ConductivityLBTEWriter.write_collision(lbte, interaction, i=0)

    assert len(calls) == 2
    assert calls[0][1]["sigma"] is None
    assert calls[1][1]["sigma"] == 0.1
    assert all(call[1]["grid_point"] == 10 for call in calls)


def test_write_collision_band_resolved(monkeypatch):
    """`write_collision` writes once per (sigma, band) when not all bands exist."""
    calls = []

    def _fake_write_collision_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.output.all_bands_exist",
        lambda _interaction: False,
    )
    monkeypatch.setattr(
        "phono3py.conductivity.output.write_collision_to_hdf5",
        _fake_write_collision_to_hdf5,
    )

    lbte = SimpleNamespace(
        grid_points=np.array([3], dtype="int64"),
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[0.05],
        sigma_cutoff_width=3.0,
        gamma=np.array([[[[1.0, 2.0]]]], dtype="double"),
        gamma_isotope=np.array([[[0.1, 0.2]]], dtype="double"),
        collision_matrix=np.array([[[[7.0, 8.0]]]], dtype="double"),
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
    )
    interaction = SimpleNamespace(
        band_indices=np.array([2, 4], dtype="int64"),
        bz_grid=SimpleNamespace(bzg2grg=np.array([0], dtype="int64")),
    )

    ConductivityLBTEWriter.write_collision(lbte, interaction, i=0)

    assert len(calls) == 2
    assert {calls[0][1]["band_index"], calls[1][1]["band_index"]} == {2, 4}
    assert all(call[1]["sigma"] == 0.05 for call in calls)
