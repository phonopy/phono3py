"""Unit tests for ConductivityLBTEWriter."""

from types import SimpleNamespace

import numpy as np

from phono3py.conductivity.base import get_unit_to_WmK
from phono3py.conductivity.lbte_output import ConductivityLBTEWriter


def test_write_collision_all_bands(monkeypatch):
    """`write_collision` writes once per sigma in all-bands mode."""
    calls = []

    def _fake_write_collision_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.all_bands_exist", lambda _interaction: True
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_collision_to_hdf5",
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
        "phono3py.conductivity.lbte_output.all_bands_exist", lambda _interaction: False
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_collision_to_hdf5",
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


def test_write_kappa_calls_hdf5_writer_per_sigma(monkeypatch):
    """`write_kappa` forwards per-sigma payload to hdf5 writer."""
    calls = []

    def _fake_write_kappa_to_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    kappa = np.arange(12, dtype="double").reshape(2, 1, 6)
    mode_kappa = np.arange(12, dtype="double").reshape(2, 1, 1, 1, 6)
    kappa_rta = np.arange(12, 24, dtype="double").reshape(2, 1, 6)
    mode_kappa_rta = np.arange(12, 24, dtype="double").reshape(2, 1, 1, 1, 6)
    gv = np.ones((1, 1, 3), dtype="double")
    gv_by_gv = np.ones((1, 1, 6), dtype="double")

    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.get_lbte_writer_kappa_data",
        lambda _lbte: (
            kappa,
            mode_kappa,
            kappa_rta,
            mode_kappa_rta,
            gv,
            gv_by_gv,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_kappa_to_hdf5",
        _fake_write_kappa_to_hdf5,
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_collision_eigenvalues_to_hdf5",
        lambda *args, **kwargs: None,
    )

    lbte = SimpleNamespace(
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[None, 0.1],
        sigma_cutoff_width=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        grid_points=np.array([0], dtype="int64"),
        grid_weights=np.array([1], dtype="int64"),
        frequencies=np.array([[4.0]], dtype="double"),
        averaged_pp_interaction=None,
        qpoints=np.array([[0.0, 0.0, 0.0]], dtype="double"),
        gamma=np.zeros((2, 1, 1, 1), dtype="double"),
        gamma_isotope=None,
        get_f_vectors=lambda: np.zeros((2, 1, 1, 3), dtype="double"),
        mode_heat_capacities=np.ones((1, 1, 1), dtype="double"),
        get_mean_free_path=lambda: np.zeros((2, 1, 1, 3), dtype="double"),
        boundary_mfp=None,
        collision_eigenvalues=None,
        collision_matrix=np.zeros((2, 1, 1, 1), dtype="double"),
        get_frequencies_all=lambda: np.array([[5.0]], dtype="double"),
    )

    ConductivityLBTEWriter.write_kappa(lbte, volume=2.0, compression="gzip")

    assert len(calls) == 2
    for i, (_args, kwargs) in enumerate(calls):
        np.testing.assert_allclose(kwargs["kappa"], kappa[i])
        np.testing.assert_allclose(kwargs["kappa_RTA"], kappa_rta[i])
        assert kwargs["sigma"] == lbte.sigmas[i]
        np.testing.assert_allclose(
            kwargs["kappa_unit_conversion"], get_unit_to_WmK() / 2.0
        )


def test_write_kappa_writes_eigen_and_unitary(monkeypatch):
    """`write_kappa` writes eigenvalues and unitary matrix when requested."""
    eigen_calls = []
    unitary_calls = []

    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.get_lbte_writer_kappa_data",
        lambda _lbte: (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_kappa_to_hdf5",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_collision_eigenvalues_to_hdf5",
        lambda *args, **kwargs: eigen_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.write_unitary_matrix_to_hdf5",
        lambda *args, **kwargs: unitary_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "phono3py.conductivity.lbte_output.select_colmat_solver", lambda _solver: 1
    )

    lbte = SimpleNamespace(
        temperatures=np.array([300.0], dtype="double"),
        sigmas=[0.2],
        sigma_cutoff_width=3.0,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        bz_grid=SimpleNamespace(),
        grid_points=np.array([0], dtype="int64"),
        grid_weights=np.array([1], dtype="int64"),
        frequencies=np.array([[4.0]], dtype="double"),
        averaged_pp_interaction=None,
        qpoints=np.array([[0.0, 0.0, 0.0]], dtype="double"),
        gamma=np.zeros((1, 1, 1, 1), dtype="double"),
        gamma_isotope=None,
        get_f_vectors=lambda: np.zeros((1, 1, 1, 3), dtype="double"),
        mode_heat_capacities=np.ones((1, 1, 1), dtype="double"),
        get_mean_free_path=lambda: np.zeros((1, 1, 1, 3), dtype="double"),
        boundary_mfp=None,
        collision_eigenvalues=np.array([[[1.0]]], dtype="double"),
        collision_matrix=np.array([[[[1.0]]]], dtype="double"),
        get_frequencies_all=lambda: np.array([[5.0]], dtype="double"),
    )

    ConductivityLBTEWriter.write_kappa(
        lbte,
        volume=1.0,
        write_LBTE_solution=True,
        pinv_solver=1,
    )

    assert len(eigen_calls) == 1
    assert len(unitary_calls) == 1
