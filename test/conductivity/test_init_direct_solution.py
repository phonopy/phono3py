"""Unit tests for lbte_init helper flows."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from phono3py.conductivity import lbte_init as ids
from phono3py.conductivity.lbte_init import CollisionFileReader


def _make_reader(log_level: int = 0) -> CollisionFileReader:
    return CollisionFileReader(
        mesh=np.array([2, 2, 2], dtype="int64"),
        indices="all",
        sigma_cutoff=None,
        filename=None,
        log_level=log_level,
    )


def test_try_full_matrix_stores_data(monkeypatch):
    """Full-matrix data is copied into collision buffers."""
    reader = _make_reader()
    collisions = (
        np.array([[[1.5]]], dtype="double"),
        np.array([[[2.5]]], dtype="double"),
        np.array([300.0], dtype="double"),
    )
    monkeypatch.setattr(reader, "read", lambda sigma, **kw: collisions)

    collision_matrix = np.zeros((2, 1, 1), dtype="double")
    gamma = np.zeros((2, 1, 1), dtype="double")

    ok = reader.try_full_matrix(None, collision_matrix, gamma, i_sigma=1)

    assert ok is True
    np.testing.assert_allclose(collision_matrix[1], np.array([[1.5]], dtype="double"))
    np.testing.assert_allclose(gamma[1], np.array([[2.5]], dtype="double"))


def test_allocate_with_fallback_prefers_grid_point_path(monkeypatch):
    """Fallback helper returns early when grid-point allocation is available."""
    reader = _make_reader()
    calls = []

    def _fake_read(sigma, *, grid_point=None, band_index=None, only_temperatures=False):
        calls.append(("gp" if band_index is None else "band", grid_point))
        return None, None, np.array([300.0], dtype="double")

    monkeypatch.setattr(reader, "read", _fake_read)

    result = reader.allocate_with_fallback(None, np.array([11], dtype="int64"))

    assert result is not False
    assert len(calls) == 1
    assert calls[0] == ("gp", 11)


def test_allocate_with_fallback_uses_band_path_when_needed(monkeypatch):
    """Fallback helper tries band path when grid-point allocation is missing."""
    reader = _make_reader()
    calls = []

    def _fake_read(sigma, *, grid_point=None, band_index=None, only_temperatures=False):
        calls.append(("gp" if band_index is None else "band", grid_point))
        if band_index is None:
            return None  # GP path fails
        return None, None, np.array([300.0], dtype="double")

    monkeypatch.setattr(reader, "read", _fake_read)

    result = reader.allocate_with_fallback(None, np.array([11], dtype="int64"))

    assert result is not False
    assert len(calls) == 2


def test_allocate_with_fallback_returns_false_when_missing(monkeypatch):
    """Fallback helper returns False when neither allocation path exists."""
    reader = _make_reader()
    monkeypatch.setattr(reader, "read", lambda sigma, **kw: None)

    result = reader.allocate_with_fallback(None, np.array([11], dtype="int64"))

    assert result is False


def test_collect_with_band_fallback_runs_all_bands(monkeypatch):
    """When gp-collection fails, band fallback iterates all bands."""
    reader = _make_reader()
    called_bands = []

    monkeypatch.setattr(reader, "collect_gp", lambda *args, **kwargs: False)

    def _fake_collect_band(
        sigma, _colmat, _gamma, _temps, _i, _gp, _bzg2grg, band_index, _is_red
    ):
        called_bands.append(band_index)
        return True

    monkeypatch.setattr(reader, "collect_band", _fake_collect_band)

    ok = reader.collect_with_band_fallback(
        None,
        np.zeros((1, 1, 3), dtype="double"),
        np.zeros((1, 1, 3), dtype="double"),
        np.array([300.0], dtype="double"),
        0,
        7,
        np.array([0], dtype="int64"),
        False,
    )

    assert ok is True
    assert called_bands == [0, 1, 2]


def test_set_collision_from_file_full_matrix_path(monkeypatch):
    """`_set_collision_from_file` prefers full-matrix data when available."""
    lbte = SimpleNamespace(
        bz_grid=SimpleNamespace(bzg2grg=np.array([0], dtype="int64")),
        sigmas=[None],
        sigma_cutoff_width=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        grid_points=np.array([5], dtype="int64"),
        collision_matrix=np.zeros((1, 1, 1), dtype="double"),
        gamma=np.zeros((1, 1, 1), dtype="double"),
    )

    collisions = (
        np.array([[[9.0]]], dtype="double"),
        np.array([[[4.0]]], dtype="double"),
        np.array([300.0], dtype="double"),
    )
    monkeypatch.setattr(
        CollisionFileReader, "read", lambda self, sigma, **kw: collisions
    )

    read_from = ids._set_collision_from_file(cast(Any, lbte), log_level=0)

    assert read_from == "full_matrix"
    np.testing.assert_allclose(
        lbte.collision_matrix[0], np.array([[9.0]], dtype="double")
    )
    np.testing.assert_allclose(lbte.gamma[0], np.array([[4.0]], dtype="double"))


def test_set_collision_from_file_fallback_grid_points_path(monkeypatch):
    """Fallback path uses per-gp collector when full-matrix read fails."""
    lbte = SimpleNamespace(
        bz_grid=SimpleNamespace(bzg2grg=np.array([0], dtype="int64")),
        sigmas=[None],
        sigma_cutoff_width=None,
        mesh_numbers=np.array([2, 2, 2], dtype="int64"),
        grid_points=np.array([5], dtype="int64"),
        collision_matrix=np.zeros((1, 1, 1), dtype="double"),
        gamma=np.zeros((1, 1, 1), dtype="double"),
    )
    calls = []

    monkeypatch.setattr(
        CollisionFileReader,
        "try_full_matrix",
        lambda self, sigma, cm, g, i: False,
    )
    monkeypatch.setattr(
        CollisionFileReader,
        "allocate_with_fallback",
        lambda self, sigma, gps: np.array([250.0], dtype="double"),
    )

    def _fake_collect(self, sigma, cm, g, temps, i, gp, bzg2grg, is_red):
        calls.append((i, gp))
        return True

    monkeypatch.setattr(
        CollisionFileReader,
        "collect_with_band_fallback",
        _fake_collect,
    )

    read_from = ids._set_collision_from_file(cast(Any, lbte), log_level=0)

    assert read_from == "grid_points"
    assert calls == [(0, 5)]
