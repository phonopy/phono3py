"""Unit tests for lbte_init helper flows."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from phono3py.conductivity import lbte_init as ids


def _collision_context(log_level: int = 0):
    return {
        "mesh": np.array([2, 2, 2], dtype="int64"),
        "indices": "all",
        "sigma": None,
        "sigma_cutoff": None,
        "filename": None,
        "log_level": log_level,
    }


def test_set_collision_from_full_matrix_if_available_sets_data():
    """Full-matrix data is copied into lbte buffers."""
    lbte = SimpleNamespace(
        collision_matrix=np.zeros((2, 1, 1), dtype="double"),
        gamma=np.zeros((2, 1, 1), dtype="double"),
    )
    collisions = (
        np.array([[[1.5]]], dtype="double"),
        np.array([[[2.5]]], dtype="double"),
        np.array([300.0], dtype="double"),
    )

    ok = ids._set_collision_from_full_matrix_if_available(
        cast(Any, lbte),
        collisions,
        i_sigma=1,
    )

    assert ok is True
    np.testing.assert_allclose(
        lbte.collision_matrix[1], np.array([[1.5]], dtype="double")
    )
    np.testing.assert_allclose(lbte.gamma[1], np.array([[2.5]], dtype="double"))


def test_allocate_collision_with_fallback_prefers_grid_point_path(monkeypatch):
    """Fallback helper returns early when grid-point allocation is available."""
    calls = []

    def _fake_allocate_collision(for_gps, _grid_points, _context):
        calls.append(for_gps)
        if for_gps:
            return None, None, np.array([300.0], dtype="double")
        return False

    monkeypatch.setattr(ids, "_allocate_collision", _fake_allocate_collision)

    vals = ids._allocate_collision_with_fallback(
        np.array([11], dtype="int64"),
        context=cast(Any, _collision_context()),
        log_level=0,
    )

    assert vals is not False
    assert calls == [True]


def test_allocate_collision_with_fallback_uses_band_path_when_needed(monkeypatch):
    """Fallback helper tries band path when grid-point allocation is missing."""
    calls = []

    def _fake_allocate_collision(for_gps, _grid_points, _context):
        calls.append(for_gps)
        if for_gps:
            return False
        return None, None, np.array([300.0], dtype="double")

    monkeypatch.setattr(ids, "_allocate_collision", _fake_allocate_collision)

    vals = ids._allocate_collision_with_fallback(
        np.array([11], dtype="int64"),
        context=cast(Any, _collision_context()),
        log_level=0,
    )

    assert vals is not False
    assert calls == [True, False]


def test_allocate_collision_with_fallback_returns_false_when_missing(monkeypatch):
    """Fallback helper returns False when neither allocation path exists."""

    def _fake_allocate_collision(_for_gps, _grid_points, _context):
        return False

    monkeypatch.setattr(ids, "_allocate_collision", _fake_allocate_collision)

    vals = ids._allocate_collision_with_fallback(
        np.array([11], dtype="int64"),
        context=cast(Any, _collision_context()),
        log_level=0,
    )

    assert vals is False


def test_collect_collision_with_band_fallback_runs_all_bands(monkeypatch):
    """When gp-collection fails, band fallback iterates all bands."""
    called_bands = []

    monkeypatch.setattr(ids, "_collect_collision_gp", lambda *args, **kwargs: False)

    def _fake_collect_collision_band(
        _colmat_at_sigma,
        _gamma_at_sigma,
        _temperatures,
        _context,
        _i,
        _gp,
        _bzg2grg,
        band_index,
        _is_reducible_collision_matrix,
    ):
        called_bands.append(band_index)
        return True

    monkeypatch.setattr(ids, "_collect_collision_band", _fake_collect_collision_band)

    ok = ids._collect_collision_with_band_fallback(
        np.zeros((1, 1, 3), dtype="double"),
        np.zeros((1, 1, 3), dtype="double"),
        np.array([300.0], dtype="double"),
        context=cast(Any, _collision_context()),
        i=0,
        gp=7,
        bzg2grg=np.array([0], dtype="int64"),
        is_reducible_collision_matrix=False,
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

    def _fake_read_collision_data(_context, **_kwargs):
        return (
            np.array([[[9.0]]], dtype="double"),
            np.array([[[4.0]]], dtype="double"),
            np.array([300.0], dtype="double"),
        )

    monkeypatch.setattr(ids, "_read_collision_data", _fake_read_collision_data)
    monkeypatch.setattr(
        ids,
        "_allocate_collision_with_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fallback must not be called")
        ),
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

    monkeypatch.setattr(ids, "_read_collision_data", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        ids,
        "_allocate_collision_with_fallback",
        lambda *_args, **_kwargs: (
            None,
            None,
            np.array([250.0], dtype="double"),
        ),
    )

    def _fake_collect_collision_with_band_fallback(
        _colmat,
        _gamma,
        _temperatures,
        _context,
        i,
        gp,
        _bzg2grg,
        _is_reducible,
    ):
        calls.append((i, gp))
        return True

    monkeypatch.setattr(
        ids,
        "_collect_collision_with_band_fallback",
        _fake_collect_collision_with_band_fallback,
    )

    read_from = ids._set_collision_from_file(cast(Any, lbte), log_level=0)

    assert read_from == "grid_points"
    assert calls == [(0, 5)]
