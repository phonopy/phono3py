"""Tests for CollisionMatrix class."""

from __future__ import annotations

import importlib.util

import numpy as np

from phono3py import Phono3py
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction

_HAS_RUST = importlib.util.find_spec("phonors") is not None
_LANGS: tuple[str, ...] = ("C", "Python", "Rust") if _HAS_RUST else ("C", "Python")


def _get_interaction(ph3: Phono3py, mesh: list[int]) -> Interaction:
    ph3.mesh_numbers = mesh
    assert ph3.grid is not None
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        cutoff_frequency=1e-4,
    )
    itr.init_dynamical_matrix(
        ph3.fc2,
        ph3.phonon_supercell,
        ph3.phonon_primitive,
    )
    itr.run_phonon_solver()
    return itr


def test_collision_matrix_py_vs_c_reducible(si_pbesol: Phono3py):
    """Python, C, and Rust implementations of reducible collision matrix agree."""
    itr = _get_interaction(si_pbesol, [4, 4, 4])
    grid_point = 1
    temperature = 300.0

    results = {}
    for lang in _LANGS:
        cm = CollisionMatrix(itr, lang=lang)
        cm.set_grid_point(grid_point)
        cm.temperature = temperature
        cm.run_integration_weights()
        cm.run()
        results[lang] = cm.get_collision_matrix().copy()

    np.testing.assert_allclose(results["Python"], results["C"], rtol=0, atol=1e-10)
    if _HAS_RUST:
        np.testing.assert_allclose(results["Rust"], results["C"], rtol=0, atol=1e-10)


def test_collision_matrix_py_vs_c_irreducible(si_pbesol: Phono3py):
    """Python, C, and Rust implementations of irreducible collision matrix agree."""
    from phonopy.phonon.grid import get_grid_points_by_rotations

    itr = _get_interaction(si_pbesol, [4, 4, 4])
    grid_point = 1
    temperature = 300.0

    bz_grid = itr.bz_grid
    rotations = bz_grid.rotations
    num_ops = len(rotations)
    rot_grid_points = np.zeros((1, num_ops), dtype="int64")
    rot_grid_points[0] = get_grid_points_by_rotations(
        grid_point,
        bz_grid,
        reciprocal_rotations=rotations,
    )

    results = {}
    for lang in _LANGS:
        cm = CollisionMatrix(itr, rot_grid_points=rot_grid_points, lang=lang)
        cm.set_grid_point(grid_point)
        cm.temperature = temperature
        cm.run_integration_weights()
        cm.run()
        results[lang] = cm.get_collision_matrix().copy()

    np.testing.assert_allclose(results["Python"], results["C"], rtol=0, atol=1e-10)
    if _HAS_RUST:
        np.testing.assert_allclose(results["Rust"], results["C"], rtol=0, atol=1e-10)


def test_get_gp2tp_map_shapes(si_pbesol: Phono3py):
    """_get_gp2tp_map returns arrays with correct shapes and valid indices."""
    itr = _get_interaction(si_pbesol, [4, 4, 4])
    grid_point = 1

    cm = CollisionMatrix(itr)
    cm.set_grid_point(grid_point)

    gp2tp, tp2s, swapped = cm._get_gp2tp_map()

    num_mesh = int(np.prod(itr.mesh_numbers))
    assert gp2tp.shape == (num_mesh,)
    assert tp2s.shape == (num_mesh,)
    assert swapped.shape == (num_mesh,)
    assert gp2tp.dtype == np.int64
    assert tp2s.dtype == np.int64
    # All entries must point to valid triplet indices
    num_triplets = len(cm._triplets_at_q)
    assert (gp2tp >= 0).all()
    assert (gp2tp < num_triplets).all()
