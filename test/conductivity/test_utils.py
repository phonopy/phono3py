"""Unit tests for conductivity.utils."""

from types import SimpleNamespace

import h5py
import numpy as np

from phono3py.conductivity.utils import write_pp_interaction


def _make_conductivity(grid_points, sigmas, sigma_cutoff_width, mesh_numbers):
    return SimpleNamespace(
        grid_points=np.array(grid_points, dtype="int64"),
        sigmas=sigmas,
        sigma_cutoff_width=sigma_cutoff_width,
        mesh_numbers=np.array(mesh_numbers, dtype="int64"),
    )


def _make_pp(interaction_strength, zero_value_positions, triplets, weights, bz_grid):
    return SimpleNamespace(
        interaction_strength=interaction_strength,
        zero_value_positions=zero_value_positions,
        bz_grid=bz_grid,
        get_triplets_at_q=lambda: (triplets, weights, None, None),
    )


def test_write_pp_interaction_writes_file(monkeypatch, tmp_path):
    """write_pp_interaction creates an hdf5 file with the correct datasets."""
    pp_data = np.ones((4, 3, 3), dtype="double")
    triplets = np.arange(12, dtype="int64").reshape(4, 3)
    weights = np.array([1, 2, 3, 4], dtype="int64")
    bz_grid = SimpleNamespace()
    all_triplets_result = np.zeros((8, 3), dtype="int64")

    monkeypatch.setattr(
        "phono3py.conductivity.utils.get_all_triplets",
        lambda gp, bz: all_triplets_result,
    )
    monkeypatch.chdir(tmp_path)

    conductivity = _make_conductivity(
        grid_points=[5, 10],
        sigmas=[0.1],
        sigma_cutoff_width=None,
        mesh_numbers=[4, 4, 4],
    )
    # zero_value_positions=None so write_pp_to_hdf5 uses the simple format
    # (stores "pp", "triplet", "weight", "triplet_all" as separate datasets)
    pp = _make_pp(pp_data, None, triplets, weights, bz_grid)

    write_pp_interaction(conductivity, pp, i=1)

    # filename is constructed by write_pp_to_hdf5: pp-m444-g10-s0.1.hdf5
    hdf5_path = tmp_path / "pp-m444-g10-s0.1.hdf5"
    assert hdf5_path.exists()

    with h5py.File(hdf5_path, "r") as f:
        np.testing.assert_array_equal(f["pp"][:], pp_data)
        np.testing.assert_array_equal(f["triplet"][:], triplets)
        np.testing.assert_array_equal(f["weight"][:], weights)
        np.testing.assert_array_equal(f["triplet_all"][:], all_triplets_result)


def test_write_pp_interaction_multiple_sigmas_prints_warning(monkeypatch, tmp_path):
    """write_pp_interaction prints a warning when multiple sigmas are given."""
    monkeypatch.setattr(
        "phono3py.conductivity.utils.get_all_triplets",
        lambda gp, bz: np.zeros((2, 3), dtype="int64"),
    )
    monkeypatch.chdir(tmp_path)

    conductivity = _make_conductivity(
        grid_points=[0],
        sigmas=[0.1, 0.2],
        sigma_cutoff_width=None,
        mesh_numbers=[2, 2, 2],
    )
    pp_data = np.ones((1, 3, 3), dtype="double")
    pp = _make_pp(
        pp_data,
        None,
        np.zeros((1, 3), dtype="int64"),
        np.ones(1, dtype="int64"),
        SimpleNamespace(),
    )

    write_pp_interaction(conductivity, pp, i=0)

    hdf5_path = tmp_path / "pp-m222-g0-s0.2.hdf5"
    assert hdf5_path.exists()
