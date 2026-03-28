"""Tests of file_IO functions."""

import os
import pathlib
import tempfile
from collections.abc import Sequence
from typing import Optional

import h5py
import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.file_IO import (
    _get_filename_suffix,
    get_filename_suffix,
    get_length_of_first_line,
    parse_FORCES_FC2,
    parse_FORCES_FC3,
    read_collision_from_hdf5,
    read_fc3_from_hdf5,
    read_gamma_from_hdf5,
    read_phonon_from_hdf5,
    read_pp_from_hdf5,
    write_collision_to_hdf5,
    write_fc2_to_hdf5,
    write_fc3_to_hdf5,
    write_FORCES_FC2,
    write_FORCES_FC3,
    write_full_collision_matrix,
    write_gamma_detail_to_hdf5,
    write_imag_self_energy_at_grid_point,
    write_joint_dos_at_t,
    write_phonon_to_hdf5,
    write_pp_to_hdf5,
    write_real_self_energy_at_grid_point,
    write_real_self_energy_to_hdf5,
    write_spectral_function_at_grid_point,
)

cwd = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# _get_filename_suffix
# ---------------------------------------------------------------------------


def test_kappa_filename():
    """Test _get_filename_suffix."""
    mesh = [4, 4, 4]
    grid_point = None
    band_indices = None
    sigma = None
    sigma_cutoff = None
    filename = None
    suffix = _get_filename_suffix(
        mesh,
        grid_point=grid_point,
        band_indices=band_indices,
        sigma=sigma,
        sigma_cutoff=sigma_cutoff,
        middle_filename=filename,
    )
    full_filename = "kappa" + suffix + ".hdf5"
    assert full_filename == "kappa-m444.hdf5"


def test_filename_suffix_with_grid_point():
    """_get_filename_suffix includes grid point."""
    suffix = _get_filename_suffix([4, 4, 4], grid_point=10)
    assert suffix == "-m444-g10"


def test_filename_suffix_with_sigma():
    """_get_filename_suffix includes sigma."""
    suffix = _get_filename_suffix([4, 4, 4], sigma=0.1)
    assert "s0.1" in suffix


def test_filename_suffix_with_band_indices():
    """_get_filename_suffix includes band indices."""
    suffix = _get_filename_suffix([4, 4, 4], band_indices=[0, 1])
    assert "b" in suffix


def test_filename_suffix_with_middle_filename():
    """_get_filename_suffix includes middle filename."""
    suffix = _get_filename_suffix([4, 4, 4], middle_filename="test")
    assert "test" in suffix


# ---------------------------------------------------------------------------
# write_fc3_to_hdf5 / read_fc3_from_hdf5
# ---------------------------------------------------------------------------


def test_fc3_roundtrip_full_format(tmp_path):
    """write_fc3_to_hdf5 and read_fc3_from_hdf5 round-trip in full format."""
    rng = np.random.default_rng(0)
    fc3 = np.array(rng.random((4, 4, 4, 3, 3, 3)), dtype="double", order="C")
    filename = str(tmp_path / "fc3.hdf5")

    write_fc3_to_hdf5(fc3, filename=filename)
    fc3_read = read_fc3_from_hdf5(filename=filename)

    assert isinstance(fc3_read, np.ndarray)
    np.testing.assert_array_equal(fc3, fc3_read)


def test_fc3_roundtrip_compact_format(tmp_path):
    """write_fc3_to_hdf5 and read_fc3_from_hdf5 round-trip in compact format."""
    rng = np.random.default_rng(1)
    fc3 = np.array(rng.random((2, 4, 4, 3, 3, 3)), dtype="double", order="C")
    fc3_nonzero_indices = np.ones((2, 4, 4), dtype="byte", order="C")
    p2s_map = np.array([0, 2], dtype=np.int64)
    filename = str(tmp_path / "fc3_compact.hdf5")

    write_fc3_to_hdf5(
        fc3,
        fc3_nonzero_indices=fc3_nonzero_indices,
        filename=filename,
        p2s_map=p2s_map,
    )
    result = read_fc3_from_hdf5(filename=filename)

    assert isinstance(result, dict)
    np.testing.assert_array_equal(fc3, result["fc3"])
    np.testing.assert_array_equal(fc3_nonzero_indices, result["fc3_nonzero_indices"])


def test_fc3_read_missing_key_raises(tmp_path):
    """read_fc3_from_hdf5 raises KeyError when 'fc3' dataset is absent."""
    filename = str(tmp_path / "empty.hdf5")
    with h5py.File(filename, "w") as w:
        w.create_dataset("dummy", data=np.array([1, 2, 3]))

    with pytest.raises(KeyError, match="fc3"):
        read_fc3_from_hdf5(filename=filename)


# ---------------------------------------------------------------------------
# write_fc2_to_hdf5 / read_fc2_from_hdf5  (via phonopy's reader)
# ---------------------------------------------------------------------------


def test_fc2_roundtrip(tmp_path):
    """write_fc2_to_hdf5 produces an hdf5 file with force_constants dataset."""
    from phono3py.file_IO import read_fc2_from_hdf5

    rng = np.random.default_rng(2)
    fc2 = np.array(rng.random((4, 4, 3, 3)), dtype="double", order="C")
    filename = str(tmp_path / "fc2.hdf5")

    write_fc2_to_hdf5(fc2, filename=filename)
    fc2_read = read_fc2_from_hdf5(filename=filename)

    np.testing.assert_array_almost_equal(fc2, fc2_read)


# ---------------------------------------------------------------------------
# write_phonon_to_hdf5 / read_phonon_from_hdf5
# ---------------------------------------------------------------------------


def test_phonon_roundtrip(tmp_path):
    """write_phonon_to_hdf5 and read_phonon_from_hdf5 round-trip."""
    mesh = np.array([2, 2, 2], dtype=np.int64)
    n_grid = 8
    n_band = 3
    rng = np.random.default_rng(3)
    frequency = np.array(rng.random((n_grid, n_band)), dtype="double", order="C")
    eigenvector = np.array(
        rng.random((n_grid, n_band, n_band))
        + 1j * rng.random((n_grid, n_band, n_band)),
        dtype="cdouble",
        order="C",
    )
    grid_address = np.array(rng.integers(0, 4, (n_grid, 3)), dtype=np.int64, order="C")

    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        write_phonon_to_hdf5(frequency, eigenvector, grid_address, mesh)
        freq_r, eigvec_r, gaddr_r = read_phonon_from_hdf5(mesh, verbose=False)
    finally:
        os.chdir(orig_dir)

    np.testing.assert_array_equal(frequency, freq_r)
    np.testing.assert_array_almost_equal(eigenvector, eigvec_r)
    np.testing.assert_array_equal(grid_address, gaddr_r)


def test_phonon_read_file_not_found(tmp_path):
    """read_phonon_from_hdf5 raises FileNotFoundError when file is absent."""
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError):
            read_phonon_from_hdf5([4, 4, 4], verbose=False)
    finally:
        os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# write_collision_to_hdf5 / read_collision_from_hdf5
# ---------------------------------------------------------------------------


def test_collision_roundtrip(tmp_path):
    """write_collision_to_hdf5 and read_collision_from_hdf5 round-trip."""
    mesh = [4, 4, 4]
    rng = np.random.default_rng(4)
    n_temp = 3
    n_gp = 10
    temperature = np.array([100.0, 200.0, 300.0])
    gamma = np.array(rng.random((n_temp, n_gp, 3)), dtype="double", order="C")
    collision_matrix = np.array(
        rng.random((n_temp, n_gp * 3, n_gp * 3)), dtype="double", order="C"
    )

    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        write_collision_to_hdf5(
            temperature,
            mesh,
            gamma=gamma,
            collision_matrix=collision_matrix,
        )
        col_r, gamma_r, temp_r = read_collision_from_hdf5(mesh, verbose=False)
    finally:
        os.chdir(orig_dir)

    np.testing.assert_array_equal(temperature, temp_r)
    np.testing.assert_array_equal(gamma, gamma_r)
    np.testing.assert_array_equal(collision_matrix, col_r[0])


def test_collision_only_temperatures(tmp_path):
    """read_collision_from_hdf5 with only_temperatures returns temperatures only."""
    mesh = [4, 4, 4]
    temperature = np.array([100.0, 200.0, 300.0])
    gamma = np.zeros((3, 5, 3))
    collision_matrix = np.zeros((3, 15, 15))

    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        write_collision_to_hdf5(
            temperature,
            mesh,
            gamma=gamma,
            collision_matrix=collision_matrix,
        )
        col_r, gamma_r, temp_r = read_collision_from_hdf5(
            mesh, only_temperatures=True, verbose=False
        )
    finally:
        os.chdir(orig_dir)

    assert col_r is None
    assert gamma_r is None
    np.testing.assert_array_equal(temperature, temp_r)


def test_collision_file_not_found(tmp_path):
    """read_collision_from_hdf5 raises FileNotFoundError when file is absent."""
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError):
            read_collision_from_hdf5([4, 4, 4], verbose=False)
    finally:
        os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# write_full_collision_matrix
# ---------------------------------------------------------------------------


def test_full_collision_matrix_roundtrip(tmp_path):
    """write_full_collision_matrix stores collision_matrix dataset."""
    rng = np.random.default_rng(5)
    colmat = np.array(rng.random((10, 10)), dtype="double")
    filename = str(tmp_path / "fcm.hdf5")

    write_full_collision_matrix(colmat, filename=filename)

    with h5py.File(filename, "r") as f:
        assert "collision_matrix" in f
        np.testing.assert_array_equal(colmat, f["collision_matrix"][:])


# ---------------------------------------------------------------------------
# write_pp_to_hdf5 / read_pp_from_hdf5
# ---------------------------------------------------------------------------


def test_pp_roundtrip_without_g_zero(tmp_path):
    """write_pp_to_hdf5 / read_pp_from_hdf5 round-trip without g_zero."""
    mesh = [4, 4, 4]
    rng = np.random.default_rng(6)
    pp = np.array(rng.random((5, 3, 3)), dtype="double", order="C")

    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        write_pp_to_hdf5(mesh, pp=pp, grid_point=0, verbose=False)
        pp_r, g_zero_r = read_pp_from_hdf5(mesh, grid_point=0, verbose=False)
    finally:
        os.chdir(orig_dir)

    np.testing.assert_array_almost_equal(pp, pp_r)
    assert g_zero_r is None


def test_pp_roundtrip_with_g_zero(tmp_path):
    """write_pp_to_hdf5 / read_pp_from_hdf5 round-trip with g_zero (packed bits).

    Positions where g_zero == 1 are treated as zero interaction strength,
    so pp must be zero there for the round-trip to be consistent.
    """
    mesh = [4, 4, 4]
    rng = np.random.default_rng(7)
    pp = np.array(rng.random((16, 3, 3)), dtype="double", order="C")
    g_zero = np.zeros((16, 3, 3), dtype="byte", order="C")
    g_zero[::3] = 1  # mark some positions as zero-weight
    pp[g_zero == 1] = 0.0  # zero out pp at those positions (semantic contract)

    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        write_pp_to_hdf5(
            mesh,
            pp=pp,
            g_zero=g_zero,
            grid_point=1,
            verbose=False,
            check_consistency=True,
        )
        pp_r, g_zero_r = read_pp_from_hdf5(
            mesh, grid_point=1, verbose=False, check_consistency=True
        )
    finally:
        os.chdir(orig_dir)

    np.testing.assert_array_almost_equal(pp, pp_r)
    np.testing.assert_array_equal(g_zero, g_zero_r)


def test_pp_file_not_found(tmp_path):
    """read_pp_from_hdf5 raises FileNotFoundError when file is absent."""
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError):
            read_pp_from_hdf5([4, 4, 4], verbose=False)
    finally:
        os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# write_imag_self_energy_at_grid_point
# ---------------------------------------------------------------------------


def test_write_imag_self_energy_at_grid_point(tmp_path):
    """write_imag_self_energy_at_grid_point writes expected .dat file."""
    frequencies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gammas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mesh = [4, 4, 4]
    gp = 10
    band_indices = [0]

    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        fname = write_imag_self_energy_at_grid_point(
            gp, band_indices, mesh, frequencies, gammas
        )
        assert pathlib.Path(fname).exists()
        data = np.loadtxt(fname)
    finally:
        os.chdir(orig_dir)

    np.testing.assert_array_almost_equal(data[:, 0], frequencies)
    np.testing.assert_array_almost_equal(data[:, 1], gammas)


# ---------------------------------------------------------------------------
# Integration: kappa hdf5 written via Phono3py.run_thermal_conductivity
# ---------------------------------------------------------------------------


def test_kappa_hdf5_with_boundary_mpf(si_pbesol: Phono3py):
    """Test boundary_mfp in kappa-*.hdf5.

    Remember to clean files created by
    Phono3py.run_thermal_conductivity(write_kappa=True).

    """
    key_ref = [
        "boundary_mfp",
        "frequency",
        "gamma",
        "grid_point",
        "group_velocity",
        "gv_by_gv",
        "heat_capacity",
        "kappa",
        "kappa_unit_conversion",
        "mesh",
        "mode_kappa",
        "qpoint",
        "temperature",
        "version",
        "weight",
    ]

    boundary_mfp = 10000.0

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            kappa_filename = _set_kappa(
                si_pbesol, [4, 4, 4], write_kappa=True, boundary_mfp=boundary_mfp
            )
            file_path = pathlib.Path(kappa_filename)
            with h5py.File(file_path, "r") as f:
                np.testing.assert_almost_equal(f["boundary_mfp"][()], boundary_mfp)
                assert set(list(f)) == set(key_ref)

            if file_path.exists():
                file_path.unlink()

        finally:
            os.chdir(original_cwd)


def _set_kappa(
    ph3: Phono3py,
    mesh: Sequence,
    is_isotope: bool = False,
    is_full_pp: bool = False,
    write_kappa: bool = False,
    boundary_mfp: Optional[float] = None,
) -> str:
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=is_isotope,
        is_full_pp=is_full_pp,
        write_kappa=write_kappa,
        boundary_mfp=boundary_mfp,
    )
    suffix = _get_filename_suffix(mesh)
    return "kappa" + suffix + ".hdf5"


# ---------------------------------------------------------------------------
# get_filename_suffix (public wrapper)
# ---------------------------------------------------------------------------


def test_get_filename_suffix_basic():
    """get_filename_suffix is a public wrapper of _get_filename_suffix."""
    assert get_filename_suffix([4, 4, 4]) == "-m444"


def test_get_filename_suffix_with_temperature():
    """get_filename_suffix includes temperature."""
    suffix = get_filename_suffix([2, 2, 2], temperature=300.0)
    assert "t300" in suffix


# ---------------------------------------------------------------------------
# get_length_of_first_line
# ---------------------------------------------------------------------------


def test_get_length_of_first_line(tmp_path):
    """get_length_of_first_line returns number of tokens on first data line."""
    f = tmp_path / "data.txt"
    f.write_text("# comment\n\n1.0 2.0 3.0\n4.0 5.0 6.0\n")
    with open(f) as fh:
        length = get_length_of_first_line(fh)
    assert length == 3


# ---------------------------------------------------------------------------
# write_FORCES_FC2 / parse_FORCES_FC2
# ---------------------------------------------------------------------------


def test_forces_fc2_roundtrip(tmp_path):
    """write_FORCES_FC2 and parse_FORCES_FC2 round-trip."""
    rng = np.random.default_rng(10)
    n_atom = 4
    forces = rng.random((n_atom, 3))
    disp_dataset = {
        "natom": n_atom,
        "first_atoms": [
            {
                "number": 0,
                "displacement": np.array([0.01, 0.0, 0.0]),
                "forces": forces,
            }
        ],
    }
    filename = tmp_path / "FORCES_FC2"
    write_FORCES_FC2(disp_dataset, filename=filename)
    assert filename.exists()

    parse_FORCES_FC2(disp_dataset, filename=filename)
    np.testing.assert_array_almost_equal(
        disp_dataset["first_atoms"][0]["forces"], forces
    )


# ---------------------------------------------------------------------------
# write_FORCES_FC3 / parse_FORCES_FC3
# ---------------------------------------------------------------------------


def test_forces_fc3_roundtrip(tmp_path):
    """write_FORCES_FC3 and parse_FORCES_FC3 round-trip."""
    rng = np.random.default_rng(11)
    n_atom = 4
    forces1 = rng.random((n_atom, 3))
    forces2 = rng.random((n_atom, 3))
    disp_dataset = {
        "natom": n_atom,
        "first_atoms": [
            {
                "number": 0,
                "displacement": np.array([0.01, 0.0, 0.0]),
                "forces": forces1,
                "second_atoms": [
                    {
                        "number": 1,
                        "displacement": np.array([0.0, 0.01, 0.0]),
                        "forces": forces2,
                    }
                ],
            }
        ],
    }
    filename = tmp_path / "FORCES_FC3"
    write_FORCES_FC3(disp_dataset, filename=filename)
    assert filename.exists()

    parse_FORCES_FC3(disp_dataset, filename=filename)
    np.testing.assert_array_almost_equal(
        disp_dataset["first_atoms"][0]["forces"], forces1
    )
    np.testing.assert_array_almost_equal(
        disp_dataset["first_atoms"][0]["second_atoms"][0]["forces"], forces2
    )


# ---------------------------------------------------------------------------
# write_joint_dos_at_t
# ---------------------------------------------------------------------------


def test_write_joint_dos_at_t(tmp_path):
    """write_joint_dos_at_t writes a .dat file with correct content."""
    frequencies = np.array([1.0, 2.0, 3.0])
    jdos = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        fname = write_joint_dos_at_t(0, [4, 4, 4], frequencies, jdos)
        assert pathlib.Path(fname).exists()
        data = np.loadtxt(fname)
    finally:
        os.chdir(orig_dir)
    np.testing.assert_array_almost_equal(data[:, 0], frequencies)
    np.testing.assert_array_almost_equal(data[:, 1:], jdos)


# ---------------------------------------------------------------------------
# write_real_self_energy_at_grid_point
# ---------------------------------------------------------------------------


def test_write_real_self_energy_at_grid_point(tmp_path):
    """write_real_self_energy_at_grid_point writes a .dat file."""
    freq_points = np.array([1.0, 2.0, 3.0])
    deltas = np.array([0.01, 0.02, 0.03])
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        fname = write_real_self_energy_at_grid_point(
            5, [0], freq_points, deltas, [4, 4, 4], epsilon=0.1, temperature=300.0
        )
        assert pathlib.Path(fname).exists()
        data = np.loadtxt(fname)
    finally:
        os.chdir(orig_dir)
    np.testing.assert_array_almost_equal(data[:, 0], freq_points)
    np.testing.assert_array_almost_equal(data[:, 1], deltas)


# ---------------------------------------------------------------------------
# write_real_self_energy_to_hdf5
# ---------------------------------------------------------------------------


def test_write_real_self_energy_to_hdf5(tmp_path):
    """write_real_self_energy_to_hdf5 writes expected datasets."""
    rng = np.random.default_rng(12)
    temperatures = np.array([100.0, 200.0])
    deltas = rng.random((2, 3))
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        fname = write_real_self_energy_to_hdf5(
            grid_point=5,
            band_indices=[0, 1, 2],
            temperatures=temperatures,
            deltas=deltas,
            mesh=[4, 4, 4],
            epsilon=0.1,
        )
        assert pathlib.Path(fname).exists()
        with h5py.File(fname, "r") as f:
            np.testing.assert_array_almost_equal(f["temperature"][:], temperatures)
            np.testing.assert_array_almost_equal(f["delta"][:], deltas)
    finally:
        os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# write_spectral_function_at_grid_point
# ---------------------------------------------------------------------------


def test_write_spectral_function_at_grid_point(tmp_path):
    """write_spectral_function_at_grid_point writes a .dat file."""
    freq_points = np.array([1.0, 2.0, 3.0])
    spectral = np.array([0.1, 0.2, 0.3])
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        fname = write_spectral_function_at_grid_point(
            5, [0], freq_points, spectral, [4, 4, 4], temperature=300.0
        )
        assert pathlib.Path(fname).exists()
        data = np.loadtxt(fname)
    finally:
        os.chdir(orig_dir)
    np.testing.assert_array_almost_equal(data[:, 0], freq_points)
    np.testing.assert_array_almost_equal(data[:, 1], spectral)


# ---------------------------------------------------------------------------
# write_gamma_detail_to_hdf5
# ---------------------------------------------------------------------------


def test_write_gamma_detail_to_hdf5(tmp_path):
    """write_gamma_detail_to_hdf5 writes expected datasets."""
    rng = np.random.default_rng(13)
    gamma_detail = rng.random((5, 3, 3))
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        fname = write_gamma_detail_to_hdf5(
            temperature=300.0,
            mesh=[4, 4, 4],
            gamma_detail=gamma_detail,
            grid_point=5,
            verbose=False,
        )
        assert pathlib.Path(fname).exists()
        with h5py.File(fname, "r") as f:
            assert "gamma_detail" in f
            np.testing.assert_array_almost_equal(f["gamma_detail"][:], gamma_detail)
    finally:
        os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# read_gamma_from_hdf5 (file absent)
# ---------------------------------------------------------------------------


def test_read_gamma_from_hdf5_file_not_found(tmp_path):
    """read_gamma_from_hdf5 returns (None, filename) when file is absent."""
    orig_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        result, fname = read_gamma_from_hdf5(np.array([4, 4, 4], dtype=np.int64))
    finally:
        os.chdir(orig_dir)
    assert result is None
    assert "kappa" in fname
