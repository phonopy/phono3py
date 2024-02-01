"""Tests of Phono3py API."""

import pathlib
from collections.abc import Sequence
from typing import Optional

import h5py
import numpy as np

from phono3py import Phono3py
from phono3py.file_IO import _get_filename_suffix

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


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
        filename=filename,
    )
    full_filename = "kappa" + suffix + ".hdf5"
    assert full_filename == "kappa-m444.hdf5"


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
    kappa_filename = _set_kappa(
        si_pbesol, [4, 4, 4], write_kappa=True, boundary_mfp=boundary_mfp
    )
    file_path = pathlib.Path(cwd_called / kappa_filename)
    with h5py.File(file_path, "r") as f:
        np.testing.assert_almost_equal(f["boundary_mfp"][()], boundary_mfp)
        assert set(list(f)) == set(key_ref)

    if file_path.exists():
        file_path.unlink()


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
