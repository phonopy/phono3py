"""Tests of Phono3py API."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Optional, Union

import h5py
import numpy as np
import pytest

from phono3py.cui.phono3py_script import main

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    filename: Optional[Sequence[os.PathLike]] = None
    conf_filename: Optional[os.PathLike] = None
    fc_calculator: Optional[str] = None
    force_sets_mode: bool = False
    force_sets_to_forces_fc2_mode: bool = False
    log_level: Optional[int] = None
    output_yaml_filename: Optional[os.PathLike] = None
    show_num_triplets: bool = False
    write_grid_points: bool = False
    fc_symmetry: bool = True
    cell_filename: Optional[str] = None
    is_bterta: Optional[bool] = None
    mesh_numbers: Optional[Sequence] = None
    temperatures: Optional[Sequence] = None
    input_filename = None
    output_filename = None
    input_output_filename = None

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))

    def __contains__(self, item):
        """Implement in operator."""
        return item in (field.name for field in fields(self))


def test_phono3py_load():
    """Test phono3py-load script."""
    # Check sys.exit(0)
    argparse_control = _get_phono3py_load_args(
        cwd / ".." / "phono3py_params_Si-111-222.yaml"
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phono3py-load script.
    for created_filename in ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5"):
        file_path = pathlib.Path(cwd_called / created_filename)
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize("fc_calculator,exit_code", [(None, 1), ("symfc", 0)])
def test_phono3py_load_with_typeII_dataset(fc_calculator, exit_code):
    """Test phono3py-load script with typeII dataset."""
    pytest.importorskip("symfc")
    argparse_control = _get_phono3py_load_args(
        cwd / ".." / "phono3py_params-Si111-rd.yaml.xz", fc_calculator=fc_calculator
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == exit_code

    # Clean files created by phono3py-load script.
    for created_filename in ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5"):
        file_path = pathlib.Path(cwd_called / created_filename)
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize("load_phono3py_yaml", [True, False])
def test_phono3py_with_QE_calculator(load_phono3py_yaml):
    """Test phono3py-load script with QE calculator."""
    argparse_control = _get_phono3py_load_args(
        cwd / "phono3py_params-qe-Si222.yaml.xz",
        load_phono3py_yaml=load_phono3py_yaml,
        is_bterta=True,
        temperatures=[
            "300",
        ],
        mesh_numbers=["11", "11", "11"],
    )
    with pytest.raises(SystemExit):
        main(**argparse_control)

    with h5py.File(cwd_called / "kappa-m111111.hdf5", "r") as f:
        np.testing.assert_almost_equal(f["kappa"][0, 0], 118.93, decimal=1)

    # Clean files created by phono3py/phono3py-load script.
    for created_filename in (
        "phono3py.yaml",
        "fc2.hdf5",
        "fc3.hdf5",
        "kappa-m111111.hdf5",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        if file_path.exists():
            file_path.unlink()


def _get_phono3py_load_args(
    phono3py_yaml_filepath: Union[str, pathlib.Path],
    fc_calculator: Optional[str] = None,
    load_phono3py_yaml: bool = True,
    is_bterta: bool = False,
    temperatures: Optional[Sequence] = None,
    mesh_numbers: Optional[Sequence] = None,
):
    # Mock of ArgumentParser.args.
    if load_phono3py_yaml:
        mockargs = MockArgs(
            filename=[phono3py_yaml_filepath],
            fc_calculator=fc_calculator,
            is_bterta=is_bterta,
            temperatures=temperatures,
            mesh_numbers=mesh_numbers,
            log_level=1,
        )
    else:
        mockargs = MockArgs(
            filename=[],
            fc_calculator=fc_calculator,
            log_level=1,
            cell_filename=phono3py_yaml_filepath,
            is_bterta=is_bterta,
            temperatures=temperatures,
            mesh_numbers=mesh_numbers,
        )

    # See phono3py-load script.
    argparse_control = {
        "fc_symmetry": load_phono3py_yaml,
        "is_nac": load_phono3py_yaml,
        "load_phono3py_yaml": load_phono3py_yaml,
        "args": mockargs,
    }
    return argparse_control
