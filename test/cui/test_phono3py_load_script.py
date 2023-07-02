"""Tests of Phono3py API."""
from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Optional

import pytest

from phono3py.cui.phono3py_script import main

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    filename: Sequence[os.PathLike]
    conf_filename: Optional[os.PathLike] = None
    fc_calculator: Optional[str] = None
    force_sets_mode: bool = False
    force_sets_to_forces_fc2_mode: bool = False
    log_level: Optional[int] = None
    output_yaml_filename: Optional[os.PathLike] = None
    show_num_triplets: bool = False
    write_grid_points: bool = False

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))


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


@pytest.mark.parametrize("fc_calculator,exit_code", [(None, 1), ("alm", 0)])
def test_phono3py_load_with_typeII_dataset(fc_calculator, exit_code):
    """Test phono3py-load script with typeII dataset."""
    # Check sys.exit(0)
    if fc_calculator == "alm":
        pytest.importorskip("alm")
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


def _get_phono3py_load_args(phono3py_yaml_filepath, fc_calculator=None):
    # Mock of ArgumentParser.args.
    mockargs = MockArgs(
        filename=[phono3py_yaml_filepath],
        fc_calculator=fc_calculator,
        log_level=1,
    )

    # See phono3py-load script.
    argparse_control = {
        "fc_symmetry": True,
        "is_nac": True,
        "load_phono3py_yaml": True,
        "args": mockargs,
    }
    return argparse_control
