"""Tests of phono3py-load script."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Optional, Union

import h5py
import numpy as np
import pytest

import phono3py
from phono3py.cui.phono3py_script import main

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    cell_filename: Optional[str] = None
    conf_filename: Optional[os.PathLike] = None
    fc_calculator: Optional[str] = None
    fc_calculator_options: Optional[str] = None
    fc_symmetry: bool = True
    filename: Optional[Sequence[os.PathLike]] = None
    force_sets_mode: bool = False
    force_sets_to_forces_fc2_mode: bool = False
    input_filename = None
    input_output_filename = None
    log_level: Optional[int] = None
    is_bterta: Optional[bool] = None
    mesh_numbers: Optional[Sequence] = None
    mlp_params: Optional[str] = None
    output_filename = None
    output_yaml_filename: Optional[os.PathLike] = None
    random_displacements: Optional[Union[int, str]] = None
    show_num_triplets: bool = False
    temperatures: Optional[Sequence] = None
    use_pypolymlp: bool = False
    write_grid_points: bool = False

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
        cwd / ".." / "phono3py_params_Si-111-222.yaml",
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    argparse_control = _get_phono3py_load_args(
        cwd_called / "phono3py.yaml",
        is_bterta=True,
        temperatures=[
            "300",
        ],
        mesh_numbers=["5", "5", "5"],
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phono3py-load script.
    for created_filename in (
        "phono3py.yaml",
        "fc2.hdf5",
        "fc3.hdf5",
        "kappa-m555.hdf5",
    ):
        file_path = cwd_called / created_filename
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize(
    "load_phono3py_yaml,fc_calculator,fc_calculator_options",
    [
        (True, None, None),
        (True, "symfc", None),
        (True, "symfc", "|cutoff=4.0"),
        (False, "symfc", "|cutoff=4.0"),
    ],
)
def test_phono3py_load_with_typeII_dataset(
    fc_calculator: str | None,
    fc_calculator_options: str | None,
    load_phono3py_yaml: bool,
):
    """Test phono3py-load script with typeII dataset.

    When None, fallback to symfc.

    """
    pytest.importorskip("symfc")
    argparse_control = _get_phono3py_load_args(
        cwd / ".." / "phono3py_params-Si111-rd.yaml.xz",
        load_phono3py_yaml=load_phono3py_yaml,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phono3py-load script.
    for created_filename in ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5"):
        file_path = pathlib.Path(cwd_called / created_filename)
        if file_path.exists():
            if created_filename == "fc3.hdf5":
                with h5py.File(file_path, "r") as f:
                    if fc_calculator_options is None:
                        assert "fc3_nonzero_indices" not in f
                    else:
                        assert "fc3_nonzero_indices" in f
                        assert "fc3_cutoff" in f
                        assert f["fc3_cutoff"][()] == pytest.approx(4.0)
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


def test_phono3py_load_with_pypolymlp_si():
    """Test phono3py-load script with pypolymlp.

    First run generates polymlp.yaml.
    Second run uses polymlp.yaml.

    """
    pytest.importorskip("pypolymlp", minversion="0.9.2")
    pytest.importorskip("symfc")

    # Create fc2.hdf5
    argparse_control = _get_phono3py_load_args(
        cwd / ".." / "phono3py_params_Si-111-222-rd.yaml.xz",
        fc_calculator="symfc",
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0
    for created_filename in ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5"):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
    pathlib.Path(cwd_called / "fc3.hdf5").unlink()

    # Create MLP (polymlp.yaml)
    argparse_control = _get_phono3py_load_args(
        cwd / ".." / "phono3py_params_Si-111-222-rd.yaml.xz",
        use_pypolymlp=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0
    for created_filename in ("phono3py.yaml", "polymlp.yaml"):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()

    # Create phono3py_mlp_eval_dataset.yaml
    argparse_control = _get_phono3py_load_args(
        cwd_called / "phono3py.yaml",
        fc_calculator="symfc",
        random_displacements="auto",
        use_pypolymlp=True,
    )

    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    ph3 = phono3py.load(cwd_called / "phono3py_mlp_eval_dataset.yaml")
    assert len(ph3.displacements) == 4

    for created_filename in (
        "phono3py.yaml",
        "fc2.hdf5",
        "fc3.hdf5",
        "polymlp.yaml",
        "phono3py_mlp_eval_dataset.yaml",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def test_phono3py_load_with_pypolymlp_nacl():
    """Test phono3py-load script with pypolymlp using NaCl.

    First run generates polymlp.yaml.
    Second run uses polymlp.yaml.

    """
    pytest.importorskip("pypolymlp", minversion="0.9.2")
    pytest.importorskip("symfc")

    # Stage1 (preparation)
    argparse_control = _get_phono3py_load_args(
        cwd / ".." / "phono3py_params_MgO-222rd-444rd.yaml.xz",
        mlp_params="cutoff=4.0,gtinv_maxl=4 4,max_p=1,gtinv_order=2",
        fc_calculator="symfc",
        random_displacements="auto",
        use_pypolymlp=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    ph3 = phono3py.load(cwd_called / "phono3py_mlp_eval_dataset.yaml")
    assert len(ph3.displacements) == 16

    for created_filename in (
        "phono3py.yaml",
        "fc2.hdf5",
        "fc3.hdf5",
        "polymlp.yaml",
        "phono3py_mlp_eval_dataset.yaml",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()

    for created_filename in (
        "fc3.hdf5",
        "phono3py_mlp_eval_dataset.yaml",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()

    # Stage2 (cutoff test)
    argparse_control = _get_phono3py_load_args(
        cwd_called / "phono3py.yaml",
        fc_calculator="symfc",
        fc_calculator_options="|cutoff=4.0",
        random_displacements="auto",
        use_pypolymlp=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    ph3 = phono3py.load(cwd_called / "phono3py_mlp_eval_dataset.yaml")
    assert len(ph3.displacements) == 4

    for created_filename in (
        "phono3py.yaml",
        "fc2.hdf5",
        "fc3.hdf5",
        "polymlp.yaml",
        "phono3py_mlp_eval_dataset.yaml",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()

    for created_filename in (
        "fc3.hdf5",
        "phono3py_mlp_eval_dataset.yaml",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()

    # Stage3 (memsize test)
    argparse_control = _get_phono3py_load_args(
        cwd_called / "phono3py.yaml",
        fc_calculator="symfc",
        fc_calculator_options="|memsize=0.05",
        random_displacements="auto",
        use_pypolymlp=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    ph3 = phono3py.load(cwd_called / "phono3py_mlp_eval_dataset.yaml")
    assert len(ph3.displacements) == 8

    for created_filename in (
        "phono3py.yaml",
        "fc2.hdf5",
        "fc3.hdf5",
        "polymlp.yaml",
        "phono3py_mlp_eval_dataset.yaml",
    ):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def _get_phono3py_load_args(
    phono3py_yaml_filepath: Union[str, pathlib.Path],
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[str] = None,
    load_phono3py_yaml: bool = True,
    is_bterta: bool = False,
    mesh_numbers: Optional[Sequence] = None,
    mlp_params: Optional[str] = None,
    random_displacements: Optional[Union[int, str]] = None,
    temperatures: Optional[Sequence] = None,
    use_pypolymlp: bool = False,
):
    # Mock of ArgumentParser.args.
    if load_phono3py_yaml:
        mockargs = MockArgs(
            filename=[phono3py_yaml_filepath],
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            is_bterta=is_bterta,
            log_level=1,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            random_displacements=random_displacements,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
        )
    else:
        mockargs = MockArgs(
            filename=[],
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            log_level=1,
            cell_filename=phono3py_yaml_filepath,
            is_bterta=is_bterta,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            random_displacements=random_displacements,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
        )

    # See phono3py-load script.
    argparse_control = {
        "fc_symmetry": load_phono3py_yaml,
        "is_nac": load_phono3py_yaml,
        "load_phono3py_yaml": load_phono3py_yaml,
        "args": mockargs,
    }
    return argparse_control
