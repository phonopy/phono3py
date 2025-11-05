"""Tests of phono3py-load script."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields

import h5py
import numpy as np
import pytest

import phono3py
from phono3py.cui.phono3py_script import main

cwd = pathlib.Path(__file__).parent


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    cell_filename: str | os.PathLike | None = None
    conf_filename: str | os.PathLike | None = None
    fc_calculator: str | None = None
    fc_calculator_options: str | None = None
    fc_symmetry: bool = True
    filename: Sequence[str | os.PathLike] | None = None
    force_sets_mode: bool = False
    force_sets_to_forces_fc2_mode: bool = False
    input_filename = None
    input_output_filename = None
    log_level: int | None = None
    is_bterta: bool | None = None
    is_lbte: bool | None = None
    is_wigner_kappa: bool | None = None
    mesh_numbers: Sequence | None = None
    mlp_params: str | None = None
    rd_number_estimation_factor: float | None = None
    read_gamma: bool = False
    output_filename = None
    output_yaml_filename: str | os.PathLike | None = None
    random_displacements: int | str | None = None
    show_num_triplets: bool = False
    temperatures: Sequence | None = None
    use_pypolymlp: bool = False
    write_gamma: bool = False
    write_grid_points: bool = False
    write_phonon: bool = False

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))

    def __contains__(self, item):
        """Implement in operator."""
        return item in (field.name for field in fields(self))


def test_phono3py_load():
    """Test phono3py-load script."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
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
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_lbte():
    """Test phono3py-load script running direct solution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                is_lbte=True,
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
                "coleigs-m555.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_wigner_rta():
    """Test phono3py-load script running wigner rta."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                is_bterta=True,
                is_wigner_kappa=True,
                temperatures=[
                    "300",
                ],
                mesh_numbers=["5", "5", "5"],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # for filename in pathlib.Path.cwd().iterdir():
            #     print(filename)

            # Clean files created by phono3py-load script.
            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "kappa-m555.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_wigner_lbte():
    """Test phono3py-load script running wigner rta."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                is_lbte=True,
                is_wigner_kappa=True,
                temperatures=[
                    "300",
                ],
                mesh_numbers=["5", "5", "5"],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # for filename in pathlib.Path.cwd().iterdir():
            #     print(filename)

            # Clean files created by phono3py-load script.
            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "kappa-m555.hdf5",
                "coleigs-m555.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


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

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
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
                file_path = pathlib.Path(created_filename)
                if file_path.exists():
                    if created_filename == "fc3.hdf5":
                        with h5py.File(file_path, "r") as f:
                            if fc_calculator_options is None:
                                assert "fc3_nonzero_indices" not in f
                            else:
                                assert "fc3_nonzero_indices" in f
                                assert "fc3_cutoff" in f
                                assert f["fc3_cutoff"][()] == pytest.approx(4.0)  # type: ignore
                    file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phono3py_yaml", [True, False])
def test_phono3py_with_QE_calculator(load_phono3py_yaml: bool):
    """Test phono3py-load script with QE calculator."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
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

            with h5py.File("kappa-m111111.hdf5", "r") as f:
                np.testing.assert_almost_equal(f["kappa"][0, 0], 118.93, decimal=1)  # type: ignore

            # Clean files created by phono3py/phono3py-load script.
            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "kappa-m111111.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_with_pypolymlp_si():
    """Test phono3py-load script with pypolymlp.

    First run generates polymlp.yaml.
    Second run uses polymlp.yaml.

    """
    pytest.importorskip("pypolymlp", minversion="0.9.2")
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Create fc2.hdf5
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222-rd.yaml.xz",
                fc_calculator="symfc",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0
            for created_filename in ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
            pathlib.Path("fc3.hdf5").unlink()

            # Create MLP (polymlp.yaml)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222-rd.yaml.xz",
                use_pypolymlp=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0
            for created_filename in ("phono3py.yaml", "polymlp.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

            # Create phono3py_mlp_eval_dataset.yaml
            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                fc_calculator="symfc",
                random_displacements="auto",
                use_pypolymlp=True,
            )

            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            ph3 = phono3py.load("phono3py_mlp_eval_dataset.yaml")
            assert len(ph3.displacements) == 8

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "polymlp.yaml",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_with_pypolymlp_nacl():
    """Test phono3py-load script with pypolymlp using NaCl.

    First run generates polymlp.yaml.
    Second run uses polymlp.yaml.

    """
    pytest.importorskip("pypolymlp", minversion="0.9.2")
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
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

            ph3 = phono3py.load("phono3py_mlp_eval_dataset.yaml")
            assert len(ph3.displacements) == 32

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "polymlp.yaml",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

            for created_filename in (
                "fc3.hdf5",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            # Stage2 (cutoff test)
            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                fc_calculator="symfc",
                fc_calculator_options="|cutoff=4.0",
                random_displacements="auto",
                use_pypolymlp=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            ph3 = phono3py.load("phono3py_mlp_eval_dataset.yaml")
            assert len(ph3.displacements) == 8

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "polymlp.yaml",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

            for created_filename in (
                "fc3.hdf5",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            # Stage3 (memsize test)
            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                fc_calculator="symfc",
                fc_calculator_options="|memsize=0.05",
                random_displacements="auto",
                use_pypolymlp=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            ph3 = phono3py.load("phono3py_mlp_eval_dataset.yaml")
            assert len(ph3.displacements) == 16

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "polymlp.yaml",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

            for created_filename in (
                "fc3.hdf5",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            # Stage4 (number_estimation_factor)
            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                fc_calculator="symfc",
                fc_calculator_options="|cutoff=4.0",
                random_displacements="auto",
                rd_number_estimation_factor=2.0,
                use_pypolymlp=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            ph3 = phono3py.load("phono3py_mlp_eval_dataset.yaml")
            assert len(ph3.displacements) == 4

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "polymlp.yaml",
                "phono3py_mlp_eval_dataset.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_fc2_fc3_cutoff():
    """Test --fc-calc-opt option for both fc2 and fc3 cutoff."""
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Stage1 (preparation)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_MgO-222rd-444rd.yaml.xz",
                fc_calculator="symfc",
                fc_calculator_options="cutoff=3.5|cutoff=3.0",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("phono3py.yaml", "fc2.hdf5", "fc3.hdf5"):
                file_path = pathlib.Path(created_filename)
                if file_path.exists():
                    if created_filename == "fc3.hdf5":
                        with h5py.File(file_path, "r") as f:
                            assert "fc3_cutoff" in f
                            assert f["fc3_cutoff"][()] == pytest.approx(3.0)  # type: ignore
                    elif created_filename == "fc2.hdf5":
                        with h5py.File(file_path, "r") as f:
                            assert "cutoff" in f
                            assert f["cutoff"][()] == pytest.approx(3.5)  # type: ignore
                    file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_write_phonon():
    """Test phono3py-load script with write_phonon."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
                write_phonon=True,
                mesh_numbers=["5", "5", "5"],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phono3py-load script.
            for created_filename in (
                "phonon-m555.hdf5",
                "fc3.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                write_phonon=True,
                mesh_numbers=["5", "5", "5"],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "phonon-m555.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_read_gamma():
    """Test phono3py-load script with read_gamma."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
                mesh_numbers=["5", "5", "5"],
                is_bterta=True,
                temperatures=[
                    "300",
                ],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            with h5py.File("kappa-m555.hdf5", "r") as f:
                kappa_val = f["kappa"][0, 0]  # type: ignore

            for created_filename in ("fc3.hdf5",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                read_gamma=True,
                mesh_numbers=["5", "5", "5"],
                is_bterta=True,
                temperatures=[
                    "300",
                ],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            assert not pathlib.Path("fc3.hdf5").exists()

            with h5py.File("kappa-m555.hdf5", "r") as f:
                np.testing.assert_almost_equal(f["kappa"][0, 0], kappa_val, decimal=3)  # type: ignore

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "kappa-m555.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_write_gamma():
    """Test phono3py-load script with read_gamma."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_Si-111-222.yaml",
                mesh_numbers=["5", "5", "5"],
                is_bterta=True,
                temperatures=[
                    "300",
                ],
                write_gamma=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            with h5py.File("kappa-m555.hdf5", "r") as f:
                kappa_val = f["kappa"][0, 0]  # type: ignore

            # Clean files created by phono3py-load script.
            for created_filename in ("fc3.hdf5", "kappa-m555.hdf5"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                read_gamma=True,
                mesh_numbers=["5", "5", "5"],
                is_bterta=True,
                temperatures=[
                    "300",
                ],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            assert not pathlib.Path("fc3.hdf5").exists()

            with h5py.File("kappa-m555.hdf5", "r") as f:
                np.testing.assert_almost_equal(f["kappa"][0, 0], kappa_val, decimal=3)  # type: ignore

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "kappa-m555.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            for gp in [0, 1, 2, 6, 7, 8, 9, 12, 13, 40]:
                created_filename = f"kappa-m555-g{gp}.hdf5"
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def _ls():
    current_dir = pathlib.Path(".")
    for file in current_dir.iterdir():
        print(file.name)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())


def _get_phono3py_load_args(
    phono3py_yaml_filepath: str | os.PathLike | None,
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    load_phono3py_yaml: bool = True,
    is_bterta: bool = False,
    is_lbte: bool = False,
    is_wigner_kappa: bool = False,
    mesh_numbers: Sequence | None = None,
    mlp_params: str | None = None,
    random_displacements: int | str | None = None,
    rd_number_estimation_factor: float | None = None,
    read_gamma: bool = False,
    temperatures: Sequence | None = None,
    use_pypolymlp: bool = False,
    write_gamma: bool = False,
    write_phonon: bool = False,
):
    # Mock of ArgumentParser.args.
    if load_phono3py_yaml:
        assert phono3py_yaml_filepath is not None
        mockargs = MockArgs(
            filename=[phono3py_yaml_filepath],
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            is_bterta=is_bterta,
            is_lbte=is_lbte,
            is_wigner_kappa=is_wigner_kappa,
            log_level=1,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            random_displacements=random_displacements,
            read_gamma=read_gamma,
            rd_number_estimation_factor=rd_number_estimation_factor,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
            write_gamma=write_gamma,
            write_phonon=write_phonon,
        )
    else:
        mockargs = MockArgs(
            filename=[],
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            log_level=1,
            cell_filename=phono3py_yaml_filepath,
            is_bterta=is_bterta,
            is_lbte=is_lbte,
            is_wigner_kappa=is_wigner_kappa,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            random_displacements=random_displacements,
            rd_number_estimation_factor=rd_number_estimation_factor,
            read_gamma=read_gamma,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
            write_gamma=write_gamma,
            write_phonon=write_phonon,
        )

    # See phono3py-load script.
    argparse_control = {
        "fc_symmetry": load_phono3py_yaml,
        "is_nac": load_phono3py_yaml,
        "load_phono3py_yaml": load_phono3py_yaml,
        "args": mockargs,
    }
    return argparse_control
