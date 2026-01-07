"""Tests of phono3py-load script."""

from __future__ import annotations

import lzma
import os
import pathlib
import tempfile
from collections.abc import Sequence

import h5py
import numpy as np
import pytest

import phono3py
from phono3py.cui.phono3py_argparse import Phono3pyMockArgs
from phono3py.cui.phono3py_script import main
from phono3py.cui.settings import Phono3pySettings
from phono3py.file_IO import write_FORCES_FC2, write_FORCES_FC3

cwd = pathlib.Path(__file__).parent


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
                if load_phono3py_yaml:
                    assert f["kappa"][0, 0] == pytest.approx(121.42, abs=0.15)  # type: ignore
                else:
                    assert f["kappa"][0, 0] == pytest.approx(100.25, abs=0.5)  # type: ignore

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


def test_phono3py_load_fc3_r0_average():
    """Test phono3py-load script with is_fc3_r0_average."""
    is_fc3_r0_average = Phono3pySettings().is_fc3_r0_average
    assert is_fc3_r0_average is True

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_params_AlN332.yaml.xz",
                load_phono3py_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                mesh_numbers="11 11 6",
                is_bterta=True,
                temperatures=[
                    "300",
                ],
                is_fc3_r0_average=not is_fc3_r0_average,
                load_phono3py_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            with h5py.File("kappa-m11116.hdf5", "r") as f:
                np.testing.assert_allclose(
                    f["kappa"][0, :3],  # type: ignore
                    [232.950, 232.950, 213.358],
                    atol=0.3,
                )

            file_path = pathlib.Path("kappa-m11116.hdf5")
            assert file_path.exists()
            file_path.unlink()

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                mesh_numbers="11 11 6",
                is_bterta=True,
                temperatures=[
                    "300",
                ],
                is_fc3_r0_average=is_fc3_r0_average,
                load_phono3py_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            with h5py.File("kappa-m11116.hdf5", "r") as f:
                np.testing.assert_allclose(
                    f["kappa"][0, :3],  # type: ignore
                    [240.807, 240.807, 219.942],
                    atol=0.3,
                )

            # Clean files created by phono3py-load script.
            for created_filename in (
                "fc3.hdf5",
                "fc2.hdf5",
                "phono3py.yaml",
                "kappa-m11116.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_FORCES_FC3_xz():
    """Test phono3py-load script with reading FORCES_FC3.xz."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        with open(cwd / ".." / "FORCES_FC3_si_pbesol", "rb") as f_in:
            with lzma.open("FORCES_FC3.xz", "wb") as f_out:
                f_out.write(f_in.read())

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                cwd / ".." / "phono3py_si_pbesol.yaml",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            pathlib.Path("FORCES_FC3.xz").unlink()

            for created_filename in ("phono3py.yaml", "fc3.hdf5", "fc2.hdf5"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_FORCES_FC2_xz():
    """Test phono3py-load script with reading FORCES_FC2.xz."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        ph3 = phono3py.load(cwd / ".." / "phono3py_params_Si-111-222-fd.yaml.xz")
        ph3.save("phono3py_disp.yaml", settings={"force_sets": False})
        assert ph3.dataset is not None
        assert ph3.forces is not None
        assert ph3.phonon_forces is not None
        write_FORCES_FC2(ph3.dataset, ph3.phonon_forces)
        write_FORCES_FC3(ph3.dataset, ph3.forces)

        with open("FORCES_FC2", "rb") as f_in:
            with lzma.open("FORCES_FC2.xz", "wb") as f_out:
                f_out.write(f_in.read())

        pathlib.Path("FORCES_FC2").unlink()

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args()
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            pathlib.Path("FORCES_FC2.xz").unlink()
            pathlib.Path("FORCES_FC3").unlink()
            pathlib.Path("phono3py_disp.yaml").unlink()

            for created_filename in ("phono3py.yaml", "fc3.hdf5", "fc2.hdf5"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("save_params", [False, True])
def test_create_forces_fc3_fc2_rd(save_params: bool):
    """Test phono3py with random displacements."""

    def check_supercell_energies(num_fc3_force_files: int, num_fc2_force_files: int):
        ph3 = phono3py.load("phono3py_params.yaml", produce_fc=False)
        assert ph3.supercell_energies is not None
        np.testing.assert_allclose(
            ph3.supercell_energies,
            [
                -223.886324,
                -223.875031,
                -223.880732,
                -223.885302,
                -223.886291,
                -223.880438,
                -223.879852,
                -223.885472,
                -223.881605,
                -223.884761,
                -223.876625,
                -223.880542,
                -223.883219,
                -223.88378,
                -223.879651,
                -223.884732,
                -223.87873,
                -223.884416,
                -223.881279,
                -223.877764,
            ][:num_fc3_force_files],
        )
        assert ph3.phonon_supercell_energies is not None
        np.testing.assert_allclose(
            ph3.phonon_supercell_energies,
            [-1791.121997, -1791.131866][:num_fc2_force_files],
        )

    if save_params:
        created_filenames = ("phono3py_params.yaml",)
    else:
        created_filenames = ("FORCES_FC3", "FORCES_FC2")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_NaCl_rd"
                / "phono3py_disp.yaml.xz",
                create_forces_fc3=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-000{i:02d}.xml.xz"
                    for i in range(1, 21)
                ],
                create_forces_fc2=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-ph000{i:02d}.xml.xz"
                    for i in range(1, 3)
                ],
                load_phono3py_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(20, 2)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            # Allows less number of force files (three disps in phonopy_disp.yaml).
            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_NaCl_rd"
                / "phono3py_disp.yaml.xz",
                create_forces_fc3=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-000{i:02d}.xml.xz"
                    for i in range(1, 20)
                ],
                create_forces_fc2=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-ph000{i:02d}.xml.xz"
                    for i in range(1, 3)
                ],
                load_phono3py_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(19, 2)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_NaCl_rd"
                / "phono3py_disp.yaml.xz",
                create_forces_fc3=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-000{i:02d}.xml.xz"
                    for i in range(1, 21)
                ],
                create_forces_fc2=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-ph000{i:02d}.xml.xz"
                    for i in range(1, 3)
                ],
                load_phono3py_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(20, 2)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            # Allows less number of force files (three disps in phonopy_disp.yaml).
            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_NaCl_rd"
                / "phono3py_disp.yaml.xz",
                create_forces_fc3=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-000{i:02d}.xml.xz"
                    for i in range(1, 20)
                ],
                create_forces_fc2=[
                    cwd / "vaspruns_NaCl_rd" / f"vasprun-ph000{i:02d}.xml.xz"
                    for i in range(1, 2)
                ],
                load_phono3py_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(19, 1)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_create_forces_fc3_fc2_fd():
    """Test phono3py with finite differentce."""
    created_filenames = ("FORCES_FC3", "FORCES_FC2")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_Si_PBEsol"
                / "phono3py_disp.yaml.xz",
                create_forces_fc3=[
                    cwd / "vaspruns_Si_PBEsol" / f"vasprun-00{i:03d}.xml.xz"
                    for i in range(1, 112)
                ],
                load_phono3py_yaml=False,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in created_filenames[:1]:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            with open("cf3_filenames.txt", "w") as f:
                for i in range(1, 112):
                    print(
                        str(cwd / "vaspruns_Si_PBEsol" / f"vasprun-00{i:03d}.xml.xz"),
                        file=f,
                    )
            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_Si_PBEsol"
                / "phono3py_disp.yaml.xz",
                create_forces_fc3_file="cf3_filenames.txt",
                load_phono3py_yaml=False,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in (created_filenames[0], "cf3_filenames.txt"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            argparse_control = _get_phono3py_load_args(
                phono3py_yaml_filepath=cwd
                / "vaspruns_Si_PBEsol"
                / "phono3py_disp_dimfc2.yaml.xz",
                create_forces_fc3=[
                    cwd / "vaspruns_Si_PBEsol" / f"vasprun-00{i:03d}.xml.xz"
                    for i in range(1, 112)
                ],
                create_forces_fc2=[
                    cwd / "vaspruns_Si_PBEsol" / "vasprun-ph00001.xml.xz"
                ],
                load_phono3py_yaml=False,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in created_filenames:
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
    phono3py_yaml_filepath: str | os.PathLike | None = None,  # =cell_filename
    create_forces_fc2: list[str | os.PathLike] | None = None,
    create_forces_fc3: list[str | os.PathLike] | None = None,
    create_forces_fc3_file: str | os.PathLike | None = None,
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    load_phono3py_yaml: bool = True,
    is_bterta: bool | None = None,
    is_fc3_r0_average: bool | None = None,
    is_lbte: bool | None = None,
    is_wigner_kappa: bool | None = None,
    mesh_numbers: Sequence | None = None,
    mlp_params: str | None = None,
    random_displacements: int | str | None = None,
    rd_number_estimation_factor: float | None = None,
    read_gamma: bool | None = None,
    save_params: bool | None = None,
    temperatures: Sequence | None = None,
    use_pypolymlp: bool | None = None,
    write_gamma: bool | None = None,
    write_phonon: bool | None = None,
):
    # Mock of ArgumentParser.args.
    if load_phono3py_yaml:
        if phono3py_yaml_filepath is None:
            filename = []
        else:
            filename = [phono3py_yaml_filepath]
        # assert phono3py_yaml_filepath is not None
        mockargs = Phono3pyMockArgs(
            filename=filename,
            create_forces_fc2=create_forces_fc2,
            create_forces_fc3=create_forces_fc3,
            create_forces_fc3_file=create_forces_fc3_file,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            is_bterta=is_bterta,
            is_fc3_r0_average=is_fc3_r0_average,
            is_lbte=is_lbte,
            is_wigner_kappa=is_wigner_kappa,
            log_level=1,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            random_displacements=random_displacements,
            read_gamma=read_gamma,
            rd_number_estimation_factor=rd_number_estimation_factor,
            save_params=save_params,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
            write_gamma=write_gamma,
            write_phonon=write_phonon,
        )
    else:
        mockargs = Phono3pyMockArgs(
            filename=[],
            create_forces_fc2=create_forces_fc2,
            create_forces_fc3=create_forces_fc3,
            create_forces_fc3_file=create_forces_fc3_file,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            log_level=1,
            cell_filename=phono3py_yaml_filepath,
            is_bterta=is_bterta,
            is_fc3_r0_average=is_fc3_r0_average,
            is_lbte=is_lbte,
            is_wigner_kappa=is_wigner_kappa,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            random_displacements=random_displacements,
            rd_number_estimation_factor=rd_number_estimation_factor,
            read_gamma=read_gamma,
            save_params=save_params,
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
