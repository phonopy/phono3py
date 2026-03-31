"""Tests of phono3py-load script for Wigner conductivity."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Sequence

import h5py
import numpy as np
import pytest

from phono3py.cui.phono3py_argparse import Phono3pyMockArgs
from phono3py.cui.phono3py_script import main

cwd = pathlib.Path(__file__).parent
_data = cwd.parent.parent  # test/ directory with yaml files


def test_phono3py_load_generates_wigner_kappa_hdf5_contents():
    """Run phono3py-load in wigner RTA mode and validate kappa hdf5 content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phono3py_load_args(
                _data / "phono3py_params_Si-111-222.yaml",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            argparse_control = _get_phono3py_load_args(
                "phono3py.yaml",
                is_bterta=True,
                is_wigner_kappa=True,
                temperatures=["300"],
                mesh_numbers=["5", "5", "5"],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            kappa_path = pathlib.Path("kappa-m555.hdf5")
            assert kappa_path.exists()

            with h5py.File(kappa_path, "r") as f:
                assert "kappa_P_RTA" in f
                assert "kappa_C" in f
                assert "kappa_TOT_RTA" in f
                assert "temperature" in f
                assert "mesh" in f
                assert f["kappa_P_RTA"].shape == (1, 6)
                assert f["kappa_C"].shape == (1, 6)
                assert f["kappa_TOT_RTA"].shape == (1, 6)
                assert f["temperature"][0] == pytest.approx(300.0)
                assert np.all(f["mesh"][:] == np.array([5, 5, 5]))
                assert np.all(np.isfinite(f["kappa_TOT_RTA"][0]))
                assert f["kappa_TOT_RTA"][0, 0] > 0

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


def test_phono3py_load_wigner_rta():
    """Test phono3py-load script running wigner rta."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                _data / "phono3py_params_Si-111-222.yaml",
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
    """Test phono3py-load script running wigner lbte."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phono3py_load_args(
                _data / "phono3py_params_Si-111-222.yaml",
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
                mesh_numbers=["9", "9", "9"],
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            with h5py.File("kappa-m999.hdf5", "r") as f:
                assert "kappa_P_exact" in f
                assert "kappa_P_RTA" in f
                assert "kappa_C" in f
                assert "temperature" in f
                assert "mesh" in f

                kappa_p_exact = f["kappa_P_exact"][:]
                kappa_p_rta = f["kappa_P_RTA"][:]
                kappa_c = f["kappa_C"][:]

                assert kappa_p_exact.shape == (1, 6)
                assert kappa_p_rta.shape == (1, 6)
                assert kappa_c.shape == (1, 6)
                assert f["temperature"][0] == pytest.approx(300.0)
                assert np.all(f["mesh"][:] == np.array([9, 9, 9]))
                assert np.all(np.isfinite(kappa_p_exact))
                assert np.all(np.isfinite(kappa_p_rta))
                assert np.all(np.isfinite(kappa_c))
                assert kappa_p_exact[0, 0] == pytest.approx(96.14464215062665, abs=0.8)
                assert kappa_p_rta[0, 0] == pytest.approx(97.30006583719721, abs=0.8)
                assert kappa_c[0, 0] == pytest.approx(0.1731832844767842, rel=2e-2)

            with h5py.File("coleigs-m999.hdf5", "r") as f:
                assert "collision_eigenvalues" in f
                coleigs = f["collision_eigenvalues"][:]
                assert coleigs.shape == (1, 630)
                assert np.all(np.isfinite(coleigs))
                assert coleigs[0, 0] == pytest.approx(-5.92530232954215e-17, rel=0.7)

            # Clean files created by phono3py-load script.
            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "kappa-m999.hdf5",
                "coleigs-m999.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_wigner_write_gamma_contains_isotope_and_N_U():
    """Test wigner write_gamma output contains gamma datasets and values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phono3py_load_args(
                _data / "phono3py_params_Si-111-222.yaml",
                mesh_numbers=["9", "9", "9"],
                is_bterta=True,
                is_wigner_kappa=True,
                is_isotope=True,
                is_N_U=True,
                temperatures=["300"],
                write_gamma=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            kappa_file = pathlib.Path("kappa-m999.hdf5")
            assert kappa_file.exists()
            with h5py.File(kappa_file, "r") as f:
                assert "gamma" in f
                assert "gamma_isotope" in f
                assert "gamma_N" in f
                assert "gamma_U" in f

                gamma = f["gamma"][:]
                gamma_isotope = f["gamma_isotope"][:]
                gamma_n = f["gamma_N"][:]
                gamma_u = f["gamma_U"][:]

                assert gamma.shape == (1, 35, 6)
                assert gamma_isotope.shape == (35, 6)
                assert gamma_n.shape == (1, 35, 6)
                assert gamma_u.shape == (1, 35, 6)
                assert np.all(np.isfinite(gamma))
                assert np.all(np.isfinite(gamma_isotope))
                assert np.all(np.isfinite(gamma_n))
                assert np.all(np.isfinite(gamma_u))

                assert np.max(gamma) == pytest.approx(0.07279890065189869, rel=0.2)
                assert np.max(gamma_n) == pytest.approx(0.04573886317305927, rel=0.2)
                assert np.max(gamma_u) == pytest.approx(0.045010833987380004, rel=0.2)
                assert np.max(gamma_isotope) == pytest.approx(
                    0.014355506081254441, rel=0.2
                )
                assert np.max(np.abs(gamma - (gamma_n + gamma_u))) < 1e-10

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "kappa-m999.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            for file_path in pathlib.Path(".").glob("kappa-m999-g*.hdf5"):
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phono3py_load_wigner_write_gamma_detail_outputs_hdf5():
    """Test wigner write_gamma_detail creates gamma_detail hdf5 files with values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phono3py_load_args(
                _data / "phono3py_params_Si-111-222.yaml",
                mesh_numbers=["9", "9", "9"],
                is_bterta=True,
                is_wigner_kappa=True,
                temperatures=["300"],
                write_gamma=True,
                write_gamma_detail=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            gamma_detail_files = sorted(
                pathlib.Path(".").glob("gamma_detail-m999*.hdf5")
            )
            assert gamma_detail_files

            with h5py.File(gamma_detail_files[0], "r") as f:
                assert "gamma_detail" in f
                assert "temperature" in f
                assert "mesh" in f
                assert "triplet" in f
                assert "triplet_all" in f
                assert f["temperature"][0] == pytest.approx(300.0)
                assert np.all(f["mesh"][:] == np.array([9, 9, 9]))

                gamma_detail = f["gamma_detail"][:]
                assert gamma_detail.shape == (1, 35, 6, 6, 6)
                assert gamma_detail.size > 0
                assert gamma_detail[0, 0, 0, 0, 0] == pytest.approx(0.0, abs=1e-16)
                assert np.max(gamma_detail) == pytest.approx(
                    2.243623483101439e-04, rel=0.2
                )
                assert np.count_nonzero(gamma_detail) > 100

                gp = int(f["grid_point"][()])
                weight = f["weight"][:]
                gamma_tp = gamma_detail.sum(axis=-1).sum(axis=-1)
                gamma_from_detail = np.dot(weight, gamma_tp[0])

            with h5py.File(f"kappa-m999-g{gp}.hdf5", "r") as f_gp:
                gamma_gp = f_gp["gamma"][0]

            assert np.sum(gamma_from_detail) == pytest.approx(
                np.sum(gamma_gp), rel=1e-10
            )

            for created_filename in (
                "phono3py.yaml",
                "fc2.hdf5",
                "fc3.hdf5",
                "kappa-m999.hdf5",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            for file_path in pathlib.Path(".").glob("gamma_detail-m999*.hdf5"):
                file_path.unlink()

            for file_path in pathlib.Path(".").glob("kappa-m999-g*.hdf5"):
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
    is_displacement: bool | None = None,
    is_fc3_r0_average: bool | None = None,
    is_isotope: bool | None = None,
    is_N_U: bool | None = None,
    is_lbte: bool | None = None,
    is_wigner_kappa: bool | None = None,
    mesh_numbers: Sequence | None = None,
    mlp_params: str | None = None,
    phonon_supercell_dimension: Sequence | None = None,
    random_displacements: int | str | None = None,
    rd_number_estimation_factor: float | None = None,
    read_gamma: bool | None = None,
    save_params: bool | None = None,
    supercell_dimension: Sequence | None = None,
    temperatures: Sequence | None = None,
    use_pypolymlp: bool | None = None,
    write_gamma: bool | None = None,
    write_gamma_detail: bool | None = None,
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
            is_displacement=is_displacement,
            is_fc3_r0_average=is_fc3_r0_average,
            is_isotope=is_isotope,
            is_N_U=is_N_U,
            is_lbte=is_lbte,
            is_wigner_kappa=is_wigner_kappa,
            log_level=1,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            phonon_supercell_dimension=phonon_supercell_dimension,
            random_displacements=random_displacements,
            read_gamma=read_gamma,
            rd_number_estimation_factor=rd_number_estimation_factor,
            save_params=save_params,
            supercell_dimension=supercell_dimension,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
            write_gamma=write_gamma,
            write_gamma_detail=write_gamma_detail,
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
            is_displacement=is_displacement,
            is_fc3_r0_average=is_fc3_r0_average,
            is_isotope=is_isotope,
            is_N_U=is_N_U,
            is_lbte=is_lbte,
            is_wigner_kappa=is_wigner_kappa,
            mesh_numbers=mesh_numbers,
            mlp_params=mlp_params,
            phonon_supercell_dimension=phonon_supercell_dimension,
            random_displacements=random_displacements,
            rd_number_estimation_factor=rd_number_estimation_factor,
            read_gamma=read_gamma,
            save_params=save_params,
            supercell_dimension=supercell_dimension,
            temperatures=temperatures,
            use_pypolymlp=use_pypolymlp,
            write_gamma=write_gamma,
            write_gamma_detail=write_gamma_detail,
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
