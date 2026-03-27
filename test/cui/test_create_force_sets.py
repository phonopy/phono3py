"""Tests for phono3py.cui.create_force_sets."""

from __future__ import annotations

import pathlib
import shutil
import tarfile

import numpy as np
import pytest

from phono3py.cui.create_force_sets import (
    _set_forces_and_nac_params,
    create_FORCE_SETS_from_FORCES_FCx,
    create_FORCES_FC2_from_FORCE_SETS,
    create_FORCES_FC3_and_FORCES_FC2,
)
from phono3py.cui.settings import Phono3pySettings
from phono3py.file_IO import write_FORCES_FC2, write_FORCES_FC3
from phono3py.interface.phono3py_yaml import Phono3pyYaml

test_dir = pathlib.Path(__file__).parent.parent


@pytest.fixture
def si_fd_work_dir(tmp_path):
    """Extract Si-111-222-fd.tar.xz and return the work directory."""
    with tarfile.open(test_dir / "Si-111-222-fd.tar.xz") as tar:
        tar.extractall(tmp_path, filter="data")
    return tmp_path / "Si-111-222-fd"


def test_create_forces_fc2_from_force_sets(tmp_path, monkeypatch):
    """FORCE_SETS is converted to FORCES_FC2 and yaml lines are printed."""
    shutil.copy(test_dir / "FORCE_SETS_NaCl", tmp_path / "FORCE_SETS")
    monkeypatch.chdir(tmp_path)
    create_FORCES_FC2_from_FORCE_SETS(log_level=1)
    assert (tmp_path / "FORCES_FC2").exists()


def test_create_force_sets_from_forces_fcx_fc2(tmp_path, monkeypatch):
    """FORCES_FC2 (type-1) is converted to FORCE_SETS."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)

    shutil.copy(params_file, tmp_path / "phono3py_disp.yaml")
    write_FORCES_FC2(p3yml.phonon_dataset, filename=str(tmp_path / "FORCES_FC2"))

    monkeypatch.chdir(tmp_path)
    create_FORCE_SETS_from_FORCES_FCx(p3yml.phonon_supercell_matrix, None, log_level=1)
    assert (tmp_path / "FORCE_SETS").exists()


def test_create_force_sets_from_forces_fcx_fc3(tmp_path, monkeypatch):
    """FORCES_FC3 (type-1) is converted to FORCE_SETS."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)

    shutil.copy(params_file, tmp_path / "phono3py_disp.yaml")
    write_FORCES_FC3(p3yml.dataset, filename=str(tmp_path / "FORCES_FC3"))

    monkeypatch.chdir(tmp_path)
    create_FORCE_SETS_from_FORCES_FCx(None, None, log_level=1)
    assert (tmp_path / "FORCE_SETS").exists()


def test_create_force_sets_from_forces_fcx_smat_mismatch(tmp_path, monkeypatch):
    """sys.exit(1) when the given phonon_smat mismatches the yaml."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)

    shutil.copy(params_file, tmp_path / "phono3py_disp.yaml")
    write_FORCES_FC2(p3yml.phonon_dataset, filename=str(tmp_path / "FORCES_FC2"))

    # phonon_supercell_matrix is 2x2x2; pass 1x1x1 to trigger mismatch
    wrong_smat = np.eye(3, dtype=int)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        create_FORCE_SETS_from_FORCES_FCx(wrong_smat, None, log_level=1)
    assert excinfo.value.code == 1


def test_set_forces_and_nac_params_type1(tmp_path, monkeypatch):
    """Forces are distributed into type-1 dataset entries."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)

    # Collect forces from the embedded dataset
    ds = p3yml.dataset
    forces: list = [ds["first_atoms"][0]["forces"]]
    for d2 in ds["first_atoms"][0]["second_atoms"]:
        forces.append(d2["forces"])
    calc_dataset_fc3 = {"forces": forces}

    # Load fresh copy without forces
    ph3_yaml = Phono3pyYaml(settings={"force_sets": True})
    ph3_yaml.read(params_file)
    for d1 in ph3_yaml.dataset["first_atoms"]:
        d1.pop("forces", None)
        for d2 in d1["second_atoms"]:
            d2.pop("forces", None)

    settings = Phono3pySettings()

    monkeypatch.chdir(tmp_path)
    _set_forces_and_nac_params(ph3_yaml, settings, calc_dataset_fc3, None)

    d1_result = ph3_yaml.dataset["first_atoms"][0]
    assert "forces" in d1_result
    np.testing.assert_array_equal(d1_result["forces"], forces[0])
    for i, d2 in enumerate(d1_result["second_atoms"]):
        assert "forces" in d2
        np.testing.assert_array_equal(d2["forces"], forces[i + 1])


def test_set_forces_and_nac_params_type2(tmp_path, monkeypatch):
    """Forces are assigned as an array into a type-2 (random-displacement) dataset."""
    params_file = test_dir / "phono3py_params-Si111-rd.yaml.xz"
    ph3_yaml = Phono3pyYaml(settings={"force_sets": True})
    ph3_yaml.read(params_file)

    ds = ph3_yaml.dataset
    assert "displacements" in ds, "Expected type-2 dataset"

    original_forces = ds["forces"].copy()

    # Remove forces so _set_forces_and_nac_params has to set them
    del ds["forces"]

    calc_dataset_fc3 = {"forces": list(original_forces)}
    settings = Phono3pySettings()

    monkeypatch.chdir(tmp_path)
    _set_forces_and_nac_params(ph3_yaml, settings, calc_dataset_fc3, None)

    assert "forces" in ph3_yaml.dataset
    np.testing.assert_array_equal(ph3_yaml.dataset["forces"], original_forces)


def test_set_forces_and_nac_params_type2_truncated(tmp_path, monkeypatch):
    """When fewer force sets than displacements, displacements are truncated."""
    params_file = test_dir / "phono3py_params-Si111-rd.yaml.xz"
    ph3_yaml = Phono3pyYaml(settings={"force_sets": True})
    ph3_yaml.read(params_file)

    ds = ph3_yaml.dataset
    truncated_forces = list(ds["forces"][:10])
    del ds["forces"]

    calc_dataset_fc3 = {"forces": truncated_forces}
    settings = Phono3pySettings()

    monkeypatch.chdir(tmp_path)
    _set_forces_and_nac_params(ph3_yaml, settings, calc_dataset_fc3, None)

    assert len(ph3_yaml.dataset["forces"]) == 10
    assert len(ph3_yaml.dataset["displacements"]) == 10


def test_create_forces_fc3_and_fc2_write_forces_fc3(si_fd_work_dir, monkeypatch):
    """FORCES_FC3 is created from VASP vasprun.xml files.

    vasprun-000.xml is the zero-displacement reference; displaced structures
    start at vasprun-001.xml (16 files for 16 displacements).
    """
    work_dir = si_fd_work_dir
    # 16 displacements (1 first + 15 second): vasprun-001..016
    vasprun_files = [str(work_dir / f"vasprun-{i:03d}.xml") for i in range(1, 17)]

    settings = Phono3pySettings()
    settings.create_forces_fc3 = vasprun_files

    monkeypatch.chdir(work_dir)
    create_FORCES_FC3_and_FORCES_FC2(settings, None, log_level=1)
    assert (work_dir / "FORCES_FC3").exists()


def test_create_forces_fc3_and_fc2_write_forces_fc2(si_fd_work_dir, monkeypatch):
    """Both FORCES_FC3 and FORCES_FC2 are created from VASP vasprun.xml files.

    vasprun-fc2-000.xml is the zero-displacement reference; the single
    phonon displacement is in vasprun-fc2-001.xml.
    """
    work_dir = si_fd_work_dir
    vasprun_fc3 = [str(work_dir / f"vasprun-{i:03d}.xml") for i in range(1, 17)]
    vasprun_fc2 = [str(work_dir / "vasprun-fc2-001.xml")]

    settings = Phono3pySettings()
    settings.create_forces_fc3 = vasprun_fc3
    settings.create_forces_fc2 = vasprun_fc2

    monkeypatch.chdir(work_dir)
    create_FORCES_FC3_and_FORCES_FC2(settings, None, log_level=1)
    assert (work_dir / "FORCES_FC3").exists()
    assert (work_dir / "FORCES_FC2").exists()


def test_create_forces_fc3_and_fc2_save_params(si_fd_work_dir, monkeypatch):
    """phono3py_params.yaml is created when save_params=True."""
    work_dir = si_fd_work_dir
    vasprun_files = [str(work_dir / f"vasprun-{i:03d}.xml") for i in range(1, 17)]

    settings = Phono3pySettings()
    settings.create_forces_fc3 = vasprun_files
    settings.save_params = True

    monkeypatch.chdir(work_dir)
    create_FORCES_FC3_and_FORCES_FC2(settings, None, log_level=1)
    assert (work_dir / "phono3py_params.yaml").exists()
    assert not (work_dir / "FORCES_FC3").exists()
