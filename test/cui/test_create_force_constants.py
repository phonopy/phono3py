"""Tests for phono3py.cui.create_force_constants."""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.cui.create_force_constants import (
    _convert_unit_in_dataset,
    parse_forces,
)
from phono3py.interface.phono3py_yaml import Phono3pyYaml

test_dir = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# _convert_unit_in_dataset
# ---------------------------------------------------------------------------


def test_convert_unit_type1_displacement_and_forces():
    """Displacements and forces are scaled in a type-1 dataset."""
    dataset = {
        "natom": 2,
        "first_atoms": [
            {
                "displacement": [1.0, 0.0, 0.0],
                "forces": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                "second_atoms": [
                    {
                        "displacement": [0.0, 1.0, 0.0],
                        "forces": [[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
                    }
                ],
            }
        ],
    }
    _convert_unit_in_dataset(dataset, distance_to_A=2.0, force_to_eVperA=0.5)
    d1 = dataset["first_atoms"][0]
    np.testing.assert_allclose(d1["displacement"], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(d1["forces"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    d2 = d1["second_atoms"][0]
    np.testing.assert_allclose(d2["displacement"], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(d2["forces"], [[0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])


def test_convert_unit_type1_displacement_only():
    """Only displacements are scaled when force_to_eVperA is None."""
    dataset = {
        "natom": 1,
        "first_atoms": [{"displacement": [1.0, 0.0, 0.0], "second_atoms": []}],
    }
    _convert_unit_in_dataset(dataset, distance_to_A=3.0, force_to_eVperA=None)
    np.testing.assert_allclose(
        dataset["first_atoms"][0]["displacement"], [3.0, 0.0, 0.0]
    )


def test_convert_unit_type2_displacement_and_forces():
    """Displacements and forces are scaled in a type-2 dataset."""
    dataset = {
        "displacements": np.ones((3, 2, 3)),
        "forces": np.ones((3, 2, 3)) * 2.0,
    }
    _convert_unit_in_dataset(dataset, distance_to_A=2.0, force_to_eVperA=0.5)
    np.testing.assert_allclose(dataset["displacements"], np.ones((3, 2, 3)) * 2.0)
    np.testing.assert_allclose(dataset["forces"], np.ones((3, 2, 3)) * 1.0)


def test_convert_unit_no_op():
    """No conversion is applied when both factors are None."""
    disp = np.array([[1.0, 0.0, 0.0]])
    dataset = {"displacements": disp.copy(), "forces": disp.copy() * 2}
    _convert_unit_in_dataset(dataset, distance_to_A=None, force_to_eVperA=None)
    np.testing.assert_allclose(dataset["displacements"], disp)
    np.testing.assert_allclose(dataset["forces"], disp * 2)


# ---------------------------------------------------------------------------
# parse_forces
# ---------------------------------------------------------------------------


def test_parse_forces_from_yaml_type1():
    """Forces are read from a phono3py yaml with an embedded type-1 dataset."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
    )
    assert "first_atoms" in dataset
    assert "forces" in dataset["first_atoms"][0]


def test_parse_forces_from_yaml_type2():
    """Forces are read from a phono3py yaml with an embedded type-2 dataset."""
    params_file = test_dir / "phono3py_params-Si111-rd.yaml.xz"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
    )
    assert "displacements" in dataset
    assert "forces" in dataset
    assert dataset["forces"].shape[0] == len(dataset["displacements"])


def test_parse_forces_from_forces_fc3_file(tmp_path, monkeypatch):
    """Forces are read from a FORCES_FC3 file when not embedded in yaml."""
    shutil.copy(test_dir / "FORCES_FC3_si_pbesol", tmp_path / "FORCES_FC3")

    p3yml = Phono3pyYaml()
    p3yml.read(test_dir / "phono3py_si_pbesol.yaml")
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    monkeypatch.chdir(tmp_path)
    dataset = parse_forces(ph3, ph3py_yaml=p3yml)
    assert "first_atoms" in dataset
    assert "forces" in dataset["first_atoms"][0]


def test_parse_forces_phonon_fc2():
    """Forces for phonon_fc2 are read from the yaml phonon_dataset."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    ph3 = Phono3py(
        p3yml.unitcell,
        p3yml.supercell_matrix,
        phonon_supercell_matrix=p3yml.phonon_supercell_matrix,
    )

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        fc_type="phonon_fc2",
    )
    assert "first_atoms" in dataset
    assert "forces" in dataset["first_atoms"][0]


def test_parse_forces_no_dataset_raises(tmp_path, monkeypatch):
    """RuntimeError is raised when neither yaml dataset nor force file is available."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Dataset is not found"):
        parse_forces(ph3, force_filename="nonexistent_file")


def test_parse_forces_cutoff_pair_distance():
    """cutoff_pair_distance is written into the returned dataset."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        cutoff_pair_distance=5.0,
    )
    assert dataset.get("cutoff_distance") == pytest.approx(5.0)


def test_parse_forces_cutoff_pair_distance_smaller_than_existing():
    """cutoff_pair_distance overwrites dataset cutoff_distance when smaller."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    p3yml.dataset["cutoff_distance"] = 8.0
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        cutoff_pair_distance=5.0,
    )
    assert dataset.get("cutoff_distance") == pytest.approx(5.0)


def test_parse_forces_cutoff_pair_distance_larger_than_existing():
    """Existing dataset cutoff_distance is kept when cutoff_pair_distance is larger."""
    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    p3yml.dataset["cutoff_distance"] = 4.0
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        cutoff_pair_distance=5.0,
    )
    assert dataset.get("cutoff_distance") == pytest.approx(4.0)


def test_parse_forces_cutoff_smaller_than_existing_with_calculator_qe():
    """cutoff_pair_distance overwrites existing cutoff_distance, then scaled by QE."""
    from phonopy.interface.calculator import get_calculator_physical_units

    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    p3yml.dataset["cutoff_distance"] = 8.0
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    units = get_calculator_physical_units("qe")
    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        cutoff_pair_distance=5.0,
        calculator="qe",
    )
    assert dataset.get("cutoff_distance") == pytest.approx(5.0 * units.distance_to_A)


def test_parse_forces_cutoff_larger_than_existing_with_calculator_qe():
    """Existing cutoff_distance is kept when larger, then scaled by QE units."""
    from phonopy.interface.calculator import get_calculator_physical_units

    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    p3yml.dataset["cutoff_distance"] = 4.0
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    units = get_calculator_physical_units("qe")
    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        cutoff_pair_distance=5.0,
        calculator="qe",
    )
    assert dataset.get("cutoff_distance") == pytest.approx(4.0 * units.distance_to_A)


def test_parse_forces_calculator_qe():
    """Displacements, forces, and cutoff_distance are converted from QE units."""
    from phonopy.interface.calculator import get_calculator_physical_units

    params_file = test_dir / "phono3py_params_Si-111-222.yaml"
    p3yml = Phono3pyYaml(settings={"force_sets": True})
    p3yml.read(params_file)
    ph3 = Phono3py(p3yml.unitcell, p3yml.supercell_matrix)

    orig_disp = p3yml.dataset["first_atoms"][0]["displacement"].copy()
    orig_forces = p3yml.dataset["first_atoms"][0]["forces"].copy()
    units = get_calculator_physical_units("qe")

    dataset = parse_forces(
        ph3,
        ph3py_yaml=p3yml,
        phono3py_yaml_filename=params_file,
        force_filename="nonexistent_file",
        cutoff_pair_distance=5.0,
        calculator="qe",
    )
    np.testing.assert_allclose(
        dataset["first_atoms"][0]["displacement"],
        orig_disp * units.distance_to_A,
    )
    np.testing.assert_allclose(
        dataset["first_atoms"][0]["forces"],
        orig_forces * units.force_to_eVperA,
    )
    assert dataset.get("cutoff_distance") == pytest.approx(5.0 * units.distance_to_A)
