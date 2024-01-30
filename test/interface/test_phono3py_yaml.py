"""Tests of PhonopyYaml."""

from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose
from phonopy.structure.dataset import get_displacements_and_forces

from phono3py import Phono3py
from phono3py.interface.phono3py_yaml import Phono3pyYaml, load_phono3py_yaml

cwd = Path(__file__).parent


def test_read_phono3py_yaml():
    """Test to parse phono3py.yaml like file."""
    filename = cwd / ".." / "phono3py_params_NaCl111.yaml"
    cell = _get_unitcell(filename)
    cell_ref = PhonopyAtoms(
        cell=[[5.64056, 0.0, 0.0], [0.0, 5.64056, 0.0], [0.0, 0.0, 5.64056]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        numbers=[11, 11, 11, 11, 17, 17, 17, 17],
    )
    assert isclose(cell, cell_ref)


def test_write_phono3py_yaml(nacl_pbe: Phono3py):
    """Test Phono3pyYaml.set_phonon_info, __str__, yaml_data, parse."""
    phonon3 = nacl_pbe
    ph3py_yaml = Phono3pyYaml(calculator="vasp")
    ph3py_yaml.set_phonon_info(phonon3)
    ph3py_yaml_test = Phono3pyYaml()
    ph3py_yaml_test._data = load_phono3py_yaml(
        yaml.safe_load(StringIO(str(ph3py_yaml))), calculator=ph3py_yaml.calculator
    )
    assert isclose(ph3py_yaml.primitive, ph3py_yaml_test.primitive)
    assert isclose(ph3py_yaml.unitcell, ph3py_yaml_test.unitcell)
    assert isclose(ph3py_yaml.supercell, ph3py_yaml_test.supercell)
    assert ph3py_yaml.version == ph3py_yaml_test.version
    np.testing.assert_allclose(
        ph3py_yaml.supercell_matrix, ph3py_yaml_test.supercell_matrix, atol=1e-8
    )
    np.testing.assert_allclose(
        ph3py_yaml.primitive_matrix, ph3py_yaml_test.primitive_matrix, atol=1e-8
    )


def test_write_phono3py_yaml_extra_NaCl222(nacl_pbe: Phono3py):
    """Test Phono3pyYaml.set_phonon_info, __str__, yaml_data, parse.

    settings parameter controls amount of yaml output. In this test,
    more data than the default are dumped and those are tested.

    """
    _test_write_phono3py_yaml_extra(nacl_pbe)


def test_write_phono3py_yaml_extra_Si111_222(si_pbesol_111_222_fd: Phono3py):
    """Test Phono3pyYaml.set_phonon_info, __str__, yaml_data, parse.

    settings parameter controls amount of yaml output. In this test,
    more data than the default are dumped and those are tested.

    """
    _test_write_phono3py_yaml_extra(si_pbesol_111_222_fd)


def _test_write_phono3py_yaml_extra(phonon3: Phono3py):
    settings = {
        "force_sets": True,
        "displacements": True,
        "force_constants": True,
        "born_effective_charge": True,
        "dielectric_constant": True,
    }
    ph3py_yaml = Phono3pyYaml(calculator="vasp", settings=settings)
    ph3py_yaml.set_phonon_info(phonon3)
    ph3py_yaml_test = Phono3pyYaml()
    ph3py_yaml_test._data = load_phono3py_yaml(
        yaml.safe_load(StringIO(str(ph3py_yaml))), calculator=ph3py_yaml.calculator
    )
    np.testing.assert_allclose(
        ph3py_yaml.force_constants, ph3py_yaml_test.force_constants, atol=1e-8
    )

    if ph3py_yaml.nac_params is not None:
        np.testing.assert_allclose(
            ph3py_yaml.nac_params["born"], ph3py_yaml_test.nac_params["born"], atol=1e-8
        )
        np.testing.assert_allclose(
            ph3py_yaml.nac_params["dielectric"],
            ph3py_yaml_test.nac_params["dielectric"],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            ph3py_yaml.nac_params["factor"],
            ph3py_yaml_test.nac_params["factor"],
            atol=1e-8,
        )

    disps, forces = get_displacements_and_forces(ph3py_yaml.dataset)
    disps_test, forces_test = get_displacements_and_forces(ph3py_yaml_test.dataset)
    np.testing.assert_allclose(forces, forces_test, atol=1e-8)
    np.testing.assert_allclose(disps, disps_test, atol=1e-8)

    if ph3py_yaml.phonon_dataset is not None:
        disps, forces = get_displacements_and_forces(ph3py_yaml.phonon_dataset)
        disps_test, forces_test = get_displacements_and_forces(
            ph3py_yaml_test.phonon_dataset
        )
        np.testing.assert_allclose(forces, forces_test, atol=1e-8)
        np.testing.assert_allclose(disps, disps_test, atol=1e-8)


def _get_unitcell(filename):
    phpy_yaml = Phono3pyYaml().read(filename)
    return phpy_yaml.unitcell
