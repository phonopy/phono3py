"""Tests of Phono3py API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phonopy.interface.pypolymlp import PypolymlpParams

from phono3py import Phono3py

cwd = Path(__file__).parent


def test_displacements_setter_NaCl(nacl_pbe: Phono3py):
    """Test Phono3py.displacements setter.

    Just check no error.

    """
    ph3_in = nacl_pbe
    displacements = ph3_in.displacements
    ph3 = Phono3py(
        ph3_in.unitcell,
        supercell_matrix=ph3_in.supercell_matrix,
        primitive_matrix=ph3_in.primitive_matrix,
    )
    ph3.displacements = displacements


def test_displacements_setter_Si(si_pbesol_111_222_fd: Phono3py):
    """Test Phono3py.displacements setter and Phono3py.phonon_displacements setter.

    Just check no error.

    """
    ph3_in = si_pbesol_111_222_fd
    displacements = ph3_in.displacements
    phonon_displacements = ph3_in.phonon_displacements
    ph3 = Phono3py(
        ph3_in.unitcell,
        supercell_matrix=ph3_in.supercell_matrix,
        phonon_supercell_matrix=ph3_in.phonon_supercell_matrix,
        primitive_matrix=ph3_in.primitive_matrix,
    )
    ph3.displacements = displacements
    ph3.phonon_displacements = phonon_displacements


def test_type1_forces_energies_setter_Si(si_111_222_fd: Phono3py):
    """Test type1 supercell_energies, phonon_supercell_energies attributes."""
    ph3_in = si_111_222_fd
    ref_ph_supercell_energies = [-346.85204143]
    ref_supercell_energies = [
        -43.3509760,
        -43.33608775,
        -43.35352904,
        -43.34370672,
        -43.34590849,
        -43.34540162,
        -43.34421408,
        -43.34481089,
        -43.34703607,
        -43.34241924,
        -43.34786243,
        -43.34168203,
        -43.34274245,
        -43.34703607,
        -43.34786243,
        -43.34184454,
    ]
    np.testing.assert_allclose(ph3_in.supercell_energies, ref_supercell_energies)
    np.testing.assert_allclose(
        ph3_in.phonon_supercell_energies, ref_ph_supercell_energies
    )

    ref_force00 = [-0.4109520800000000, 0.0000000100000000, 0.0000000300000000]
    ref_force_last = [0.1521426300000000, 0.0715600600000000, -0.0715600700000000]
    ref_ph_force00 = [-0.4027479600000000, 0.0000000200000000, 0.0000001000000000]
    np.testing.assert_allclose(ph3_in.forces[0, 0], ref_force00)
    np.testing.assert_allclose(ph3_in.forces[-1, -1], ref_force_last)
    np.testing.assert_allclose(ph3_in.phonon_forces[0, 0], ref_ph_force00)

    ph3 = Phono3py(
        ph3_in.unitcell,
        supercell_matrix=ph3_in.supercell_matrix,
        phonon_supercell_matrix=ph3_in.phonon_supercell_matrix,
        primitive_matrix=ph3_in.primitive_matrix,
    )
    ph3.dataset = ph3_in.dataset
    ph3.phonon_dataset = ph3_in.phonon_dataset

    ph3.supercell_energies = ph3_in.supercell_energies + 1
    ph3.phonon_supercell_energies = ph3_in.phonon_supercell_energies + 1
    np.testing.assert_allclose(ph3_in.supercell_energies + 1, ph3.supercell_energies)
    np.testing.assert_allclose(
        ph3_in.phonon_supercell_energies + 1, ph3.phonon_supercell_energies
    )

    ph3.forces = ph3_in.forces + 1
    ph3.phonon_forces = ph3_in.phonon_forces + 1
    np.testing.assert_allclose(ph3_in.forces + 1, ph3.forces)
    np.testing.assert_allclose(ph3_in.phonon_forces + 1, ph3.phonon_forces)


def test_type2_forces_energies_setter_Si(si_111_222_rd: Phono3py):
    """Test type2 supercell_energies, phonon_supercell_energies attributes."""
    ph3_in = si_111_222_rd
    ref_ph_supercell_energies = [
        -346.81061270,  # 1
        -346.81263617,  # 2
    ]
    ref_supercell_energies = [
        -43.35270268,  # 1
        -43.35211687,  # 2
        -43.35122776,  # 3
        -43.35226673,  # 4
        -43.35146358,  # 5
        -43.35133209,  # 6
        -43.35042212,  # 7
        -43.35008442,  # 8
        -43.34968796,  # 9
        -43.35348999,  # 10
        -43.35134937,  # 11
        -43.35335251,  # 12
        -43.35160892,  # 13
        -43.35009115,  # 14
        -43.35202797,  # 15
        -43.35076370,  # 16
        -43.35174477,  # 17
        -43.35107001,  # 18
        -43.35037949,  # 19
        -43.35126123,  # 20
    ]
    np.testing.assert_allclose(ph3_in.supercell_energies, ref_supercell_energies)
    np.testing.assert_allclose(
        ph3_in.phonon_supercell_energies, ref_ph_supercell_energies
    )

    ref_force00 = [0.0445647800000000, 0.1702929900000000, 0.0913398200000000]
    ref_force_last = [-0.1749668700000000, 0.0146997300000000, -0.1336066300000000]
    ref_ph_force00 = [-0.0161598900000000, -0.1161657500000000, 0.1399128100000000]
    ref_ph_force_last = [0.1049486700000000, 0.0795870900000000, 0.1062164600000000]

    np.testing.assert_allclose(ph3_in.forces[0, 0], ref_force00)
    np.testing.assert_allclose(ph3_in.forces[-1, -1], ref_force_last)
    np.testing.assert_allclose(ph3_in.phonon_forces[0, 0], ref_ph_force00)
    np.testing.assert_allclose(ph3_in.phonon_forces[-1, -1], ref_ph_force_last)

    ph3 = Phono3py(
        ph3_in.unitcell,
        supercell_matrix=ph3_in.supercell_matrix,
        phonon_supercell_matrix=ph3_in.phonon_supercell_matrix,
        primitive_matrix=ph3_in.primitive_matrix,
    )
    ph3.dataset = ph3_in.dataset
    ph3.phonon_dataset = ph3_in.phonon_dataset
    ph3.supercell_energies = ph3_in.supercell_energies + 1
    ph3.phonon_supercell_energies = ph3_in.phonon_supercell_energies + 1

    np.testing.assert_allclose(ph3_in.supercell_energies + 1, ph3.supercell_energies)
    np.testing.assert_allclose(
        ph3_in.phonon_supercell_energies + 1, ph3.phonon_supercell_energies
    )

    ph3.forces = ph3_in.forces + 1
    ph3.phonon_forces = ph3_in.phonon_forces + 1
    np.testing.assert_allclose(ph3_in.forces + 1, ph3.forces)
    np.testing.assert_allclose(ph3_in.phonon_forces + 1, ph3.phonon_forces)


def test_use_pypolymlp_mgo(mgo_222rd_444rd: Phono3py):
    """Test use_pypolymlp in produce_fc3."""
    pytest.importorskip("pypolymlp")

    ph3_in = mgo_222rd_444rd
    ph3 = Phono3py(
        ph3_in.unitcell,
        supercell_matrix=ph3_in.supercell_matrix,
        phonon_supercell_matrix=ph3_in.phonon_supercell_matrix,
        primitive_matrix=ph3_in.primitive_matrix,
        log_level=2,
    )
    ph3.mlp_dataset = {
        "displacements": ph3_in.displacements[:10],
        "forces": ph3_in.forces[:10],
        "supercell_energies": ph3_in.supercell_energies[:10],
    }
    ph3.phonon_dataset = ph3_in.phonon_dataset
    ph3.nac_params = ph3_in.nac_params
    ph3.displacements = ph3_in.displacements[:100]

    atom_energies = {"Mg": -0.00896717, "O": -0.95743902}
    params = PypolymlpParams(gtinv_maxl=(4, 4), atom_energies=atom_energies)

    ph3.generate_displacements(distance=0.001, is_plusminus=True)
    ph3.develop_mlp(params=params)
    ph3.evaluate_mlp()
    ph3.produce_fc3(fc_calculator="symfc")
    ph3.produce_fc2(fc_calculator="symfc")

    ph3.mesh_numbers = 30
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300])
    assert (
        pytest.approx(63.0018546, abs=1e-1) == ph3.thermal_conductivity.kappa[0, 0, 0]
    )
