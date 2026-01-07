"""Tests of Phono3py API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phonopy.harmonic.force_constants import get_drift_force_constants
from phonopy.interface.pypolymlp import PypolymlpParams
from phonopy.structure.atoms import PhonopyAtoms

from phono3py import Phono3py
from phono3py.conductivity.rta import ConductivityRTA
from phono3py.phonon3.fc3 import get_drift_fc3

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


def test_supercell_displacements_AlN(aln_cell: PhonopyAtoms):
    """Test Phono3py.(phonon_)supercells_with_displacements.

    Just check no error.

    """
    ph3 = Phono3py(
        aln_cell, supercell_matrix=[2, 2, 2], phonon_supercell_matrix=[4, 4, 4]
    )
    ph3.generate_displacements()
    assert len(ph3.supercells_with_displacements) == 582
    assert len(ph3.phonon_supercells_with_displacements) == 6


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
    assert ph3_in.supercell_energies is not None
    np.testing.assert_allclose(ph3_in.supercell_energies, ref_supercell_energies)
    assert ph3_in.phonon_supercell_energies is not None
    np.testing.assert_allclose(
        ph3_in.phonon_supercell_energies, ref_ph_supercell_energies
    )

    ref_force00 = [-0.4109520800000000, 0.0000000100000000, 0.0000000300000000]
    ref_force_last = [0.1521426300000000, 0.0715600600000000, -0.0715600700000000]
    ref_ph_force00 = [-0.4027479600000000, 0.0000000200000000, 0.0000001000000000]
    assert ph3_in.forces is not None
    np.testing.assert_allclose(ph3_in.forces[0, 0], ref_force00)
    np.testing.assert_allclose(ph3_in.forces[-1, -1], ref_force_last)
    assert ph3_in.phonon_forces is not None
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
    assert ph3_in.supercell_energies is not None
    np.testing.assert_allclose(ph3_in.supercell_energies, ref_supercell_energies)
    assert ph3_in.phonon_supercell_energies is not None
    np.testing.assert_allclose(
        ph3_in.phonon_supercell_energies, ref_ph_supercell_energies
    )

    ref_force00 = [0.0445647800000000, 0.1702929900000000, 0.0913398200000000]
    ref_force_last = [-0.1749668700000000, 0.0146997300000000, -0.1336066300000000]
    ref_ph_force00 = [-0.0161598900000000, -0.1161657500000000, 0.1399128100000000]
    ref_ph_force_last = [0.1049486700000000, 0.0795870900000000, 0.1062164600000000]

    assert ph3_in.forces is not None
    np.testing.assert_allclose(ph3_in.forces[0, 0], ref_force00)
    np.testing.assert_allclose(ph3_in.forces[-1, -1], ref_force_last)
    assert ph3_in.phonon_forces is not None
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
    pytest.importorskip("symfc")

    ph3_in = mgo_222rd_444rd
    ph3 = Phono3py(
        ph3_in.unitcell,
        supercell_matrix=ph3_in.supercell_matrix,
        phonon_supercell_matrix=ph3_in.phonon_supercell_matrix,
        primitive_matrix=ph3_in.primitive_matrix,
        log_level=2,
    )
    assert ph3_in.forces is not None
    assert ph3_in.supercell_energies is not None
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
    assert ph3.thermal_conductivity is not None
    assert isinstance(ph3.thermal_conductivity, ConductivityRTA)
    assert ph3.thermal_conductivity.kappa is not None
    assert pytest.approx(63.547, abs=1e-1) == ph3.thermal_conductivity.kappa[0, 0, 0]


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_symmetrize_fc_traditional(si_pbesol: Phono3py, is_compact_fc: bool):
    """Test symmetrize_fc3 and symmetrize_fc2 with traditional approach."""
    ph3 = Phono3py(
        si_pbesol.unitcell,
        supercell_matrix=si_pbesol.supercell_matrix,
        primitive_matrix=si_pbesol.primitive_matrix,
        log_level=2,
    )
    ph3.dataset = si_pbesol.dataset
    ph3.produce_fc3(is_compact_fc=is_compact_fc)
    assert ph3.fc3 is not None
    assert ph3.fc2 is not None

    v1, v2, v3, _, _, _ = get_drift_fc3(ph3.fc3, primitive=ph3.primitive)
    np.testing.assert_allclose(
        np.abs([v1, v2, v3]), [1.755065e-01, 1.749287e-01, 3.333333e-05], atol=1e-6
    )
    ph3.symmetrize_fc3(options="level=3")
    v1_sym, v2_sym, v3_sym, _, _, _ = get_drift_fc3(ph3.fc3, primitive=ph3.primitive)
    if is_compact_fc:
        np.testing.assert_allclose(
            np.abs([v1_sym, v2_sym, v3_sym]), 1.217081e-05, atol=1e-6
        )
    else:
        np.testing.assert_allclose(
            np.abs([v1_sym, v2_sym, v3_sym]), 1.421085e-14, atol=1e-6
        )

    v1_sym, v2_sym, _, _ = get_drift_force_constants(ph3.fc2, primitive=ph3.primitive)
    if is_compact_fc:
        np.testing.assert_allclose(np.abs([v1_sym, v2_sym]), 1.0e-06, atol=1e-6)
    else:
        np.testing.assert_allclose(np.abs([v1_sym, v2_sym]), 1.0e-06, atol=1e-6)


@pytest.mark.parametrize("is_compact_fc", [True, False])
def test_symmetrize_fc_symfc(si_pbesol: Phono3py, is_compact_fc: bool):
    """Test symmetrize_fc3 and symmetrize_fc2 with symfc."""
    pytest.importorskip("symfc")

    ph3 = Phono3py(
        si_pbesol.unitcell,
        supercell_matrix=si_pbesol.supercell_matrix,
        primitive_matrix=si_pbesol.primitive_matrix,
        log_level=2,
    )
    ph3.dataset = si_pbesol.dataset
    ph3.produce_fc3(is_compact_fc=is_compact_fc)
    assert ph3.fc3 is not None
    assert ph3.fc2 is not None

    ph3.symmetrize_fc3(use_symfc_projector=True)
    v1_sym, v2_sym, v3_sym, _, _, _ = get_drift_fc3(ph3.fc3, primitive=ph3.primitive)
    np.testing.assert_allclose([v1_sym, v2_sym, v3_sym], 0, atol=1e-6)
    ph3.symmetrize_fc2(use_symfc_projector=True)
    v1_sym, v2_sym, _, _ = get_drift_force_constants(ph3.fc2, primitive=ph3.primitive)
    np.testing.assert_allclose([v1_sym, v2_sym], 0, atol=1e-6)
