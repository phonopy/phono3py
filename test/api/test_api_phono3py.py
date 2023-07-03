"""Tests of Phono3py API."""
from pathlib import Path

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
