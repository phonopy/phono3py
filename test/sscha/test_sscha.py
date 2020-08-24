import numpy as np
from phono3py.sscha.sscha import (
    DispCorrMatrix, DispCorrMatrixMesh, SupercellPhonon, LambdaTensor)
from phonopy.phonon.qpoints import QpointsPhonon


si_pbesol_gamma0_0 = [[3.849187e+02, 0, 0],
                      [0, 3.849187e+02, 0],
                      [0, 0, 3.849187e+02]]
si_pbesol_gamma1_34 = [[1.886404, -1.549705, -1.126055],
                       [-1.549705, 1.886404, -1.126055],
                       [-1.126055, -1.126055, -0.006187]]
si_pbesol_111_freqs = [
    0.00000, 0.00000, 0.00000, 4.02839, 4.02839, 4.02839,
    4.02839, 4.02839, 4.02839, 12.13724, 12.13724, 12.13724,
    12.13724, 12.13724, 12.13724, 13.71746, 13.71746, 13.71746,
    13.71746, 13.71746, 13.71746, 15.24974, 15.24974, 15.24974]


def get_supercell_phonon(ph3):
    ph3.mesh_numbers = [1, 1, 1]
    ph3.init_phph_interaction()
    fc2 = ph3.dynamical_matrix.force_constants
    supercell = ph3.phonon_supercell
    factor = ph3.unit_conversion_factor
    return SupercellPhonon(supercell, fc2, factor=factor)


# def test_run_Lambda(si_pbesol_111):
#     lt = LambdaTensor()

def test_SupercellPhonon(si_pbesol_111):
    sph = get_supercell_phonon(si_pbesol_111)
    np.testing.assert_allclose(
        si_pbesol_111_freqs, sph.frequencies, atol=1e-4)


def test_gamma_matrix_mesh(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    dynmat = si_pbesol.dynamical_matrix
    gmat = DispCorrMatrixMesh(dynmat.primitive, dynmat.supercell)
    qpoints_phonon = QpointsPhonon(gmat.commensurate_points,
                                   dynmat,
                                   with_eigenvectors=True)
    freqs = qpoints_phonon.frequencies
    eigvecs = qpoints_phonon.eigenvectors
    gmat.create_gamma_matrix(freqs, eigvecs, 300.0)
    gmat.run()
    np.testing.assert_allclose(
        si_pbesol_gamma0_0, gmat.gamma_matrix[0, 0], atol=1e-4)
    np.testing.assert_allclose(
        si_pbesol_gamma1_34, gmat.gamma_matrix[1, 34], atol=1e-4)


def test_gamma_matrix(si_pbesol):
    supercell_phonon = get_supercell_phonon(si_pbesol)
    gmat = DispCorrMatrix(supercell_phonon)
    gmat.run(300.0)
    np.testing.assert_allclose(
        si_pbesol_gamma0_0, gmat.gamma_matrix[0:3, 0:3], atol=1e-4)
    np.testing.assert_allclose(
        si_pbesol_gamma1_34, gmat.gamma_matrix[1 * 3: 2 * 3, 34 * 3: 35 * 3],
        atol=1e-4)
