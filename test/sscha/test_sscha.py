import numpy as np
from phono3py.sscha.sscha import DispCorrMatrix
from phonopy.phonon.qpoints import QpointsPhonon


si_pbesol_gamma0_0 = [[96.22804, 0, 0], [0, 96.22804, 0], [0, 0, 96.22804]]
si_pbesol_gamma1_34 = [[0.067368, -0.336319, -0.162847],
                       [-0.336319, 0.067368, -0.162847],
                       [-0.162847, -0.162847, -0.599353]]


def test_gamma_matrix(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    dynmat = si_pbesol.dynamical_matrix
    gmat = DispCorrMatrix(dynmat.primitive, dynmat.supercell)
    qpoints_phonon = QpointsPhonon(gmat.commensurate_points,
                                   dynmat,
                                   with_eigenvectors=True)
    freqs = qpoints_phonon.frequencies
    eigvecs = qpoints_phonon.eigenvectors
    gmat.create_gamma_matrix(freqs, eigvecs, 300.0)
    gmat.run()
    np.testing.assert_allclose(si_pbesol_gamma0_0,
                               gmat.gamma_matrix[0, 0],
                               atol=1e-5)
    np.testing.assert_allclose(si_pbesol_gamma1_34,
                               gmat.gamma_matrix[1, 34],
                               atol=1e-5)
    print(gmat.gamma_matrix[1, 2])
