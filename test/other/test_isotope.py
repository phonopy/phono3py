import numpy as np
from phono3py import Phono3pyIsotope

si_pbesol_iso = [
    [8.32325038e-07, 9.45389739e-07, 1.57942189e-05, 1.28121297e-03,
     1.13842605e-03, 3.84915211e-04],
    [2.89457649e-05, 1.57841863e-04, 3.97462227e-04, 1.03489892e-02,
     4.45981554e-03, 2.67184355e-03]]
si_pbesol_iso_sigma = [
    [1.57262391e-06, 1.64031282e-06, 2.02007165e-05, 1.41999212e-03,
     1.26361419e-03, 7.91243161e-04],
    [3.10266472e-05, 1.53059329e-04, 3.80963936e-04, 1.05238031e-02,
     6.72552880e-03, 3.21592329e-03]]


def test_Phono3pyIsotope(si_pbesol):
    si_pbesol.mesh_numbers = [21, 21, 21]
    iso = Phono3pyIsotope(
        si_pbesol.mesh_numbers,
        si_pbesol.phonon_primitive,
        symprec=si_pbesol.symmetry.tolerance)
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params)
    iso.run([23, 103])
    # print(iso.gamma[0])
    np.testing.assert_allclose(si_pbesol_iso, iso.gamma[0], atol=2e-4)


def test_Phono3pyIsotope_with_sigma(si_pbesol):
    si_pbesol.mesh_numbers = [21, 21, 21]
    iso = Phono3pyIsotope(
        si_pbesol.mesh_numbers,
        si_pbesol.phonon_primitive,
        sigmas=[0.1, ],
        symprec=si_pbesol.symmetry.tolerance)
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params)
    iso.run([23, 103])
    # print(iso.gamma[0])
    np.testing.assert_allclose(si_pbesol_iso_sigma, iso.gamma[0], atol=2e-4)
