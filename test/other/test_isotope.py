import numpy as np
from phono3py import Phono3pyIsotope

si_pbesol_iso = [
    [5.91671111e-07, 5.58224287e-07, 6.43147255e-05, 1.55630249e-03,
     3.94732090e-04, 4.71724676e-04],
    [2.85604812e-04, 3.11142983e-04, 2.27454364e-04, 9.63215998e-04,
     1.03882853e-02, 1.28926668e-02]]
si_pbesol_iso_sigma = [
    [4.97062464e-06, 4.97062464e-06, 1.16928959e-04, 2.08391870e-03,
     1.22859101e-03, 1.22859101e-03],
    [5.54968007e-04, 3.82314550e-04, 4.79051030e-04, 2.85301149e-03,
     9.62156392e-03, 1.33646381e-02]]

def test_Phono3pyIsotope(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    iso = Phono3pyIsotope(
        si_pbesol.mesh_numbers,
        si_pbesol.phonon_primitive,
        symprec=si_pbesol.symmetry.tolerance)
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params)
    iso.run([1, 103])
    np.testing.assert_allclose(si_pbesol_iso, iso.gamma[0], atol=1e-3)


def test_Phono3pyIsotope_with_sigma(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
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
    iso.run([1, 103])
    np.testing.assert_allclose(si_pbesol_iso_sigma, iso.gamma[0], atol=1e-3)
