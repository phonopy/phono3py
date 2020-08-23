import numpy as np
from phono3py import Phono3pyIsotope

si_pbesol_kappa_RTA = [107.991, 107.991, 107.991, 0, 0, 0]
si_pbesol_kappa_LBTE = [111.802, 111.802, 111.802, 0, 0, 0]

si_pbesol_iso = [
    [5.91671111e-07, 5.58224287e-07, 6.43147255e-05, 1.55630249e-03,
     3.94732090e-04, 4.71724676e-04],
    [2.85604812e-04, 3.11142983e-04, 2.27454364e-04, 9.63215998e-04,
     1.03882853e-02, 1.28926668e-02]]

si_pbesol_Delta = [
    [-0.0057666, -0.0057666, -0.01639729, -0.14809965,
     -0.15091765, -0.15091765],
    [-0.02078728, -0.02102094, -0.06573269, -0.11432603,
     -0.1366966, -0.14371315]]


def test_kappa_RTA(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(temperatures=[300, ])
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_LBTE(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(is_LBTE=True, temperatures=[300, ])
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(si_pbesol_kappa_LBTE, kappa, atol=0.5)


def test_frequency_shift(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    delta = si_pbesol.run_frequency_shift(
        [1, 103],
        temperatures=[300, ],
        write_Delta_hdf5=False)
    np.testing.assert_allclose(si_pbesol_Delta, delta[0, :, 0], atol=1e-5)


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
