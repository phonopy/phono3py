import numpy as np

si_pbesol_kappa_RTA = [107.991, 107.991, 107.991, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas = [109.6985, 109.6985, 109.6985, 0, 0, 0]


def test_kappa_RTA(si_pbesol):
    kappa = _get_kappa(si_pbesol, [9, 9, 9]).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_with_sigma(si_pbesol):
    si_pbesol.sigmas = [0.1, ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9]).ravel()
    np.testing.assert_allclose(
        si_pbesol_kappa_RTA_with_sigmas, kappa, atol=0.5)


def test_kappa_RTA_compact_fc(si_pbesol_compact_fc):
    kappa = _get_kappa(si_pbesol_compact_fc, [9, 9, 9]).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def _get_kappa(ph3, mesh):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300, ])
    return ph3.thermal_conductivity.kappa
