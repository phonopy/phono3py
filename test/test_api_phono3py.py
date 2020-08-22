import numpy as np

si_pbesol_kappa = [107.991, 107.991, 107.991, 0, 0, 0]


def test_Phono3py(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(temperatures=[300, ])
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(si_pbesol_kappa, kappa, atol=0.1)
