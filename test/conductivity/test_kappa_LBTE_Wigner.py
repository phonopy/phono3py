"""Tests for direct solution of LBTE."""
import numpy as np

from phono3py.api_phono3py import Phono3py

si_pbesol_kappa_P_LBTE = [111.123, 111.123, 111.123, 0, 0, 0]  # old value 111.117
si_pbesol_kappa_C = [0.167, 0.167, 0.167, 0.000, 0.000, 0.000]

si_pbesol_kappa_P_LBTE_redcol = [62.783, 62.783, 62.783, 0, 0, 0]  # old value 63.019
si_pbesol_kappa_C_redcol = (
    -1
)  # coherences conductivity is not implemented for is_reducible_collision_matrix=True,


def test_kappa_LBTE(si_pbesol: Phono3py):
    """Test for symmetry reduced collision matrix."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
        conductivity_type="wigner",
    )
    # kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    kappa_P = si_pbesol.thermal_conductivity.kappa_P_exact.ravel()
    kappa_C = si_pbesol.thermal_conductivity.kappa_C.ravel()
    np.testing.assert_allclose(si_pbesol_kappa_P_LBTE, kappa_P, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C, kappa_C, atol=0.02)


'''
#coherences conductivity is not implemented for is_reducible_collision_matrix=True,
def test_kappa_LBTE_full_colmat(si_pbesol: Phono3py):
    """Test for full collision matrix."""
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
        is_reducible_collision_matrix=True,
    )
    kappa_P = si_pbesol.thermal_conductivity.kappa_P_exact.ravel()
    kappa_C = si_pbesol.thermal_conductivity.kappa_C.ravel()
    np.testing.assert_allclose(si_pbesol_kappa_P_LBTE_redcol, kappa_P, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C_redcol, kappa_C, atol=0.02)
'''
