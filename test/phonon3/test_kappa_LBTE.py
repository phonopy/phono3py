"""Tests for direct solution of LBTE."""
import numpy as np

from phono3py.api_phono3py import Phono3py

si_pbesol_kappa_LBTE = [111.117, 111.117, 111.117, 0, 0, 0]
si_pbesol_kappa_LBTE_redcol = [63.019, 63.019, 63.019, 0, 0, 0]


def test_kappa_LBTE(si_pbesol: Phono3py):
    """Test for symmetry reduced collision matrix."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
    )
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(si_pbesol_kappa_LBTE, kappa, atol=0.5)


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
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(si_pbesol_kappa_LBTE_redcol, kappa, atol=0.5)
