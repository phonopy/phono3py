"""Tests for direct solution of LBTE."""

import numpy as np
import pytest

from phono3py.api_phono3py import Phono3py


def test_kappa_LBTE(si_pbesol: Phono3py):
    """Test for symmetry reduced collision matrix."""
    if si_pbesol._make_r0_average:
        ref_kappa_P_LBTE = [110.896, 110.896, 110.896, 0, 0, 0]
        ref_kappa_C = [0.166, 0.166, 0.166, 0.000, 0.000, 0.000]
    else:
        ref_kappa_P_LBTE = [111.149, 111.149, 111.149, 0, 0, 0]
        ref_kappa_C = [0.166, 0.166, 0.166, 0.000, 0.000, 0.000]

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
    np.testing.assert_allclose(ref_kappa_P_LBTE, kappa_P, atol=0.5)
    np.testing.assert_allclose(ref_kappa_C, kappa_C, atol=0.02)


@pytest.mark.skip(
    reason=(
        "coherences conductivity is not implemented for "
        "is_reducible_collision_matrix=True"
    )
)
def test_kappa_LBTE_full_colmat(si_pbesol: Phono3py):
    """Test for full collision matrix."""
    si_pbesol_kappa_P_LBTE_redcol = [62.783, 62.783, 62.783, 0, 0, 0]
    si_pbesol_kappa_C_redcol = -1
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
