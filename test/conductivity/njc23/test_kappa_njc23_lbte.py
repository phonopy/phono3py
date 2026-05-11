"""Tests for NJC23-LBTE thermal conductivity."""

import numpy as np

from phono3py.api_phono3py import Phono3py


def test_kappa_njc23_lbte_si(si_pbesol: Phono3py):
    """Test NJC23-LBTE by Si."""
    ref_kappa_intra_exact = [110.846, 110.846, 110.846, 0, 0, 0]
    ref_kappa_inter = [0.518, 0.518, 0.518, 0, 0, 0]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        transport_type="NJC23",
    )
    tc = si_pbesol.thermal_conductivity
    kappa_intra = tc.kappa_intra_exact.ravel()
    kappa_inter = tc.kappa_inter.ravel()
    np.testing.assert_allclose(ref_kappa_intra_exact, kappa_intra, atol=0.5)
    np.testing.assert_allclose(ref_kappa_inter, kappa_inter, atol=0.02)


def test_kappa_njc23_lbte_si_multi_temp(si_pbesol: Phono3py):
    """Test NJC23-LBTE with multiple temperatures by Si.

    Regression test for a bug where heat_capacity_matrix had wrong axis
    order (num_ir, num_temp, ...) instead of (num_temp, num_ir, ...),
    causing incorrect results when more than one temperature was given.

    """
    # Reference values: kappa_intra_exact and kappa_inter at 200, 300, 400 K
    ref_kappa_intra_exact = [
        [193.1, 193.1, 193.1, 0, 0, 0],
        [110.8, 110.8, 110.8, 0, 0, 0],
        [79.0, 79.0, 79.0, 0, 0, 0],
    ]
    ref_kappa_inter = [
        [0.792, 0.792, 0.792, 0, 0, 0],
        [0.518, 0.518, 0.518, 0, 0, 0],
        [0.431, 0.431, 0.431, 0, 0, 0],
    ]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[200, 300, 400],
        transport_type="NJC23",
    )
    tc = si_pbesol.thermal_conductivity

    # Check each temperature independently
    for i_temp in range(3):
        kappa_intra = tc.kappa_intra_exact[0, i_temp]
        kappa_inter = tc.kappa_inter[0, i_temp]
        np.testing.assert_allclose(ref_kappa_intra_exact[i_temp], kappa_intra, atol=0.5)
        np.testing.assert_allclose(ref_kappa_inter[i_temp], kappa_inter, atol=0.02)

    # Verify 300 K slice matches the single-temperature test
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        transport_type="NJC23",
    )
    tc_single = si_pbesol.thermal_conductivity
    np.testing.assert_allclose(
        tc.kappa_intra_exact[0, 1], tc_single.kappa_intra_exact[0, 0], atol=1e-5
    )
    np.testing.assert_allclose(
        tc.kappa_inter[0, 1], tc_single.kappa_inter[0, 0], atol=1e-5
    )
