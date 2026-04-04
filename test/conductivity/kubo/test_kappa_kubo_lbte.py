"""Tests for Kubo-LBTE thermal conductivity."""

import numpy as np

from phono3py.api_phono3py import Phono3py


def test_kappa_kubo_lbte_si(si_pbesol: Phono3py):
    """Test Kubo-LBTE by Si."""
    ref_kappa_intra_exact = [110.846, 110.846, 110.846, 0, 0, 0]
    ref_kappa_inter = [0.083, 0.083, 0.083, 0, 0, 0]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        transport_type="kubo",
    )
    tc = si_pbesol.thermal_conductivity
    kappa_intra = tc.kappa_intra_exact.ravel()
    kappa_inter = tc.kappa_inter.ravel()
    np.testing.assert_allclose(ref_kappa_intra_exact, kappa_intra, atol=0.5)
    np.testing.assert_allclose(ref_kappa_inter, kappa_inter, atol=0.02)
