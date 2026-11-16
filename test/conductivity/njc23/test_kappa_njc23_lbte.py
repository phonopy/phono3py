"""Tests for NJC23-LBTE thermal conductivity."""

import numpy as np

from phono3py.api_phono3py import Phono3py


def test_kappa_njc23_lbte_si_multi_temp(si_pbesol: Phono3py):
    """Test NJC23-LBTE with multiple temperatures by Si.

    Regression test for a bug where heat_capacity_matrix had wrong axis
    order (num_ir, num_temp, ...) instead of (num_temp, num_ir, ...),
    causing incorrect results when more than one temperature was given.

    Rather than hardcoding mesh-dependent reference kappa values, this test
    checks that the 300 K slice of a multi-temperature run matches a separate
    single-temperature run at 300 K. This catches the axis-order bug without
    depending on any reference value.

    """
    temperatures = [200, 300, 400]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=temperatures,
        transport_type="NJC23",
    )
    tc = si_pbesol.thermal_conductivity

    # The 300 K slice of the multi-temperature run must match a single
    # 300 K run, which guards against the heat_capacity_matrix axis-order bug.
    i_300 = temperatures.index(300)
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        transport_type="NJC23",
    )
    tc_single = si_pbesol.thermal_conductivity
    np.testing.assert_allclose(
        tc.kappa_intra_exact[0, i_300], tc_single.kappa_intra_exact[0, 0], atol=1e-5
    )
    np.testing.assert_allclose(
        tc.kappa_inter[0, i_300], tc_single.kappa_inter[0, 0], atol=1e-5
    )


def test_kappa_njc23_lbte_si_boundary_mfp(si_pbesol: Phono3py):
    """Test NJC23-LBTE with boundary scattering by Si.

    The inter-band kappa depends only on the linewidths of RTA, so it has
    to be the same between NJC23-LBTE and NJC23-RTA, including boundary
    scattering.

    """
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction(
        exclude_gamma_acoustic=True, average_degenerate_weights=True
    )
    kappa_inter = {}
    for is_LBTE in (False, True):
        si_pbesol.run_thermal_conductivity(
            is_LBTE=is_LBTE,
            temperatures=[300],
            boundary_mfp=0.001,
            transport_type="NJC23",
        )
        kappa_inter[is_LBTE] = si_pbesol.thermal_conductivity.kappa_inter[0, 0]
    np.testing.assert_allclose(kappa_inter[True], kappa_inter[False], atol=1e-5)
