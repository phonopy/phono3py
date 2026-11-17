"""Tests of kappa of inter-band transport variants with LBTE."""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py


@pytest.mark.parametrize("is_reducible_collision_matrix", [False, True])
@pytest.mark.parametrize("transport_type", ["NJC23", "IBDB19", "SMM19"])
def test_interband_lbte_kappa_is_total(
    si_pbesol: Phono3py, transport_type: str, is_reducible_collision_matrix: bool
):
    """Test that kappa of inter-band LBTE includes the inter-band kappa.

    The short boundary mean free path broadens the linewidths, which makes
    the inter-band kappa large enough to be detected.

    """
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction(
        exclude_gamma_acoustic=True, average_degenerate_weights=True
    )
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        boundary_mfp=0.001,
        transport_type=transport_type,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
    )
    tc = si_pbesol.thermal_conductivity
    assert tc.kappa_inter[0, 0, 0] > 0.1
    np.testing.assert_allclose(
        tc.kappa, tc.kappa_intra_exact + tc.kappa_inter, atol=1e-8
    )
