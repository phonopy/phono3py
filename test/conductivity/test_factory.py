"""Tests of the conductivity calculator factory."""

from __future__ import annotations

import pytest

from phono3py import Phono3py


@pytest.mark.parametrize("is_LBTE", [False, True])
@pytest.mark.parametrize("transport_type", ["SMM19", "smm19"])
def test_removed_smm19(si_pbesol: Phono3py, transport_type: str, is_LBTE: bool):
    """Test that the removed SMM19 transport type points to phono3py-wte."""
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    with pytest.raises(NotImplementedError, match="phono3py-wte"):
        si_pbesol.run_thermal_conductivity(
            is_LBTE=is_LBTE, temperatures=[300], transport_type=transport_type
        )
