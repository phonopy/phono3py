"""Tests for direct solution of LBTE."""

import numpy as np
import pytest

import phono3py._phono3py as phono3c
from phono3py.api_phono3py import Phono3py


@pytest.mark.skipif(
    not phono3c.include_lapacke(), reason="test for phono3py compliled with lapacke"
)
@pytest.mark.parametrize("pinv_solver", [1, 2])
def test_kappa_LBTE_12(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    _test_kappa_LBTE(si_pbesol, pinv_solver)


@pytest.mark.parametrize("pinv_solver", [3, 4, 5])
def test_kappa_LBTE_345(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    _test_kappa_LBTE(si_pbesol, pinv_solver)


def _test_kappa_LBTE(si_pbesol: Phono3py, pinv_solver: int):
    if si_pbesol._make_r0_average:
        ref_kappa = [110.896, 110.896, 110.896, 0, 0, 0]
    else:
        ref_kappa = [111.149, 111.149, 111.149, 0, 0, 0]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
        pinv_solver=pinv_solver,
    )
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.3)


@pytest.mark.skipif(
    phono3c.include_lapacke(), reason="test for phono3py compliled without lapacke"
)
@pytest.mark.parametrize("pinv_solver", [1, 2])
def test_kappa_LBTE_witout_lapacke(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    with pytest.raises(RuntimeError):
        si_pbesol.run_thermal_conductivity(
            is_LBTE=True,
            temperatures=[
                300,
            ],
            pinv_solver=pinv_solver,
        )


def test_kappa_LBTE_full_colmat(si_pbesol: Phono3py):
    """Test for full collision matrix."""
    if si_pbesol._make_r0_average:
        ref_kappa = [62.497, 62.497, 62.497, 0, 0, 0]
    else:
        ref_kappa = [62.777, 62.777, 62.777, 0, 0, 0]

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
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.5)


def test_kappa_LBTE_aln(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    if aln_lda._make_r0_average:
        ref_kappa = [234.141, 234.141, 254.006, 0, 0, 0]
    else:
        ref_kappa = [231.191, 231.191, 248.367, 0, 0, 0]

    aln_lda.mesh_numbers = [7, 7, 5]
    aln_lda.init_phph_interaction()
    aln_lda.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
    )
    kappa = aln_lda.thermal_conductivity.kappa.ravel()
    # print(", ".join([f"{k:e}" for k in kappa]))
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.3)


def test_kappa_LBTE_aln_with_sigma(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    if aln_lda._make_r0_average:
        ref_kappa = [254.111, 254.111, 271.406, 0, 0, 0]
    else:
        ref_kappa = [250.030, 250.030, 269.405, 0, 0, 0]
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    aln_lda.mesh_numbers = [7, 7, 5]
    aln_lda.init_phph_interaction()
    aln_lda.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
    )
    kappa = aln_lda.thermal_conductivity.kappa.ravel()
    # np.testing.assert_allclose(aln_lda_kappa_RTA_with_sigmas, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None
    # print(", ".join([f"{k:e}" for k in kappa]))
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.3)
