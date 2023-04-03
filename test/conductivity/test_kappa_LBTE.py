"""Tests for direct solution of LBTE."""
import numpy as np

from phono3py.api_phono3py import Phono3py

si_pbesol_kappa_LBTE = [111.117, 111.117, 111.117, 0, 0, 0]
si_pbesol_kappa_LBTE_redcol = [63.019, 63.019, 63.019, 0, 0, 0]
aln_lda_kappa_LBTE = [2.313066e02, 2.313066e02, 2.483627e02, 0, 0, 0]
aln_lda_kappa_LBTE_with_sigma = [2.500303e02, 2.500303e02, 2.694047e02, 0, 0, 0]
aln_lda_kappa_LBTE_with_r0_ave = [2.342499e02, 2.342499e02, 2.540009e02, 0, 0, 0]


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


def test_kappa_LBTE_aln(aln_lda: Phono3py):
    """Test direct solution by AlN."""
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
    np.testing.assert_allclose(aln_lda_kappa_LBTE, kappa, atol=0.5)


def test_kappa_LBTE_aln_with_r0_ave(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    aln_lda.mesh_numbers = [7, 7, 5]
    aln_lda.init_phph_interaction(make_r0_average=True)
    aln_lda.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
    )
    kappa = aln_lda.thermal_conductivity.kappa.ravel()
    # print(", ".join([f"{k:e}" for k in kappa]))
    np.testing.assert_allclose(aln_lda_kappa_LBTE_with_r0_ave, kappa, atol=0.5)


def test_kappa_LBTE_aln_with_sigma(aln_lda: Phono3py):
    """Test direct solution by AlN."""
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
    np.testing.assert_allclose(aln_lda_kappa_LBTE_with_sigma, kappa, atol=0.5)
