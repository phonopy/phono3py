"""Tests for direct solution of LBTE."""

import numpy as np
import phono3py._phono3py as phono3c
import pytest

from phono3py.api_phono3py import Phono3py


@pytest.mark.skipif(
    not phono3c.include_lapacke(), reason="test for phono3py compiled with lapacke"
)
@pytest.mark.parametrize("pinv_solver", [1, 2, 6])
def test_kappa_LBTE_126(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    _test_kappa_LBTE(si_pbesol, pinv_solver)


@pytest.mark.parametrize("pinv_solver", [3, 4, 5, 7])
def test_kappa_LBTE_3457(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    _test_kappa_LBTE(si_pbesol, pinv_solver)


def _test_kappa_LBTE(si_pbesol: Phono3py, pinv_solver: int):
    ref_kappa = [110.896, 110.896, 110.896, 0, 0, 0]
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
    phono3c.include_lapacke(), reason="test for phono3py compiled without lapacke"
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
    ref_kappa = [62.497, 62.497, 62.497, 0, 0, 0]
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


def test_kappa_LBTE_no_kappa_star(si_pbesol: Phono3py):
    """Test LBTE with is_kappa_star=False iterates all grid points."""
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[1000],
        is_kappa_star=True,
    )
    kappa_star = si_pbesol.thermal_conductivity.kappa.ravel()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[1000],
        is_kappa_star=False,
    )
    kappa_nostar = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(kappa_star[:3], kappa_nostar[:3], atol=0.5)


def test_kappa_LBTE_aln(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    ref_kappa = [234.141, 234.141, 254.006, 0, 0, 0]
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
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.5)


def test_kappa_LBTE_aln_with_sigma(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    ref_kappa = [254.111, 254.111, 271.406, 0, 0, 0]
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


def test_kappa_LBTE_read_collision_per_gp(si_pbesol: Phono3py, tmp_path, monkeypatch):
    """Write per-GP collision files, then --read-collision round-trip.

    Exercises the fallback to per-GP files (``collision-m*-g*.hdf5``) and
    the gv/cv/isotope recomputation path in ``set_kappa_at_sigmas``.

    """
    monkeypatch.chdir(tmp_path)
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()

    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        write_collision=True,
    )
    ref_kappa = si_pbesol.thermal_conductivity.kappa.ravel().copy()
    assert list(tmp_path.glob("collision-m555-g*.hdf5"))

    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        read_collision="all",
    )
    read_kappa = si_pbesol.thermal_conductivity.kappa.ravel()

    np.testing.assert_allclose(ref_kappa, read_kappa, atol=1e-3)
