"""Test for Conductivity_RTA.py."""

import itertools

import numpy as np
import pytest

from phono3py import Phono3py


@pytest.mark.parametrize(
    "openmp_per_triplets,is_full_pp,is_compact_fc",
    itertools.product([False, True], [False, True], [False, True]),
)
def test_kappa_RTA_si(
    si_pbesol: Phono3py,
    si_pbesol_compact_fc: Phono3py,
    openmp_per_triplets: bool,
    is_full_pp: bool,
    is_compact_fc: bool,
):
    """Test RTA by Si."""
    if is_compact_fc:
        ph3 = si_pbesol_compact_fc
    else:
        ph3 = si_pbesol
    if is_compact_fc:
        ref_kappa_RTA = [107.794, 107.794, 107.794, 0, 0, 0]
    else:
        ref_kappa_RTA = [107.694, 107.694, 107.694, 0, 0, 0]

    kappa = _get_kappa(
        ph3,
        [9, 9, 9],
        is_full_pp=is_full_pp,
        openmp_per_triplets=openmp_per_triplets,
    ).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_iso(si_pbesol: Phono3py):
    """Test RTA with isotope scattering by Si."""
    ref_kappa_RTA_iso = [97.296, 97.296, 97.296, 0, 0, 0]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_iso, kappa, atol=0.5)


@pytest.mark.parametrize(
    "openmp_per_triplets,is_full_pp", itertools.product([False, True], [False, True])
)
def test_kappa_RTA_si_with_sigma(
    si_pbesol: Phono3py, openmp_per_triplets: bool, is_full_pp: bool
):
    """Test RTA with smearing method by Si."""
    ref_kappa_RTA_with_sigmas = [109.999, 109.999, 109.999, 0, 0, 0]
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa = _get_kappa(
        si_pbesol,
        [9, 9, 9],
        is_full_pp=is_full_pp,
        openmp_per_triplets=openmp_per_triplets,
    ).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_with_sigmas, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_iso(si_pbesol: Phono3py):
    """Test RTA with smearing method and isotope scattering by Si."""
    ref_kappa_RTA_with_sigmas_iso = [96.368, 96.368, 96.368, 0, 0, 0]
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_with_sigmas_iso, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_nosym(si_pbesol: Phono3py, si_pbesol_nosym: Phono3py):
    """Test RTA without considering symmetry by Si."""
    ref_kappa_RTA_si_nosym = [38.315, 38.616, 39.093, 0.221, 0.166, 0.284]
    si_pbesol_nosym.fc2 = si_pbesol.fc2
    si_pbesol_nosym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nosym, [4, 4, 4]).reshape(-1, 3).sum(axis=1)
    kappa_ref = np.reshape(ref_kappa_RTA_si_nosym, (-1, 3)).sum(axis=1)
    np.testing.assert_allclose(kappa_ref / 3, kappa / 3, atol=0.5)


def test_kappa_RTA_si_nomeshsym(si_pbesol: Phono3py, si_pbesol_nomeshsym: Phono3py):
    """Test RTA without considering mesh symmetry by Si."""
    ref_kappa_RTA_si_nomeshsym = [81.147, 81.147, 81.147, 0.000, 0.000, 0.000]
    si_pbesol_nomeshsym.fc2 = si_pbesol.fc2
    si_pbesol_nomeshsym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nomeshsym, [7, 7, 7]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_si_nomeshsym, kappa, atol=0.5)


def test_kappa_RTA_si_grg(si_pbesol_grg: Phono3py):
    """Test RTA by Si with GR-grid."""
    ref_kappa_RTA_grg = [111.204, 111.204, 111.204, 0, 0, 0]
    mesh = 30
    ph3 = si_pbesol_grg
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
    )
    kappa = ph3.thermal_conductivity.kappa.ravel()
    np.testing.assert_equal(
        ph3.thermal_conductivity.bz_grid.grid_matrix,
        [[-6, 6, 6], [6, -6, 6], [6, 6, -6]],
    )
    np.testing.assert_equal(
        ph3.grid.grid_matrix,
        [[-6, 6, 6], [6, -6, 6], [6, 6, -6]],
    )
    A = ph3.grid.grid_matrix
    D_diag = ph3.grid.D_diag
    P = ph3.grid.P
    Q = ph3.grid.Q
    np.testing.assert_equal(np.dot(P, np.dot(A, Q)), np.diag(D_diag))

    np.testing.assert_allclose(ref_kappa_RTA_grg, kappa, atol=0.5)


def test_kappa_RTA_si_grg_iso(si_pbesol_grg: Phono3py):
    """Test RTA with isotope scattering by Si with GR-grid.."""
    ref_kappa_RTA_grg_iso = [104.290, 104.290, 104.290, 0, 0, 0]
    mesh = 30
    ph3 = si_pbesol_grg
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=True,
    )
    kappa = ph3.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(ref_kappa_RTA_grg_iso, kappa, atol=0.5)
    np.testing.assert_equal(ph3.grid.grid_matrix, [[-6, 6, 6], [6, -6, 6], [6, 6, -6]])


def test_kappa_RTA_si_grg_sigma_iso(si_pbesol_grg: Phono3py):
    """Test RTA with isotope scattering by Si with GR-grid.."""
    ref_kappa_RTA_grg_sigma_iso = [107.264, 107.264, 107.264, 0, 0, 0]
    mesh = 30
    ph3 = si_pbesol_grg
    ph3.sigmas = [
        0.1,
    ]
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=True,
    )
    kappa = ph3.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(ref_kappa_RTA_grg_sigma_iso, kappa, atol=0.5)
    ph3.sigmas = None


def test_kappa_RTA_si_N_U(si_pbesol):
    """Test RTA with N and U scatterings by Si."""
    ph3 = si_pbesol
    mesh = [4, 4, 4]
    is_N_U = True
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_N_U=is_N_U,
    )
    gN, gU = ph3.thermal_conductivity.get_gamma_N_U()
    # for line in gN.reshape(-1, 6):
    #     print("[", ",".join([f"{val:.8f}" for val in line]), "],")
    # for line in gU.reshape(-1, 6):
    #     print("[", ",".join([f"{val:.8f}" for val in line]), "],")

    gN_ref = [
        [0.00000000, 0.00000000, 0.00000000, 0.07898606, 0.07898606, 0.07898606],
        [0.00079647, 0.00079647, 0.00913611, 0.01911102, 0.04553001, 0.04553001],
        [0.00173868, 0.00173868, 0.01404937, 0.00201732, 0.03354033, 0.03354033],
        [0.00223616, 0.00223616, 0.01039331, 0.02860916, 0.02860916, 0.01987485],
        [0.00291788, 0.00356241, 0.02858543, 0.00367742, 0.02065990, 0.01533763],
        [0.00146333, 0.00343175, 0.01596851, 0.00626596, 0.02431620, 0.01091592],
        [0.00396766, 0.00396766, 0.00159161, 0.00159161, 0.01479018, 0.01479018],
        [0.00682740, 0.00682740, 0.03983399, 0.03983399, 0.02728522, 0.02728522],
    ]
    gU_ref = [
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        [0.00015184, 0.00015184, 0.00075965, 0.00736940, 0.00114177, 0.00114177],
        [0.00022400, 0.00022400, 0.00072237, 0.00000112, 0.00022016, 0.00022016],
        [0.00079188, 0.00079188, 0.00106579, 0.00418717, 0.00418717, 0.00712761],
        [0.00219252, 0.00262840, 0.01927670, 0.00491388, 0.01254730, 0.00519414],
        [0.00146999, 0.00168024, 0.01596274, 0.00641979, 0.00597353, 0.00859841],
        [0.00307881, 0.00307881, 0.00036554, 0.00036554, 0.01176737, 0.01176737],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
    ]

    np.testing.assert_allclose(np.sum(gN_ref, axis=1), gN[0, 0].sum(axis=1), atol=0.05)
    np.testing.assert_allclose(np.sum(gU_ref, axis=1), gU[0, 0].sum(axis=1), atol=0.05)


def test_kappa_RTA_nacl(nacl_pbe: Phono3py):
    """Test RTA by NaCl."""
    ref_kappa_RTA = [7.881, 7.881, 7.881, 0, 0, 0]
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_mgo(
    mgo_222rd_444rd_symfc: Phono3py, mgo_222rd_444rd_symfc_compact_fc: Phono3py
):
    """Test RTA by MgO cutoff 4."""
    for ph3 in (mgo_222rd_444rd_symfc, mgo_222rd_444rd_symfc_compact_fc):
        ref_kappa_RTA = [63.75, 63.75, 63.75, 0, 0, 0]
        kappa = _get_kappa(ph3, [11, 11, 11]).ravel()
        np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)
        kappa = _get_kappa(ph3, [11, 11, 11], is_full_pp=True).ravel()
        np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_nacl_with_sigma(nacl_pbe: Phono3py):
    """Test RTA with smearing method by NaCl."""
    ref_kappa_RTA_with_sigma = [7.895, 7.895, 7.895, 0, 0, 0]
    nacl_pbe.sigmas = [
        0.1,
    ]
    nacl_pbe.sigma_cutoff = 3
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_with_sigma, kappa, atol=0.5)
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None


def test_kappa_RTA_aln(aln_lda: Phono3py):
    """Test RTA by AlN."""
    ref_kappa_RTA = [206.379, 206.379, 219.786, 0, 0, 0]
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_aln_with_sigma(aln_lda: Phono3py):
    """Test RTA with smearing method by AlN."""
    ref_kappa_RTA_with_sigmas = [217.598, 217.598, 230.099, 0, 0, 0]
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_with_sigmas, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None


def test_kappa_RTA_si_no_r0avg(si_pbesol_no_r0avg: Phono3py):
    """Test RTA by Si with make_r0_average=False."""
    ref_kappa_RTA = [107.844, 107.844, 107.844, 0, 0, 0]
    kappa = _get_kappa(si_pbesol_no_r0avg, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_nacl_no_r0avg(nacl_pbe_no_r0avg: Phono3py):
    """Test RTA by NaCl with make_r0_average=False."""
    ref_kappa_RTA = [7.741, 7.741, 7.741, 0, 0, 0]
    kappa = _get_kappa(nacl_pbe_no_r0avg, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_aln_no_r0avg(aln_lda_no_r0avg: Phono3py):
    """Test RTA by AlN with make_r0_average=False."""
    ref_kappa_RTA = [203.278, 203.278, 212.965, 0, 0, 0]
    kappa = _get_kappa(aln_lda_no_r0avg, [7, 7, 5]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def _get_kappa(
    ph3: Phono3py,
    mesh,
    is_isotope=False,
    is_full_pp=False,
    openmp_per_triplets=None,
    transport_type=None,
):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction(openmp_per_triplets=openmp_per_triplets)
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=is_isotope,
        is_full_pp=is_full_pp,
        transport_type=transport_type,
    )
    return ph3.thermal_conductivity.kappa
