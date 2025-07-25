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
    if ph3._make_r0_average:
        if is_compact_fc:
            ref_kappa_RTA = [107.794, 107.794, 107.794, 0, 0, 0]
        else:
            ref_kappa_RTA = [107.694, 107.694, 107.694, 0, 0, 0]
    else:
        if is_compact_fc:
            ref_kappa_RTA = [107.956, 107.956, 107.956, 0, 0, 0]
        else:
            ref_kappa_RTA = [107.844, 107.844, 107.844, 0, 0, 0]

    kappa = _get_kappa(
        ph3,
        [9, 9, 9],
        is_full_pp=is_full_pp,
        openmp_per_triplets=openmp_per_triplets,
    ).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_iso(si_pbesol: Phono3py):
    """Test RTA with isotope scattering by Si."""
    if si_pbesol._make_r0_average:
        ref_kappa_RTA_iso = [97.296, 97.296, 97.296, 0, 0, 0]
    else:
        ref_kappa_RTA_iso = [97.346, 97.346, 97.346, 0, 0, 0]

    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_iso, kappa, atol=0.5)


@pytest.mark.parametrize(
    "openmp_per_triplets,is_full_pp", itertools.product([False, True], [False, True])
)
def test_kappa_RTA_si_with_sigma(
    si_pbesol: Phono3py, openmp_per_triplets: bool, is_full_pp: bool
):
    """Test RTA with smearing method by Si."""
    if si_pbesol._make_r0_average:
        ref_kappa_RTA_with_sigmas = [109.999, 109.999, 109.999, 0, 0, 0]
    else:
        ref_kappa_RTA_with_sigmas = [109.699, 109.699, 109.699, 0, 0, 0]

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
    if si_pbesol._make_r0_average:
        ref_kappa_RTA_with_sigmas_iso = [96.368, 96.368, 96.368, 0, 0, 0]
    else:
        ref_kappa_RTA_with_sigmas_iso = [96.032, 96.032, 96.032, 0, 0, 0]

    si_pbesol.sigmas = [
        0.1,
    ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_with_sigmas_iso, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_nosym(si_pbesol: Phono3py, si_pbesol_nosym: Phono3py):
    """Test RTA without considering symmetry by Si."""
    if si_pbesol_nosym._make_r0_average:
        ref_kappa_RTA_si_nosym = [38.315, 38.616, 39.093, 0.221, 0.166, 0.284]
    else:
        ref_kappa_RTA_si_nosym = [38.342, 38.650, 39.105, 0.224, 0.170, 0.288]

    si_pbesol_nosym.fc2 = si_pbesol.fc2
    si_pbesol_nosym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nosym, [4, 4, 4]).reshape(-1, 3).sum(axis=1)
    kappa_ref = np.reshape(ref_kappa_RTA_si_nosym, (-1, 3)).sum(axis=1)
    np.testing.assert_allclose(kappa_ref / 3, kappa / 3, atol=0.5)


def test_kappa_RTA_si_nomeshsym(si_pbesol: Phono3py, si_pbesol_nomeshsym: Phono3py):
    """Test RTA without considering mesh symmetry by Si."""
    if si_pbesol_nomeshsym._make_r0_average:
        ref_kappa_RTA_si_nomeshsym = [81.147, 81.147, 81.147, 0.000, 0.000, 0.000]
    else:
        ref_kappa_RTA_si_nomeshsym = [81.263, 81.263, 81.263, 0.000, 0.000, 0.000]
    si_pbesol_nomeshsym.fc2 = si_pbesol.fc2
    si_pbesol_nomeshsym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nomeshsym, [7, 7, 7]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_si_nomeshsym, kappa, atol=0.5)


def test_kappa_RTA_si_grg(si_pbesol_grg: Phono3py):
    """Test RTA by Si with GR-grid."""
    if si_pbesol_grg._make_r0_average:
        ref_kappa_RTA_grg = [111.204, 111.204, 111.204, 0, 0, 0]
    else:
        ref_kappa_RTA_grg = [111.349, 111.349, 111.349, 0, 0, 0]
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
    if si_pbesol_grg._make_r0_average:
        ref_kappa_RTA_grg_iso = [104.290, 104.290, 104.290, 0, 0, 0]
    else:
        ref_kappa_RTA_grg_iso = [104.425, 104.425, 104.425, 0, 0, 0]

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
    if si_pbesol_grg._make_r0_average:
        ref_kappa_RTA_grg_sigma_iso = [107.264, 107.264, 107.264, 0, 0, 0]
    else:
        ref_kappa_RTA_grg_sigma_iso = [107.283, 107.283, 107.283, 0, 0, 0]
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

    if si_pbesol._make_r0_average:
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
    else:
        gN_ref = [
            [0.00000000, 0.00000000, 0.00000000, 0.07832198, 0.07832198, 0.07832198],
            [0.00079578, 0.00079578, 0.00909025, 0.01917012, 0.04557656, 0.04557656],
            [0.00176235, 0.00176235, 0.01414436, 0.00204092, 0.03361112, 0.03361112],
            [0.00221919, 0.00221919, 0.01020133, 0.02889554, 0.02889554, 0.01995543],
            [0.00292189, 0.00356099, 0.02855954, 0.00370530, 0.02071850, 0.01533334],
            [0.00147656, 0.00342335, 0.01589430, 0.00630792, 0.02427768, 0.01099287],
            [0.00400675, 0.00400675, 0.00162186, 0.00162186, 0.01478489, 0.01478489],
            [0.00676576, 0.00676576, 0.03984290, 0.03984290, 0.02715102, 0.02715102],
        ]
        gU_ref = [
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00015178, 0.00015178, 0.00076936, 0.00727539, 0.00113112, 0.00113112],
            [0.00022696, 0.00022696, 0.00072558, 0.00000108, 0.00021968, 0.00021968],
            [0.00079397, 0.00079397, 0.00111068, 0.00424761, 0.00424761, 0.00697760],
            [0.00219456, 0.00261878, 0.01928629, 0.00490046, 0.01249235, 0.00517685],
            [0.00149539, 0.00161230, 0.01594274, 0.00653088, 0.00593572, 0.00849890],
            [0.00311169, 0.00311169, 0.00036610, 0.00036610, 0.01171667, 0.01171667],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ]

    np.testing.assert_allclose(np.sum(gN_ref, axis=1), gN[0, 0].sum(axis=1), atol=0.05)
    np.testing.assert_allclose(np.sum(gU_ref, axis=1), gU[0, 0].sum(axis=1), atol=0.05)


def test_kappa_RTA_nacl(nacl_pbe: Phono3py):
    """Test RTA by NaCl."""
    if nacl_pbe._make_r0_average:
        ref_kappa_RTA = [7.881, 7.881, 7.881, 0, 0, 0]
    else:
        ref_kappa_RTA = [7.741, 7.741, 7.741, 0, 0, 0]
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
    if nacl_pbe._make_r0_average:
        ref_kappa_RTA_with_sigma = [7.895, 7.895, 7.895, 0, 0, 0]
    else:
        ref_kappa_RTA_with_sigma = [7.719, 7.719, 7.719, 0, 0, 0]
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
    if aln_lda._make_r0_average:
        ref_kappa_RTA = [206.379, 206.379, 219.786, 0, 0, 0]
    else:
        ref_kappa_RTA = [203.278, 203.278, 212.965, 0, 0, 0]
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_aln_with_sigma(aln_lda: Phono3py):
    """Test RTA with smearing method by AlN."""
    if aln_lda._make_r0_average:
        ref_kappa_RTA_with_sigmas = [217.598, 217.598, 230.099, 0, 0, 0]
    else:
        ref_kappa_RTA_with_sigmas = [213.820, 213.820, 224.800, 0, 0, 0]
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(ref_kappa_RTA_with_sigmas, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None


def _get_kappa(
    ph3: Phono3py,
    mesh,
    is_isotope=False,
    is_full_pp=False,
    openmp_per_triplets=None,
):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction(openmp_per_triplets=openmp_per_triplets)
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=is_isotope,
        is_full_pp=is_full_pp,
    )
    return ph3.thermal_conductivity.kappa
