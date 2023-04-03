"""Test for Conductivity_RTA.py."""
import numpy as np
import pytest

from phono3py import Phono3py

si_pbesol_kappa_RTA = [107.991, 107.991, 107.991, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas = [109.6985, 109.6985, 109.6985, 0, 0, 0]
si_pbesol_kappa_RTA_iso = [96.92419, 96.92419, 96.92419, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas_iso = [96.03248, 96.03248, 96.03248, 0, 0, 0]
si_pbesol_kappa_RTA_si_nosym = [
    38.242347,
    38.700219,
    39.198018,
    0.3216,
    0.207731,
    0.283,
]
si_pbesol_kappa_RTA_si_nomeshsym = [81.31304, 81.31304, 81.31304, 0, 0, 0]
si_pbesol_kappa_RTA_grg = [93.99526, 93.99526, 93.99526, 0, 0, 0]
si_pbesol_kappa_RTA_grg_iso = [104.281556, 104.281556, 104.281556, 0, 0, 0]
si_pbesol_kappa_RTA_grg_sigma_iso = [107.2834, 107.2834, 107.2834, 0, 0, 0]
nacl_pbe_kappa_RTA = [7.72798252, 7.72798252, 7.72798252, 0, 0, 0]
nacl_pbe_kappa_RTA_with_sigma = [7.71913708, 7.71913708, 7.71913708, 0, 0, 0]


aln_lda_kappa_RTA = [203.304059, 203.304059, 213.003125, 0, 0, 0]
aln_lda_kappa_RTA_r0_ave = [2.06489355e02, 2.06489355e02, 2.19821864e02, 0, 0, 0]
aln_lda_kappa_RTA_with_sigmas = [213.820000, 213.820000, 224.800121, 0, 0, 0]
aln_lda_kappa_RTA_with_sigmas_r0_ave = [
    2.17597885e02,
    2.17597885e02,
    2.30098660e02,
    0,
    0,
    0,
]


@pytest.mark.parametrize("openmp_per_triplets", [True, False])
def test_kappa_RTA_si(
    si_pbesol: Phono3py,
    openmp_per_triplets: bool,
):
    """Test RTA by Si."""
    kappa = _get_kappa(
        si_pbesol,
        [9, 9, 9],
        openmp_per_triplets=openmp_per_triplets,
    ).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


@pytest.mark.parametrize("openmp_per_triplets", [True, False])
def test_kappa_RTA_si_full_pp(si_pbesol: Phono3py, openmp_per_triplets: bool):
    """Test RTA with full-pp by Si."""
    kappa = _get_kappa(
        si_pbesol, [9, 9, 9], is_full_pp=True, openmp_per_triplets=openmp_per_triplets
    ).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_iso(si_pbesol: Phono3py):
    """Test RTA with isotope scattering by Si."""
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA_iso, kappa, atol=0.5)


@pytest.mark.parametrize("openmp_per_triplets", [True, False])
def test_kappa_RTA_si_with_sigma(si_pbesol: Phono3py, openmp_per_triplets: bool):
    """Test RTA with smearing method by Si."""
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa = _get_kappa(
        si_pbesol, [9, 9, 9], openmp_per_triplets=openmp_per_triplets
    ).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA_with_sigmas, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_full_pp(si_pbesol: Phono3py):
    """Test RTA with smearing method and full-pp by Si."""
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_full_pp=True).ravel()
    print(kappa)
    np.testing.assert_allclose(si_pbesol_kappa_RTA_with_sigmas, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_iso(si_pbesol: Phono3py):
    """Test RTA with smearing method and isotope scattering by Si."""
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA_with_sigmas_iso, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_compact_fc(si_pbesol_compact_fc: Phono3py):
    """Test RTA with compact-fc by Si."""
    kappa = _get_kappa(si_pbesol_compact_fc, [9, 9, 9]).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_nosym(si_pbesol: Phono3py, si_pbesol_nosym: Phono3py):
    """Test RTA without considering symmetry by Si."""
    si_pbesol_nosym.fc2 = si_pbesol.fc2
    si_pbesol_nosym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nosym, [4, 4, 4]).reshape(-1, 3).sum(axis=1)
    kappa_ref = np.reshape(si_pbesol_kappa_RTA_si_nosym, (-1, 3)).sum(axis=1)
    np.testing.assert_allclose(kappa_ref / 3, kappa / 3, atol=0.5)


def test_kappa_RTA_si_nomeshsym(si_pbesol: Phono3py, si_pbesol_nomeshsym: Phono3py):
    """Test RTA without considering mesh symmetry by Si."""
    si_pbesol_nomeshsym.fc2 = si_pbesol.fc2
    si_pbesol_nomeshsym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nomeshsym, [7, 7, 7]).ravel()
    kappa_ref = si_pbesol_kappa_RTA_si_nomeshsym
    np.testing.assert_allclose(kappa_ref, kappa, atol=0.5)


def test_kappa_RTA_si_grg(si_pbesol_grg: Phono3py):
    """Test RTA by Si with GR-grid."""
    mesh = 20
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
        [[-4, 4, 4], [4, -4, 4], [4, 4, -4]],
    )
    np.testing.assert_equal(
        ph3.grid.grid_matrix,
        [[-4, 4, 4], [4, -4, 4], [4, 4, -4]],
    )
    A = ph3.grid.grid_matrix
    D_diag = ph3.grid.D_diag
    P = ph3.grid.P
    Q = ph3.grid.Q
    np.testing.assert_equal(np.dot(P, np.dot(A, Q)), np.diag(D_diag))

    np.testing.assert_allclose(si_pbesol_kappa_RTA_grg, kappa, atol=0.5)


def test_kappa_RTA_si_grg_iso(si_pbesol_grg: Phono3py):
    """Test RTA with isotope scattering by Si with GR-grid.."""
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
    np.testing.assert_allclose(si_pbesol_kappa_RTA_grg_iso, kappa, atol=0.5)
    np.testing.assert_equal(ph3.grid.grid_matrix, [[-6, 6, 6], [6, -6, 6], [6, 6, -6]])


def test_kappa_RTA_si_grg_sigma_iso(si_pbesol_grg: Phono3py):
    """Test RTA with isotope scattering by Si with GR-grid.."""
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
    np.testing.assert_allclose(si_pbesol_kappa_RTA_grg_sigma_iso, kappa, atol=0.5)
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

    gN_ref = [
        0.00000000,
        0.00000000,
        0.00000000,
        0.07402084,
        0.07402084,
        0.07402084,
        0.00078535,
        0.00078535,
        0.00917995,
        0.02178049,
        0.04470075,
        0.04470075,
        0.00173337,
        0.00173337,
        0.01240191,
        0.00198981,
        0.03165195,
        0.03165195,
        0.00224713,
        0.00224713,
        0.00860026,
        0.03083611,
        0.03083611,
        0.02142118,
        0.00277534,
        0.00330170,
        0.02727451,
        0.00356415,
        0.01847744,
        0.01320643,
        0.00155072,
        0.00365611,
        0.01641919,
        0.00650083,
        0.02576069,
        0.01161589,
        0.00411969,
        0.00411969,
        0.00168211,
        0.00168211,
        0.01560092,
        0.01560092,
        0.00620091,
        0.00620091,
        0.03764912,
        0.03764912,
        0.02668523,
        0.02668523,
    ]
    gU_ref = [
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00015178,
        0.00015178,
        0.00076936,
        0.00727539,
        0.00113112,
        0.00113112,
        0.00022696,
        0.00022696,
        0.00072558,
        0.00000108,
        0.00021968,
        0.00021968,
        0.00079397,
        0.00079397,
        0.00111068,
        0.00424761,
        0.00424761,
        0.00697760,
        0.00221593,
        0.00259510,
        0.01996296,
        0.00498962,
        0.01258375,
        0.00513825,
        0.00148802,
        0.00161955,
        0.01589219,
        0.00646134,
        0.00577275,
        0.00849711,
        0.00313208,
        0.00313208,
        0.00036610,
        0.00036610,
        0.01135335,
        0.01135335,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
    ]

    # print(np.sum(gN), np.sum(gU))
    np.testing.assert_allclose(
        np.sum([gN_ref, gU_ref], axis=0), gN.ravel() + gU.ravel(), atol=1e-2
    )
    np.testing.assert_allclose(gN_ref, gN.ravel(), atol=1e-2)
    np.testing.assert_allclose(gU_ref, gU.ravel(), atol=1e-2)


def test_kappa_RTA_nacl(nacl_pbe: Phono3py):
    """Test RTA by NaCl."""
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(nacl_pbe_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_nacl_with_sigma(nacl_pbe: Phono3py):
    """Test RTA with smearing method by NaCl."""
    nacl_pbe.sigmas = [
        0.1,
    ]
    nacl_pbe.sigma_cutoff = 3
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(nacl_pbe_kappa_RTA_with_sigma, kappa, atol=0.5)
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None


def test_kappa_RTA_aln(aln_lda: Phono3py):
    """Test RTA by AlN."""
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(aln_lda_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_aln_with_r0_ave(aln_lda: Phono3py):
    """Test RTA by AlN."""
    kappa = _get_kappa(aln_lda, [7, 7, 5], make_r0_average=True).ravel()
    np.testing.assert_allclose(aln_lda_kappa_RTA_r0_ave, kappa, atol=0.5)


def test_kappa_RTA_aln_with_sigma(aln_lda: Phono3py):
    """Test RTA with smearing method by AlN."""
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(aln_lda_kappa_RTA_with_sigmas, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None


def test_kappa_RTA_aln_with_sigma_and_r0_ave(aln_lda: Phono3py):
    """Test RTA with smearing method by AlN."""
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    kappa = _get_kappa(aln_lda, [7, 7, 5], make_r0_average=True).ravel()
    np.testing.assert_allclose(aln_lda_kappa_RTA_with_sigmas_r0_ave, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None


def _get_kappa(
    ph3,
    mesh,
    is_isotope=False,
    is_full_pp=False,
    openmp_per_triplets=None,
    make_r0_average=False,
):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction(
        make_r0_average=make_r0_average, openmp_per_triplets=openmp_per_triplets
    )
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=is_isotope,
        is_full_pp=is_full_pp,
    )
    return ph3.thermal_conductivity.kappa
