"""Test for Conductivity_RTA.py."""
import numpy as np

# first list is k_P, second list is k_C
si_pbesol_kappa_P_RTA = [108.723, 108.723, 108.723, 0.000, 0.000, 0.000]
si_pbesol_kappa_C = [0.167, 0.167, 0.167, 0.000, 0.000, 0.000]

si_pbesol_kappa_P_RTA_iso = [98.008, 98.008, 98.008, 0.000, 0.000, 0.000]
si_pbesol_kappa_C_iso = [0.177, 0.177, 0.177, 0.000, 0.000, 0.000]

si_pbesol_kappa_P_RTA_with_sigmas = [110.534, 110.534, 110.534, 0, 0, 0]
si_pbesol_kappa_C_with_sigmas = [0.163, 0.163, 0.163, 0.000, 0.000, 0.000]

si_pbesol_kappa_P_RTA_with_sigmas_iso = [97.268, 97.268, 97.268, 0, 0, 0]
si_pbesol_kappa_C_with_sigmas_iso = [0.179, 0.179, 0.179, 0.000, 0.000, 0.000]

si_pbesol_kappa_P_RTA_si_nosym = [39.325, 39.323, 39.496, -0.004, 0.020, 0.018]

si_pbesol_kappa_C_si_nosym = [0.009, 0.009, 0.009, 0.000, 0.000, 0.000]

si_pbesol_kappa_P_RTA_si_nomeshsym = [39.411, 39.411, 39.411, 0, 0, 0]
si_pbesol_kappa_C_si_nomeshsym = [0.009, 0.009, 0.009, 0.000, 0.000, 0.000]

nacl_pbe_kappa_P_RTA = [7.753, 7.753, 7.753, 0.000, 0.000, 0.000]
nacl_pbe_kappa_C = [0.081, 0.081, 0.081, 0.000, 0.000, 0.000]

nacl_pbe_kappa_RTA_with_sigma = [7.742, 7.742, 7.742, 0, 0, 0]  # old value 7.71913708
nacl_pbe_kappa_C_with_sigma = [0.081, 0.081, 0.081, 0.000, 0.000, 0.000]

aln_lda_kappa_P_RTA = [203.304, 203.304, 213.003, 0, 0, 0]
aln_lda_kappa_C = [0.084, 0.084, 0.037, 0, 0, 0]

aln_lda_kappa_P_RTA_with_sigmas = [213.820000, 213.820000, 224.800000, 0, 0, 0]
aln_lda_kappa_C_with_sigmas = [0.084, 0.084, 0.036, 0, 0, 0]


def test_kappa_RTA_si(si_pbesol):
    """Test RTA by Si."""
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol, [9, 9, 9])
    np.testing.assert_allclose(si_pbesol_kappa_P_RTA, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C, kappa_C, atol=0.02)


def test_kappa_RTA_si_full_pp(si_pbesol):
    """Test RTA with full-pp by Si."""
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol, [9, 9, 9], is_full_pp=True)
    np.testing.assert_allclose(si_pbesol_kappa_P_RTA, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C, kappa_C, atol=0.02)


def test_kappa_RTA_si_iso(si_pbesol):
    """Test RTA with isotope scattering by Si."""
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol, [9, 9, 9], is_isotope=True)
    np.testing.assert_allclose(si_pbesol_kappa_P_RTA_iso, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C_iso, kappa_C, atol=0.02)


def test_kappa_RTA_si_with_sigma(si_pbesol):
    """Test RTA with smearing method by Si."""
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol, [9, 9, 9])
    np.testing.assert_allclose(si_pbesol_kappa_P_RTA_with_sigmas, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C_with_sigmas, kappa_C, atol=0.02)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_full_pp(si_pbesol):
    """Test RTA with smearing method and full-pp by Si."""
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol, [9, 9, 9], is_full_pp=True)
    np.testing.assert_allclose(si_pbesol_kappa_P_RTA_with_sigmas, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C_with_sigmas, kappa_C, atol=0.02)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_iso(si_pbesol):
    """Test RTA with smearing method and isotope scattering by Si."""
    si_pbesol.sigmas = [
        0.1,
    ]
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol, [9, 9, 9], is_isotope=True)
    np.testing.assert_allclose(
        si_pbesol_kappa_P_RTA_with_sigmas_iso, kappa_P_RTA, atol=0.5
    )
    np.testing.assert_allclose(si_pbesol_kappa_C_with_sigmas_iso, kappa_C, atol=0.02)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_compact_fc(si_pbesol_compact_fc):
    """Test RTA with compact-fc by Si."""
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol_compact_fc, [9, 9, 9])
    np.testing.assert_allclose(si_pbesol_kappa_P_RTA, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(si_pbesol_kappa_C, kappa_C, atol=0.02)


def test_kappa_RTA_si_nosym(si_pbesol, si_pbesol_nosym):
    """Test RTA without considering symmetry by Si."""
    si_pbesol_nosym.fc2 = si_pbesol.fc2
    si_pbesol_nosym.fc3 = si_pbesol.fc3
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol_nosym, [4, 4, 4])
    kappa_P_RTA_r = kappa_P_RTA.reshape(-1, 3).sum(axis=1)
    kappa_C_r = kappa_C.reshape(-1, 3).sum(axis=1)
    kappa_P_ref = np.reshape(si_pbesol_kappa_P_RTA_si_nosym, (-1, 3)).sum(axis=1)
    kappa_C_ref = np.reshape(si_pbesol_kappa_C_si_nosym, (-1, 3)).sum(axis=1)
    np.testing.assert_allclose(kappa_P_ref / 3, kappa_P_RTA_r / 3, atol=0.8)
    np.testing.assert_allclose(kappa_C_ref / 3, kappa_C_r / 3, atol=0.02)


def test_kappa_RTA_si_nomeshsym(si_pbesol, si_pbesol_nomeshsym):
    """Test RTA without considering mesh symmetry by Si."""
    si_pbesol_nomeshsym.fc2 = si_pbesol.fc2
    si_pbesol_nomeshsym.fc3 = si_pbesol.fc3
    kappa_P_RTA, kappa_C = _get_kappa_RTA(si_pbesol_nomeshsym, [4, 4, 4])
    np.testing.assert_allclose(
        si_pbesol_kappa_P_RTA_si_nomeshsym, kappa_P_RTA, atol=1.0
    )
    np.testing.assert_allclose(si_pbesol_kappa_C_si_nomeshsym, kappa_C, atol=0.02)


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
        conductivity_type="wigner",
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


def test_kappa_RTA_nacl(nacl_pbe):
    """Test RTA by NaCl."""
    kappa_P_RTA, kappa_C = _get_kappa_RTA(nacl_pbe, [9, 9, 9])
    np.testing.assert_allclose(nacl_pbe_kappa_P_RTA, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(nacl_pbe_kappa_C, kappa_C, atol=0.02)


def test_kappa_RTA_nacl_with_sigma(nacl_pbe):
    """Test RTA with smearing method by NaCl."""
    nacl_pbe.sigmas = [
        0.1,
    ]
    nacl_pbe.sigma_cutoff = 3
    kappa_P_RTA, kappa_C = _get_kappa_RTA(nacl_pbe, [9, 9, 9])
    np.testing.assert_allclose(nacl_pbe_kappa_RTA_with_sigma, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(nacl_pbe_kappa_C_with_sigma, kappa_C, atol=0.02)
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None


def test_kappa_RTA_aln(aln_lda):
    """Test RTA by AlN."""
    kappa_P_RTA, kappa_C = _get_kappa_RTA(aln_lda, [7, 7, 5])
    np.testing.assert_allclose(aln_lda_kappa_P_RTA, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(aln_lda_kappa_C, kappa_C, atol=0.02)


def test_kappa_RTA_aln_with_sigma(aln_lda):
    """Test RTA with smearing method by AlN."""
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    kappa_P_RTA, kappa_C = _get_kappa_RTA(aln_lda, [7, 7, 5])
    np.testing.assert_allclose(aln_lda_kappa_P_RTA_with_sigmas, kappa_P_RTA, atol=0.5)
    np.testing.assert_allclose(aln_lda_kappa_C_with_sigmas, kappa_C, atol=0.02)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None


def _get_kappa_RTA(ph3, mesh, is_isotope=False, is_full_pp=False):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ],
        is_isotope=is_isotope,
        is_full_pp=is_full_pp,
        conductivity_type="wigner",
    )
    return (
        ph3.thermal_conductivity.kappa_P_RTA.ravel(),
        ph3.thermal_conductivity.kappa_C.ravel(),
    )
