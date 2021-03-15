import numpy as np

si_pbesol_kappa_RTA = [107.991, 107.991, 107.991, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas = [109.6985, 109.6985, 109.6985, 0, 0, 0]
si_pbesol_kappa_RTA_iso = [96.92419, 96.92419, 96.92419, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas_iso = [96.03248, 96.03248, 96.03248, 0, 0, 0]
si_pbesol_kappa_RTA_si_nosym = [38.242347, 38.700219, 39.198018,
                                0.3216, 0.207731, 0.283]
si_pbesol_kappa_RTA_si_nomeshsym = [38.90918, 38.90918, 38.90918, 0, 0, 0]
nacl_pbe_kappa_RTA = [7.72798252, 7.72798252, 7.72798252, 0, 0, 0]
nacl_pbe_kappa_RTA_with_sigma = [7.71913708, 7.71913708, 7.71913708, 0, 0, 0]


aln_lda_kappa_RTA = [203.304059, 203.304059, 213.003125, 0, 0, 0]
aln_lda_kappa_RTA_with_sigmas = [213.820000, 213.820000, 224.800121, 0, 0, 0]


def test_kappa_RTA_si(si_pbesol):
    kappa = _get_kappa(si_pbesol, [9, 9, 9]).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_full_pp(si_pbesol):
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_full_pp=True).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_iso(si_pbesol):
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA_iso, kappa, atol=0.5)


def test_kappa_RTA_si_with_sigma(si_pbesol):
    si_pbesol.sigmas = [0.1, ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9]).ravel()
    np.testing.assert_allclose(
        si_pbesol_kappa_RTA_with_sigmas, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_full_pp(si_pbesol):
    si_pbesol.sigmas = [0.1, ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_full_pp=True).ravel()
    print(kappa)
    np.testing.assert_allclose(
        si_pbesol_kappa_RTA_with_sigmas, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_with_sigma_iso(si_pbesol):
    si_pbesol.sigmas = [0.1, ]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(
        si_pbesol_kappa_RTA_with_sigmas_iso, kappa, atol=0.5)
    si_pbesol.sigmas = None


def test_kappa_RTA_si_compact_fc(si_pbesol_compact_fc):
    kappa = _get_kappa(si_pbesol_compact_fc, [9, 9, 9]).ravel()
    np.testing.assert_allclose(si_pbesol_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_si_nosym(si_pbesol, si_pbesol_nosym):
    si_pbesol_nosym.fc2 = si_pbesol.fc2
    si_pbesol_nosym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nosym, [4, 4, 4]).reshape(-1, 3).sum(axis=1)
    kappa_ref = np.reshape(si_pbesol_kappa_RTA_si_nosym, (-1, 3)).sum(axis=1)
    np.testing.assert_allclose(kappa_ref / 3, kappa / 3, atol=0.5)


def test_kappa_RTA_si_nomeshsym(si_pbesol, si_pbesol_nomeshsym):
    si_pbesol_nomeshsym.fc2 = si_pbesol.fc2
    si_pbesol_nomeshsym.fc3 = si_pbesol.fc3
    kappa = _get_kappa(si_pbesol_nomeshsym, [4, 4, 4]).ravel()
    kappa_ref = si_pbesol_kappa_RTA_si_nomeshsym
    np.testing.assert_allclose(kappa_ref, kappa, atol=0.5)


def test_kappa_RTA_si_N_U(si_pbesol):
    ph3 = si_pbesol
    mesh = [4, 4, 4]
    is_N_U = True
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300, ], is_N_U=is_N_U)
    gN, gU = ph3.thermal_conductivity.get_gamma_N_U()
    # for g in (gN, gU):
    #     print("".join(["%10.8f, " % x for x in g.ravel()]))
    gN_ref = [
        0.00000000, 0.00000000, 0.00000000, 0.07435213, 0.07435213, 0.07435213,
        0.00079445, 0.00079445, 0.00919937, 0.01856908, 0.04369030, 0.04369030,
        0.00170305, 0.00170305, 0.01495127, 0.00205249, 0.03254221, 0.03254221,
        0.00222008, 0.00222008, 0.00929966, 0.03122158, 0.03122158, 0.01919934,
        0.00272149, 0.00331375, 0.02719147, 0.00356367, 0.01986296, 0.01429359,
        0.00150297, 0.00363122, 0.01620556, 0.00681661, 0.02497616, 0.01055094,
        0.00387128, 0.00387128, 0.00162993, 0.00162993, 0.01691065, 0.01691065,
        0.00641890, 0.00641890, 0.03800105, 0.03800105, 0.02700023, 0.02700023]
    gU_ref = [
        0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
        0.00015178, 0.00015178, 0.00076936, 0.00727539, 0.00113112, 0.00113112,
        0.00022696, 0.00022696, 0.00072558, 0.00000108, 0.00021968, 0.00021968,
        0.00079397, 0.00079397, 0.00111068, 0.00424761, 0.00424761, 0.00697760,
        0.00224483, 0.00260118, 0.01912242, 0.00539855, 0.01271630, 0.00515927,
        0.00147066, 0.00162491, 0.01558419, 0.00614029, 0.00569639, 0.00849517,
        0.00312509, 0.00312509, 0.00036610, 0.00036610, 0.01155516, 0.01155516,
        0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]

    np.testing.assert_allclose(gN_ref, gN.ravel(), atol=1e-2)
    np.testing.assert_allclose(gU_ref, gU.ravel(), atol=1e-2)


def test_kappa_RTA_nacl(nacl_pbe):
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(nacl_pbe_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_nacl_with_sigma(nacl_pbe):
    nacl_pbe.sigmas = [0.1, ]
    nacl_pbe.sigma_cutoff = 3
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(nacl_pbe_kappa_RTA_with_sigma, kappa, atol=0.5)
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None


def test_kappa_RTA_aln(aln_lda):
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(aln_lda_kappa_RTA, kappa, atol=0.5)


def test_kappa_RTA_aln_with_sigma(aln_lda):
    aln_lda.sigmas = [0.1, ]
    aln_lda.sigma_cutoff = 3
    kappa = _get_kappa(aln_lda, [7, 7, 5]).ravel()
    np.testing.assert_allclose(aln_lda_kappa_RTA_with_sigmas, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None


def _get_kappa(ph3, mesh, is_isotope=False, is_full_pp=False):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300, ],
                                 is_isotope=is_isotope,
                                 is_full_pp=is_full_pp)
    return ph3.thermal_conductivity.kappa
