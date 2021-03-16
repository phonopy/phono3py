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
    for g in (gN, gU):
        print("".join(["%10.8f, " % x for x in g.ravel()]))

    gN_ref = [
        0.00000000, 0.00000000, 0.00000000, 0.06180334, 0.06180334, 0.06180334,
        0.00053926, 0.00053926, 0.00792356, 0.00951148, 0.02592651, 0.02592651,
        0.00117934, 0.00117934, 0.00698456, 0.00170658, 0.01118689, 0.01118689,
        0.00166414, 0.00166414, 0.00885484, 0.01888396, 0.01888396, 0.00897665,
        0.00131452, 0.00235779, 0.01566565, 0.00286692, 0.00429834, 0.00530297,
        0.00077627, 0.00299155, 0.01015759, 0.00521881, 0.01128501, 0.00700690,
        0.00161764, 0.00161764, 0.00121183, 0.00121183, 0.00244780, 0.00244780,
        0.00331803, 0.00331803, 0.01516610, 0.01516610, 0.01577839, 0.01577839]
    gU_ref = [
        0.00000000, 0.00000000, 0.00000000, 0.01247437, 0.01247437, 0.01247437,
        0.00041217, 0.00041217, 0.00205338, 0.01673556, 0.01923664, 0.01923664,
        0.00073456, 0.00073456, 0.00854797, 0.00035641, 0.02120137, 0.02120137,
        0.00134860, 0.00134860, 0.00155953, 0.01663700, 0.01663700, 0.01772709,
        0.00364242, 0.00356620, 0.03074762, 0.00610700, 0.02850547, 0.01370453,
        0.00220781, 0.00224245, 0.02145013, 0.00773445, 0.02014343, 0.01205437,
        0.00531674, 0.00531674, 0.00078421, 0.00078421, 0.02643499, 0.02643499,
        0.00288569, 0.00288569, 0.02288481, 0.02288481, 0.01102395, 0.01102395]

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
