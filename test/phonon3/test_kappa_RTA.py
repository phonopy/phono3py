import numpy as np

si_pbesol_kappa_RTA = [107.991, 107.991, 107.991, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas = [109.6985, 109.6985, 109.6985, 0, 0, 0]
si_pbesol_kappa_RTA_iso = [96.92419, 96.92419, 96.92419, 0, 0, 0]
si_pbesol_kappa_RTA_with_sigmas_iso = [96.03248, 96.03248, 96.03248, 0, 0, 0]
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
