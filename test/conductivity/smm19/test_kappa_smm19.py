"""Tests for SMM19-RTA thermal conductivity."""

import numpy as np

from phono3py import Phono3py

TOLERANCE = 0.2


def test_kappa_smm19_si(si_pbesol: Phono3py):
    """Test SMM19-RTA by Si."""
    ref_kappa = [108.330, 108.330, 108.330, 0.0, 0.0, 0.0]
    ref_kappa_intra = [107.794, 107.794, 107.794, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.537, 0.537, 0.537, 0.0, 0.0, 0.0]
    tc = _run_smm19_rta(si_pbesol, [9, 9, 9])
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE)


def test_kappa_smm19_si_with_sigma(si_pbesol: Phono3py):
    """Test SMM19-RTA with smearing method by Si."""
    ref_kappa = [110.592, 110.592, 110.592, 0.0, 0.0, 0.0]
    ref_kappa_intra = [109.999, 109.999, 109.999, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.592, 0.592, 0.592, 0.0, 0.0, 0.0]
    si_pbesol.sigmas = [0.1]
    tc = _run_smm19_rta(si_pbesol, [9, 9, 9])
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE)
    si_pbesol.sigmas = None


def test_kappa_smm19_si_iso(si_pbesol: Phono3py):
    """Test SMM19-RTA with isotope scattering by Si."""
    ref_kappa = [97.758, 97.758, 97.758, 0.0, 0.0, 0.0]
    ref_kappa_intra = [97.213, 97.213, 97.213, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.545, 0.545, 0.545, 0.0, 0.0, 0.0]
    tc = _run_smm19_rta(si_pbesol, [9, 9, 9], is_isotope=True)
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE)


def test_kappa_smm19_nacl(nacl_pbe: Phono3py):
    """Test SMM19-RTA by NaCl."""
    ref_kappa = [7.929, 7.929, 7.929, 0.0, 0.0, 0.0]
    ref_kappa_intra = [7.881, 7.881, 7.881, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.048, 0.048, 0.048, 0.0, 0.0, 0.0]
    tc = _run_smm19_rta(nacl_pbe, [9, 9, 9])
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE)


def test_kappa_smm19_nacl_with_sigma(nacl_pbe: Phono3py):
    """Test SMM19-RTA with smearing method by NaCl."""
    ref_kappa = [7.943, 7.943, 7.943, 0.0, 0.0, 0.0]
    ref_kappa_intra = [7.895, 7.895, 7.895, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.049, 0.049, 0.049, 0.0, 0.0, 0.0]
    nacl_pbe.sigmas = [0.1]
    nacl_pbe.sigma_cutoff = 3
    tc = _run_smm19_rta(nacl_pbe, [9, 9, 9])
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE)
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None


def _run_smm19_rta(ph3: Phono3py, mesh, is_isotope: bool = False):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[300],
        is_isotope=is_isotope,
        transport_type="SMM19",
    )
    return ph3.thermal_conductivity
