"""Tests for Kubo-RTA thermal conductivity."""

import numpy as np

from phono3py import Phono3py

TOLERANCE = 0.02


def test_kappa_kubo_si(si_pbesol: Phono3py):
    """Test Kubo-RTA by Si."""
    ref_kappa = [107.877, 107.877, 107.877, 0.0, 0.0, 0.0]
    kappa = _get_kappa(si_pbesol, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=TOLERANCE)


def test_kappa_kubo_si_with_sigma(si_pbesol: Phono3py):
    """Test Kubo-RTA with smearing method by Si."""
    ref_kappa = [110.08, 110.08, 110.08, 0.0, 0.0, 0.0]
    si_pbesol.sigmas = [0.1]
    kappa = _get_kappa(si_pbesol, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=TOLERANCE)
    si_pbesol.sigmas = None


def test_kappa_kubo_si_iso(si_pbesol: Phono3py):
    """Test Kubo-RTA with isotope scattering by Si."""
    ref_kappa = [97.301, 97.301, 97.301, 0.0, 0.0, 0.0]
    kappa = _get_kappa(si_pbesol, [9, 9, 9], is_isotope=True).ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=TOLERANCE)


def test_kappa_kubo_nacl(nacl_pbe: Phono3py):
    """Test Kubo-RTA by NaCl."""
    ref_kappa = [7.93, 7.93, 7.93, 0.0, 0.0, 0.0]
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=TOLERANCE)


def test_kappa_kubo_nacl_with_sigma(nacl_pbe: Phono3py):
    """Test Kubo-RTA with smearing method by NaCl."""
    ref_kappa = [7.944, 7.944, 7.944, 0.0, 0.0, 0.0]
    nacl_pbe.sigmas = [0.1]
    nacl_pbe.sigma_cutoff = 3
    kappa = _get_kappa(nacl_pbe, [9, 9, 9]).ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=TOLERANCE)
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None


def _get_kappa(ph3: Phono3py, mesh, is_isotope: bool = False):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[300],
        is_isotope=is_isotope,
        conductivity_type="kubo",
    )
    return ph3.thermal_conductivity.kappa
