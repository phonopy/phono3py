"""Tests for NJC23-RTA thermal conductivity."""

import numpy as np

from phono3py import Phono3py

# kappa and kappa_intra follow the RTA solution, which varies among
# architectures by up to ~0.06 W/m-K for Si with the tetrahedron method. 0.25 is
# the value calibrated for the conda Windows build.
TOLERANCE = 0.25
# Isotope scattering is built from the eigenvectors of degenerate bands, whose
# basis is not fixed by the eigensolver, so it varies more.
TOLERANCE_ISO = 0.3
# The inter-band part is what these tests exist to pin down, and it is stable
# among architectures, so it is checked tightly.
TOLERANCE_INTER = 0.05


def test_kappa_njc23_si(si_pbesol: Phono3py):
    """Test NJC23-RTA by Si."""
    ref_kappa = [108.325, 108.325, 108.325, 0.0, 0.0, 0.0]
    ref_kappa_intra = [107.794, 107.794, 107.794, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.531, 0.531, 0.531, 0.0, 0.0, 0.0]
    tc = _run_njc23_rta(si_pbesol, [9, 9, 9])
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(
        ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE_INTER
    )


def test_kappa_njc23_si_with_sigma(si_pbesol: Phono3py):
    """Test NJC23-RTA with smearing method by Si."""
    ref_kappa = [110.586, 110.586, 110.586, 0.0, 0.0, 0.0]
    ref_kappa_intra = [109.999, 109.999, 109.999, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.587, 0.587, 0.587, 0.0, 0.0, 0.0]
    si_pbesol.sigmas = [0.1]
    tc = _run_njc23_rta(si_pbesol, [9, 9, 9])
    si_pbesol.sigmas = None
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(
        ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE_INTER
    )


def test_kappa_njc23_si_iso(si_pbesol: Phono3py):
    """Test NJC23-RTA with isotope scattering by Si."""
    ref_kappa = [97.753, 97.753, 97.753, 0.0, 0.0, 0.0]
    ref_kappa_intra = [97.213, 97.213, 97.213, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.540, 0.540, 0.540, 0.0, 0.0, 0.0]
    tc = _run_njc23_rta(si_pbesol, [9, 9, 9], is_isotope=True)
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE_ISO)
    np.testing.assert_allclose(
        ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE_ISO
    )
    np.testing.assert_allclose(
        ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE_INTER
    )


def test_kappa_njc23_nacl(nacl_pbe: Phono3py):
    """Test NJC23-RTA by NaCl."""
    ref_kappa = [7.956, 7.956, 7.956, 0.0, 0.0, 0.0]
    ref_kappa_intra = [7.862, 7.862, 7.862, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.094, 0.094, 0.094, 0.0, 0.0, 0.0]
    tc = _run_njc23_rta(nacl_pbe, [9, 9, 9])
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(
        ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE_INTER
    )


def test_kappa_njc23_nacl_with_sigma(nacl_pbe: Phono3py):
    """Test NJC23-RTA with smearing method by NaCl."""
    ref_kappa = [7.988, 7.988, 7.988, 0.0, 0.0, 0.0]
    ref_kappa_intra = [7.895, 7.895, 7.895, 0.0, 0.0, 0.0]
    ref_kappa_inter = [0.094, 0.094, 0.094, 0.0, 0.0, 0.0]
    nacl_pbe.sigmas = [0.1]
    nacl_pbe.sigma_cutoff = 3
    tc = _run_njc23_rta(nacl_pbe, [9, 9, 9])
    nacl_pbe.sigmas = None
    nacl_pbe.sigma_cutoff = None
    np.testing.assert_allclose(ref_kappa, tc.kappa.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(ref_kappa_intra, tc.kappa_intra.ravel(), atol=TOLERANCE)
    np.testing.assert_allclose(
        ref_kappa_inter, tc.kappa_inter.ravel(), atol=TOLERANCE_INTER
    )


def _run_njc23_rta(ph3: Phono3py, mesh, is_isotope: bool = False):
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[300],
        is_isotope=is_isotope,
        transport_type="NJC23",
    )
    return ph3.thermal_conductivity
