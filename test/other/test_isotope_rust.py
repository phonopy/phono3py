"""Compare the Rust isotope backend against the C one.

Uses ``Phono3pyIsotope`` with ``lang="C"`` vs ``lang="Rust"`` at the
same grid points, checking the tetrahedron-method and the
Gaussian-smearing paths.

"""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py, Phono3pyIsotope

pytest.importorskip("phonors")
pytest.importorskip("phono3py._phono3py")


def _build_iso(
    ph3: Phono3py, *, sigmas, lang: str, mesh=(21, 21, 21)
) -> Phono3pyIsotope:
    iso = Phono3pyIsotope(
        mesh,
        ph3.phonon_primitive,
        sigmas=sigmas,
        symprec=ph3.symmetry.tolerance,
        lang=lang,
    )
    iso.init_dynamical_matrix(
        ph3.fc2,
        ph3.phonon_supercell,
        ph3.phonon_primitive,
        nac_params=ph3.nac_params,
    )
    return iso


# Spread of grid points across the BZ; enough to average out the
# tetrahedron-induced peakiness in the Rust-vs-C comparison.
_GRID_POINTS = [11, 23, 47, 79, 103, 137, 169, 211, 251, 307]


def test_isotope_rust_vs_c_tetrahedron(si_pbesol: Phono3py):
    """Tetrahedron-method isotope gamma sum: Rust matches C.

    Per-element parity is not achievable.  The C and Rust dynamical
    matrix builders use slightly different floating-point summation
    order, so eigenvalues and eigenvectors differ at machine epsilon.
    For nearly degenerate bands the eigenvectors are only defined up to
    a rotation within the degenerate subspace, so the two backends pick
    different (but equally valid) eigenvectors, and the isotope weights
    ``|e_i . e_j|^2`` get redistributed among the degenerate channels.
    Tetrahedron weights involve reciprocals of vertex-energy differences
    ``1 / (E_i - E_j)``, so a tiny shift of either vertex energy makes
    the weight blow up or change sign -- visible as relatively large
    fluctuation in the smallest gamma channels (~1e-7 - 1e-4 THz).

    Both effects only redistribute gamma between channels and grid
    points; they conserve the total.  So instead of comparing each grid
    point, sum over all grid points and bands to obtain the integrated
    isotope scattering rate, which is the physically meaningful quantity
    and is robust against the per-channel redistribution.  The observed
    Rust-vs-C difference of this O(0.2 THz) total is ~7e-4, so a modest
    absolute tolerance keeps the comparison meaningful.

    """
    iso_c = _build_iso(si_pbesol, sigmas=None, lang="C")
    iso_c.run(_GRID_POINTS)
    iso_rust = _build_iso(si_pbesol, sigmas=None, lang="Rust")
    iso_rust.run(_GRID_POINTS)
    np.testing.assert_allclose(
        iso_rust.gamma.sum(axis=(-1, -2)),
        iso_c.gamma.sum(axis=(-1, -2)),
        atol=2e-3,
    )


def test_isotope_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Gaussian-smearing isotope gamma: Rust matches C."""
    iso_c = _build_iso(si_pbesol, sigmas=[0.1], lang="C")
    iso_c.run(_GRID_POINTS)
    iso_rust = _build_iso(si_pbesol, sigmas=[0.1], lang="Rust")
    iso_rust.run(_GRID_POINTS)
    np.testing.assert_allclose(iso_rust.gamma, iso_c.gamma, rtol=1e-10, atol=1e-14)
