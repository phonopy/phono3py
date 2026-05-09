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
    order, so eigenvalues differ at machine epsilon.  Tetrahedron
    weights involve reciprocals of vertex-energy differences
    ``1 / (E_i - E_j)``, so when two vertex energies happen to be
    nearly degenerate, a tiny shift of either makes the weight blow
    up or change sign -- visible as relatively large fluctuation in
    the smallest gamma channels (~1e-7 - 1e-4 THz).  Summing over
    bands cancels the per-channel redistribution and yields the
    integrated isotope scattering rate, which is the physically
    meaningful quantity.  ``gamma`` is itself a physical rate so an
    absolute tolerance is more informative than a relative one.

    """
    iso_c = _build_iso(si_pbesol, sigmas=None, lang="C")
    iso_c.run(_GRID_POINTS)
    iso_rust = _build_iso(si_pbesol, sigmas=None, lang="Rust")
    iso_rust.run(_GRID_POINTS)
    np.testing.assert_allclose(
        iso_rust.gamma.sum(axis=-1),
        iso_c.gamma.sum(axis=-1),
        atol=1e-3,
    )


def test_isotope_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Gaussian-smearing isotope gamma: Rust matches C."""
    iso_c = _build_iso(si_pbesol, sigmas=[0.1], lang="C")
    iso_c.run(_GRID_POINTS)
    iso_rust = _build_iso(si_pbesol, sigmas=[0.1], lang="Rust")
    iso_rust.run(_GRID_POINTS)
    np.testing.assert_allclose(iso_rust.gamma, iso_c.gamma, rtol=1e-10, atol=1e-14)
