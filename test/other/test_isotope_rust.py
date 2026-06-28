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

    Per-element parity is not achievable.  Even though both backends
    solve the phonons with the same C routine, the C and Rust grid
    handling feed slightly different (but equivalent) q images into the
    dynamical matrix, so the eigenvalues differ at machine epsilon
    (~1e-14 THz) over most of the spectrum and the eigenvectors differ
    by an O(1) rotation within each degenerate subspace.

    The tetrahedron integration weight is a piecewise-rational function
    of the vertex energies.  By crystal symmetry the probe frequency at
    a grid point coincides to the last bit with the vertex energies at
    the symmetry-equivalent grid points, so the weight sits exactly on a
    branch boundary: a machine-epsilon shift of a vertex energy flips it
    and changes that grid point's band sum by as much as ~15%.  These
    flips do not cancel cleanly -- they leave a small systematic bias
    plus architecture-dependent scatter -- so even the total summed over
    grid points and bands is only reproducible to ~1% across BLAS / CPU
    architectures (the C total alone moves by ~1e-3 between machines,
    while the Rust total stays put).

    There is therefore no point in comparing per grid point.  Sum over
    all grid points and bands to get the integrated isotope scattering
    rate and compare that O(0.2 THz) total with a loose absolute
    tolerance that absorbs the ~1% knife-edge irreproducibility.

    """
    iso_c = _build_iso(si_pbesol, sigmas=None, lang="C")
    iso_c.run(_GRID_POINTS)
    iso_rust = _build_iso(si_pbesol, sigmas=None, lang="Rust")
    iso_rust.run(_GRID_POINTS)
    np.testing.assert_allclose(
        iso_rust.gamma.sum(),
        iso_c.gamma.sum(),
        atol=1e-2,
    )


def test_isotope_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Gaussian-smearing isotope gamma: Rust matches C."""
    iso_c = _build_iso(si_pbesol, sigmas=[0.1], lang="C")
    iso_c.run(_GRID_POINTS)
    iso_rust = _build_iso(si_pbesol, sigmas=[0.1], lang="Rust")
    iso_rust.run(_GRID_POINTS)
    np.testing.assert_allclose(iso_rust.gamma, iso_c.gamma, rtol=1e-10, atol=1e-14)
