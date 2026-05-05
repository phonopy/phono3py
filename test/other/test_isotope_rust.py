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


def test_isotope_rust_vs_c_tetrahedron(si_pbesol: Phono3py):
    """Tetrahedron-method isotope gamma: Rust matches C."""
    grid_points = [23, 103]
    iso_c = _build_iso(si_pbesol, sigmas=None, lang="C")
    iso_c.run(grid_points)
    iso_rust = _build_iso(si_pbesol, sigmas=None, lang="Rust")
    iso_rust.run(grid_points)
    np.testing.assert_allclose(iso_rust.gamma, iso_c.gamma, rtol=1e-10, atol=1e-14)


def test_isotope_rust_vs_c_sigma(si_pbesol: Phono3py):
    """Gaussian-smearing isotope gamma: Rust matches C."""
    grid_points = [23, 103]
    iso_c = _build_iso(si_pbesol, sigmas=[0.1], lang="C")
    iso_c.run(grid_points)
    iso_rust = _build_iso(si_pbesol, sigmas=[0.1], lang="Rust")
    iso_rust.run(grid_points)
    np.testing.assert_allclose(iso_rust.gamma, iso_c.gamma, rtol=1e-10, atol=1e-14)
