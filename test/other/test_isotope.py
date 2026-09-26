"""Tests for isotope scatterings."""

import numpy as np
import pytest

from phono3py import Phono3pyIsotope
from phono3py.other.isotope import Isotope, get_mass_variances

si_pbesol_iso = [
    [
        9.29165496e-07,
        8.91095194e-07,
        1.59076880e-05,
        1.37564273e-03,
        1.22769818e-03,
        6.78849406e-04,
    ],
    [
        3.02404246e-05,
        1.58721060e-04,
        3.99215630e-04,
        1.03909232e-02,
        4.58409106e-03,
        2.89547273e-03,
    ],
]
si_pbesol_iso_sigma = [
    [
        1.57262391e-06,
        1.64031282e-06,
        2.02007165e-05,
        1.41999212e-03,
        1.26361419e-03,
        7.91243161e-04,
    ],
    [
        3.10266472e-05,
        1.53059329e-04,
        3.80963936e-04,
        1.05238031e-02,
        6.72552880e-03,
        3.21592329e-03,
    ],
]
si_pbesol_grg_iso = [
    [0.000140, 0.000162, 0.000524, 0.001383, 0.017868, 0.015051],
    [0.000228, 0.000379, 0.000203, 0.001178, 0.010589, 0.013985],
]
si_pbesol_grg_iso_sigma = [
    [0.000129, 0.000154, 0.000677, 0.001306, 0.011859, 0.010465],
    [0.000227, 0.000395, 0.000181, 0.001216, 0.010474, 0.012425],
]


def test_get_mass_variances_from_primitive(si_pbesol):
    """get_mass_variances returns correct shape and regression value from primitive."""
    mv = get_mass_variances(primitive=si_pbesol.phonon_primitive)
    num_atoms = len(si_pbesol.phonon_primitive)
    assert mv.shape == (num_atoms,)
    # Both Si atoms in primitive have the same variance
    np.testing.assert_allclose(mv[0], mv[-1])
    np.testing.assert_allclose(mv[0], 2.0070046e-04, rtol=1e-5)


def test_get_mass_variances_from_symbols():
    """get_mass_variances from symbols returns correct regression values."""
    mv_sym = get_mass_variances(symbols=["Si", "Si"])
    assert mv_sym.shape == (2,)
    assert mv_sym.dtype == np.double
    np.testing.assert_allclose(mv_sym, [2.0070046e-04, 2.0070046e-04], rtol=1e-5)


def test_get_mass_variances_custom_isotope_data():
    """get_mass_variances with custom isotope_data overrides default."""
    # Pure Si-28 (no isotope spread) should give variance ~0
    custom = {"Si": [(14, 28.0, 1.0)]}  # single isotope, variance = 0
    mv = get_mass_variances(symbols=["Si"], isotope_data=custom)
    np.testing.assert_allclose(mv, [0.0], atol=1e-10)


def test_get_mass_variances_no_args_raises():
    """get_mass_variances raises RuntimeError with no primitive or symbols."""
    with pytest.raises(RuntimeError):
        get_mass_variances()


@pytest.mark.parametrize("lang", ["C", "Rust"])
def test_Phono3pyIsotope(si_pbesol, lang):
    """Phono3pyIsotope with tetrahedron method."""
    if lang == "C":
        pytest.importorskip("phonopy._phonopy")
    si_pbesol.mesh_numbers = [21, 21, 21]
    iso = Phono3pyIsotope(
        si_pbesol.mesh_numbers,
        si_pbesol.phonon_primitive,
        symprec=si_pbesol.symmetry.tolerance,
        lang=lang,
    )
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params,
    )
    iso.run([23, 103])
    # print(iso.gamma[0])
    np.testing.assert_allclose(si_pbesol_iso, iso.gamma[0], atol=3e-4)


@pytest.mark.parametrize("lang", ["C", "Python", "Rust"])
def test_Phono3pyIsotope_with_sigma(si_pbesol, lang):
    """Phono3pyIsotope with smearing method."""
    si_pbesol.mesh_numbers = [21, 21, 21]
    iso = Phono3pyIsotope(
        si_pbesol.mesh_numbers,
        si_pbesol.phonon_primitive,
        sigmas=[
            0.1,
        ],
        symprec=si_pbesol.symmetry.tolerance,
        lang=lang,
    )
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params,
    )
    iso.run([23, 103])
    # print(iso.gamma[0])
    np.testing.assert_allclose(si_pbesol_iso_sigma, iso.gamma[0], atol=3e-4)


@pytest.mark.parametrize("lang", ["C", "Rust"])
def test_Phono3pyIsotope_grg(si_pbesol_grg, lang):
    """Phono3pyIsotope with tetrahedron method and GR-grid."""
    if lang == "C":
        pytest.importorskip("phonopy._phonopy")
    ph3 = si_pbesol_grg
    iso = Phono3pyIsotope(
        80,
        ph3.phonon_primitive,
        symprec=ph3.symmetry.tolerance,
        use_grg=True,
        lang=lang,
    )
    iso.init_dynamical_matrix(
        ph3.fc2,
        ph3.phonon_supercell,
        ph3.phonon_primitive,
        nac_params=ph3.nac_params,
    )
    np.testing.assert_equal(
        iso.grid.grid_matrix, [[-15, 15, 15], [15, -15, 15], [15, 15, -15]]
    )
    iso.run([23, 103])
    np.testing.assert_allclose(si_pbesol_grg_iso, iso.gamma[0], atol=3e-3)


@pytest.mark.parametrize("lang", ["C", "Python", "Rust"])
def test_Phono3pyIsotope_grg_with_sigma(si_pbesol_grg, lang):
    """Phono3pyIsotope with smearing method and GR-grid."""
    ph3 = si_pbesol_grg
    iso = Phono3pyIsotope(
        80,
        ph3.phonon_primitive,
        sigmas=[
            0.1,
        ],
        symprec=ph3.symmetry.tolerance,
        use_grg=True,
        lang=lang,
    )
    iso.init_dynamical_matrix(
        ph3.fc2,
        ph3.phonon_supercell,
        ph3.phonon_primitive,
        nac_params=ph3.nac_params,
    )
    iso.run([23, 103])
    np.testing.assert_equal(
        iso.grid.grid_matrix, [[-15, 15, 15], [15, -15, 15], [15, 15, -15]]
    )
    np.testing.assert_allclose(si_pbesol_grg_iso_sigma, iso.gamma[0], atol=3e-4)


def test_Phono3pyIsotope_symmetrize_tetrahedra(aln_lda):
    """Phono3pyIsotope passes symmetrize_tetrahedra to Isotope."""
    gammas = []
    for symmetrize_tetrahedra in (False, True):
        iso = Phono3pyIsotope(
            [6, 6, 4],
            aln_lda.phonon_primitive,
            symprec=aln_lda.symmetry.tolerance,
            symmetrize_tetrahedra=symmetrize_tetrahedra,
        )
        iso.init_dynamical_matrix(
            aln_lda.fc2,
            aln_lda.phonon_supercell,
            aln_lda.phonon_primitive,
            nac_params=aln_lda.nac_params,
        )
        iso.run([1])
        gammas.append(iso.gamma[0])
    assert abs(gammas[0] - gammas[1]).max() > 1e-6


@pytest.mark.parametrize(
    "ph3_name,mesh,use_grg,symmetrize_tetrahedra",
    [
        ("si_pbesol", [7, 7, 7], False, False),
        ("si_pbesol_grg", 20, True, False),
        ("aln_lda", [4, 4, 2], False, True),
    ],
)
def test_Isotope_python_matches_rust(
    request, ph3_name, mesh, use_grg, symmetrize_tetrahedra
):
    """The pure-Python tetrahedron path gives the gamma of the Rust path.

    The Python path is the prototype of the Rust one, and it is slow, so it is
    checked against Rust on a small mesh rather than against the references.

    Both paths get the same phonons. The frequency points are the frequencies
    at the grid point itself, which vertices of symmetrically equivalent points
    share, so phonons solved separately, apart by 1e-7, move the weights.

    """
    ph3 = request.getfixturevalue(ph3_name)
    isotopes = {}
    for lang in ("Rust", "Python"):
        iso = Isotope(
            mesh,
            ph3.phonon_primitive,
            symprec=ph3.symmetry.tolerance,
            use_grg=use_grg,
            symmetrize_tetrahedra=symmetrize_tetrahedra,
            lang=lang,
        )
        iso.init_dynamical_matrix(
            ph3.fc2,
            ph3.phonon_supercell,
            ph3.phonon_primitive,
            nac_params=ph3.nac_params,
        )
        isotopes[lang] = iso
    for grid_point in (1, 10):
        isotopes["Rust"].set_grid_point(grid_point)
        isotopes["Rust"].run()
        frequencies, eigenvectors, phonon_done = isotopes["Rust"].get_phonons()
        isotopes["Python"].set_phonons(
            frequencies.copy(), eigenvectors.copy(), phonon_done.copy()
        )
        isotopes["Python"].set_grid_point(grid_point)
        isotopes["Python"].run()
        np.testing.assert_allclose(
            isotopes["Python"].gamma, isotopes["Rust"].gamma, rtol=1e-10, atol=1e-16
        )
