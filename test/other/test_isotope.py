"""Tests for isotope scatterings."""

import numpy as np
import pytest

from phono3py import Phono3pyIsotope

si_pbesol_iso = [
    [
        8.32325038e-07,
        9.45389739e-07,
        1.57942189e-05,
        1.28121297e-03,
        1.13842605e-03,
        3.84915211e-04,
    ],
    [
        2.89457649e-05,
        1.57841863e-04,
        3.97462227e-04,
        1.03489892e-02,
        4.45981554e-03,
        2.67184355e-03,
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
    [0.000141, 0.000161, 0.000599, 0.001332, 0.017676, 0.012157],
    [0.000227, 0.00039, 0.000187, 0.001136, 0.01043, 0.01381],
]
si_pbesol_grg_iso_sigma = [
    [0.000129, 0.000154, 0.000677, 0.001306, 0.011859, 0.010465],
    [0.000227, 0.000395, 0.000181, 0.001216, 0.010474, 0.012425],
]


@pytest.mark.parametrize("lang", ["C", "Py"])
def test_Phono3pyIsotope(si_pbesol, lang):
    """Phono3pyIsotope with tetrahedron method."""
    si_pbesol.mesh_numbers = [21, 21, 21]
    iso = Phono3pyIsotope(
        si_pbesol.mesh_numbers,
        si_pbesol.phonon_primitive,
        symprec=si_pbesol.symmetry.tolerance,
    )
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params,
    )
    iso.run([23, 103], lang=lang)
    # print(iso.gamma[0])
    np.testing.assert_allclose(si_pbesol_iso, iso.gamma[0], atol=3e-4)


@pytest.mark.parametrize("lang", ["C", "Py"])
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
    )
    iso.init_dynamical_matrix(
        si_pbesol.fc2,
        si_pbesol.phonon_supercell,
        si_pbesol.phonon_primitive,
        nac_params=si_pbesol.nac_params,
    )
    iso.run([23, 103], lang=lang)
    # print(iso.gamma[0])
    np.testing.assert_allclose(si_pbesol_iso_sigma, iso.gamma[0], atol=3e-4)


@pytest.mark.parametrize("lang", ["C", "Py"])
def test_Phono3pyIsotope_grg(si_pbesol_grg, lang):
    """Phono3pyIsotope with tetrahedron method and GR-grid."""
    ph3 = si_pbesol_grg
    iso = Phono3pyIsotope(
        80,
        ph3.phonon_primitive,
        symprec=ph3.symmetry.tolerance,
        use_grg=True,
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
    iso.run([23, 103], lang=lang)
    np.testing.assert_allclose(si_pbesol_grg_iso, iso.gamma[0], atol=2e-3)


@pytest.mark.parametrize("lang", ["C", "Py"])
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
    )
    iso.init_dynamical_matrix(
        ph3.fc2,
        ph3.phonon_supercell,
        ph3.phonon_primitive,
        nac_params=ph3.nac_params,
    )
    iso.run([23, 103], lang=lang)
    np.testing.assert_equal(
        iso.grid.grid_matrix, [[-15, 15, 15], [15, -15, 15], [15, 15, -15]]
    )
    np.testing.assert_allclose(si_pbesol_grg_iso_sigma, iso.gamma[0], atol=3e-4)
