"""Tests for fc3."""

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.fc3 import (
    cutoff_fc3_by_zero,
    get_fc3,
    set_permutation_symmetry_fc3,
    set_translational_invariance_fc3,
)


def test_cutoff_fc3(nacl_pbe_cutoff_fc3: Phono3py, nacl_pbe: Phono3py):
    """Test for cutoff-pair-distance option.

    Only supercell forces that satisfy specified cutoff pairs are set in
    dataset preparation.

    """
    fc3_cut = nacl_pbe_cutoff_fc3.fc3
    fc3 = nacl_pbe.fc3
    abs_delta = np.abs(fc3_cut - fc3).sum()

    assert fc3.shape == (64, 64, 64, 3, 3, 3)
    assert np.abs(1901.0248613 - abs_delta) < 1e-3


def test_cutoff_fc3_all_forces(
    nacl_pbe_cutoff_fc3: Phono3py, nacl_pbe_cutoff_fc3_all_forces: Phono3py
):
    """Test for cutoff-pair-distance option with all forces are set.

    By definition, displacement datasets are kept unchanged when
    cutoff-pair-distance is specified.

    This test checkes only supercell forces that satisfy specified cutoff pairs
    are chosen properly.

    """
    fc3_cut = nacl_pbe_cutoff_fc3.fc3
    fc3_cut_all_forces = nacl_pbe_cutoff_fc3_all_forces.fc3
    np.testing.assert_allclose(fc3_cut, fc3_cut_all_forces, atol=1e-8)


def test_cutoff_fc3_compact_fc(
    nacl_pbe_cutoff_fc3_compact_fc: Phono3py, nacl_pbe_cutoff_fc3: Phono3py
):
    """Test for cutoff-pair-distance option with compact-fc."""
    fc3_cfc = nacl_pbe_cutoff_fc3_compact_fc.fc3
    fc3_full = nacl_pbe_cutoff_fc3.fc3
    p2s_map = nacl_pbe_cutoff_fc3.primitive.p2s_map
    assert fc3_cfc.shape == (2, 64, 64, 3, 3, 3)
    assert fc3_full.shape == (64, 64, 64, 3, 3, 3)
    np.testing.assert_allclose(fc3_cfc, fc3_full[p2s_map], atol=1e-8)


def test_cutoff_fc3_zero(nacl_pbe: Phono3py):
    """Test for abrupt cut of fc3 by distance."""
    ph = nacl_pbe
    fc3 = ph.fc3.copy()
    cutoff_fc3_by_zero(fc3, ph.supercell, 5)
    abs_delta = np.abs(ph.fc3 - fc3).sum()
    assert np.abs(5259.2234163 - abs_delta) < 1e-3


def test_cutoff_fc3_zero_compact_fc(nacl_pbe_compact_fc: Phono3py):
    """Test for abrupt cut of fc3 by distance."""
    ph = nacl_pbe_compact_fc
    fc3 = ph.fc3.copy()
    cutoff_fc3_by_zero(fc3, ph.supercell, 5, p2s_map=ph.primitive.p2s_map)
    abs_delta = np.abs(ph.fc3 - fc3).sum()
    assert np.abs(164.359250 - abs_delta) < 1e-3


def test_fc3(si_pbesol_111: Phono3py):
    """Test fc3 with Si PBEsol 1x1x1."""
    ph = si_pbesol_111
    fc3_ref = [
        [
            [0.10725082233071165, -3.17309835814091e-17, 9.184999404573031e-17],
            [0.008964145692710241, -0.14304691148751, -0.13849893745060796],
            [-0.008964145692710304, -0.13849893745060804, -0.14304691148750995],
        ],
        [
            [-0.008964145692710266, -0.14304691148750992, -0.13849893745060804],
            [-0.03394571572679527, -1.5305320668253703e-17, -2.419577848263484e-17],
            [-0.3317461672212566, -0.026002572441157376, -0.026002572441157404],
        ],
        [
            [0.008964145692710323, -0.13849893745060782, -0.14304691148750995],
            [-0.3317461672212566, 0.026002572441157404, 0.026002572441157387],
            [-0.033945715726795195, -1.4289784633358948e-16, 1.3426036902612163e-17],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-8, rtol=0)


def test_phonon_smat_fd(si_pbesol_111_222_fd: Phono3py):
    """Test phonon smat with Si PBEsol 1x1x1-2x2x2."""
    ph = si_pbesol_111_222_fd
    fc3_ref = [
        [
            [1.07250822e-01, 1.86302073e-17, -4.26452855e-18],
            [8.96414569e-03, -1.43046911e-01, -1.38498937e-01],
            [-8.96414569e-03, -1.38498937e-01, -1.43046911e-01],
        ],
        [
            [-8.96414569e-03, -1.43046911e-01, -1.38498937e-01],
            [-3.39457157e-02, -4.63315728e-17, -4.17779237e-17],
            [-3.31746167e-01, -2.60025724e-02, -2.60025724e-02],
        ],
        [
            [8.96414569e-03, -1.38498937e-01, -1.43046911e-01],
            [-3.31746167e-01, 2.60025724e-02, 2.60025724e-02],
            [-3.39457157e-02, 3.69351540e-17, 5.94504191e-18],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-8, rtol=0)

    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_symfc(si_pbesol_111_222_symfc: Phono3py):
    """Test phonon smat and symfc with Si PBEsol 1x1x1-2x2x2."""
    ph = si_pbesol_111_222_symfc
    fc3_ref = [
        [
            [0.10725082, 0.0, 0.0],
            [-0.04225275, -0.09187669, -0.1386571],
            [0.04225275, -0.1386571, -0.09187669],
        ],
        [
            [0.04225275, -0.09187669, -0.1386571],
            [-0.17073504, 0.0, 0.0],
            [-0.33192165, 0.02516976, 0.02516976],
        ],
        [
            [-0.04225275, -0.1386571, -0.09187669],
            [-0.33192165, -0.02516976, -0.02516976],
            [-0.17073504, 0.0, 0.0],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_symfc_fd(si_pbesol_111_222_symfc_fd: Phono3py):
    """Test phonon smat and symfc (fc2) FD (fc3) with Si PBEsol 1x1x1-2x2x2."""
    ph = si_pbesol_111_222_symfc_fd
    fc3_ref = [
        [
            [1.07250822e-01, 1.86302073e-17, -4.26452855e-18],
            [8.96414569e-03, -1.43046911e-01, -1.38498937e-01],
            [-8.96414569e-03, -1.38498937e-01, -1.43046911e-01],
        ],
        [
            [-8.96414569e-03, -1.43046911e-01, -1.38498937e-01],
            [-3.39457157e-02, -4.63315728e-17, -4.17779237e-17],
            [-3.31746167e-01, -2.60025724e-02, -2.60025724e-02],
        ],
        [
            [8.96414569e-03, -1.38498937e-01, -1.43046911e-01],
            [-3.31746167e-01, 2.60025724e-02, 2.60025724e-02],
            [-3.39457157e-02, 3.69351540e-17, 5.94504191e-18],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_fd_symfc(si_pbesol_111_222_fd_symfc: Phono3py):
    """Test phonon smat and FD (fc2) symfc (fc3) with Si PBEsol 1x1x1-2x2x2."""
    ph = si_pbesol_111_222_fd_symfc
    fc3_ref = [
        [
            [0.10725082, 0.0, 0.0],
            [-0.04225275, -0.09187669, -0.1386571],
            [0.04225275, -0.1386571, -0.09187669],
        ],
        [
            [0.04225275, -0.09187669, -0.1386571],
            [-0.17073504, 0.0, 0.0],
            [-0.33192165, 0.02516976, 0.02516976],
        ],
        [
            [-0.04225275, -0.1386571, -0.09187669],
            [-0.33192165, -0.02516976, -0.02516976],
            [-0.17073504, 0.0, 0.0],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_alm_cutoff(si_pbesol_111_222_alm_cutoff: Phono3py):
    """Test phonon smat and alm with Si PBEsol 1x1x1-2x2x2 cutoff."""
    ph = si_pbesol_111_222_alm_cutoff
    np.testing.assert_allclose(ph.fc3[0, 1, 7], 0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph.fc2[0, 33], 0, atol=1e-6, rtol=0)


def test_phonon_smat_alm_cutoff_fc2(si_pbesol_111_222_alm_cutoff_fc2: Phono3py):
    """Test phonon smat and alm with Si PBEsol 1x1x1-2x2x2 cutoff fc2."""
    ph = si_pbesol_111_222_alm_cutoff_fc2
    fc3_ref = [
        [
            [0.10725082, 0.0, 0.0],
            [-0.04225275, -0.09187669, -0.1386571],
            [0.04225275, -0.1386571, -0.09187669],
        ],
        [
            [0.04225275, -0.09187669, -0.1386571],
            [-0.17073504, 0.0, 0.0],
            [-0.33192165, 0.02516976, 0.02516976],
        ],
        [
            [-0.04225275, -0.1386571, -0.09187669],
            [-0.33192165, -0.02516976, -0.02516976],
            [-0.17073504, 0.0, 0.0],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph.fc2[0, 33], 0, atol=1e-6, rtol=0)


def test_phonon_smat_alm_cutoff_fc3(si_pbesol_111_222_alm_cutoff_fc3: Phono3py):
    """Test phonon smat and alm with Si PBEsol 1x1x1-2x2x2 cutoff fc3."""
    ph = si_pbesol_111_222_alm_cutoff_fc3
    np.testing.assert_allclose(ph.fc3[0, 1, 7], 0, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


@pytest.mark.parametrize("pinv_solver", ["numpy", "lapacke"])
def test_fc3_lapacke_solver(si_pbesol_111: Phono3py, pinv_solver: str):
    """Test fc3 with Si PBEsol 1x1x1 using lapacke solver."""
    ph = si_pbesol_111
    _, fc3 = get_fc3(
        ph.supercell,
        ph.primitive,
        ph.dataset,
        ph.symmetry,
        pinv_solver=pinv_solver,
        verbose=True,
    )
    set_translational_invariance_fc3(fc3)
    set_permutation_symmetry_fc3(fc3)

    fc3_ref = [
        [
            [1.07250822e-01, 1.86302073e-17, -4.26452855e-18],
            [8.96414569e-03, -1.43046911e-01, -1.38498937e-01],
            [-8.96414569e-03, -1.38498937e-01, -1.43046911e-01],
        ],
        [
            [-8.96414569e-03, -1.43046911e-01, -1.38498937e-01],
            [-3.39457157e-02, -4.63315728e-17, -4.17779237e-17],
            [-3.31746167e-01, -2.60025724e-02, -2.60025724e-02],
        ],
        [
            [8.96414569e-03, -1.38498937e-01, -1.43046911e-01],
            [-3.31746167e-01, 2.60025724e-02, 2.60025724e-02],
            [-3.39457157e-02, 3.69351540e-17, 5.94504191e-18],
        ],
    ]

    np.testing.assert_allclose(fc3[0, 1, 7], fc3_ref, atol=1e-8, rtol=0)


def test_fc3_symfc(si_pbesol_111_symfc: Phono3py):
    """Test fc3 with Si PBEsol 1x1x1 calcualted using symfc."""
    ph = si_pbesol_111_symfc
    fc3_ref = [
        [
            [0.10725082233069763, 0.0, 0.0],
            [-0.04225274805794354, -0.09187668739926935, -0.13865710308133664],
            [0.04225274805794354, -0.13865710308133664, -0.09187668739926935],
        ],
        [
            [0.04225274805794354, -0.09187668739926935, -0.13865710308133664],
            [-0.17073503897042558, 0.0, 0.0],
            [-0.33192165463027573, 0.02516976132993421, 0.02516976132993421],
        ],
        [
            [-0.04225274805794354, -0.13865710308133664, -0.09187668739926935],
            [-0.33192165463027573, -0.02516976132993421, -0.02516976132993421],
            [-0.17073503897042558, 0.0, 0.0],
        ],
    ]
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-8, rtol=0)
