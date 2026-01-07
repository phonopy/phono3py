"""Tests for fc3."""

from __future__ import annotations

import numpy as np
import phono3py._phono3py as phono3c
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

    assert fc3 is not None
    assert fc3_cut is not FileNotFoundError
    abs_delta = np.abs(fc3_cut - fc3).sum()

    assert fc3.shape == (64, 64, 64, 3, 3, 3)
    assert np.abs(1901.0248613 - abs_delta) < 1e-3


def test_cutoff_fc3_all_forces(
    nacl_pbe_cutoff_fc3: Phono3py, nacl_pbe_cutoff_fc3_all_forces: Phono3py
):
    """Test for cutoff-pair-distance option with all forces are set.

    By definition, displacement datasets are kept unchanged when
    cutoff-pair-distance is specified.

    This test checks only supercell forces that satisfy specified cutoff pairs
    are chosen properly.

    """
    fc3_cut = nacl_pbe_cutoff_fc3.fc3
    fc3_cut_all_forces = nacl_pbe_cutoff_fc3_all_forces.fc3
    assert fc3_cut is not None
    assert fc3_cut_all_forces is not None
    np.testing.assert_allclose(fc3_cut, fc3_cut_all_forces, atol=1e-8)


def test_cutoff_fc3_compact_fc(
    nacl_pbe_cutoff_fc3_compact_fc: Phono3py, nacl_pbe_cutoff_fc3: Phono3py
):
    """Test for cutoff-pair-distance option with compact-fc."""
    fc3_cfc = nacl_pbe_cutoff_fc3_compact_fc.fc3
    fc3_full = nacl_pbe_cutoff_fc3.fc3
    p2s_map = nacl_pbe_cutoff_fc3.primitive.p2s_map
    assert fc3_cfc is not None
    assert fc3_full is not None
    assert fc3_cfc.shape == (2, 64, 64, 3, 3, 3)
    assert fc3_full.shape == (64, 64, 64, 3, 3, 3)
    np.testing.assert_allclose(fc3_cfc, fc3_full[p2s_map], atol=1e-8)


def test_cutoff_fc3_zero(nacl_pbe: Phono3py):
    """Test for abrupt cut of fc3 by distance."""
    ph = nacl_pbe
    assert ph.fc3 is not None
    fc3 = ph.fc3.copy()
    cutoff_fc3_by_zero(fc3, ph.supercell, 5)
    abs_delta = np.abs(ph.fc3 - fc3).sum()
    assert np.abs(5259.2234163 - abs_delta) < 1e-3


def test_cutoff_fc3_zero_compact_fc(nacl_pbe_compact_fc: Phono3py):
    """Test for abrupt cut of fc3 by distance."""
    ph = nacl_pbe_compact_fc
    assert ph.fc3 is not None
    fc3 = ph.fc3.copy()
    cutoff_fc3_by_zero(fc3, ph.supercell, 5, p2s_map=ph.primitive.p2s_map)
    abs_delta = np.abs(ph.fc3 - fc3).sum()
    assert np.abs(164.350663 - abs_delta) < 1e-3


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
    assert ph.fc3 is not None
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
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-8, rtol=0)

    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    assert ph.fc2 is not None
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_symfc(si_pbesol_111_222_symfc: Phono3py):
    """Test phonon smat and symfc with Si PBEsol 1x1x1-2x2x2."""
    ph = si_pbesol_111_222_symfc
    fc3_ref = [
        [
            [0.10725082233070223, 0.0, 0.0],
            [0.008901878238903862, -0.14303131369612382, -0.1385764832859841],
            [-0.008901878238903862, -0.1385764832859841, -0.14303131369612382],
        ],
        [
            [-0.008901878238903862, -0.14303131369612382, -0.1385764832859841],
            [-0.03432270217882153, 0.0, 0.0],
            [-0.33182990143137375, -0.025984864966909885, -0.025984864966909885],
        ],
        [
            [0.008901878238903862, -0.1385764832859841, -0.14303131369612382],
            [-0.33182990143137375, 0.025984864966909885, 0.025984864966909885],
            [-0.03432270217882153, 0.0, 0.0],
        ],
    ]
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    assert ph.fc2 is not None
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
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    assert ph.fc2 is not None
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_fd_symfc(si_pbesol_111_222_fd_symfc: Phono3py):
    """Test phonon smat and FD (fc2) symfc (fc3) with Si PBEsol 1x1x1-2x2x2."""
    ph = si_pbesol_111_222_fd_symfc
    fc3_ref = [
        [
            [0.10725082233070223, 0.0, 0.0],
            [0.008901878238903862, -0.14303131369612382, -0.1385764832859841],
            [-0.008901878238903862, -0.1385764832859841, -0.14303131369612382],
        ],
        [
            [-0.008901878238903862, -0.14303131369612382, -0.1385764832859841],
            [-0.03432270217882153, 0.0, 0.0],
            [-0.33182990143137375, -0.025984864966909885, -0.025984864966909885],
        ],
        [
            [0.008901878238903862, -0.1385764832859841, -0.14303131369612382],
            [-0.33182990143137375, 0.025984864966909885, 0.025984864966909885],
            [-0.03432270217882153, 0.0, 0.0],
        ],
    ]
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    assert ph.fc2 is not None
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


def test_phonon_smat_alm_cutoff(si_pbesol_111_222_alm_cutoff: Phono3py):
    """Test phonon smat and alm with Si PBEsol 1x1x1-2x2x2 cutoff."""
    ph = si_pbesol_111_222_alm_cutoff
    assert ph.fc3 is not None
    assert ph.fc2 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], 0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph.fc2[0, 33], 0, atol=1e-6, rtol=0)


def test_phonon_smat_symfc_cutoff(si_pbesol_111_222_symfc_cutoff: Phono3py):
    """Test phonon smat and symfc with Si PBEsol 1x1x1-2x2x2 cutoff."""
    ph = si_pbesol_111_222_symfc_cutoff
    assert ph.fc3 is not None
    assert ph.fc2 is not None
    assert ph.fc3_nonzero_indices is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], 0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph.fc2[0, 33], 0, atol=1e-6, rtol=0)
    assert ph.fc3.shape == (8, 8, 8, 3, 3, 3)
    assert ph.fc3_nonzero_indices.shape == (8, 8, 8)
    assert ph.fc3_nonzero_indices.sum() == 104


def test_phonon_smat_symfc_cutoff_compact_fc(
    si_pbesol_111_222_symfc_cutoff_compact_fc: Phono3py,
):
    """Test phonon smat and symfc with Si PBEsol 1x1x1-2x2x2 cutoff."""
    ph = si_pbesol_111_222_symfc_cutoff_compact_fc
    assert ph.fc3 is not None
    assert ph.fc2 is not None
    assert ph.fc3_nonzero_indices is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], 0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph.fc2[0, 33], 0, atol=1e-6, rtol=0)
    assert ph.fc3.shape == (2, 8, 8, 3, 3, 3)
    assert ph.fc3_nonzero_indices.shape == (2, 8, 8)
    assert ph.fc3_nonzero_indices.sum() == 26


def test_phonon_smat_alm_cutoff_fc2(si_pbesol_111_222_alm_cutoff_fc2: Phono3py):
    """Test phonon smat and alm with Si PBEsol 1x1x1-2x2x2 cutoff fc2."""
    ph = si_pbesol_111_222_alm_cutoff_fc2
    fc3_ref = [
        [
            [0.10725082233070826, 0.0, 0.0],
            [0.008901878238907217, -0.1430313136961221, -0.13857648328597594],
            [-0.008901878238907217, -0.13857648328597594, -0.1430313136961221],
        ],
        [
            [-0.008901878238907217, -0.1430313136961221, -0.13857648328597594],
            [-0.03432270217882977, 0.0, 0.0],
            [-0.33182990143136837, -0.02598486496691343, -0.02598486496691343],
        ],
        [
            [0.008901878238907217, -0.13857648328597594, -0.1430313136961221],
            [-0.33182990143136837, 0.02598486496691343, 0.02598486496691343],
            [-0.03432270217882977, 0.0, 0.0],
        ],
    ]
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-6, rtol=0)
    assert ph.fc2 is not None
    np.testing.assert_allclose(ph.fc2[0, 33], 0, atol=1e-6, rtol=0)


def test_phonon_smat_alm_cutoff_fc3(si_pbesol_111_222_alm_cutoff_fc3: Phono3py):
    """Test phonon smat and alm with Si PBEsol 1x1x1-2x2x2 cutoff fc3."""
    ph = si_pbesol_111_222_alm_cutoff_fc3
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], 0, atol=1e-6, rtol=0)
    fc2_ref = [
        [-0.20333398, -0.0244225, -0.0244225],
        [-0.0244225, -0.02219682, -0.024112],
        [-0.0244225, -0.024112, -0.02219682],
    ]
    assert ph.fc2 is not None
    np.testing.assert_allclose(ph.fc2[0, 33], fc2_ref, atol=1e-6, rtol=0)


@pytest.mark.skipif(
    not phono3c.include_lapacke(), reason="requires to compile with lapacke"
)
def test_fc3_lapacke_solver(si_pbesol_111: Phono3py):
    """Test fc3 with Si PBEsol 1x1x1 using lapacke solver."""
    for pinv_solver in ["lapacke", "numpy"]:
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
    """Test fc3 with Si PBEsol 1x1x1 calculated using symfc."""
    ph = si_pbesol_111_symfc
    fc3_ref = [
        [
            [0.10725082233070223, 0.0, 0.0],
            [0.008901878238903862, -0.14303131369612382, -0.1385764832859841],
            [-0.008901878238903862, -0.1385764832859841, -0.14303131369612382],
        ],
        [
            [-0.008901878238903862, -0.14303131369612382, -0.1385764832859841],
            [-0.03432270217882153, 0.0, 0.0],
            [-0.33182990143137375, -0.025984864966909885, -0.025984864966909885],
        ],
        [
            [0.008901878238903862, -0.1385764832859841, -0.14303131369612382],
            [-0.33182990143137375, 0.025984864966909885, 0.025984864966909885],
            [-0.03432270217882153, 0.0, 0.0],
        ],
    ]
    assert ph.fc3 is not None
    np.testing.assert_allclose(ph.fc3[0, 1, 7], fc3_ref, atol=1e-8, rtol=0)
