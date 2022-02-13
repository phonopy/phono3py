"""Tests for fc3."""
import numpy as np

from phono3py.phonon3.fc3 import cutoff_fc3_by_zero


def test_cutoff_fc3(nacl_pbe_cutoff_fc3, nacl_pbe):
    """Test for cutoff pair option."""
    fc3_cut = nacl_pbe_cutoff_fc3.fc3
    fc3 = nacl_pbe.fc3
    abs_delta = np.abs(fc3_cut - fc3).sum()
    assert np.abs(1894.2058837 - abs_delta) < 1e-3


def test_cutoff_fc3_zero(nacl_pbe):
    """Test for abrupt cut of fc3 by distance."""
    ph = nacl_pbe
    fc3 = ph.fc3.copy()
    cutoff_fc3_by_zero(fc3, ph.supercell, 5)
    abs_delta = np.abs(ph.fc3 - fc3).sum()
    assert np.abs(5259.2234163 - abs_delta) < 1e-3


def test_fc3(si_pbesol_111):
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


# @pytest.mark.skipif(not FC_CALCULATOR_ALM_AVAILABLE, reason="not found ALM package")
def test_fc3_alm(si_pbesol_111_alm):
    """Test fc3 with Si PBEsol 1x1x1 calcualted using ALM."""
    ph = si_pbesol_111_alm
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
