import numpy as np
from phono3py.phonon3.fc3 import cutoff_fc3_by_zero


def test_cutoff_fc3(nacl_pbe_cutoff_fc3, nacl_pbe):
    fc3_cut = nacl_pbe_cutoff_fc3.fc3
    fc3 = nacl_pbe.fc3
    abs_delta = np.abs(fc3_cut - fc3).sum()
    assert np.abs(1894.2058837 - abs_delta) < 1e-3


def test_cutoff_fc3_zero(nacl_pbe):
    ph = nacl_pbe
    fc3 = ph.fc3.copy()
    cutoff_fc3_by_zero(fc3, ph.supercell, 5)
    abs_delta = np.abs(ph.fc3 - fc3).sum()
    assert np.abs(5259.2234163 - abs_delta) < 1e-3
