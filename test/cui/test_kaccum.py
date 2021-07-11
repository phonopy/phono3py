import numpy as np
from phono3py.cui.phono3py_kaccum import KappaDOS
from phono3py.phonon.grid import get_ir_grid_points

kappados_si = [-0.0000002, 0.0000000, 0.0000000,
               1.6966400, 2.1977566, 5.1814323,
               3.3932803, 25.8022392, 15.5096766,
               5.0899206, 56.6994259, 19.4995156,
               6.7865608, 68.7759426, 3.2465477,
               8.4832011, 72.8398965, 1.6583881,
               10.1798413, 74.8143686, 0.7945952,
               11.8764816, 77.2489625, 5.4385183,
               13.5731219, 80.9162245, 0.5998735,
               15.2697621, 81.4303646, 0.0000000]
kappados_nacl = [-0.0000002, 0.0000000, 0.0000000,
                 0.8051732, 0.0366488, 0.1820668,
                 1.6103466, 0.7748514, 1.5172957,
                 2.4155199, 2.0165794, 2.0077744,
                 3.2206933, 4.6670801, 2.8357892,
                 4.0258667, 6.6123781, 32.8560281,
                 4.8310401, 7.7105916, 0.6136893,
                 5.6362134, 7.9112790, 0.2391300,
                 6.4413868, 8.0272187, 0.0604842,
                 7.2465602, 8.0430831, 0.0000000]


def test_kappados_si(si_pbesol):
    ph3 = si_pbesol
    freq_points, kdos = _calculate_kappados_all_frequencies(ph3)
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_si,
        np.vstack((freq_points, kdos.T)).T.ravel(),
        rtol=0, atol=1e-5)


def test_kappados_nacl(nacl_pbe):
    ph3 = nacl_pbe
    freq_points, kdos = _calculate_kappados_all_frequencies(ph3)
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_nacl,
        np.vstack((freq_points, kdos.T)).T.ravel(),
        rtol=0, atol=1e-5)


def _calculate_kappados_all_frequencies(ph3):
    ph3.mesh_numbers = [7, 7, 7]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300, ])
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    frequencies, _, _ = ph3.get_phonon_data()
    kappados = KappaDOS(
        tc.mode_kappa[0],
        frequencies,
        bz_grid,
        bz_grid.bzg2grg[tc.grid_points],
        num_sampling_points=10)
    freq_points, kdos = kappados.get_kdos()

    ir_grid_points, _, ir_grid_map = get_ir_grid_points(bz_grid)
    kappados = KappaDOS(
        tc.mode_kappa[0],
        tc.frequencies,
        bz_grid,
        ir_grid_points,
        ir_grid_map=ir_grid_map,
        num_sampling_points=10)
    ir_freq_points, ir_kdos = kappados.get_kdos()
    np.testing.assert_allclose(ir_freq_points, freq_points, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ir_kdos, kdos, rtol=0, atol=1e-5)

    return freq_points, kdos[0, :, :, 0]
