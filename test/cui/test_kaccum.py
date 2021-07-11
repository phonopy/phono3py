import numpy as np
from phono3py.cui.kaccum import KappaDOS, _get_mfp
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
mfpdos_si = [0.0000000, 0.0000000, 0.0000000,
             806.8089241, 33.7703552, 0.0225548,
             1613.6178483, 45.0137786, 0.0103479,
             2420.4267724, 53.3456168, 0.0106724,
             3227.2356966, 62.4915811, 0.0107850,
             4034.0446207, 69.8839011, 0.0075919,
             4840.8535449, 74.8662085, 0.0049228,
             5647.6624690, 78.2273252, 0.0035758,
             6454.4713932, 80.5493065, 0.0020836,
             7261.2803173, 81.4303646, 0.0000000]
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
mfpdos_nacl = [0.0000000, 0.0000000, 0.0000000,
               117.4892903, 3.1983595, 0.0266514,
               234.9785806, 5.7974129, 0.0153383,
               352.4678709, 7.2012603, 0.0075057,
               469.9571612, 7.5964440, 0.0017477,
               587.4464515, 7.7823291, 0.0013915,
               704.9357418, 7.9195460, 0.0009363,
               822.4250321, 8.0024702, 0.0004844,
               939.9143223, 8.0375053, 0.0001382,
               1057.4036126, 8.0430831, 0.0000000]


def test_kappados_si(si_pbesol):
    ph3 = si_pbesol
    ph3.mesh_numbers = [7, 7, 7]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300, ])
    freq_points_in = np.array(kappados_si).reshape(-1, 3)[:, 0]
    freq_points, kdos = _calculate_kappados(
        ph3, freq_points=freq_points_in)
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_si,
        np.vstack((freq_points, kdos.T)).T.ravel(),
        rtol=0, atol=0.5)

    mfp_points_in = np.array(mfpdos_si).reshape(-1, 3)[:, 0]
    mfp_points, mfpdos = _calculate_mfpdos(ph3, mfp_points_in)
    # for f, (jval, ival) in zip(freq_points, mfpdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        mfpdos_si,
        np.vstack((mfp_points, mfpdos.T)).T.ravel(),
        rtol=0, atol=0.5)


def test_kappados_nacl(nacl_pbe):
    ph3 = nacl_pbe
    ph3.mesh_numbers = [7, 7, 7]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300, ])
    freq_points_in = np.array(kappados_nacl).reshape(-1, 3)[:, 0]
    freq_points, kdos = _calculate_kappados(
        ph3, freq_points=freq_points_in)
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_nacl,
        np.vstack((freq_points, kdos.T)).T.ravel(),
        rtol=0, atol=0.5)

    mfp_points_in = np.array(mfpdos_nacl).reshape(-1, 3)[:, 0]
    mfp_points, mfpdos = _calculate_mfpdos(ph3, mfp_points_in)
    # for f, (jval, ival) in zip(freq_points, mfpdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        mfpdos_nacl,
        np.vstack((mfp_points, mfpdos.T)).T.ravel(),
        rtol=0, atol=0.5)


def _calculate_kappados(ph3, freq_points=None):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    frequencies, _, _ = ph3.get_phonon_data()
    kappados = KappaDOS(
        tc.mode_kappa[0],
        frequencies,
        bz_grid,
        bz_grid.bzg2grg[tc.grid_points],
        frequency_points=freq_points,
        num_sampling_points=10)
    freq_points, kdos = kappados.get_kdos()

    ir_grid_points, _, ir_grid_map = get_ir_grid_points(bz_grid)
    kappados = KappaDOS(
        tc.mode_kappa[0],
        tc.frequencies,
        bz_grid,
        ir_grid_points,
        ir_grid_map=ir_grid_map,
        frequency_points=freq_points,
        num_sampling_points=10)
    ir_freq_points, ir_kdos = kappados.get_kdos()
    np.testing.assert_allclose(ir_freq_points, freq_points, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ir_kdos, kdos, rtol=0, atol=1e-5)

    return freq_points, kdos[0, :, :, 0]


def _calculate_mfpdos(ph3, mfp_points=None):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    mean_freepath = _get_mfp(tc.gamma[0], tc.group_velocities)
    ir_grid_points, _, ir_grid_map = get_ir_grid_points(bz_grid)
    mfpdos = KappaDOS(
        tc.mode_kappa[0],
        mean_freepath[0],
        bz_grid,
        ir_grid_points,
        ir_grid_map=ir_grid_map,
        frequency_points=mfp_points,
        num_sampling_points=10)
    freq_points, kdos = mfpdos.get_kdos()

    return freq_points, kdos[0, :, :, 0]
