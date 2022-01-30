"""Test for kaccum.py."""
import numpy as np

from phono3py.cui.kaccum import KappaDOS, _get_mfp
from phono3py.phonon.grid import get_ir_grid_points

kappados_si = [
    -1.999999999999999909e-07,  #    -0.0000002,
    2.578465216692497550e-29,  #    0.0000000,
    8.733717316163621463e-22,  #    0.0000000,
    1.696639999999999926e00,  #    1.6966400,
    2.286484528752652068e00,  #    2.1977566,
    5.390617290726341437e00,  #    5.1814323,
    3.393280299999999805e00,  #    3.3932803,
    2.643100559458443755e01,  #    25.8022392,
    1.559163392450822805e01,  #    15.5096766,
    5.089920600000000128e00,  #    5.0899206,
    5.738322258727477987e01,  #    56.6994259,
    1.951159362195583924e01,  #    19.4995156,
    6.786560800000000171e00,  #    6.7865608,
    6.947641641262839585e01,  #    68.7759426,
    3.261427807434183368e00,  #    3.2465477,
    8.483201100000000494e00,  #    8.4832011,
    7.355519194206652855e01,  #    72.8398965,
    1.654583483191118809e00,  #    1.6583881,
    1.017984129999999965e01,  #    10.1798413,
    7.552409658808292647e01,  #    74.8143686,
    7.911545616302700923e-01,  #    0.7945952,
    1.187648159999999997e01,  #    11.8764816,
    7.793862623570640835e01,  #    77.2489625,
    5.387081349970075372e00,  #    5.4385183,
    1.357312190000000029e01,  #    13.5731219,
    8.160771332675264489e01,  #    80.9162245,
    6.021934300980872345e-01,  #    0.5998735,
    1.526976209999999945e01,  #    15.2697621,
    8.212398405798980150e01,  #    81.4303646,
    7.562440844785435130e-21,  #    0.0000000,
]
mfpdos_si = [
    0.000000000000000000e00,  # 0.0000000,
    0.000000000000000000e00,  # 0.0000000,
    0.000000000000000000e00,  # 0.0000000,
    8.068089241000000129e02,  # 806.8089241,
    8.212398405798980150e01,  # 33.7703552,
    0.000000000000000000e00,  # 0.0225548,
    1.613617848300000105e03,  # 1613.6178483,
    8.212398405798980150e01,  # 45.0137786,
    0.000000000000000000e00,  # 0.0103479,
    2.420426772400000118e03,  # 2420.4267724,
    8.212398405798980150e01,  # 53.3456168,
    0.000000000000000000e00,  # 0.0106724,
    3.227235696600000210e03,  # 3227.2356966,
    8.212398405798980150e01,  # 62.4915811,
    0.000000000000000000e00,  # 0.0107850,
    4.034044620699999996e03,  # 4034.0446207,
    8.212398405798980150e01,  # 69.8839011,
    0.000000000000000000e00,  # 0.0075919,
    4.840853544900000088e03,  # 4840.8535449,
    8.212398405798980150e01,  # 74.8662085,
    0.000000000000000000e00,  # 0.0049228,
    5.647662468999999874e03,  # 5647.6624690,
    8.212398405798980150e01,  # 78.2273252,
    0.000000000000000000e00,  # 0.0035758,
    6.454471393200000421e03,  # 6454.4713932,
    8.212398405798980150e01,  # 80.5493065,
    0.000000000000000000e00,  # 0.0020836,
    7.261280317300000206e03,  # 7261.2803173,
    8.212398405798980150e01,  # 81.4303646,
    0.000000000000000000e00,  # 0.0000000,
]
gammados_si = [
    -0.0000002,
    0.0000000,
    0.0000000,
    1.6966400,
    0.0000063,
    0.0000149,
    3.3932803,
    0.0004133,
    0.0012312,
    5.0899206,
    0.0071709,
    0.0057356,
    6.7865608,
    0.0099381,
    0.0006492,
    8.4832011,
    0.0133390,
    0.0049604,
    10.1798413,
    0.0394030,
    0.0198106,
    11.8764816,
    0.0495160,
    0.0044113,
    13.5731219,
    0.0560223,
    0.0050103,
    15.2697621,
    0.1300596,
    0.0000000,
]
kappados_nacl = [
    -0.0000002,
    0.0000000,
    0.0000000,
    0.8051732,
    0.0366488,
    0.1820668,
    1.6103466,
    0.7748514,
    1.5172957,
    2.4155199,
    2.0165794,
    2.0077744,
    3.2206933,
    4.6670801,
    2.8357892,
    4.0258667,
    6.6123781,
    32.8560281,
    4.8310401,
    7.7105916,
    0.6136893,
    5.6362134,
    7.9112790,
    0.2391300,
    6.4413868,
    8.0272187,
    0.0604842,
    7.2465602,
    8.0430831,
    0.0000000,
]
mfpdos_nacl = [
    0.000000000000000000e00,  #    0.0000000,
    0.000000000000000000e00,  #    0.0000000,
    0.000000000000000000e00,  #    0.0000000,
    1.174892902999999933e02,  #    117.4892903,
    8.061007407297983818e00,  #    3.1983595,
    0.000000000000000000e00,  #    0.0266514,
    2.349785805999999866e02,  #    234.9785806,
    8.061007407297983818e00,  #    5.7974129,
    0.000000000000000000e00,  #    0.0153383,
    3.524678708999999799e02,  #    352.4678709,
    8.061007407297983818e00,  #    7.2012603,
    0.000000000000000000e00,  #    0.0075057,
    4.699571611999999732e02,  #    469.9571612,
    8.061007407297983818e00,  #    7.5964440,
    0.000000000000000000e00,  #    0.0017477,
    5.874464514999999665e02,  #    587.4464515,
    8.061007407297983818e00,  #    7.7823291,
    0.000000000000000000e00,  #    0.0013915,
    7.049357417999999598e02,  #    704.9357418,
    8.061007407297983818e00,  #    7.9195460,
    0.000000000000000000e00,  #    0.0009363,
    8.224250320999999531e02,  #    822.4250321,
    8.061007407297983818e00,  #    8.0024702,
    0.000000000000000000e00,  #    0.0004844,
    9.399143222999999807e02,  #    939.9143223,
    8.061007407297983818e00,  #    8.0375053,
    0.000000000000000000e00,  #    0.0001382,
    1.057403612600000088e03,  #    1057.4036126,
    8.061007407297983818e00,  #    8.0430831,
    0.000000000000000000e00,  #    0.0000000,
]
gammados_nacl = [
    -0.0000002,
    0.0000000,
    0.0000000,
    0.8051732,
    0.0000822,
    0.0004081,
    1.6103466,
    0.0018975,
    0.0053389,
    2.4155199,
    0.0114668,
    0.0182495,
    3.2206933,
    0.0353621,
    0.0329440,
    4.0258667,
    0.0604996,
    0.1138884,
    4.8310401,
    0.1038315,
    0.0716216,
    5.6362134,
    0.1481243,
    0.0468421,
    6.4413868,
    0.1982823,
    0.0662494,
    7.2465602,
    0.2429551,
    0.0000000,
]


def test_kappados_si(si_pbesol):
    """Test KappaDOS class with Si.

    * 3x3 tensor vs frequency
    * scalar vs frequency
    * kappa vs mean free path

    """
    ph3 = si_pbesol
    ph3.mesh_numbers = [7, 7, 7]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ]
    )
    tc = ph3.thermal_conductivity
    freq_points_in = np.array(kappados_si).reshape(-1, 3)[:, 0]
    freq_points, kdos = _calculate_kappados(
        ph3, tc.mode_kappa_P_RTA[0], freq_points=freq_points_in
    )
    for f, (jval, ival) in zip(freq_points, kdos):
        print("%.7f, %.7f, %.7f," % (f, jval, ival))

    np.testing.assert_allclose(
        kappados_si, np.vstack((freq_points, kdos.T)).T.ravel(), rtol=0, atol=0.5
    )

    freq_points, kdos = _calculate_kappados(
        ph3, tc.gamma[0, :, :, :, None], freq_points=freq_points_in
    )
    np.testing.assert_allclose(
        gammados_si, np.vstack((freq_points, kdos.T)).T.ravel(), rtol=0, atol=0.5
    )

    mfp_points_in = np.array(mfpdos_si).reshape(-1, 3)[:, 0]
    mfp_points, mfpdos = _calculate_mfpdos(ph3, mfp_points_in)
    # for f, (jval, ival) in zip(freq_points, mfpdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))

    np.testing.assert_allclose(
        mfpdos_si, np.vstack((mfp_points, mfpdos.T)).T.ravel(), rtol=0, atol=0.5
    )


def test_kappados_nacl(nacl_pbe):
    """Test KappaDOS class with NaCl.

    * 3x3 tensor vs frequency
    * scalar vs frequency
    * kappa vs mean free path

    """
    ph3 = nacl_pbe
    ph3.mesh_numbers = [7, 7, 7]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ]
    )
    tc = ph3.thermal_conductivity
    freq_points_in = np.array(kappados_nacl).reshape(-1, 3)[:, 0]
    freq_points, kdos = _calculate_kappados(
        ph3, tc.mode_kappa_P_RTA[0], freq_points=freq_points_in
    )
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_nacl, np.vstack((freq_points, kdos.T)).T.ravel(), rtol=0, atol=0.5
    )

    freq_points, kdos = _calculate_kappados(
        ph3, tc.gamma[0, :, :, :, None], freq_points=freq_points_in
    )
    np.testing.assert_allclose(
        gammados_nacl, np.vstack((freq_points, kdos.T)).T.ravel(), rtol=0, atol=0.5
    )

    mfp_points_in = np.array(mfpdos_nacl).reshape(-1, 3)[:, 0]
    mfp_points, mfpdos = _calculate_mfpdos(ph3, mfp_points_in)
    # for f, (jval, ival) in zip(freq_points, mfpdos):
    #     print("%.7f, %.7f, %.7f," % (f, jval, ival))

    np.testing.assert_allclose(
        mfpdos_nacl, np.vstack((mfp_points, mfpdos.T)).T.ravel(), rtol=0, atol=0.5
    )


def _calculate_kappados(ph3, mode_prop, freq_points=None):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    frequencies, _, _ = ph3.get_phonon_data()
    kappados = KappaDOS(
        mode_prop, frequencies, bz_grid, tc.grid_points, frequency_points=freq_points
    )
    freq_points, kdos = kappados.get_kdos()

    ir_grid_points, _, ir_grid_map = get_ir_grid_points(bz_grid)
    kappados = KappaDOS(
        mode_prop,
        tc.frequencies,
        bz_grid,
        tc.grid_points,
        ir_grid_map=ir_grid_map,
        frequency_points=freq_points,
    )
    ir_freq_points, ir_kdos = kappados.get_kdos()
    np.testing.assert_equal(bz_grid.bzg2grg[tc.grid_points], ir_grid_points)
    np.testing.assert_allclose(ir_freq_points, freq_points, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ir_kdos, kdos, rtol=0, atol=1e-5)
    return freq_points, kdos[0, :, :, 0]


def _calculate_mfpdos(ph3, mfp_points=None):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    mean_freepath = _get_mfp(tc.gamma[0], tc.group_velocities)
    _, _, ir_grid_map = get_ir_grid_points(bz_grid)
    mfpdos = KappaDOS(
        tc.mode_kappa_P_RTA[0],
        mean_freepath[0],
        bz_grid,
        tc.grid_points,
        ir_grid_map=ir_grid_map,
        frequency_points=mfp_points,
        num_sampling_points=10,
    )
    freq_points, kdos = mfpdos.get_kdos()

    return freq_points, kdos[0, :, :, 0]
