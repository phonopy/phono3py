"""Test for kaccum.py."""
import numpy as np

from phono3py import Phono3py
from phono3py.other.kaccum import GammaDOSsmearing, KappaDOS, get_mfp
from phono3py.phonon.grid import get_ir_grid_points

kappados_si = [
    -0.0000002,
    0.0000000,
    0.0000000,
    1.6966400,
    2.1977566,
    5.1814323,
    3.3932803,
    25.8022392,
    15.5096766,
    5.0899206,
    56.6994259,
    19.4995156,
    6.7865608,
    68.7759426,
    3.2465477,
    8.4832011,
    72.8398965,
    1.6583881,
    10.1798413,
    74.8143686,
    0.7945952,
    11.8764816,
    77.2489625,
    5.4385183,
    13.5731219,
    80.9162245,
    0.5998735,
    15.2697621,
    81.4303646,
    0.0000000,
]
mfpdos_si = [
    0.0000000,
    0.0000000,
    0.0000000,
    806.8089241,
    33.7703552,
    0.0225548,
    1613.6178483,
    45.0137786,
    0.0103479,
    2420.4267724,
    53.3456168,
    0.0106724,
    3227.2356966,
    62.4915811,
    0.0107850,
    4034.0446207,
    69.8839011,
    0.0075919,
    4840.8535449,
    74.8662085,
    0.0049228,
    5647.6624690,
    78.2273252,
    0.0035758,
    6454.4713932,
    80.5493065,
    0.0020836,
    7261.2803173,
    81.4303646,
    0.0000000,
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
    0.0000000,
    0.0000000,
    0.0000000,
    117.4892903,
    3.1983595,
    0.0266514,
    234.9785806,
    5.7974129,
    0.0153383,
    352.4678709,
    7.2012603,
    0.0075057,
    469.9571612,
    7.5964440,
    0.0017477,
    587.4464515,
    7.7823291,
    0.0013915,
    704.9357418,
    7.9195460,
    0.0009363,
    822.4250321,
    8.0024702,
    0.0004844,
    939.9143223,
    8.0375053,
    0.0001382,
    1057.4036126,
    8.0430831,
    0.0000000,
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


def test_kappados_si(si_pbesol: Phono3py):
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
        ph3, tc.mode_kappa[0], freq_points=freq_points_in
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


def test_kappados_nacl(nacl_pbe: Phono3py):
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
        ph3, tc.mode_kappa[0], freq_points=freq_points_in
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


def test_GammaDOSsmearing(nacl_pbe: Phono3py):
    """Test for GammaDOSsmearing by computing phonon-DOS."""
    ph3 = nacl_pbe
    ph3.mesh_numbers = [21, 21, 21]
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()
    bz_grid = ph3.grid
    frequencies, _, _ = ph3.get_phonon_data()
    ir_grid_points, ir_weights, _ = get_ir_grid_points(bz_grid)
    ir_frequencies = frequencies[bz_grid.grg2bzg[ir_grid_points]]
    phonon_states = np.ones((1,) + ir_frequencies.shape, dtype="double", order="C")
    gdos = GammaDOSsmearing(
        phonon_states, ir_frequencies, ir_weights, num_sampling_points=10
    )
    fpoints, gdos_vals = gdos.get_gdos()
    gdos_ref = [
        [-1.4312845953710325e-07, 0.001748289450006],
        [0.8213328685698041, 0.04545825822129761],
        [1.6426658802680678, 0.2533557541451728],
        [2.463998891966331, 0.9005575010964907],
        [3.285331903664595, 1.6202936411038107],
        [4.106664915362859, 1.9916061367478763],
        [4.9279979270611225, 2.5977728237205513],
        [5.749330938759386, 0.4504707799027985],
        [6.57066395045765, 0.2936475034684396],
        [7.391996962155914, 0.02869983288053483],
    ]

    np.testing.assert_allclose(
        np.c_[fpoints, gdos_vals[0, :, 1]], gdos_ref, rtol=0, atol=1e-5
    )


def _calculate_kappados(ph3: Phono3py, mode_prop, freq_points=None):
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


def _calculate_mfpdos(ph3: Phono3py, mfp_points=None):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    mean_freepath = get_mfp(tc.gamma[0], tc.group_velocities)
    _, _, ir_grid_map = get_ir_grid_points(bz_grid)
    mfpdos = KappaDOS(
        tc.mode_kappa[0],
        mean_freepath[0],
        bz_grid,
        tc.grid_points,
        ir_grid_map=ir_grid_map,
        frequency_points=mfp_points,
        num_sampling_points=10,
    )
    freq_points, kdos = mfpdos.get_kdos()

    return freq_points, kdos[0, :, :, 0]
