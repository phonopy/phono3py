"""Test for kaccum.py."""

from __future__ import annotations

from typing import Optional

import numpy as np

from phono3py import Phono3py
from phono3py.other.kaccum import (
    GammaDOSsmearing,
    KappaDOSTHM,
    get_mfp,
    run_mfp_dos,
    run_prop_dos,
)
from phono3py.phonon.grid import get_ir_grid_points


def test_kappados_si(si_pbesol: Phono3py):
    """Test KappaDOS class with Si.

    * 3x3 tensor vs frequency
    * scalar vs frequency
    * kappa vs mean free path

    """
    if si_pbesol._make_r0_average:
        kappados_si = [
            [-0.0000002, 0.0000000, 0.0000000],
            [1.6966400, 2.1916229, 5.1669722],
            [3.3932803, 25.7283368, 15.5208121],
            [5.0899206, 56.6273812, 19.4749436],
            [6.7865608, 68.6447676, 3.2126609],
            [8.4832011, 72.6556224, 1.6322844],
            [10.1798413, 74.6095011, 0.7885669],
            [11.8764816, 77.0212439, 5.3742100],
            [13.5731219, 80.7020498, 0.6098605],
            [15.2697621, 81.2167777, 0.0000000],
        ]
        gammados_si = [
            [-0.0000002, 0.0000000, 0.0000000],
            [1.6966400, 0.0000009, 0.0000022],
            [3.3932803, 0.0000344, 0.0000772],
            [5.0899206, 0.0003875, 0.0002461],
            [6.7865608, 0.0005180, 0.0000450],
            [8.4832011, 0.0007156, 0.0002556],
            [10.1798413, 0.0017878, 0.0008881],
            [11.8764816, 0.0023096, 0.0001782],
            [13.5731219, 0.0026589, 0.0003556],
            [15.2697621, 0.0077743, 0.0000000],
        ]
        mfpdos_si = [
            [0.0000000, 0.0000000, 0.0000000],
            [806.8089241, 33.7286816, 0.0222694],
            [1613.6178483, 44.9685295, 0.0104179],
            [2420.4267724, 53.3416745, 0.0106935],
            [3227.2356966, 62.5115915, 0.0108348],
            [4034.0446207, 69.8830824, 0.0075011],
            [4840.8535449, 74.7736977, 0.0048188],
            [5647.6624690, 78.0965121, 0.0035467],
            [6454.4713932, 80.3863740, 0.0020324],
            [7261.2803173, 81.2167777, 0.0000000],
        ]
    else:
        kappados_si = [
            [-0.0000002, 0.0000000, 0.0000000],
            [1.6966400, 2.1929621, 5.1701294],
            [3.3932803, 25.7483415, 15.5091053],
            [5.0899206, 56.6694055, 19.5050829],
            [6.7865608, 68.7310303, 3.2377297],
            [8.4832011, 72.7739143, 1.6408582],
            [10.1798413, 74.7329367, 0.7889027],
            [11.8764816, 77.1441825, 5.3645613],
            [13.5731219, 80.8235276, 0.6098150],
            [15.2697621, 81.3384416, 0.0000000],
        ]
        gammados_si = [
            [-0.0000002, 0.0000000, 0.0000000],
            [1.6966400, 0.0000009, 0.0000022],
            [3.3932803, 0.0000346, 0.0000776],
            [5.0899206, 0.0003887, 0.0002460],
            [6.7865608, 0.0005188, 0.0000447],
            [8.4832011, 0.0007153, 0.0002547],
            [10.1798413, 0.0017871, 0.0008887],
            [11.8764816, 0.0023084, 0.0001784],
            [13.5731219, 0.0026578, 0.0003562],
            [15.2697621, 0.0077689, 0.0000000],
        ]
        mfpdos_si = [
            [0.0000000, 0.0000000, 0.0000000],
            [806.8089241, 33.7483611, 0.0223642],
            [1613.6178483, 44.9984189, 0.0104116],
            [2420.4267724, 53.3597126, 0.0106858],
            [3227.2356966, 62.5301968, 0.0108511],
            [4034.0446207, 69.9297685, 0.0075505],
            [4840.8535449, 74.8603881, 0.0048576],
            [5647.6624690, 78.1954706, 0.0035546],
            [6454.4713932, 80.4941074, 0.0020465],
            [7261.2803173, 81.3384416, 0.0000000],
        ]

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
        ph3,
        tc.mode_kappa[0],
        freq_points=freq_points_in,
    )
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("[%.7f, %.7f, %.7f]," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_si, np.vstack((freq_points, kdos.T)).T, rtol=0, atol=0.5
    )

    freq_points, kdos = _calculate_kappados(
        ph3,
        tc.gamma[0, :, :, :, None],
        freq_points=freq_points_in,
    )
    # for f, (jval, ival) in zip(freq_points, kdos):
    #     print("[%.7f, %.7f, %.7f]," % (f, jval, ival))
    np.testing.assert_allclose(
        gammados_si, np.vstack((freq_points, kdos.T)).T, rtol=0, atol=1e-4
    )

    mfp_points_in = np.array(mfpdos_si).reshape(-1, 3)[:, 0]
    mfp_points, mfpdos = _calculate_mfpdos(ph3, mfp_points_in)
    # for f, (jval, ival) in zip(mfp_points, mfpdos):
    #     print("[%.7f, %.7f, %.7f]," % (f, jval, ival))
    np.testing.assert_allclose(
        mfpdos_si, np.vstack((mfp_points, mfpdos.T)).T, rtol=0, atol=0.5
    )


def test_kappados_nacl(nacl_pbe: Phono3py):
    """Test KappaDOS class with NaCl.

    * 3x3 tensor vs frequency
    * scalar vs frequency
    * kappa vs mean free path

    """
    if nacl_pbe._make_r0_average:
        kappados_nacl = [
            [-0.0000002, 0.0000000, 0.0000000],
            [0.8051732, 0.0399444, 0.1984390],
            [1.6103466, 0.8500862, 1.6651565],
            [2.4155199, 2.1611612, 2.0462826],
            [3.2206933, 4.8252014, 2.7906917],
            [4.0258667, 6.7455774, 32.0221250],
            [4.8310401, 7.8342086, 0.6232244],
            [5.6362134, 8.0342122, 0.2370738],
            [6.4413868, 8.1491279, 0.0600325],
            [7.2465602, 8.1649079, 0.0000000],
        ]
        gammados_nacl = [
            [-0.0000002, 0.0000000, 0.0000000],
            [0.8051732, 0.0000106, 0.0000528],
            [1.6103466, 0.0002046, 0.0004709],
            [2.4155199, 0.0009472, 0.0012819],
            [3.2206933, 0.0022622, 0.0016645],
            [4.0258667, 0.0034103, 0.0054783],
            [4.8310401, 0.0061284, 0.0029336],
            [5.6362134, 0.0080135, 0.0019550],
            [6.4413868, 0.0106651, 0.0046371],
            [7.2465602, 0.0151994, 0.0000000],
        ]
        mfpdos_nacl = [
            [0.0000000, 0.0000000, 0.0000000],
            [117.4892903, 3.1996975, 0.0260716],
            [234.9785806, 5.7553155, 0.0153949],
            [352.4678709, 7.1540040, 0.0077163],
            [469.9571612, 7.5793413, 0.0018178],
            [587.4464515, 7.7799946, 0.0015717],
            [704.9357418, 7.9439439, 0.0012052],
            [822.4250321, 8.0613451, 0.0007915],
            [939.9143223, 8.1309598, 0.0004040],
            [1057.4036126, 8.1601561, 0.0001157],
        ]
    else:
        kappados_nacl = [
            [-0.0000002, 0.0000000, 0.0000000],
            [0.8051732, 0.0367419, 0.1825292],
            [1.6103466, 0.7836072, 1.5469421],
            [2.4155199, 2.0449081, 2.0062821],
            [3.2206933, 4.6731679, 2.7888186],
            [4.0258667, 6.6041834, 32.7256000],
            [4.8310401, 7.6993258, 0.6289821],
            [5.6362134, 7.8997102, 0.2365916],
            [6.4413868, 8.0146450, 0.0603293],
            [7.2465602, 8.0305633, 0.0000000],
        ]
        gammados_nacl = [
            [-0.0000002, 0.0000000, 0.0000000],
            [0.8051732, 0.0000106, 0.0000524],
            [1.6103466, 0.0002041, 0.0004715],
            [2.4155199, 0.0009495, 0.0012874],
            [3.2206933, 0.0022743, 0.0016787],
            [4.0258667, 0.0034299, 0.0053880],
            [4.8310401, 0.0061710, 0.0029085],
            [5.6362134, 0.0080493, 0.0019511],
            [6.4413868, 0.0106809, 0.0045989],
            [7.2465602, 0.0151809, 0.0000000],
        ]
        mfpdos_nacl = [
            [0.0000000, 0.0000000, 0.0000000],
            [117.4892903, 3.2044884, 0.0265249],
            [234.9785806, 5.8068154, 0.0153182],
            [352.4678709, 7.1822717, 0.0071674],
            [469.9571612, 7.5691736, 0.0017935],
            [587.4464515, 7.7601125, 0.0014313],
            [704.9357418, 7.9015132, 0.0009674],
            [822.4250321, 7.9875088, 0.0005054],
            [939.9143223, 8.0243816, 0.0001485],
            [1057.4036126, 8.0305631, 0.0000001],
        ]

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
    #     print("[%.7f, %.7f, %.7f]," % (f, jval, ival))
    np.testing.assert_allclose(
        kappados_nacl, np.vstack((freq_points, kdos.T)).T, rtol=0, atol=0.5
    )

    freq_points, kdos = _calculate_kappados(
        ph3,
        tc.gamma[0, :, :, :, None],
        freq_points=freq_points_in,
    )
    for f, (jval, ival) in zip(freq_points, kdos):
        print("[%.7f, %.7f, %.7f]," % (f, jval, ival))
    np.testing.assert_allclose(
        gammados_nacl, np.vstack((freq_points, kdos.T)).T, rtol=0, atol=1e-4
    )

    mfp_points_in = np.array(mfpdos_nacl).reshape(-1, 3)[:, 0]
    mfp_points, mfpdos = _calculate_mfpdos(ph3, mfp_points_in)
    # for f, (jval, ival) in zip(mfp_points, mfpdos):
    #     print("[%.7f, %.7f, %.7f]," % (f, jval, ival))
    np.testing.assert_allclose(
        mfpdos_nacl, np.vstack((mfp_points, mfpdos.T)).T, rtol=0, atol=0.5
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
        [-1.30357573e-07, 1.74828946e-03],
        [8.21332876e-01, 4.54582590e-02],
        [1.64266588e00, 2.53356134e-01],
        [2.46399889e00, 9.00558131e-01],
        [3.28533190e00, 1.62029335e00],
        [4.10666490e00, 1.99160666e00],
        [4.92799791e00, 2.59777233e00],
        [5.74933092e00, 4.50470780e-01],
        [6.57066392e00, 2.93647488e-01],
        [7.39199693e00, 2.86997789e-02],
    ]

    np.testing.assert_allclose(
        np.c_[fpoints, gdos_vals[0, :, 1]], gdos_ref, rtol=0, atol=1e-5
    )


def test_run_prop_dos(si_pbesol: Phono3py):
    """Test of run_prop_dos."""
    ph3 = si_pbesol
    ph3.mesh_numbers = [7, 7, 7]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[
            300,
        ]
    )
    bz_grid = ph3.grid
    ir_grid_points, _, ir_grid_map = get_ir_grid_points(bz_grid)
    tc = ph3.thermal_conductivity

    kdos, sampling_points = run_prop_dos(
        tc.frequencies, tc.mode_kappa[0], ir_grid_map, ir_grid_points, 10, bz_grid
    )
    mean_freepath = get_mfp(tc.gamma[0], tc.group_velocities)
    mfp, sampling_points_mfp = run_mfp_dos(
        mean_freepath, tc.mode_kappa[0], ir_grid_map, ir_grid_points, 10, bz_grid
    )

    # print(",".join([f"{v:10.5f}" for v in kdos[0, :, :, 0].ravel()]))
    ref_kdos = [
        0.00000,
        0.00000,
        2.19162,
        5.16697,
        28.22125,
        18.97280,
        58.56343,
        12.19206,
        69.05896,
        3.47035,
        73.17626,
        1.48915,
        74.74544,
        0.43485,
        75.87064,
        1.74135,
        79.08179,
        2.30428,
        81.21678,
        0.00000,
    ]
    # print(",".join([f"{v:10.5f}" for v in mfp[0, :, :, 0].ravel()]))
    ref_mfp = [
        0.00000,
        0.00000,
        29.19150,
        0.02604,
        42.80717,
        0.01202,
        52.09457,
        0.01158,
        61.79908,
        0.01140,
        69.49177,
        0.00784,
        74.57499,
        0.00501,
        77.99145,
        0.00364,
        80.33477,
        0.00210,
        81.21678,
        0.00000,
    ]
    # print(",".join([f"{v:10.5f}" for v in sampling_points[0]]))
    ref_sampling_points = [
        -0.00000,
        1.69664,
        3.39328,
        5.08992,
        6.78656,
        8.48320,
        10.17984,
        11.87648,
        13.57312,
        15.26976,
    ]
    # print(",".join([f"{v:10.5f}" for v in sampling_points_mfp[0]]))
    ref_sampling_points_mfp = [
        0.00000,
        803.91710,
        1607.83420,
        2411.75130,
        3215.66841,
        4019.58551,
        4823.50261,
        5627.41971,
        6431.33681,
        7235.25391,
    ]
    np.testing.assert_allclose(ref_kdos, kdos[0, :, :, 0].ravel(), atol=0.5)
    np.testing.assert_allclose(ref_mfp, mfp[0, :, :, 0].ravel(), atol=0.5)
    np.testing.assert_allclose(ref_sampling_points, sampling_points[0], atol=1e-4)
    np.testing.assert_allclose(ref_sampling_points_mfp, sampling_points_mfp[0], rtol=10)


def _calculate_kappados(
    ph3: Phono3py,
    mode_prop: np.ndarray,
    freq_points: Optional[np.ndarray] = None,
):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    ir_grid_points, _, ir_grid_map = get_ir_grid_points(bz_grid)
    kappados = KappaDOSTHM(
        mode_prop,
        tc.frequencies,
        bz_grid,
        bz_grid.bzg2grg[tc.grid_points],
        ir_grid_map=ir_grid_map,
        frequency_points=freq_points,
    )
    ir_freq_points, ir_kdos = kappados.get_kdos()
    np.testing.assert_equal(bz_grid.bzg2grg[tc.grid_points], ir_grid_points)
    np.testing.assert_allclose(ir_freq_points, freq_points, rtol=0, atol=1e-5)

    return freq_points, ir_kdos[0, :, :, 0]


def _calculate_mfpdos(
    ph3: Phono3py,
    mfp_points=None,
):
    tc = ph3.thermal_conductivity
    bz_grid = ph3.grid
    mean_freepath = get_mfp(tc.gamma[0], tc.group_velocities)
    _, _, ir_grid_map = get_ir_grid_points(bz_grid)
    mfpdos = KappaDOSTHM(
        tc.mode_kappa[0],
        mean_freepath[0],
        bz_grid,
        bz_grid.bzg2grg[tc.grid_points],
        ir_grid_map=ir_grid_map,
        frequency_points=mfp_points,
        num_sampling_points=10,
    )
    freq_points, kdos = mfpdos.get_kdos()

    return freq_points, kdos[0, :, :, 0]
