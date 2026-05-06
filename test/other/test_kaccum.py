"""Test for kaccum.py."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
from phonopy.other.kaccum import (
    GammaDOSsmearing,
    KappaDOSTHM,
    get_mfp,
    run_mfp_dos,
    run_prop_dos,
)
from phonopy.phonon.grid import get_ir_grid_points

from phono3py import Phono3py


def test_kappados_si(si_pbesol: Phono3py):
    """Test KappaDOS class with Si.

    * 3x3 tensor vs frequency
    * scalar vs frequency
    * kappa vs mean free path

    """
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
    for f, (jval, ival) in zip(freq_points, kdos, strict=True):
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


@pytest.mark.parametrize(
    "sigma,gdos_ref",
    [
        (
            0.5,
            [
                [-1.30235600e-07, 9.50477186e-03],
                [8.21332877e-01, 7.69536176e-02],
                [1.64266588e00, 3.13334519e-01],
                [2.46399889e00, 8.82406425e-01],
                [3.28533190e00, 1.40817939e00],
                [4.10666490e00, 1.85919540e00],
                [4.92799791e00, 1.70898105e00],
                [5.74933092e00, 6.65224558e-01],
                [6.57066392e00, 2.93999184e-01],
                [7.39199693e00, 8.24248337e-02],
            ],
        ),
        (
            0.3,
            [
                [-1.30235600e-07, 3.29817062e-03],
                [8.21332877e-01, 5.97326058e-02],
                [1.64266588e00, 2.64290543e-01],
                [2.46399889e00, 8.88151288e-01],
                [3.28533190e00, 1.41527252e00],
                [4.10666490e00, 1.90607496e00],
                [4.92799791e00, 1.98375766e00],
                [5.74933092e00, 4.97576075e-01],
                [6.57066392e00, 3.00025228e-01],
                [7.39199693e00, 5.74070952e-02],
            ],
        ),
    ],
)
def test_GammaDOSsmearing_with_sigma(nacl_pbe: Phono3py, sigma: float, gdos_ref: list):
    """Test for GammaDOSsmearing with user-specified sigma.

    Verifies that the sigma parameter is actually used (regression test for
    a bug where sigma was silently ignored and 0.1 was used instead).
    """
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
        phonon_states, ir_frequencies, ir_weights, sigma=sigma, num_sampling_points=10
    )
    fpoints, gdos_vals = gdos.get_gdos()
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
        2.19027,
        5.16378,
        25.73771,
        15.50548,
        56.63845,
        19.48634,
        68.64061,
        3.16016,
        72.58422,
        1.62370,
        74.53304,
        0.78480,
        76.94719,
        5.37456,
        80.63569,
        0.61517,
        81.15021,
        0.00000,
    ]
    # print(",".join([f"{v:10.5f}" for v in mfp[0, :, :, 0].ravel()]))
    ref_mfp = [
        0.00000,
        0.00000,
        33.66617,
        0.02216,
        44.81836,
        0.01032,
        53.09252,
        0.01062,
        62.18077,
        0.01085,
        69.59018,
        0.00765,
        74.58994,
        0.00496,
        77.95139,
        0.00359,
        80.27019,
        0.00209,
        81.15021,
        0.00000,
    ]
    # print(",".join([f"{v:10.5f}" for v in sampling_points[0]]))
    ref_sampling_points = [
        0.00000,
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
        803.28312,
        1606.56625,
        2409.84937,
        3213.13249,
        4016.41562,
        4819.69874,
        5622.98186,
        6426.26499,
        7229.54811,
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
