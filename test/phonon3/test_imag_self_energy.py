"""Test for imag_free_energy.py."""
import numpy as np

from phono3py import Phono3py

gammas = [
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0005412,
    0.0005412,
    0.0008843,
    0.0191694,
    0.0206316,
    0.0206316,
    0.0019424,
    0.0019424,
    0.0067566,
    0.0548967,
    0.0506115,
    0.0506115,
    0.0062204,
    0.0062204,
    0.0088148,
    0.0426150,
    0.0417223,
    0.0417223,
    0.0016263,
    0.0016263,
    0.0017293,
    0.0279509,
    0.0289259,
    0.0289259,
    0.0097926,
    0.0097926,
    0.0170092,
    0.0438828,
    0.0523105,
    0.0523105,
    0.0035542,
    0.0035542,
    0.0135109,
    0.0623533,
    0.0343746,
    0.0343746,
    0.0073140,
    0.0073140,
    0.0289659,
    0.5006760,
    0.5077932,
    0.5077932,
    0.0016144,
    0.0016144,
    0.0126326,
    0.2731933,
    0.2791702,
    0.2791702,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0023304,
    0.0026469,
    0.0052513,
    0.0209641,
    0.0220092,
    0.0234752,
    0.0035532,
    0.0038158,
    0.0087882,
    0.0276654,
    0.0315055,
    0.0286975,
    0.0345193,
    0.0277533,
    0.0495734,
    0.0511798,
    0.0465938,
    0.0436605,
    0.0071705,
    0.0081615,
    0.0139063,
    0.0204058,
    0.0307320,
    0.0237855,
    0.0202095,
    0.0197716,
    0.0316074,
    0.0402461,
    0.0438103,
    0.0394924,
    0.0171448,
    0.0176446,
    0.0567310,
    0.0930479,
    0.0570520,
    0.0622142,
    0.0292639,
    0.0328821,
    0.0667957,
    0.2541887,
    0.4592188,
    0.4234131,
    0.0104887,
    0.0179753,
    0.0827533,
    0.2659557,
    0.3242633,
    0.3189804,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
    0.0000000,
]
gammas_sigma = [
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00046029,
    0.00046029,
    0.00071545,
    0.02242054,
    0.01975435,
    0.01975435,
    0.00143860,
    0.00143860,
    0.00715263,
    0.05481156,
    0.04396936,
    0.04396936,
    0.00826301,
    0.00826301,
    0.00950813,
    0.04304817,
    0.04400210,
    0.04400210,
    0.00203560,
    0.00203560,
    0.00207048,
    0.02226551,
    0.03531839,
    0.03531839,
    0.00746195,
    0.00746195,
    0.01268396,
    0.02380441,
    0.03074892,
    0.03074892,
    0.00389360,
    0.00389360,
    0.01154058,
    0.05602348,
    0.04034627,
    0.04034627,
    0.00642767,
    0.00642767,
    0.02338437,
    0.43710790,
    0.48306584,
    0.48306584,
    0.00291728,
    0.00291728,
    0.11718631,
    0.84620157,
    0.80881708,
    0.80881708,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00224835,
    0.00288498,
    0.00554574,
    0.02261273,
    0.02349047,
    0.02647988,
    0.00330612,
    0.00430468,
    0.00975355,
    0.02954525,
    0.03242621,
    0.03052183,
    0.03210358,
    0.02583317,
    0.04906091,
    0.04609366,
    0.04064508,
    0.04250035,
    0.00888799,
    0.00936948,
    0.01541312,
    0.02079095,
    0.03001210,
    0.02721119,
    0.02593986,
    0.02559304,
    0.04760672,
    0.04958274,
    0.04942973,
    0.03703768,
    0.01005313,
    0.01125217,
    0.05423798,
    0.10135670,
    0.06021902,
    0.09005459,
    0.02358822,
    0.03737522,
    0.06633807,
    0.22190369,
    0.41562743,
    0.32601504,
    0.01240071,
    0.02372173,
    0.20217767,
    0.49239981,
    0.52883866,
    0.50769018,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
]
gammas_class1 = [
    0.00000000,
    0.00000000,
    0.00000000,
    -0.00000000,
    0.00000000,
    0.00000000,
    0.00053387,
    0.00053387,
    0.00086230,
    0.01894313,
    0.02034210,
    0.02034210,
    0.00155506,
    0.00155506,
    0.00260125,
    0.01821681,
    0.01820381,
    0.01820381,
    0.00571765,
    0.00571765,
    0.00544460,
    0.01325570,
    0.01118428,
    0.01118428,
    0.00016153,
    0.00016153,
    0.00032679,
    0.00020002,
    0.00020927,
    0.00020927,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00233036,
    0.00264690,
    0.00525130,
    0.02096414,
    0.02200915,
    0.02347515,
    0.00297698,
    0.00348529,
    0.00638118,
    0.01776255,
    0.02740917,
    0.02217207,
    0.03234423,
    0.02580162,
    0.03682891,
    0.03904463,
    0.01942315,
    0.02072384,
    0.00004097,
    0.00005101,
    0.00007457,
    0.00003508,
    0.00004210,
    0.00003803,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
]
gammas_class2 = [
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000728,
    0.00000728,
    0.00002201,
    0.00022624,
    0.00028946,
    0.00028946,
    0.00038736,
    0.00038736,
    0.00415534,
    0.03667993,
    0.03240766,
    0.03240766,
    0.00050274,
    0.00050274,
    0.00337024,
    0.02935928,
    0.03053801,
    0.03053801,
    0.00146473,
    0.00146473,
    0.00140248,
    0.02775086,
    0.02871662,
    0.02871662,
    0.00979262,
    0.00979262,
    0.01700920,
    0.04388280,
    0.05231049,
    0.05231049,
    0.00355424,
    0.00355424,
    0.01351094,
    0.06235333,
    0.03437465,
    0.03437465,
    0.00731397,
    0.00731397,
    0.02896588,
    0.50067605,
    0.50779324,
    0.50779324,
    0.00161440,
    0.00161440,
    0.01263256,
    0.27319333,
    0.27917018,
    0.27917018,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00057618,
    0.00033051,
    0.00240702,
    0.00990280,
    0.00409632,
    0.00652547,
    0.00217505,
    0.00195163,
    0.01274449,
    0.01213516,
    0.02717067,
    0.02293662,
    0.00712953,
    0.00811051,
    0.01383178,
    0.02037067,
    0.03068992,
    0.02374747,
    0.02020952,
    0.01977157,
    0.03160744,
    0.04024612,
    0.04381027,
    0.03949241,
    0.01714475,
    0.01764459,
    0.05673104,
    0.09304789,
    0.05705200,
    0.06221421,
    0.02926385,
    0.03288210,
    0.06679574,
    0.25418868,
    0.45921877,
    0.42341309,
    0.01048868,
    0.01797532,
    0.08275328,
    0.26595568,
    0.32426329,
    0.31898043,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
]
freq_points = [
    0.0,
    3.41024688,
    6.82049376,
    10.23074063,
    13.64098751,
    17.05123439,
    20.46148127,
    23.87172814,
    27.28197502,
    30.6922219,
]
freq_points_sigma = [
    0.0,
    3.45491354,
    6.90982709,
    10.36474063,
    13.81965418,
    17.27456772,
    20.72948127,
    24.18439481,
    27.63930835,
    31.09422190,
]

detailed_gamma = [
    0.00000000,
    0.00653193,
    0.02492913,
    0.01682092,
    0.01001680,
    0.02181888,
    0.01858641,
    0.16208762,
    0.09598706,
    0.00000000,
]

gammas_nacl = [
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.03396688,
    0.03396688,
    0.00687452,
    0.21001764,
    0.21001764,
    0.12310439,
    0.00297385,
    0.00297385,
    0.00227915,
    0.10673763,
    0.10673763,
    0.06918881,
    0.01003326,
    0.01003326,
    0.00996780,
    0.03414868,
    0.03414868,
    0.02258494,
    0.04027592,
    0.04027592,
    0.03603612,
    0.57995646,
    0.57995646,
    0.39737731,
    0.12705253,
    0.12705253,
    0.09246595,
    0.88750309,
    0.88750309,
    0.60334780,
    0.29968747,
    0.29968747,
    0.14257862,
    0.22134950,
    0.22134950,
    0.09606896,
    0.03941985,
    0.03941985,
    0.01632766,
    0.00222574,
    0.00222574,
    0.00627294,
    0.00240808,
    0.00240808,
    0.00688951,
    0.00008074,
    0.00008074,
    0.00003641,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.02850846,
    0.09000833,
    0.19582553,
    0.13715943,
    0.19892888,
    0.14203618,
    0.00861856,
    0.02747203,
    0.05000735,
    0.04441740,
    0.11080545,
    0.04172184,
    0.00738182,
    0.01722875,
    0.03273830,
    0.04517923,
    0.02441539,
    0.03277688,
    0.03233818,
    0.08459289,
    0.19264167,
    0.11281266,
    0.45667245,
    0.18491212,
    0.10846241,
    0.47768641,
    1.04554356,
    0.64678566,
    0.83834225,
    0.61795504,
    0.19485590,
    0.43708391,
    0.24896003,
    0.35882984,
    0.30654914,
    0.22471014,
    0.03624311,
    0.13350831,
    0.12479592,
    0.06750776,
    0.02503182,
    0.04543786,
    0.00155614,
    0.01088453,
    0.00064712,
    0.00392933,
    0.00058749,
    0.00022448,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
]
freq_points_nacl = [
    0.0,
    1.63223063,
    3.26446125,
    4.89669188,
    6.5289225,
    8.16115313,
    9.79338375,
    11.42561438,
    13.057845,
    14.69007563,
]
gammas_nacl_nac = [
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.20482566,
    0.20482566,
    0.12447648,
    0.00000000,
    0.00000000,
    0.00000000,
    0.10819754,
    0.10819754,
    0.06679962,
    0.00000000,
    0.00000000,
    0.00000000,
    0.03735364,
    0.03735364,
    0.02305203,
    0.00000000,
    0.00000000,
    0.00000000,
    0.69026924,
    0.69026924,
    0.42880009,
    0.00000000,
    0.00000000,
    0.00000000,
    1.05462484,
    1.05462484,
    0.64579913,
    0.00000000,
    0.00000000,
    0.00000000,
    0.24280747,
    0.24280747,
    0.15052565,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00013397,
    0.00013397,
    0.00008461,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
    0.00000000,
]
freq_points_nacl_nac = [
    0.0,
    1.63223063,
    3.26446125,
    4.89669188,
    6.5289225,
    8.16115313,
    9.79338375,
    11.42561438,
    13.057845,
    14.69007563,
]


def test_imag_self_energy_at_bands(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * at frequencies of band indices.

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    gammas_ref = np.reshape(
        [
            0.00021553,
            0.00021553,
            0.00084329,
            0.04693498,
            0.04388354,
            0.04388354,
            0.00383646,
            0.00494357,
            0.02741665,
            0.01407101,
            0.04133322,
            0.03013125,
        ],
        (2, -1),
    )
    for i, grgp in enumerate((1, 103)):
        _fpoints, _gammas = si_pbesol.run_imag_self_energy(
            [
                si_pbesol.grid.grg2bzg[grgp],
            ],
            [
                300,
            ],
            frequency_points_at_bands=True,
        )
        np.testing.assert_allclose(_gammas.ravel(), gammas_ref[i], rtol=0, atol=1e-2)


def test_imag_self_energy_at_bands_detailed(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * at frequencies of band indices.
    * contribution from each triplet is returned.

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas, _detailed_gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        frequency_points_at_bands=True,
        keep_gamma_detail=True,
    )

    weights_1 = [
        2,
        2,
        2,
        2,
        1,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        12,
        12,
        12,
        12,
        6,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        6,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        6,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        6,
    ]
    weights_103 = [2] * 364 + [1]

    gammas_1_ref = _gammas[:, :, 0].ravel()
    gammas_103_ref = _gammas[:, :, 1].ravel()
    gammas_1 = np.dot(weights_1, _detailed_gammas[0][0, 0].sum(axis=-1).sum(axis=-1))
    gammas_103 = np.dot(
        weights_103, _detailed_gammas[1][0, 0].sum(axis=-1).sum(axis=-1)
    )
    np.testing.assert_allclose(
        gammas_1[:2].sum(), gammas_1_ref[:2].sum(), rtol=0, atol=1e-2
    )
    np.testing.assert_allclose(
        gammas_1[-2:].sum(), gammas_1_ref[-2:].sum(), rtol=0, atol=1e-2
    )
    np.testing.assert_allclose(gammas_1[2:4], gammas_1_ref[2:4], rtol=0, atol=1e-2)
    np.testing.assert_allclose(gammas_103, gammas_103_ref, rtol=0, atol=1e-2)


def test_imag_self_energy_npoints(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * at 10 frequency points sampled uniformly.

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        num_frequency_points=10,
    )
    np.testing.assert_allclose(
        gammas, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=1e-2
    )
    np.testing.assert_allclose(freq_points, _fpoints.ravel(), rtol=0, atol=1e-5)


def test_imag_self_energy_npoints_with_sigma(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * at 10 frequency points sampled uniformly.
    * with smearing method

    """
    si_pbesol.sigmas = [
        0.1,
    ]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        num_frequency_points=10,
    )
    # for _g_line in np.swapaxes(_gammas, -1, -2).reshape(-1, 6):
    #     print("".join(["%.8f, " % g for g in _g_line]))
    # print("".join(["%.8f, " % f for f in _fpoints]))
    np.testing.assert_allclose(
        gammas_sigma, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=1e-2
    )
    np.testing.assert_allclose(freq_points_sigma, _fpoints.ravel(), rtol=0, atol=1e-5)
    si_pbesol.sigmas = None


def test_imag_self_energy_freq_points(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * specified frquency points

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        frequency_points=freq_points,
    )
    np.testing.assert_allclose(
        gammas, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=1e-2
    )
    np.testing.assert_allclose(freq_points, _fpoints.ravel(), rtol=0, atol=1e-5)


def test_imag_self_energy_detailed(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * specified frquency points
    * contribution from each triplet is returned.

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas, _detailed_gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[
            [
                1,
            ]
        ],
        [
            300,
        ],
        frequency_points=freq_points,
        keep_gamma_detail=True,
    )
    np.testing.assert_allclose(
        detailed_gamma,
        _detailed_gammas[0][0, 0].sum(axis=(1, 2, 3, 4)),
        rtol=0,
        atol=1e-2,
    )


def test_imag_self_energy_scat_class1(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * specified frquency points
    * scattering event class 1

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        frequency_points=freq_points,
        scattering_event_class=1,
    )
    # for line in si_pbesol.gammas.reshape(-1, 6):
    #     print(("%10.8f, " * 6) % tuple(line))
    np.testing.assert_allclose(
        gammas_class1, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=1e-2
    )


def test_imag_self_energy_scat_class2(si_pbesol):
    """Imaginary part of self energy spectrum of Si.

    * specified frquency points
    * scattering event class 2

    """
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    _fpoints, _gammas = si_pbesol.run_imag_self_energy(
        si_pbesol.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        frequency_points=freq_points,
        scattering_event_class=2,
    )
    # for line in si_pbesol.gammas.reshape(-1, 6):
    #     print(("%10.8f, " * 6) % tuple(line))
    np.testing.assert_allclose(
        gammas_class2, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=1e-2
    )


def test_imag_self_energy_nacl_npoints(nacl_pbe):
    """Imaginary part of self energy spectrum of NaCl.

    * at 10 frequency points sampled uniformly.

    """
    nacl_pbe.mesh_numbers = [9, 9, 9]
    nacl_pbe.init_phph_interaction()
    _fpoints, _gammas = nacl_pbe.run_imag_self_energy(
        nacl_pbe.grid.grg2bzg[[1, 103]],
        [
            300,
        ],
        num_frequency_points=10,
    )
    # for line in np.swapaxes(_gammas, -1, -2).ravel().reshape(-1, 6):
    #     print(("%10.8f, " * 6) % tuple(line))
    # print(_fpoints.ravel())
    np.testing.assert_allclose(
        gammas_nacl, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=2e-2
    )
    np.testing.assert_allclose(freq_points_nacl, _fpoints.ravel(), rtol=0, atol=1e-5)


def test_imag_self_energy_nacl_nac_npoints(nacl_pbe: Phono3py):
    """Imaginary part of self energy spectrum of NaCl.

    * at 10 frequency points sampled uniformly.
    * at q->0

    """
    nacl_pbe.mesh_numbers = [9, 9, 9]
    nacl_pbe.init_phph_interaction(nac_q_direction=[1, 0, 0])
    _fpoints, _gammas = nacl_pbe.run_imag_self_energy(
        [nacl_pbe.grid.gp_Gamma], [300], num_frequency_points=10
    )
    # for line in np.swapaxes(_gammas, -1, -2).ravel().reshape(-1, 6):
    #     print(("%10.8f, " * 6) % tuple(line))
    # print(_fpoints.ravel())
    np.testing.assert_allclose(
        freq_points_nacl_nac, _fpoints.ravel(), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        gammas_nacl_nac, np.swapaxes(_gammas, -1, -2).ravel(), rtol=0, atol=2e-2
    )
