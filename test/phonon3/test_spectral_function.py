"""Test spectral_function.py."""

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.spectral_function import SpectralFunction


@pytest.mark.parametrize("check_bands", [False, True])
def test_SpectralFunction(si_pbesol: Phono3py, check_bands: bool):
    """Spectral function of Si."""
    if si_pbesol._make_r0_average:
        ref_shifts = [
            [
                [
                    -0.00491549,
                    -0.00513379,
                    -0.00549296,
                    -0.00377646,
                    -0.00342792,
                    -0.00178271,
                    0.00063823,
                    0.00208652,
                    0.00376214,
                    0.00262103,
                ],
                [
                    -0.00491549,
                    -0.00513379,
                    -0.00549296,
                    -0.00377646,
                    -0.00342792,
                    -0.00178271,
                    0.00063823,
                    0.00208652,
                    0.00376214,
                    0.00262103,
                ],
                [
                    -0.01200352,
                    -0.01272740,
                    -0.01213832,
                    -0.00902171,
                    -0.01079296,
                    -0.01033632,
                    -0.00661888,
                    0.00196845,
                    0.01212704,
                    0.01037479,
                ],
                [
                    -0.12224881,
                    -0.12186476,
                    -0.10950402,
                    -0.09619987,
                    -0.10705652,
                    -0.13503653,
                    -0.20145401,
                    -0.07468261,
                    0.15954608,
                    0.16311580,
                ],
                [
                    -0.12117175,
                    -0.11966157,
                    -0.10786945,
                    -0.09635481,
                    -0.10686396,
                    -0.12762500,
                    -0.20163789,
                    -0.08374266,
                    0.15834565,
                    0.16375939,
                ],
                [
                    -0.12117175,
                    -0.11966157,
                    -0.10786945,
                    -0.09635481,
                    -0.10686396,
                    -0.12762500,
                    -0.20163789,
                    -0.08374266,
                    0.15834565,
                    0.16375939,
                ],
            ],
            [
                [
                    -0.01895098,
                    -0.01944799,
                    -0.02343492,
                    -0.01391542,
                    -0.00562335,
                    -0.00675556,
                    -0.00299433,
                    0.00568717,
                    0.01407410,
                    0.01103152,
                ],
                [
                    -0.01883020,
                    -0.01904414,
                    -0.02189420,
                    -0.01497364,
                    -0.00887001,
                    -0.00907876,
                    -0.00600166,
                    0.00297018,
                    0.01485332,
                    0.01364674,
                ],
                [
                    -0.04158644,
                    -0.04204489,
                    -0.04660642,
                    -0.03449069,
                    -0.02581911,
                    -0.03329617,
                    -0.02458896,
                    -0.00519089,
                    0.03119712,
                    0.04283032,
                ],
                [
                    -0.09542060,
                    -0.09131037,
                    -0.08642350,
                    -0.07536168,
                    -0.07744657,
                    -0.10100751,
                    -0.11855897,
                    -0.05851935,
                    0.08976609,
                    0.12789279,
                ],
                [
                    -0.11827973,
                    -0.11431900,
                    -0.10875016,
                    -0.10196855,
                    -0.10719767,
                    -0.13236285,
                    -0.19702133,
                    -0.09252739,
                    0.14584477,
                    0.17195801,
                ],
                [
                    -0.11277741,
                    -0.10757643,
                    -0.10150954,
                    -0.09524697,
                    -0.10202478,
                    -0.12884057,
                    -0.18556233,
                    -0.08932224,
                    0.13482432,
                    0.16483631,
                ],
            ],
        ]
        ref_spec_funcs = [
            [
                [
                    0.00000000,
                    0.00001578,
                    0.00000272,
                    0.00000161,
                    0.00000013,
                    0.00000032,
                    0.00000006,
                    0.00000006,
                    0.00000001,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00001578,
                    0.00000272,
                    0.00000161,
                    0.00000013,
                    0.00000032,
                    0.00000006,
                    0.00000006,
                    0.00000001,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00223977,
                    0.00005791,
                    0.00001160,
                    0.00000066,
                    0.00000259,
                    0.00000098,
                    0.00000112,
                    0.00000029,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00012503,
                    0.00049470,
                    0.00089500,
                    0.00632551,
                    0.00256627,
                    0.00044659,
                    0.00118653,
                    0.00029733,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00013193,
                    0.00044796,
                    0.00083448,
                    0.00540572,
                    0.00352615,
                    0.00026154,
                    0.00125032,
                    0.00031326,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00013193,
                    0.00044796,
                    0.00083448,
                    0.00540572,
                    0.00352615,
                    0.00026154,
                    0.00125032,
                    0.00031326,
                    0.00000000,
                ],
            ],
            [
                [
                    -0.00000000,
                    0.00320096,
                    0.00007269,
                    0.00008693,
                    0.00000484,
                    0.00000526,
                    0.00000203,
                    0.00000189,
                    0.00000039,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00055197,
                    0.00022117,
                    0.00012818,
                    0.00000952,
                    0.00000843,
                    0.00000335,
                    0.00000334,
                    0.00000105,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00014311,
                    0.00162453,
                    0.00335197,
                    0.00009023,
                    0.00005696,
                    0.00004058,
                    0.00002353,
                    0.00001610,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00018525,
                    0.00041136,
                    0.00272557,
                    0.01393003,
                    0.00057594,
                    0.00031515,
                    0.00034235,
                    0.00017612,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00016156,
                    0.00034722,
                    0.00132145,
                    0.04066875,
                    0.00134953,
                    0.00030062,
                    0.00087157,
                    0.00029123,
                    0.00000000,
                ],
                [
                    -0.00000000,
                    0.00016722,
                    0.00030198,
                    0.00114459,
                    0.01745510,
                    0.00141671,
                    0.00035262,
                    0.00084328,
                    0.00029789,
                    0.00000000,
                ],
            ],
        ]
    else:
        ref_shifts = [
            [
                [
                    -0.00496121,
                    -0.00517009,
                    -0.00553453,
                    -0.00380260,
                    -0.00343693,
                    -0.00177772,
                    0.00067068,
                    0.00210892,
                    0.00377279,
                    0.00262923,
                ],
                [
                    -0.00496121,
                    -0.00517009,
                    -0.00553453,
                    -0.00380260,
                    -0.00343693,
                    -0.00177772,
                    0.00067068,
                    0.00210892,
                    0.00377279,
                    0.00262923,
                ],
                [
                    -0.01202453,
                    -0.01275071,
                    -0.01219115,
                    -0.00903023,
                    -0.01073474,
                    -0.01031603,
                    -0.00662244,
                    0.00195097,
                    0.01211624,
                    0.01038809,
                ],
                [
                    -0.12223984,
                    -0.12186467,
                    -0.10941590,
                    -0.09614195,
                    -0.10716126,
                    -0.13519843,
                    -0.20168604,
                    -0.07470229,
                    0.15969340,
                    0.16315933,
                ],
                [
                    -0.12114484,
                    -0.11964957,
                    -0.10778748,
                    -0.09630527,
                    -0.10697739,
                    -0.12776042,
                    -0.20180332,
                    -0.08373516,
                    0.15847215,
                    0.16377770,
                ],
                [
                    -0.12114484,
                    -0.11964957,
                    -0.10778748,
                    -0.09630527,
                    -0.10697739,
                    -0.12776042,
                    -0.20180332,
                    -0.08373516,
                    0.15847215,
                    0.16377770,
                ],
            ],
            [
                [
                    -0.01893616,
                    -0.01942744,
                    -0.02339230,
                    -0.01392117,
                    -0.00568286,
                    -0.00676300,
                    -0.00298881,
                    0.00566943,
                    0.01407270,
                    0.01103835,
                ],
                [
                    -0.01882494,
                    -0.01903756,
                    -0.02188396,
                    -0.01496531,
                    -0.00891242,
                    -0.00912826,
                    -0.00600410,
                    0.00298462,
                    0.01486886,
                    0.01365297,
                ],
                [
                    -0.04155678,
                    -0.04201981,
                    -0.04661298,
                    -0.03446840,
                    -0.02577765,
                    -0.03332493,
                    -0.02460421,
                    -0.00520459,
                    0.03117184,
                    0.04283480,
                ],
                [
                    -0.09551912,
                    -0.09141204,
                    -0.08650838,
                    -0.07531933,
                    -0.07736040,
                    -0.10097208,
                    -0.11850788,
                    -0.05857319,
                    0.08971321,
                    0.12793090,
                ],
                [
                    -0.11821481,
                    -0.11425389,
                    -0.10865996,
                    -0.10189830,
                    -0.10716084,
                    -0.13231357,
                    -0.19690540,
                    -0.09252776,
                    0.14571718,
                    0.17189918,
                ],
                [
                    -0.11276994,
                    -0.10757084,
                    -0.10142181,
                    -0.09519851,
                    -0.10205844,
                    -0.12882962,
                    -0.18549798,
                    -0.08931099,
                    0.13476362,
                    0.16481222,
                ],
            ],
        ]
        ref_spec_funcs = [
            [
                [
                    0.00000000,
                    0.00001654,
                    0.00000271,
                    0.00000163,
                    0.00000013,
                    0.00000032,
                    0.00000006,
                    0.00000006,
                    0.00000001,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00001654,
                    0.00000271,
                    0.00000163,
                    0.00000013,
                    0.00000032,
                    0.00000006,
                    0.00000006,
                    0.00000001,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00223584,
                    0.00005772,
                    0.00001181,
                    0.00000067,
                    0.00000258,
                    0.00000098,
                    0.00000112,
                    0.00000029,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00012499,
                    0.00049591,
                    0.00088903,
                    0.00629350,
                    0.00256216,
                    0.00044658,
                    0.00118784,
                    0.00029727,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00013178,
                    0.00044892,
                    0.00082915,
                    0.00537908,
                    0.00352736,
                    0.00026142,
                    0.00125148,
                    0.00031313,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00013178,
                    0.00044892,
                    0.00082915,
                    0.00537908,
                    0.00352736,
                    0.00026142,
                    0.00125148,
                    0.00031313,
                    0.00000000,
                ],
            ],
            [
                [
                    -0.00000000,
                    0.00320916,
                    0.00007278,
                    0.00008659,
                    0.00000484,
                    0.00000528,
                    0.00000202,
                    0.00000189,
                    0.00000039,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00055185,
                    0.00022088,
                    0.00012806,
                    0.00000942,
                    0.00000847,
                    0.00000336,
                    0.00000334,
                    0.00000105,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00014260,
                    0.00161534,
                    0.00335775,
                    0.00008970,
                    0.00005681,
                    0.00004062,
                    0.00002350,
                    0.00001611,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00018553,
                    0.00041303,
                    0.00273911,
                    0.01389025,
                    0.00057626,
                    0.00031509,
                    0.00034209,
                    0.00017625,
                    0.00000000,
                ],
                [
                    0.00000000,
                    0.00016153,
                    0.00034741,
                    0.00131818,
                    0.04062696,
                    0.00134833,
                    0.00030074,
                    0.00087080,
                    0.00029123,
                    0.00000000,
                ],
                [
                    -0.00000000,
                    0.00016730,
                    0.00030334,
                    0.00113844,
                    0.01748363,
                    0.00141678,
                    0.00035292,
                    0.00084285,
                    0.00029790,
                    0.00000000,
                ],
            ],
        ]

    si_pbesol.mesh_numbers = [9, 9, 9]
    if check_bands:
        si_pbesol.band_indices = [[4, 5]]
    si_pbesol.init_phph_interaction()
    sf = SpectralFunction(
        si_pbesol.phph_interaction,
        si_pbesol.grid.grg2bzg[[1, 103]],
        temperatures=[
            300,
        ],
        num_frequency_points=10,
        log_level=1,
    )
    sf.run()

    if check_bands:
        np.testing.assert_allclose(
            np.array(ref_shifts)[:, [4, 5], :],
            sf.shifts[0, 0],
            atol=1e-2,
        )
        np.testing.assert_allclose(
            np.array(ref_spec_funcs)[:, [4, 5], :],
            sf.spectral_functions[0, 0],
            atol=1e-2,
            rtol=1e-2,
        )
    else:
        # for line in sf.shifts[0, 0, 0]:
        #     print("[", ",".join([f"{val:.8f}" for val in line]), "],")
        # print("")
        # for line in sf.shifts[0, 0, 1]:
        #     print("[", ",".join([f"{val:.8f}" for val in line]), "],")

        for line in sf.spectral_functions[0, 0, 0]:
            print("[", ",".join([f"{val:.8f}" for val in line]), "],")
        print("")
        for line in sf.spectral_functions[0, 0, 1]:
            print("[", ",".join([f"{val:.8f}" for val in line]), "],")

        np.testing.assert_allclose(ref_shifts, sf.shifts[0, 0], atol=1e-2)
        np.testing.assert_allclose(
            ref_spec_funcs, sf.spectral_functions[0, 0], atol=1e-2, rtol=1e-2
        )

    if check_bands:
        si_pbesol.band_indices = None
