import numpy as np

gammas = [
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0005412, 0.0005412, 0.0008843, 0.0191694, 0.0206316, 0.0206316,
    0.0019424, 0.0019424, 0.0067566, 0.0548967, 0.0506115, 0.0506115,
    0.0062204, 0.0062204, 0.0088148, 0.0426150, 0.0417223, 0.0417223,
    0.0016263, 0.0016263, 0.0017293, 0.0279509, 0.0289259, 0.0289259,
    0.0097926, 0.0097926, 0.0170092, 0.0438828, 0.0523105, 0.0523105,
    0.0035542, 0.0035542, 0.0135109, 0.0623533, 0.0343746, 0.0343746,
    0.0073140, 0.0073140, 0.0289659, 0.5006760, 0.5077932, 0.5077932,
    0.0016144, 0.0016144, 0.0126326, 0.2731933, 0.2791702, 0.2791702,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0023304, 0.0026469, 0.0052513, 0.0209641, 0.0220092, 0.0234752,
    0.0035532, 0.0038158, 0.0087882, 0.0276654, 0.0315055, 0.0286975,
    0.0345193, 0.0277533, 0.0495734, 0.0511798, 0.0465938, 0.0436605,
    0.0071705, 0.0081615, 0.0139063, 0.0204058, 0.0307320, 0.0237855,
    0.0202095, 0.0197716, 0.0316074, 0.0402461, 0.0438103, 0.0394924,
    0.0171448, 0.0176446, 0.0567310, 0.0930479, 0.0570520, 0.0622142,
    0.0292639, 0.0328821, 0.0667957, 0.2541887, 0.4592188, 0.4234131,
    0.0104887, 0.0179753, 0.0827533, 0.2659557, 0.3242633, 0.3189804,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]
freq_points = [0., 3.41024688, 6.82049376, 10.23074063, 13.64098751,
               17.05123439, 20.46148127, 23.87172814, 27.28197502, 30.6922219]
detailed_gamma = [0.00000000, 0.00653193, 0.02492913, 0.01682092, 0.01001680,
                  0.02181888, 0.01858641, 0.16208762, 0.09598706, 0.00000000]


def test_imag_self_energy_npoints(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_imag_self_energy(
        [1, 103],
        temperatures=[300, ],
        num_frequency_points=10)
    print(np.array(si_pbesol.gammas).shape)
    np.testing.assert_allclose(
        gammas, si_pbesol.gammas.ravel(), atol=1e-2)
    np.testing.assert_allclose(
        freq_points, si_pbesol.frequency_points.ravel(),
        atol=1e-5)


def test_imag_self_energy_freq_points(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_imag_self_energy(
        [1, 103],
        temperatures=[300, ],
        frequency_points=freq_points)
    np.testing.assert_allclose(
        gammas, si_pbesol.gammas.ravel(), atol=1e-2)
    np.testing.assert_allclose(
        freq_points, si_pbesol.frequency_points.ravel(), atol=1e-5)


def test_imag_self_energy_detailed(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_imag_self_energy(
        [1, ],
        temperatures=[300, ],
        frequency_points=freq_points,
        keep_gamma_detail=True)
    np.testing.assert_allclose(
        detailed_gamma,
        si_pbesol.detailed_gammas[0][0, 0].sum(axis=(1, 2, 3, 4)),
        atol=1e-2)


# def test_imag_self_energy_scat_class1(si_pbesol):
#     si_pbesol.mesh_numbers = [9, 9, 9]
#     si_pbesol.init_phph_interaction()
#     si_pbesol.run_imag_self_energy(
#         [1, 103],
#         temperatures=[300, ],
#         frequency_points=freq_points,
#         scattering_event_class=1)
#     print(si_pbesol.gammas)
#     np.testing.assert_allclose(
#         gammas, np.array(si_pbesol.gammas).ravel(), atol=1e-2)
#     np.testing.assert_allclose(
#         freq_points * 2, np.array(si_pbesol.frequency_points).ravel(),
#         atol=1e-5)
#     raise
