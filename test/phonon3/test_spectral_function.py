import numpy as np
from phono3py.phonon3.spectral_function import SpectralFunction


def test_SpectralFunction(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    sf = SpectralFunction(si_pbesol.phph_interaction,
                          [1, 103],
                          temperatures=[300, ],
                          num_frequency_points=10)
    sf.run()
    print(sf.half_linewidths)

    # np.testing.assert_allclose(
    #     gammas, si_pbesol.gammas.ravel(), atol=1e-2)
    # np.testing.assert_allclose(
    #     freq_points, si_pbesol.frequency_points.ravel(),
    #     atol=1e-5)
