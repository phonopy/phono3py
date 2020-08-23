import numpy as np

si_pbesol_Delta = [
    [-0.0057666, -0.0057666, -0.01639729, -0.14809965,
     -0.15091765, -0.15091765],
    [-0.02078728, -0.02102094, -0.06573269, -0.11432603,
     -0.1366966, -0.14371315]]


def test_frequency_shift(si_pbesol):
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    delta = si_pbesol.run_frequency_shift(
        [1, 103],
        temperatures=[300, ],
        write_Delta_hdf5=False)
    np.testing.assert_allclose(si_pbesol_Delta, delta[0, :, 0], atol=1e-5)
