import os
import numpy as np

from phono3py.phonon3.triplets import (get_grid_point_from_address,
                                       get_grid_point_from_address_py,
                                       get_triplets_at_q)

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_grid_point_from_address(agno2_cell):
    """

    Compare get_grid_point_from_address from spglib and that
    written in python with mesh numbers.

    """

    mesh = (10, 10, 10)

    for address in list(np.ndindex(mesh)):
        gp_spglib = get_grid_point_from_address(address, mesh)
        gp_py = get_grid_point_from_address_py(address, mesh)
        # print("%s %d %d" % (address, gp_spglib, gp_py))
        np.testing.assert_equal(gp_spglib, gp_py)


def test_get_triplets_at_q(si_pbesol_111):
    pcell = si_pbesol_111.primitive
    reclat = np.linalg.inv(pcell.cell)
    psym = si_pbesol_111.primitive_symmetry
    grid_point = 1
    mesh = [4, 4, 4]

    triplets, weights = get_triplets_at_q(
        grid_point,
        mesh,
        psym.get_pointgroup_operations(),
        reclat)[:2]

    triplets_ref = [1, 0, 3, 1, 1, 64, 1, 4, 15, 1,
                    5, 14, 1, 6, 13, 1, 7, 12, 1, 8,
                    11, 1, 9, 66, 1, 24, 59, 1, 26, 88]
    weights_ref = [2, 2, 6, 6, 6, 6, 6, 6, 12, 12]
    np.testing.assert_equal(triplets.ravel(), triplets_ref)
    np.testing.assert_equal(weights, weights_ref)
