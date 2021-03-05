import os
import numpy as np

from phono3py.phonon3.triplets import (get_grid_point_from_address,
                                       get_grid_point_from_address_py,
                                       get_triplets_at_q,
                                       BZGrid)

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


def test_BZGrid(si_pbesol_111):
    gp_map2 = [1, 0, 1, 1, 2, 2, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 2, 9, 1, 11,
               2, 12, 1, 14, 1, 15, 1, 16, 1, 17, 1, 18, 1, 19, 1, 20, 1, 21,
               1, 22, 1, 23, 1, 24, 1, 25, 1, 26, 1, 27, 1, 28, 1, 29, 4, 30,
               1, 34, 1, 35, 4, 36, 1, 40, 2, 41, 1, 43, 2, 44, 1, 46, 1, 47,
               1, 48, 1, 49, 4, 50, 2, 54, 1, 56, 2, 57, 1, 59, 1, 60, 4, 61,
               1, 65, 1, 66, 1, 67, 1, 68, 1, 69, 1, 70, 1, 71, 1, 72, 4, 73,
               1, 77, 1, 78, 4, 79, 1, 83, 1, 84, 1, 85, 1, 86, 1, 87, 1, 88]

    reclat = np.linalg.inv(si_pbesol_111.primitive.cell)
    mesh = [4, 4, 4]

    bzgrid1 = BZGrid()
    bzgrid1.set_bz_grid(mesh, reclat, is_dense_bz_map=False)
    bzgrid2 = BZGrid()
    bzgrid2.set_bz_grid(mesh, reclat, is_dense_bz_map=True)

    adrs1 = bzgrid1.addresses[:np.prod(mesh)]
    adrs2 = bzgrid2.addresses[bzgrid2.gp_map[:, 1]]
    assert ((adrs1 - adrs2) % mesh == 0).all()
    np.testing.assert_equal(bzgrid1.addresses.shape, bzgrid2.addresses.shape)
    np.testing.assert_equal(bzgrid2.gp_map.ravel(), gp_map2)
