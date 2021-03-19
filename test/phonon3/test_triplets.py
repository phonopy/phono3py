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


def test_get_triplets_at_q_type1(si_pbesol_111):
    pcell = si_pbesol_111.primitive
    psym = si_pbesol_111.primitive_symmetry
    grid_point = 1
    mesh = [4, 4, 4]

    bz_grid = BZGrid(mesh,
                     lattice=pcell.cell,
                     is_dense_gp_map=False)
    triplets, weights = get_triplets_at_q(
        grid_point,
        psym.pointgroup_operations,
        bz_grid)[:2]

    triplets_ref = [1, 0, 3, 1, 1, 2, 1, 4, 15, 1,
                    5, 14, 1, 6, 13, 1, 7, 12, 1, 8,
                    11, 1, 9, 10, 1, 24, 59, 1, 26, 57,]
    weights_ref = [2, 2, 6, 6, 6, 6, 6, 6, 12, 12]

    # _show_triplets_info(mesh, bzgrid, triplets, reclat, bztype=1)

    np.testing.assert_equal(triplets.ravel(), triplets_ref)
    np.testing.assert_equal(weights, weights_ref)


def test_get_triplets_at_q_type2(si_pbesol_111):
    pcell = si_pbesol_111.primitive
    psym = si_pbesol_111.primitive_symmetry
    grid_point = 1
    mesh = [4, 4, 4]

    bz_grid = BZGrid(mesh,
                     lattice=pcell.cell,
                     is_dense_gp_map=True)
    triplets, weights = get_triplets_at_q(
        grid_point,
        psym.pointgroup_operations,
        bz_grid)[:2]

    triplets_ref = [1, 0, 4, 1, 1, 2, 1, 5, 18, 1, 6, 17, 1, 7, 16, 1,
                    8, 15, 1, 10, 14, 1, 11, 12, 1, 27, 84, 1, 29, 82]
    weights_ref = [2, 2, 6, 6, 6, 6, 6, 6, 12, 12]

    # _show_triplets_info(mesh, bzgrid, triplets, reclat, bztype=2)
    # print("".join(["%d, " % i for i in triplets.ravel()]))
    # print("".join(["%d, " % i for i in weights]))

    np.testing.assert_equal(triplets.ravel(), triplets_ref)
    np.testing.assert_equal(weights, weights_ref)


def _show_triplets_info(mesh, bz_grid, triplets, reclat, bztype=2):
    if bztype == 2:
        for i in range(np.prod(mesh)):
            bzgp = bz_grid.gp_map[i]
            print(bz_grid.addresses[bzgp],
                  bz_grid.gp_map[i + 1] - bz_grid.gp_map[i])

    for tp in triplets:
        multis = []
        if bztype == 2:
            for tp_adrs in bz_grid.addresses[tp]:
                gp = get_grid_point_from_address(tp_adrs, mesh)
                multis.append(bz_grid.gp_map[gp + 1] - bz_grid.gp_map[gp])
        bztp = bz_grid.addresses[tp]
        gadrs = bz_grid.addresses[tp].sum(axis=0) / mesh
        d = np.sqrt(np.linalg.norm(np.dot(reclat, gadrs)))
        print("[", bztp[0], bztp[1], bztp[2], "]", multis, bztp.sum(axis=0), d)


def test_BZGrid(si_pbesol_111):
    """Basis test of BZGrid type1 and type2"""

    lat = si_pbesol_111.primitive.cell
    reclat = np.linalg.inv(lat)
    mesh = [4, 4, 4]

    gp_map2 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18,
               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35,
               36, 40, 41, 43, 44, 46, 47, 48, 49, 50, 54, 56, 57, 59,
               60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 77, 78, 79,
               83, 84, 85, 86, 87, 88, 89]

    bzgrid1 = BZGrid(mesh, lattice=lat, is_dense_gp_map=False)
    bzgrid2 = BZGrid(mesh, lattice=lat, is_dense_gp_map=True)

    adrs1 = bzgrid1.addresses[:np.prod(mesh)]
    adrs2 = bzgrid2.addresses[bzgrid2.gp_map[:-1]]
    assert ((adrs1 - adrs2) % mesh == 0).all()
    np.testing.assert_equal(bzgrid1.addresses.shape, bzgrid2.addresses.shape)
    # print("".join(["%d, " % i for i in bzgrid2.gp_map.ravel()]))
    np.testing.assert_equal(bzgrid2.gp_map.ravel(), gp_map2)

    dist1 = np.sqrt((np.dot(adrs1, reclat.T) ** 2).sum(axis=1))
    dist2 = np.sqrt((np.dot(adrs2, reclat.T) ** 2).sum(axis=1))
    np.testing.assert_allclose(dist1, dist2, atol=1e-8)


def test_BZGrid_bzg2grg(si_pbesol_111):
    """BZGrid to GRGrid

    This mapping table is stored in BZGrid, but also determined by
    get_grid_point_from_address. This test checks the consistency.

    """

    lat = si_pbesol_111.primitive.cell
    mesh = [4, 4, 4]
    bzgrid1 = BZGrid(mesh, lattice=lat, is_dense_gp_map=False)
    grg = []
    for i in range(len(bzgrid1.addresses)):
        grg.append(get_grid_point_from_address(bzgrid1.addresses[i], mesh))
    np.testing.assert_equal(grg, bzgrid1.bzg2grg)

    bzgrid2 = BZGrid(mesh, lattice=lat, is_dense_gp_map=True)
    grg = []
    for i in range(len(bzgrid2.addresses)):
        grg.append(get_grid_point_from_address(bzgrid2.addresses[i], mesh))
    np.testing.assert_equal(grg, bzgrid2.bzg2grg)
