import numpy as np

from phono3py.phonon3.triplets import get_triplets_at_q
from phono3py.phonon.grid import BZGrid


def test_get_triplets_at_q_type1(si_pbesol_111):
    pcell = si_pbesol_111.primitive
    psym = si_pbesol_111.primitive_symmetry
    grid_point = 1
    mesh = [4, 4, 4]

    bz_grid = BZGrid(mesh,
                     lattice=pcell.cell,
                     symmetry_dataset=psym.dataset,
                     is_dense_gp_map=False)
    triplets, weights = get_triplets_at_q(grid_point, bz_grid)[:2]

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
                     symmetry_dataset=psym.dataset,
                     is_dense_gp_map=True)
    triplets, weights = get_triplets_at_q(grid_point, bz_grid)[:2]

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
