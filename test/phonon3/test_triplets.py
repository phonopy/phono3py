"""Test for triplets.py."""
import numpy as np

from phono3py.phonon3.triplets import get_triplets_at_q
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address


def test_get_triplets_at_q_type1(si_pbesol_111):
    """Test triplets under type1 grid."""
    pcell = si_pbesol_111.primitive
    psym = si_pbesol_111.primitive_symmetry
    grid_point = 1
    mesh = [4, 4, 4]

    bz_grid = BZGrid(
        mesh,
        lattice=pcell.cell,
        symmetry_dataset=psym.dataset,
        store_dense_gp_map=False,
    )
    triplets, weights = get_triplets_at_q(grid_point, bz_grid)[:2]
    triplets_ref = [
        1,
        0,
        3,
        1,
        1,
        2,
        1,
        4,
        15,
        1,
        5,
        14,
        1,
        6,
        13,
        1,
        7,
        12,
        1,
        65,
        11,
        1,
        9,
        10,
        1,
        24,
        59,
        1,
        26,
        88,
    ]
    weights_ref = [2, 2, 6, 6, 6, 6, 6, 6, 12, 12]
    # print("".join(["%d, " % i for i in triplets.ravel()]))
    # print("".join(["%d, " % i for i in weights]))
    _show_triplets_info(mesh, bz_grid, triplets, np.linalg.inv(pcell.cell))

    np.testing.assert_equal(triplets.ravel(), triplets_ref)
    np.testing.assert_equal(weights, weights_ref)


def test_get_triplets_at_q_type2(si_pbesol_111):
    """Test triplets under type2 grid."""
    pcell = si_pbesol_111.primitive
    psym = si_pbesol_111.primitive_symmetry
    grid_point = 1
    mesh = [4, 4, 4]

    bz_grid = BZGrid(
        mesh, lattice=pcell.cell, symmetry_dataset=psym.dataset, store_dense_gp_map=True
    )
    triplets, weights = get_triplets_at_q(grid_point, bz_grid)[:2]

    triplets_ref = [
        1,
        0,
        4,
        1,
        1,
        2,
        1,
        5,
        18,
        1,
        6,
        17,
        1,
        7,
        16,
        1,
        8,
        15,
        1,
        10,
        14,
        1,
        11,
        12,
        1,
        27,
        84,
        1,
        29,
        82,
    ]
    weights_ref = [2, 2, 6, 6, 6, 6, 6, 6, 12, 12]

    _show_triplets_info(mesh, bz_grid, triplets, np.linalg.inv(pcell.cell))
    # print("".join(["%d, " % i for i in triplets.ravel()]))
    # print("".join(["%d, " % i for i in weights]))

    np.testing.assert_equal(triplets.ravel(), triplets_ref)
    np.testing.assert_equal(weights, weights_ref)


def _show_triplets_info(
    mesh: list, bz_grid: BZGrid, triplets: np.ndarray, reclat: np.ndarray
) -> None:
    """Show triplets details in grid type-1 and 2."""
    shift = np.prod(mesh)
    double_shift = np.prod(mesh) * 8

    for i in np.arange(np.prod(mesh)):
        adrs = []
        if bz_grid.store_dense_gp_map:
            bzgp = bz_grid.gp_map[i]
            multi = bz_grid.gp_map[i + 1] - bz_grid.gp_map[i]
            for j in range(multi):
                adrs.append(bz_grid.addresses[bzgp + j].tolist())
        else:
            bzgp = i
            multi = (
                bz_grid.gp_map[double_shift + i + 1]
                - bz_grid.gp_map[double_shift + i]
                + 1
            )
            adrs.append(bz_grid.addresses[bzgp].tolist())
            for j in range(multi - 1):
                adrs.append(
                    bz_grid.addresses[
                        shift + bz_grid.gp_map[double_shift + i] + j
                    ].tolist()
                )
        print(bzgp, adrs, multi)

    for tp in triplets:
        multis = []
        for tp_adrs in bz_grid.addresses[tp]:
            gp = get_grid_point_from_address(tp_adrs, mesh)
            if bz_grid.store_dense_gp_map:
                multis.append(bz_grid.gp_map[gp + 1] - bz_grid.gp_map[gp])
            else:
                shift = np.prod(mesh) * 8
                multis.append(
                    bz_grid.gp_map[shift + gp + 1] - bz_grid.gp_map[shift + gp] + 1
                )
        bztp = bz_grid.addresses[tp]
        gadrs = bz_grid.addresses[tp].sum(axis=0) / mesh
        d = np.sqrt(np.linalg.norm(np.dot(reclat, gadrs)))
        print(tp, "[", bztp[0], bztp[1], bztp[2], "]", multis, bztp.sum(axis=0), d)
