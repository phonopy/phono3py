"""Tests for grids."""
import numpy as np
import pytest
from phonopy.structure.tetrahedron_method import TetrahedronMethod

from phono3py.phonon.grid import (
    BZGrid,
    _get_grid_points_by_bz_rotations_c,
    _get_grid_points_by_bz_rotations_py,
    _get_grid_points_by_rotations,
    can_use_std_lattice,
    get_grid_point_from_address,
    get_grid_point_from_address_py,
)


def test_get_grid_point_from_address(agno2_cell):
    """Test for get_grid_point_from_address.

    Compare get_grid_point_from_address from spglib and that
    written in python with mesh numbers.

    """
    mesh = (10, 10, 10)

    for address in list(np.ndindex(mesh)):
        gp_spglib = get_grid_point_from_address(address, mesh)
        gp_py = get_grid_point_from_address_py(address, mesh)
        # print("%s %d %d" % (address, gp_spglib, gp_py))
        np.testing.assert_equal(gp_spglib, gp_py)


def test_BZGrid(si_pbesol_111):
    """Tests of BZGrid type1 and type2."""
    lat = si_pbesol_111.primitive.cell
    reclat = np.linalg.inv(lat)
    mesh = [4, 4, 4]

    gp_map2 = [
        0,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        34,
        35,
        36,
        40,
        41,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        54,
        56,
        57,
        59,
        60,
        61,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        77,
        78,
        79,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
    ]

    bzgrid1 = BZGrid(mesh, lattice=lat, store_dense_gp_map=False)
    bzgrid2 = BZGrid(mesh, lattice=lat, store_dense_gp_map=True)

    adrs1 = bzgrid1.addresses[: np.prod(mesh)]
    adrs2 = bzgrid2.addresses[bzgrid2.gp_map[:-1]]
    assert ((adrs1 - adrs2) % mesh == 0).all()
    np.testing.assert_equal(bzgrid1.addresses.shape, bzgrid2.addresses.shape)
    # print("".join(["%d, " % i for i in bzgrid2.gp_map.ravel()]))
    np.testing.assert_equal(bzgrid2.gp_map.ravel(), gp_map2)

    dist1 = np.sqrt((np.dot(adrs1, reclat.T) ** 2).sum(axis=1))
    dist2 = np.sqrt((np.dot(adrs2, reclat.T) ** 2).sum(axis=1))
    np.testing.assert_allclose(dist1, dist2, atol=1e-8)


def test_BZGrid_bzg2grg(si_pbesol_111):
    """Test of mapping of BZGrid to GRGrid.

    This mapping table is stored in BZGrid, but also determined by
    get_grid_point_from_address. This test checks the consistency.

    """
    lat = si_pbesol_111.primitive.cell
    mesh = [4, 4, 4]
    bzgrid1 = BZGrid(mesh, lattice=lat, store_dense_gp_map=False)
    grg = []
    for i in range(len(bzgrid1.addresses)):
        grg.append(get_grid_point_from_address(bzgrid1.addresses[i], mesh))
    np.testing.assert_equal(grg, bzgrid1.bzg2grg)

    bzgrid2 = BZGrid(mesh, lattice=lat, store_dense_gp_map=True)
    grg = []
    for i in range(len(bzgrid2.addresses)):
        grg.append(get_grid_point_from_address(bzgrid2.addresses[i], mesh))
    np.testing.assert_equal(grg, bzgrid2.bzg2grg)


def test_BZGrid_SNF(si_pbesol_111):
    """Test of SNF in BZGrid."""
    lat = si_pbesol_111.primitive.cell
    mesh = 10
    bzgrid1 = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=si_pbesol_111.primitive_symmetry.dataset,
        use_grg=True,
        store_dense_gp_map=False,
    )
    _test_BZGrid_SNF(bzgrid1)

    bzgrid2 = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=si_pbesol_111.primitive_symmetry.dataset,
        use_grg=True,
        store_dense_gp_map=True,
    )
    _test_BZGrid_SNF(bzgrid2)


def _test_BZGrid_SNF(bzgrid):
    # from phonopy.structure.atoms import PhonopyAtoms
    # from phonopy.interface.vasp import get_vasp_structure_lines

    A = bzgrid.grid_matrix
    D_diag = bzgrid.D_diag
    P = bzgrid.P
    Q = bzgrid.Q
    np.testing.assert_equal(np.dot(P, np.dot(A, Q)), np.diag(D_diag))

    # print(D_diag)
    # grg2bzg = bzgrid.grg2bzg
    # qpoints = np.dot(bzgrid.addresses[grg2bzg], bzgrid.QDinv.T)
    # cell = PhonopyAtoms(cell=np.linalg.inv(lat).T,
    #                     scaled_positions=qpoints,
    #                     numbers=[1,] * len(qpoints))
    # print("\n".join(get_vasp_structure_lines(cell)))

    gr_addresses = bzgrid.addresses[bzgrid.grg2bzg]
    # print(D_diag)
    # print(len(gr_addresses))
    # for line in gr_addresses.reshape(-1, 12):
    #     print("".join(["%d, " % i for i in line]))

    ref = [
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        -2,
        0,
        -1,
        -2,
        0,
        0,
        -1,
        0,
        -1,
        -1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        2,
        1,
        1,
        2,
        1,
        0,
        -1,
        1,
        -1,
        -1,
        1,
        0,
        0,
        -2,
        -1,
        0,
        -2,
        0,
        1,
        2,
        1,
        1,
        2,
        0,
        -2,
        -2,
        -1,
        -2,
        -2,
        0,
        -1,
        -2,
        -1,
        -1,
        -2,
        0,
        0,
        -1,
        -1,
        0,
        -1,
        0,
        1,
        -1,
        -1,
        1,
        -1,
        0,
        -2,
        -1,
        -1,
        -2,
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]

    assert (
        ((np.reshape(ref, (-1, 3)) - gr_addresses) % bzgrid.D_diag).ravel() == 0
    ).all()

    ref_rots = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        2,
        1,
        -1,
        2,
        2,
        -1,
        1,
        0,
        0,
        2,
        -1,
        0,
        4,
        0,
        -1,
        1,
        0,
        0,
        0,
        -1,
        1,
        2,
        -2,
        1,
        -1,
        0,
        0,
        0,
        -1,
        0,
        -2,
        -2,
        1,
        -1,
        0,
        0,
        0,
        1,
        -1,
        0,
        0,
        -1,
        -1,
        0,
        0,
        -2,
        1,
        0,
        -2,
        2,
        -1,
        -1,
        0,
        0,
        -2,
        -1,
        1,
        -4,
        0,
        1,
        -1,
        -1,
        1,
        0,
        -1,
        1,
        -2,
        -1,
        2,
        -1,
        -1,
        1,
        0,
        -2,
        1,
        0,
        -3,
        2,
        -1,
        -1,
        1,
        -2,
        -1,
        1,
        -2,
        -3,
        2,
        -1,
        -1,
        1,
        -2,
        0,
        1,
        -4,
        -1,
        2,
        1,
        1,
        -1,
        0,
        1,
        -1,
        0,
        3,
        -2,
        1,
        1,
        -1,
        2,
        0,
        -1,
        2,
        1,
        -2,
        1,
        1,
        -1,
        2,
        1,
        -1,
        4,
        1,
        -2,
        1,
        1,
        -1,
        0,
        2,
        -1,
        2,
        3,
        -2,
        -1,
        1,
        0,
        -2,
        0,
        1,
        -2,
        1,
        1,
        -1,
        1,
        0,
        -2,
        1,
        0,
        -4,
        1,
        1,
        -1,
        1,
        0,
        0,
        2,
        -1,
        -2,
        3,
        -1,
        -1,
        1,
        0,
        0,
        1,
        0,
        0,
        3,
        -1,
        1,
        -1,
        0,
        2,
        0,
        -1,
        4,
        -1,
        -1,
        1,
        -1,
        0,
        0,
        -1,
        0,
        2,
        -1,
        -1,
        1,
        -1,
        0,
        0,
        -2,
        1,
        0,
        -3,
        1,
        1,
        -1,
        0,
        2,
        -1,
        0,
        2,
        -3,
        1,
        -1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1,
        -1,
        0,
        0,
        -2,
        -1,
        1,
        -2,
        -2,
        1,
        -1,
        0,
        0,
        -2,
        1,
        0,
        -4,
        0,
        1,
        -1,
        0,
        0,
        0,
        1,
        -1,
        -2,
        2,
        -1,
        1,
        0,
        0,
        0,
        1,
        0,
        2,
        2,
        -1,
        1,
        0,
        0,
        0,
        -1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        2,
        -1,
        0,
        2,
        -2,
        1,
        1,
        0,
        0,
        2,
        1,
        -1,
        4,
        0,
        -1,
        1,
        1,
        -1,
        0,
        1,
        -1,
        2,
        1,
        -2,
        1,
        1,
        -1,
        0,
        2,
        -1,
        0,
        3,
        -2,
        1,
        1,
        -1,
        2,
        1,
        -1,
        2,
        3,
        -2,
        1,
        1,
        -1,
        2,
        0,
        -1,
        4,
        1,
        -2,
        -1,
        -1,
        1,
        0,
        -1,
        1,
        0,
        -3,
        2,
        -1,
        -1,
        1,
        -2,
        0,
        1,
        -2,
        -1,
        2,
        -1,
        -1,
        1,
        -2,
        -1,
        1,
        -4,
        -1,
        2,
        -1,
        -1,
        1,
        0,
        -2,
        1,
        -2,
        -3,
        2,
        1,
        -1,
        0,
        2,
        0,
        -1,
        2,
        -1,
        -1,
        1,
        -1,
        0,
        2,
        -1,
        0,
        4,
        -1,
        -1,
        1,
        -1,
        0,
        0,
        -2,
        1,
        2,
        -3,
        1,
        1,
        -1,
        0,
        0,
        -1,
        0,
        0,
        -3,
        1,
        -1,
        1,
        0,
        -2,
        0,
        1,
        -4,
        1,
        1,
        -1,
        1,
        0,
        0,
        1,
        0,
        -2,
        1,
        1,
        -1,
        1,
        0,
        0,
        2,
        -1,
        0,
        3,
        -1,
        -1,
        1,
        0,
        -2,
        1,
        0,
        -2,
        3,
        -1,
    ]

    np.testing.assert_equal(ref_rots, bzgrid.rotations.ravel())


def test_BZGrid_SNF_hexagonal(aln_lda):
    """Test of SNF in BZGrid."""
    lat = aln_lda.primitive.cell
    mesh = 20
    bzgrid = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=aln_lda.primitive_symmetry.dataset,
    )
    np.testing.assert_equal(bzgrid.D_diag, [7, 7, 4])

    bzgrid = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=aln_lda.primitive_symmetry.dataset,
        use_grg=True,
    )
    np.testing.assert_equal(bzgrid.D_diag, [7, 7, 4])

    bzgrid = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=aln_lda.primitive_symmetry.dataset,
        use_grg=True,
        force_SNF=True,
    )
    np.testing.assert_equal(bzgrid.D_diag, [1, 7, 28])

    bzgrid = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=aln_lda.primitive_symmetry.dataset,
        use_grg=True,
        force_SNF=True,
        SNF_coordinates="direct",
    )
    np.testing.assert_equal(bzgrid.D_diag, [1, 6, 30])


def test_BZGrid_SNF_nonprimitive(si_pbesol_111):
    """Test of SNF in BZGrid."""
    lat = si_pbesol_111.supercell.cell
    mesh = 20
    with pytest.warns(
        RuntimeWarning, match="Non primitive cell input. Unable to use GR-grid."
    ):
        bzgrid = BZGrid(
            mesh,
            lattice=lat,
            symmetry_dataset=si_pbesol_111.symmetry.dataset,
            use_grg=True,
        )
    np.testing.assert_equal(bzgrid.D_diag, [4, 4, 4])
    identity = np.eye(3, dtype=int)
    np.testing.assert_equal(bzgrid.P, identity)
    np.testing.assert_equal(bzgrid.Q, identity)


def test_SNF_tetrahedra_relative_grid(aln_lda):
    """Test relative grid addresses under GR-grid.

    Under GR-grid, grid point addressing becomes different from ordinal uniform
    grid. But P and Q matrices can be used to map betweewn these grid systems.
    In this test, the agreement is checked by representing them in Cartesian
    coordinates.

    """
    lat = aln_lda.primitive.cell
    mesh = 25

    for snf_coordinates, d_diag in zip(
        ("direct", "reciprocal"), ([2, 8, 24], [1, 9, 45])
    ):
        bzgrid = BZGrid(
            mesh,
            lattice=lat,
            symmetry_dataset=aln_lda.primitive_symmetry.dataset,
            use_grg=True,
            force_SNF=True,
            SNF_coordinates=snf_coordinates,
        )

        np.testing.assert_equal(bzgrid.D_diag, d_diag)

        plat = np.linalg.inv(aln_lda.primitive.cell)
        mlat = bzgrid.microzone_lattice
        thm = TetrahedronMethod(mlat)
        snf_tetrahedra = np.dot(thm.get_tetrahedra(), bzgrid.P.T)

        for mtet, ptet in zip(thm.get_tetrahedra(), snf_tetrahedra):
            np.testing.assert_allclose(
                np.dot(mtet, mlat.T),
                np.dot(np.dot(ptet, bzgrid.QDinv.T), plat.T),
                atol=1e-8,
            )


def test_get_grid_points_by_bz_rotations(si_pbesol_111):
    """Rotate grid point by rotations with and without considering BZ surface.

    The following three methods are tested between type-1 and type-2.

        _get_grid_points_by_rotations
        _get_grid_points_by_bz_rotations_c
        _get_grid_points_by_bz_rotations_py

    """
    ref10_type1 = [
        10,
        26,
        10,
        26,
        26,
        10,
        26,
        10,
        88,
        80,
        200,
        208,
        200,
        208,
        88,
        80,
        208,
        88,
        80,
        200,
        208,
        88,
        80,
        200,
        26,
        10,
        26,
        10,
        10,
        26,
        10,
        26,
        200,
        208,
        88,
        80,
        88,
        80,
        200,
        208,
        80,
        200,
        208,
        88,
        80,
        200,
        208,
        88,
    ]
    ref12_type2 = [
        12,
        39,
        12,
        39,
        39,
        12,
        39,
        12,
        122,
        109,
        265,
        278,
        265,
        278,
        122,
        109,
        278,
        122,
        109,
        265,
        278,
        122,
        109,
        265,
        39,
        12,
        39,
        12,
        12,
        39,
        12,
        39,
        265,
        278,
        122,
        109,
        122,
        109,
        265,
        278,
        109,
        265,
        278,
        122,
        109,
        265,
        278,
        122,
    ]

    ref10_bz_type1 = [
        10,
        26,
        260,
        270,
        269,
        258,
        271,
        259,
        88,
        285,
        200,
        328,
        322,
        208,
        291,
        286,
        327,
        292,
        287,
        321,
        326,
        290,
        80,
        323,
        269,
        258,
        271,
        259,
        10,
        26,
        260,
        270,
        200,
        328,
        88,
        285,
        291,
        286,
        322,
        208,
        80,
        323,
        326,
        290,
        287,
        321,
        327,
        292,
    ]
    ref12_bz_type2 = [
        12,
        39,
        15,
        41,
        40,
        13,
        42,
        14,
        122,
        110,
        265,
        281,
        267,
        278,
        124,
        111,
        280,
        125,
        112,
        266,
        279,
        123,
        109,
        268,
        40,
        13,
        42,
        14,
        12,
        39,
        15,
        41,
        265,
        281,
        122,
        110,
        124,
        111,
        267,
        278,
        109,
        268,
        279,
        123,
        112,
        266,
        280,
        125,
    ]

    lat = si_pbesol_111.primitive.cell
    mesh = 20

    bz_grid_type1 = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=si_pbesol_111.primitive_symmetry.dataset,
        use_grg=True,
        store_dense_gp_map=False,
    )
    bz_grid_type2 = BZGrid(
        mesh,
        lattice=lat,
        symmetry_dataset=si_pbesol_111.primitive_symmetry.dataset,
        use_grg=True,
        store_dense_gp_map=True,
    )

    # Check data consistency by reducing to GR-grid.
    # Grid point 10 in type-1 and 12 in type-2 are the same points in GR-grid.
    assert bz_grid_type1.bzg2grg[10] == bz_grid_type2.bzg2grg[12]
    np.testing.assert_equal(
        bz_grid_type1.bzg2grg[ref10_type1], bz_grid_type2.bzg2grg[ref12_type2]
    )
    np.testing.assert_equal(
        bz_grid_type1.bzg2grg[ref10_type1], bz_grid_type1.bzg2grg[ref10_bz_type1]
    )
    np.testing.assert_equal(
        bz_grid_type1.bzg2grg[ref10_type1], bz_grid_type2.bzg2grg[ref12_bz_type2]
    )

    bzgps = _get_grid_points_by_rotations(10, bz_grid_type1, bz_grid_type1.rotations)
    np.testing.assert_equal(bzgps, ref10_type1)

    bzgps = _get_grid_points_by_rotations(12, bz_grid_type2, bz_grid_type2.rotations)
    np.testing.assert_equal(bzgps, ref12_type2)

    bzgps = _get_grid_points_by_bz_rotations_c(
        10, bz_grid_type1, bz_grid_type1.rotations
    )
    np.testing.assert_equal(bzgps, ref10_bz_type1)

    bzgps = _get_grid_points_by_bz_rotations_c(
        12, bz_grid_type2, bz_grid_type2.rotations
    )
    np.testing.assert_equal(bzgps, ref12_bz_type2)

    bzgps = _get_grid_points_by_bz_rotations_py(
        10, bz_grid_type1, bz_grid_type1.rotations
    )
    np.testing.assert_equal(bzgps, ref10_bz_type1)

    bzgps = _get_grid_points_by_bz_rotations_py(
        12, bz_grid_type2, bz_grid_type2.rotations
    )
    np.testing.assert_equal(bzgps, ref12_bz_type2)

    # Exhaustive consistency check among methods
    for grgp in range(np.prod(bz_grid_type1.D_diag)):
        bzgp_type1 = bz_grid_type1.grg2bzg[grgp]
        bzgp_type2 = bz_grid_type2.grg2bzg[grgp]

        rot_grgps = bz_grid_type1.bzg2grg[
            _get_grid_points_by_rotations(
                bzgp_type1, bz_grid_type1, bz_grid_type1.rotations
            )
        ]

        np.testing.assert_equal(
            rot_grgps,
            bz_grid_type2.bzg2grg[
                _get_grid_points_by_rotations(
                    bzgp_type2, bz_grid_type2, bz_grid_type2.rotations
                )
            ],
        )

        np.testing.assert_equal(
            _get_grid_points_by_bz_rotations_c(
                bzgp_type1, bz_grid_type1, bz_grid_type1.rotations
            ),
            _get_grid_points_by_bz_rotations_py(
                bzgp_type1, bz_grid_type1, bz_grid_type1.rotations
            ),
        )

        np.testing.assert_equal(
            _get_grid_points_by_bz_rotations_c(
                bzgp_type2, bz_grid_type2, bz_grid_type2.rotations
            ),
            _get_grid_points_by_bz_rotations_py(
                bzgp_type2, bz_grid_type2, bz_grid_type2.rotations
            ),
        )

        np.testing.assert_equal(
            rot_grgps,
            bz_grid_type1.bzg2grg[
                _get_grid_points_by_bz_rotations_c(
                    bzgp_type1, bz_grid_type1, bz_grid_type1.rotations
                )
            ],
        )

        np.testing.assert_equal(
            rot_grgps,
            bz_grid_type2.bzg2grg[
                _get_grid_points_by_bz_rotations_c(
                    bzgp_type2, bz_grid_type2, bz_grid_type2.rotations
                )
            ],
        )

    # for gps in bzgps.reshape(-1, 12):
    #     print("".join(["%d, " % gp for gp in gps]))


def test_can_use_std_lattice():
    """Test of can_use_std_lattice."""
    conv_lat = [[6.06531185, 0.0, 0.0], [0.0, 0.0, 6.06531185], [0.0, -6.06531185, 0.0]]
    std_lattice = [
        [6.06531185, 0.0, 0.0],
        [0.0, 6.06531185, 0.0],
        [0.0, 0.0, 6.06531185],
    ]
    tmat = [[0.0, 0.5, 0.5], [-0.5, -0.5, 0.0], [0.5, 0.0, 0.5]]
    rotations = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1], [-1, -1, -1]],
        [[0, 0, 1], [-1, -1, -1], [1, 0, 0]],
        [[-1, -1, -1], [1, 0, 0], [0, 1, 0]],
        [[-1, -1, -1], [0, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [-1, -1, -1], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [-1, -1, -1]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[-1, -1, -1], [1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [-1, -1, -1]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, 1], [-1, -1, -1], [0, 1, 0]],
        [[1, 0, 0], [-1, -1, -1], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 1, 0], [-1, -1, -1]],
        [[-1, -1, -1], [0, 0, 1], [1, 0, 0]],
        [[0, 1, 0], [-1, -1, -1], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [-1, -1, -1]],
        [[-1, -1, -1], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[-1, -1, -1], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [-1, -1, -1]],
        [[0, 1, 0], [-1, -1, -1], [1, 0, 0]],
    ]

    assert can_use_std_lattice(conv_lat, tmat, std_lattice, rotations)
