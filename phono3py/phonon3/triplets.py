import numpy as np
from phonopy.units import THzToEv, Kb
import phonopy.structure.spglib as spg
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phonopy.structure.grid_points import extract_ir_grid_points


def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)


def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1)


def get_triplets_at_q(grid_point,
                      mesh,
                      point_group,  # real space point group of space group
                      reciprocal_lattice,  # column vectors
                      is_time_reversal=True,
                      swappable=True,
                      stores_triplets_map=False):
    """Parameters
    ----------
    grid_point : int
        A grid point
    mesh : array_like
        Mesh numbers
        dtype='intc'
        shape=(3,)
    point_group : array_like
        Rotation matrices in real space. Note that those in reciprocal space
        mean these matrices transposed (local terminology).
        dtype='intc'
        shape=(n_rot, 3, 3)
    reciprocal_lattice : array_like
        Reciprocal primitive basis vectors given as column vectors
        dtype='double'
        shape=(3, 3)
    is_time_reversal : bool
        Inversion symemtry is added if it doesn't exist.
    swappable : bool
        q1 and q2 can be swapped. By this number of triplets decreases.

    Returns
    -------
    triplets_at_q : ndarray
        Symmetry reduced number of triplets are stored as grid point
        integer numbers.
        dtype='uintp'
        shape=(n_triplets, 3)
    weights : ndarray
        Weights of triplets in Brillouin zone
        dtype='intc'
        shape=(n_triplets,)
    bz_grid_address : ndarray
        Integer grid address of the points in Brillouin zone including
        surface.  The first prod(mesh) numbers of points are
        independent. But the rest of points are
        translational-symmetrically equivalent to some other points.
        dtype='intc'
        shape=(n_grid_points, 3)
    bz_map : ndarray
        Grid point mapping table containing BZ surface. See more
        detail in spglib docstring.
        dtype='uintp'
        shape=(prod(mesh*2),)
    map_tripelts : ndarray or None
        Returns when stores_triplets_map=True, otherwise None is
        returned.  Mapping table of all triplets to symmetrically
        independent tripelts. More precisely, this gives a list of
        index mapping from all q-points to independent q' of
        q+q'+q''=G. Considering q' is enough because q is fixed and
        q''=G-q-q' where G is automatically determined to choose
        smallest |G|.
        dtype='uintp'
        shape=(prod(mesh),)
    map_q : ndarray or None
        Returns when stores_triplets_map=True, otherwise None is
        returned.  Irreducible q-points stabilized by q-point of
        specified grid_point.
        dtype='uintp'
        shape=(prod(mesh),)

    """

    map_triplets, map_q, grid_address = _get_triplets_reciprocal_mesh_at_q(
        grid_point,
        mesh,
        point_group,
        is_time_reversal=is_time_reversal,
        swappable=swappable)
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           reciprocal_lattice,
                                                           is_dense=True)
    triplets_at_q, weights = _get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        map_triplets,
        mesh)

    assert np.prod(mesh) == weights.sum(), \
        "Num grid points %d, sum of weight %d" % (
                    np.prod(mesh), weights.sum())

    # These maps are required for collision matrix calculation.
    if not stores_triplets_map:
        map_triplets = None
        map_q = None

    return triplets_at_q, weights, bz_grid_address, bz_map, map_triplets, map_q


def get_all_triplets(grid_point,
                     bz_grid_address,
                     bz_map,
                     mesh):
    triplets_at_q, _ = _get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        np.arange(np.prod(mesh), dtype=bz_map.dtype),
        mesh)

    return triplets_at_q


def get_nosym_triplets_at_q(grid_point,
                            mesh,
                            reciprocal_lattice,
                            stores_triplets_map=False):
    grid_address = get_grid_address(mesh)
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           reciprocal_lattice,
                                                           is_dense=True)
    map_triplets = np.arange(len(grid_address), dtype=bz_map.dtype)
    triplets_at_q, weights = _get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        map_triplets,
        mesh)

    if not stores_triplets_map:
        map_triplets = None
        map_q = None
    else:
        map_q = map_triplets.copy()

    return triplets_at_q, weights, bz_grid_address, bz_map, map_triplets, map_q


def get_grid_address(mesh):
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        is_time_reversal=False,
        is_dense=True)

    return grid_address


def get_bz_grid_address(mesh, reciprocal_lattice, with_boundary=False):
    grid_address = get_grid_address(mesh)
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           reciprocal_lattice,
                                                           is_dense=True)
    if with_boundary:
        return bz_grid_address, bz_map
    else:
        return bz_grid_address[:np.prod(mesh)]


def get_grid_point_from_address_py(address, mesh):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    m = mesh
    return (address[0] % m[0] +
            (address[1] % m[1]) * m[0] +
            (address[2] % m[2]) * m[0] * m[1])


def get_grid_point_from_address(address, mesh):
    """Grid point number is given by grid address.

    Parameters
    ----------
    address : array_like
        Grid address.
        dtype='intc'
        shape=(3,)
    mesh : array_like
        Mesh numbers.
        dtype='intc'
        shape=(3,)

    Returns
    -------
    int
        Grid point number.

    """

    return spg.get_grid_point_from_address(address, mesh)


def get_bz_grid_point_from_address(address, mesh, bz_map):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    # 2m is defined in kpoint.c of spglib.
    m = 2 * np.array(mesh, dtype='intc')
    return bz_map[get_grid_point_from_address(address, m)]


def invert_grid_point(grid_point, mesh, grid_address, bz_map):
    # gp --> [address] --> [-address] --> inv_gp
    address = grid_address[grid_point]
    return get_bz_grid_point_from_address(-address, mesh, bz_map)


def get_ir_grid_points(mesh, rotations, mesh_shifts=None):
    if mesh_shifts is None:
        mesh_shifts = [False, False, False]
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        rotations,
        is_shift=np.where(mesh_shifts, 1, 0),
        is_dense=True)
    (ir_grid_points,
     ir_grid_weights) = extract_ir_grid_points(grid_mapping_table)

    return ir_grid_points, ir_grid_weights, grid_address, grid_mapping_table


def get_grid_points_by_rotations(grid_point,
                                 reciprocal_rotations,
                                 mesh,
                                 mesh_shifts=None):
    if mesh_shifts is None:
        mesh_shifts = [False, False, False]
    return spg.get_grid_points_by_rotations(
        grid_point,
        reciprocal_rotations,
        mesh,
        is_shift=np.where(mesh_shifts, 1, 0),
        is_dense=True)


def get_BZ_grid_points_by_rotations(grid_point,
                                    reciprocal_rotations,
                                    mesh,
                                    bz_map,
                                    mesh_shifts=None):
    if mesh_shifts is None:
        mesh_shifts = [False, False, False]
    return spg.get_BZ_grid_points_by_rotations(
        grid_point,
        reciprocal_rotations,
        mesh,
        bz_map,
        is_shift=np.where(mesh_shifts, 1, 0),
        is_dense=True)


def reduce_grid_points(mesh_divisors,
                       grid_address,
                       dense_grid_points,
                       dense_grid_weights=None,
                       coarse_mesh_shifts=None):
    divisors = np.array(mesh_divisors, dtype='intc')
    if (divisors == 1).all():
        coarse_grid_points = np.array(dense_grid_points, dtype='uintp')
        if dense_grid_weights is not None:
            coarse_grid_weights = np.array(dense_grid_weights, dtype='intc')
    else:
        if coarse_mesh_shifts is None:
            shift = [0, 0, 0]
        else:
            shift = np.where(coarse_mesh_shifts, divisors // 2, [0, 0, 0])
        modulo = grid_address[dense_grid_points] % divisors
        condition = (modulo == shift).all(axis=1)
        coarse_grid_points = np.extract(condition, dense_grid_points)
        if dense_grid_weights is not None:
            coarse_grid_weights = np.extract(condition, dense_grid_weights)

    if dense_grid_weights is None:
        return coarse_grid_points
    else:
        return coarse_grid_points, coarse_grid_weights


def from_coarse_to_dense_grid_points(dense_mesh,
                                     mesh_divisors,
                                     coarse_grid_points,
                                     coarse_grid_address,
                                     coarse_mesh_shifts=None):
    if coarse_mesh_shifts is None:
        coarse_mesh_shifts = [False, False, False]
    shifts = np.where(coarse_mesh_shifts, 1, 0)
    dense_grid_points = []
    for cga in coarse_grid_address[coarse_grid_points]:
        dense_address = cga * mesh_divisors + shifts * (mesh_divisors // 2)
        dense_grid_points.append(get_grid_point_from_address(dense_address,
                                                             dense_mesh))
    return np.array(dense_grid_points, dtype='uintp')


def get_coarse_ir_grid_points(primitive,
                              mesh,
                              mesh_divisors,
                              coarse_mesh_shifts,
                              is_kappa_star=True,
                              symprec=1e-5):
    mesh = np.array(mesh, dtype='intc')

    symmetry = Symmetry(primitive, symprec)
    point_group = symmetry.get_pointgroup_operations()

    if mesh_divisors is None:
        (ir_grid_points,
         ir_grid_weights,
         grid_address,
         grid_mapping_table) = get_ir_grid_points(mesh, point_group)
    else:
        mesh_divs = np.array(mesh_divisors, dtype='intc')
        coarse_mesh = mesh // mesh_divs
        if coarse_mesh_shifts is None:
            coarse_mesh_shifts = [False, False, False]

        if not is_kappa_star:
            coarse_grid_address = get_grid_address(coarse_mesh)
            coarse_grid_points = np.arange(np.prod(coarse_mesh), dtype='uintp')
        else:
            (coarse_ir_grid_points,
             coarse_ir_grid_weights,
             coarse_grid_address,
             coarse_grid_mapping_table) = get_ir_grid_points(
                 coarse_mesh,
                 point_group,
                 mesh_shifts=coarse_mesh_shifts)
        ir_grid_points = from_coarse_to_dense_grid_points(
            mesh,
            mesh_divs,
            coarse_grid_points,
            coarse_grid_address,
            coarse_mesh_shifts=coarse_mesh_shifts)
        grid_address = get_grid_address(mesh)
        ir_grid_weights = ir_grid_weights

    reciprocal_lattice = np.linalg.inv(primitive.get_cell())
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           reciprocal_lattice,
                                                           is_dense=True)

    return (ir_grid_points,
            ir_grid_weights,
            bz_grid_address,
            grid_mapping_table)


def get_number_of_triplets(primitive,
                           mesh,
                           grid_point,
                           swappable=True,
                           symprec=1e-5):
    mesh = np.array(mesh, dtype='intc')
    symmetry = Symmetry(primitive, symprec)
    point_group = symmetry.get_pointgroup_operations()
    reciprocal_lattice = np.linalg.inv(primitive.get_cell())
    triplets_at_q, _, _, _, _, _ = get_triplets_at_q(
        grid_point,
        mesh,
        point_group,
        reciprocal_lattice,
        swappable=swappable)

    return len(triplets_at_q)


def get_triplets_integration_weights(interaction,
                                     frequency_points,
                                     sigma,
                                     sigma_cutoff=None,
                                     is_collision_matrix=False,
                                     neighboring_phonons=False,
                                     lang='C'):
    triplets = interaction.get_triplets_at_q()[0]
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    g_zero = None

    if is_collision_matrix:
        g = np.empty(
            (3, len(triplets), len(frequency_points), num_band, num_band),
            dtype='double', order='C')
    else:
        g = np.empty(
            (2, len(triplets), len(frequency_points), num_band, num_band),
            dtype='double', order='C')
    g[:] = 0

    if sigma:
        if lang == 'C':
            import phono3py._phono3py as phono3c
            g_zero = np.zeros(g.shape[1:], dtype='byte', order='C')
            if sigma_cutoff is None:
                cutoff = -1
            else:
                cutoff = float(sigma_cutoff)
            # cutoff < 0 disables g_zero feature.
            phono3c.triplets_integration_weights_with_sigma(
                g,
                g_zero,
                frequency_points,
                triplets,
                frequencies,
                sigma,
                cutoff)
        else:
            for i, tp in enumerate(triplets):
                f1s = frequencies[tp[1]]
                f2s = frequencies[tp[2]]
                for j, k in list(np.ndindex((num_band, num_band))):
                    f1 = f1s[j]
                    f2 = f2s[k]
                    g0 = gaussian(frequency_points - f1 - f2, sigma)
                    g[0, i, :, j, k] = g0
                    g1 = gaussian(frequency_points + f1 - f2, sigma)
                    g2 = gaussian(frequency_points - f1 + f2, sigma)
                    g[1, i, :, j, k] = g1 - g2
                    if len(g) == 3:
                        g[2, i, :, j, k] = g0 + g1 + g2
    else:
        if lang == 'C':
            g_zero = np.zeros(g.shape[1:], dtype='byte', order='C')
            _set_triplets_integration_weights_c(
                g,
                g_zero,
                interaction,
                frequency_points,
                neighboring_phonons=neighboring_phonons)
        else:
            _set_triplets_integration_weights_py(
                g, interaction, frequency_points)

    return g, g_zero


def get_tetrahedra_vertices(relative_address,
                            mesh,
                            triplets_at_q,
                            bz_grid_address,
                            bz_map):
    bzmesh = mesh * 2
    grid_order = [1, mesh[0], mesh[0] * mesh[1]]
    bz_grid_order = [1, bzmesh[0], bzmesh[0] * bzmesh[1]]
    num_triplets = len(triplets_at_q)
    vertices = np.zeros((num_triplets, 2, 24, 4), dtype='uintp')
    for i, tp in enumerate(triplets_at_q):
        for j, adrs_shift in enumerate(
                (relative_address, -relative_address)):
            adrs = bz_grid_address[tp[j + 1]] + adrs_shift
            bz_gp = np.dot(adrs % bzmesh, bz_grid_order)
            gp = np.dot(adrs % mesh, grid_order)
            vgp = bz_map[bz_gp]
            vertices[i, j] = vgp + (vgp == -1) * (gp + 1)
    return vertices


def _get_triplets_reciprocal_mesh_at_q(fixed_grid_number,
                                       mesh,
                                       rotations,
                                       is_time_reversal=True,
                                       swappable=True):
    """Search symmetry reduced triplets fixing one q-point

    Triplets of (q0, q1, q2) are searched.

    Parameters
    ----------
    fixed_grid_number : int
        Grid point of q0
    mesh : array_like
        Mesh numbers
        dtype='intc'
        shape=(3,)
    rotations : array_like
        Rotation matrices in real space. Note that those in reciprocal space
        mean these matrices transposed (local terminology).
        dtype='intc'
        shape=(n_rot, 3, 3)
    is_time_reversal : bool
        Inversion symemtry is added if it doesn't exist.
    swappable : bool
        q1 and q2 can be swapped. By this number of triplets decreases.

    """

    import phono3py._phono3py as phono3c

    map_triplets = np.zeros(np.prod(mesh), dtype='uintp')
    map_q = np.zeros(np.prod(mesh), dtype='uintp')
    grid_address = np.zeros((np.prod(mesh), 3), dtype='intc')

    phono3c.triplets_reciprocal_mesh_at_q(
        map_triplets,
        map_q,
        grid_address,
        fixed_grid_number,
        np.array(mesh, dtype='intc'),
        is_time_reversal * 1,
        np.array(rotations, dtype='intc', order='C'),
        swappable * 1)

    return map_triplets, map_q, grid_address


def _get_BZ_triplets_at_q(grid_point,
                          bz_grid_address,
                          bz_map,
                          map_triplets,
                          mesh):
    import phono3py._phono3py as phono3c

    weights = np.zeros(len(map_triplets), dtype='intc')
    for g in map_triplets:
        weights[g] += 1
    ir_weights = np.extract(weights > 0, weights)
    triplets = np.zeros((len(ir_weights), 3), dtype=bz_map.dtype)
    # triplets are overwritten.
    num_ir_ret = phono3c.BZ_triplets_at_q(triplets,
                                          grid_point,
                                          bz_grid_address,
                                          bz_map,
                                          map_triplets,
                                          np.array(mesh, dtype='intc'))
    assert num_ir_ret == len(ir_weights)

    return triplets, np.array(ir_weights, dtype='intc')


def _set_triplets_integration_weights_c(g,
                                        g_zero,
                                        interaction,
                                        frequency_points,
                                        neighboring_phonons=False):
    import phono3py._phono3py as phono3c

    reciprocal_lattice = np.linalg.inv(interaction.get_primitive().get_cell())
    mesh = interaction.get_mesh_numbers()
    thm = TetrahedronMethod(reciprocal_lattice, mesh=mesh)
    grid_address = interaction.get_grid_address()
    bz_map = interaction.get_bz_map()
    triplets_at_q = interaction.get_triplets_at_q()[0]

    if neighboring_phonons:
        unique_vertices = thm.get_unique_tetrahedra_vertices()
        for i, j in zip((1, 2), (1, -1)):
            neighboring_grid_points = np.zeros(
                len(unique_vertices) * len(triplets_at_q), dtype=bz_map.dtype)
            phono3c.neighboring_grid_points(
                neighboring_grid_points,
                np.array(triplets_at_q[:, i], dtype='uintp').ravel(),
                j * unique_vertices,
                mesh,
                grid_address,
                bz_map)
            interaction.set_phonons(np.unique(neighboring_grid_points))

    phono3c.triplets_integration_weights(
        g,
        g_zero,
        frequency_points,
        thm.get_tetrahedra(),
        mesh,
        triplets_at_q,
        interaction.get_phonons()[0],
        grid_address,
        bz_map)


def _set_triplets_integration_weights_py(g, interaction, frequency_points):
    reciprocal_lattice = np.linalg.inv(interaction.get_primitive().get_cell())
    mesh = interaction.get_mesh_numbers()
    thm = TetrahedronMethod(reciprocal_lattice, mesh=mesh)
    grid_address = interaction.get_grid_address()
    bz_map = interaction.get_bz_map()
    triplets_at_q = interaction.get_triplets_at_q()[0]
    tetrahedra_vertices = get_tetrahedra_vertices(
        thm.get_tetrahedra(),
        mesh,
        triplets_at_q,
        grid_address,
        bz_map)
    interaction.set_phonons(np.unique(tetrahedra_vertices))
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    for i, vertices in enumerate(tetrahedra_vertices):
        for j, k in list(np.ndindex((num_band, num_band))):
            f1_v = frequencies[vertices[0], j]
            f2_v = frequencies[vertices[1], k]
            thm.set_tetrahedra_omegas(f1_v + f2_v)
            thm.run(frequency_points)
            g0 = thm.get_integration_weight()
            g[0, i, :, j, k] = g0
            thm.set_tetrahedra_omegas(-f1_v + f2_v)
            thm.run(frequency_points)
            g1 = thm.get_integration_weight()
            thm.set_tetrahedra_omegas(f1_v - f2_v)
            thm.run(frequency_points)
            g2 = thm.get_integration_weight()
            g[1, i, :, j, k] = g1 - g2
            if len(g) == 3:
                g[2, i, :, j, k] = g0 + g1 + g2
