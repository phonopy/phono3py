import sys
import logging
import numpy as np
from phonopy.harmonic.force_constants import (get_fc2,
                                              similarity_transformation,
                                              distribute_force_constants,
                                              solve_force_constants,
                                              get_rotated_displacement,
                                              get_positions_sent_by_rot_inv,
                                              get_nsym_list_and_s2pp)
from phono3py.phonon3.displacement_fc3 import (get_reduced_site_symmetry,
                                               get_bond_symmetry,
                                               get_equivalent_smallest_vectors)
from phonopy.structure.cells import compute_all_sg_permutations

logger = logging.getLogger(__name__)


def get_fc3(supercell,
            primitive,
            disp_dataset,
            symmetry,
            is_compact_fc=False,
            verbose=False):
    # fc2 has to be full matrix to compute delta-fc2
    # p2s_map elements are extracted if is_compact_fc=True at the last part.
    fc2 = get_fc2(supercell, symmetry, disp_dataset)
    fc3 = _get_fc3_least_atoms(supercell,
                               primitive,
                               disp_dataset,
                               fc2,
                               symmetry,
                               is_compact_fc=is_compact_fc,
                               verbose=verbose)
    if verbose:
        print("Expanding fc3")

    first_disp_atoms = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    rotations = symmetry.get_symmetry_operations()['rotations']
    lattice = supercell.get_cell().T
    permutations = symmetry.get_atomic_permutations()

    if is_compact_fc:
        s2p_map = primitive.get_supercell_to_primitive_map()
        p2s_map = primitive.get_primitive_to_supercell_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        s2compact = np.array([p2p_map[i] for i in s2p_map], dtype='intc')
        for i in first_disp_atoms:
            assert i in p2s_map
        target_atoms = [i for i in p2s_map if i not in first_disp_atoms]
    else:
        s2compact = np.arange(supercell.get_number_of_atoms(), dtype='intc')
        target_atoms = [i for i in s2compact if i not in first_disp_atoms]

    distribute_fc3(fc3,
                   first_disp_atoms,
                   target_atoms,
                   lattice,
                   rotations,
                   permutations,
                   s2compact,
                   verbose=verbose)

    if 'cutoff_distance' in disp_dataset:
        if verbose:
            print("Cutting-off fc3 (cut-off distance: %f)" %
                  disp_dataset['cutoff_distance'])
        if is_compact_fc:
            print("cutoff_fc3 doesn't support compact-fc3 yet.")
            raise ValueError
        cutoff_fc3(fc3,
                   supercell,
                   disp_dataset,
                   symmetry,
                   verbose=verbose)

    if is_compact_fc:
        p2s_map = primitive.get_primitive_to_supercell_map()
        fc2 = np.array(fc2[p2s_map], dtype='double', order='C')

    return fc2, fc3


def distribute_fc3(fc3,
                   first_disp_atoms,
                   target_atoms,
                   lattice,
                   rotations,
                   permutations,
                   s2compact,
                   verbose=False):
    """Distribute fc3

    fc3[i, :, :, 0:3, 0:3, 0:3] where i=indices done are distributed to
    symmetrically equivalent fc3 elements by tensor rotations.

    Search symmetry operation (R, t) that performs
        i_target -> i_done
    and
        atom_mapping[i_target] = i_done
        fc3[i_target, j_target, k_target] = R_inv[i_done, j, k]

    Parameters
    ----------

    target_atoms: list or ndarray
        Supercell atom indices to which fc3 are distributed.
    s2compact: ndarray
        Maps supercell index to compact index. For full-fc3,
        s2compact=np.arange(n_satom).
        shape=(n_satom,)
        dtype=intc

    """

    n_satom = fc3.shape[1]
    for i_target in target_atoms:
        for i_done in first_disp_atoms:
            rot_indices = np.where(permutations[:, i_target] == i_done)[0]
            if len(rot_indices) > 0:
                atom_mapping = np.array(permutations[rot_indices[0]],
                                        dtype='intc')
                rot = rotations[rot_indices[0]]
                rot_cart_inv = np.array(
                    similarity_transformation(lattice, rot).T,
                    dtype='double', order='C')
                break

        if len(rot_indices) == 0:
            print("Position or symmetry may be wrong.")
            raise RuntimeError

        if verbose > 2:
            print("    [ %d, x, x ] to [ %d, x, x ]" %
                  (i_done + 1, i_target + 1))
            sys.stdout.flush()

        try:
            import phono3py._phono3py as phono3c
            phono3c.distribute_fc3(fc3,
                                   int(s2compact[i_target]),
                                   int(s2compact[i_done]),
                                   atom_mapping,
                                   rot_cart_inv)
        except ImportError:
            print("Phono3py C-routine is not compiled correctly.")
            for j in range(n_satom):
                j_rot = atom_mapping[j]
                for k in range(n_satom):
                    k_rot = atom_mapping[k]
                    fc3[i_target, j, k] = third_rank_tensor_rotation(
                        rot_cart_inv, fc3[i_done, j_rot, k_rot])


def set_permutation_symmetry_fc3(fc3):
    try:
        import phono3py._phono3py as phono3c
        phono3c.permutation_symmetry_fc3(fc3)
    except ImportError:
        print("Phono3py C-routine is not compiled correctly.")
        num_atom = fc3.shape[0]
        for i in range(num_atom):
            for j in range(i, num_atom):
                for k in range(j, num_atom):
                    fc3_elem = set_permutation_symmetry_fc3_elem(fc3, i, j, k)
                    copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, i, j, k)


def set_permutation_symmetry_compact_fc3(fc3, primitive):
    try:
        import phono3py._phono3py as phono3c
        s2p_map = primitive.get_supercell_to_primitive_map()
        p2s_map = primitive.get_primitive_to_supercell_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        permutations = primitive.get_atomic_permutations()
        s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map,
                                                     p2p_map,
                                                     permutations)
        phono3c.permutation_symmetry_compact_fc3(fc3,
                                                 permutations,
                                                 s2pp_map,
                                                 p2s_map,
                                                 nsym_list)
    except ImportError:
        text = ("Import error at phono3c.permutation_symmetry_compact_fc3. "
                "Corresponding python code is not implemented.")
        raise RuntimeError(text)


def copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, a, b, c):
    for (i, j, k) in list(np.ndindex(3, 3, 3)):
        fc3[a, b, c, i, j, k] = fc3_elem[i, j, k]
        fc3[c, a, b, k, i, j] = fc3_elem[i, j, k]
        fc3[b, c, a, j, k, i] = fc3_elem[i, j, k]
        fc3[a, c, b, i, k, j] = fc3_elem[i, j, k]
        fc3[b, a, c, j, i, k] = fc3_elem[i, j, k]
        fc3[c, b, a, k, j, i] = fc3_elem[i, j, k]


def set_permutation_symmetry_fc3_elem(fc3, a, b, c, divisor=6):
    tensor3 = np.zeros((3, 3, 3), dtype='double')
    for (i, j, k) in list(np.ndindex(3, 3, 3)):
        tensor3[i, j, k] = (fc3[a, b, c, i, j, k] +
                            fc3[c, a, b, k, i, j] +
                            fc3[b, c, a, j, k, i] +
                            fc3[a, c, b, i, k, j] +
                            fc3[b, a, c, j, i, k] +
                            fc3[c, b, a, k, j, i]) / divisor
    return tensor3


def set_translational_invariance_fc3(fc3):
    for i in range(3):
        set_translational_invariance_fc3_per_index(fc3, index=i)


def set_translational_invariance_compact_fc3(fc3, primitive):
    try:
        import phono3py._phono3py as phono3c
        s2p_map = primitive.get_supercell_to_primitive_map()
        p2s_map = primitive.get_primitive_to_supercell_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        permutations = primitive.get_atomic_permutations()
        s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map,
                                                     p2p_map,
                                                     permutations)
        phono3c.transpose_compact_fc3(fc3,
                                      permutations,
                                      s2pp_map,
                                      p2s_map,
                                      nsym_list,
                                      0)  # dim[0] <--> dim[1]
        set_translational_invariance_fc3_per_index(fc3, index=1)
        phono3c.transpose_compact_fc3(fc3,
                                      permutations,
                                      s2pp_map,
                                      p2s_map,
                                      nsym_list,
                                      0)  # dim[0] <--> dim[1]
        set_translational_invariance_fc3_per_index(fc3, index=1)
        set_translational_invariance_fc3_per_index(fc3, index=2)

    except ImportError:
        text = ("Import error at phono3c.tranpose_compact_fc3. "
                "Corresponding python code is not implemented.")
        raise RuntimeError(text)


def set_translational_invariance_fc3_per_index(fc3, index=0):
    for i in range(fc3.shape[(1 + index) % 3]):
        for j in range(fc3.shape[(2 + index) % 3]):
            for k, l, m in list(np.ndindex(3, 3, 3)):
                if index == 0:
                    fc3[:, i, j, k, l, m] -= np.sum(
                        fc3[:, i, j, k, l, m]) / fc3.shape[0]
                elif index == 1:
                    fc3[j, :, i, k, l, m] -= np.sum(
                        fc3[j, :, i, k, l, m]) / fc3.shape[1]
                elif index == 2:
                    fc3[i, j, :, k, l, m] -= np.sum(
                        fc3[i, j, :, k, l, m]) / fc3.shape[2]


def third_rank_tensor_rotation(rot_cart, tensor):
    rot_tensor = np.zeros((3, 3, 3), dtype='double')
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                rot_tensor[i, j, k] = _third_rank_tensor_rotation_elem(
                    rot_cart, tensor, i, j, k)
    return rot_tensor


def get_delta_fc2(dataset_second_atoms,
                  atom1,
                  fc2,
                  supercell,
                  reduced_site_sym,
                  symprec):
    logger.debug("get_delta_fc2")
    disp_fc2 = get_constrained_fc2(supercell,
                                   dataset_second_atoms,
                                   atom1,
                                   reduced_site_sym,
                                   symprec)
    return disp_fc2 - fc2


def get_constrained_fc2(supercell,
                        dataset_second_atoms,
                        atom1,
                        reduced_site_sym,
                        symprec):
    """
    dataset_second_atoms: [{'number': 7,
                            'displacement': [],
                            'delta_forces': []}, ...]
    """
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    num_atom = supercell.get_number_of_atoms()

    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
    atom_list = np.unique([x['number'] for x in dataset_second_atoms])
    for atom2 in atom_list:
        disps2 = []
        sets_of_forces = []
        for disps_second in dataset_second_atoms:
            if atom2 != disps_second['number']:
                continue
            bond_sym = get_bond_symmetry(
                reduced_site_sym,
                lattice,
                positions,
                atom1,
                atom2,
                symprec)

            disps2.append(disps_second['displacement'])
            sets_of_forces.append(disps_second['delta_forces'])

        solve_force_constants(fc2,
                              atom2,
                              disps2,
                              sets_of_forces,
                              supercell,
                              bond_sym,
                              symprec)

    # Shift positions according to set atom1 is at origin
    pos_center = positions[atom1].copy()
    positions -= pos_center
    rotations = np.array(reduced_site_sym, dtype='intc', order='C')
    translations = np.zeros((len(reduced_site_sym), 3),
                            dtype='double', order='C')
    permutations = compute_all_sg_permutations(positions,
                                               rotations,
                                               translations,
                                               lattice,
                                               symprec)
    distribute_force_constants(fc2,
                               atom_list,
                               lattice,
                               rotations,
                               permutations)
    return fc2


def solve_fc3(first_atom_num,
              supercell,
              site_symmetry,
              displacements_first,
              delta_fc2s,
              symprec,
              pinv_solver="numpy",
              verbose=False):

    logger.debug("solve_fc3")

    if pinv_solver == "numpy":
        solver = "numpy.linalg.pinv"
    else:
        try:
            import phono3py._lapackepy as lapackepy
            solver = "lapacke-dgesvd"
        except ImportError:
            print("Phono3py C-routine is not compiled correctly.")
            solver = "numpy.linalg.pinv"

    if verbose:
        text = ("Computing fc3[ %d, x, x ] using %s with " %
                (first_atom_num + 1, solver))
        if len(displacements_first) > 1:
            text += "displacements:"
        else:
            text += "a displacement:"
        print(text)
        for i, v in enumerate(displacements_first):
            print("    [%7.4f %7.4f %7.4f]" % tuple(v))
            sys.stdout.flush()
        if verbose > 2:
            print("  Site symmetry:")
            for i, v in enumerate(site_symmetry):
                print("    [%2d %2d %2d] #%2d" % tuple(list(v[0])+[i + 1]))
                print("    [%2d %2d %2d]" % tuple(v[1]))
                print("    [%2d %2d %2d]\n" % tuple(v[2]))
                sys.stdout.flush()

    lattice = supercell.get_cell().T
    site_sym_cart = np.array([similarity_transformation(lattice, sym)
                              for sym in site_symmetry],
                             dtype='double', order='C')
    num_atom = supercell.get_number_of_atoms()
    positions = supercell.get_scaled_positions()
    pos_center = positions[first_atom_num].copy()
    positions -= pos_center

    logger.debug("get_positions_sent_by_rot_inv")

    rot_map_syms = get_positions_sent_by_rot_inv(lattice,
                                                 positions,
                                                 site_symmetry,
                                                 symprec)
    rot_disps = get_rotated_displacement(displacements_first, site_sym_cart)

    logger.debug("pinv")

    if "numpy" in solver:
        inv_U = np.array(np.linalg.pinv(rot_disps), dtype='double', order='C')
    else:
        inv_U = np.zeros((rot_disps.shape[1], rot_disps.shape[0]),
                         dtype='double', order='C')
        lapackepy.pinv(inv_U, rot_disps, 1e-13)

    fc3 = np.zeros((num_atom, num_atom, 3, 3, 3), dtype='double', order='C')

    logger.debug("rotate_delta_fc2s")

    try:
        import phono3py._phono3py as phono3c
        phono3c.rotate_delta_fc2s(fc3,
                                  delta_fc2s,
                                  inv_U,
                                  site_sym_cart,
                                  rot_map_syms)
    except ImportError:
        for i, j in np.ndindex(num_atom, num_atom):
            fc3[i, j] = np.dot(inv_U, _get_rotated_fc2s(
                    i, j, delta_fc2s, rot_map_syms, site_sym_cart)
            ).reshape(3, 3, 3)

    return fc3


def cutoff_fc3(fc3,
               supercell,
               disp_dataset,
               symmetry,
               verbose=False):
    if verbose:
        print("Building atom mapping table...")
    fc3_done = _get_fc3_done(supercell, disp_dataset, symmetry, fc3.shape[:3])

    if verbose:
        print("Creating contracted fc3...")
    num_atom = supercell.get_number_of_atoms()
    for i in range(num_atom):
        for j in range(i, num_atom):
            for k in range(j, num_atom):
                ave_fc3 = _set_permutation_symmetry_fc3_elem_with_cutoff(
                    fc3, fc3_done, i, j, k)
                copy_permutation_symmetry_fc3_elem(fc3, ave_fc3, i, j, k)


def cutoff_fc3_by_zero(fc3, supercell, cutoff_distance, symprec=1e-5):
    num_atom = supercell.get_number_of_atoms()
    lattice = supercell.get_cell().T
    min_distances = np.zeros((num_atom, num_atom), dtype='double')
    for i in range(num_atom):  # run in supercell
        for j in range(num_atom):  # run in primitive
            min_distances[i, j] = np.linalg.norm(
                np.dot(lattice,
                       get_equivalent_smallest_vectors(
                           i, j, supercell, symprec)[0]))

    for i, j, k in np.ndindex(num_atom, num_atom, num_atom):
        for pair in ((i, j), (j, k), (k, i)):
            if min_distances[pair] > cutoff_distance:
                fc3[i, j, k] = 0
                break


def show_drift_fc3(fc3,
                   primitive=None,
                   name="fc3"):
    if fc3.shape[0] == fc3.shape[1]:
        num_atom = fc3.shape[0]
        maxval1 = 0
        maxval2 = 0
        maxval3 = 0
        klm1 = [0, 0, 0]
        klm2 = [0, 0, 0]
        klm3 = [0, 0, 0]
        for i, j, k, l, m in list(np.ndindex((num_atom, num_atom, 3, 3, 3))):
            val1 = fc3[:, i, j, k, l, m].sum()
            val2 = fc3[i, :, j, k, l, m].sum()
            val3 = fc3[i, j, :, k, l, m].sum()
            if abs(val1) > abs(maxval1):
                maxval1 = val1
                klm1 = [k, l, m]
            if abs(val2) > abs(maxval2):
                maxval2 = val2
                klm2 = [k, l, m]
            if abs(val3) > abs(maxval3):
                maxval3 = val3
                klm3 = [k, l, m]
    else:
        try:
            import phono3py._phono3py as phono3c
            s2p_map = primitive.get_supercell_to_primitive_map()
            p2s_map = primitive.get_primitive_to_supercell_map()
            p2p_map = primitive.get_primitive_to_primitive_map()
            permutations = primitive.get_atomic_permutations()
            s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map,
                                                         p2p_map,
                                                         permutations)
            num_patom = fc3.shape[0]
            num_satom = fc3.shape[1]
            maxval1 = 0
            maxval2 = 0
            maxval3 = 0
            klm1 = [0, 0, 0]
            klm2 = [0, 0, 0]
            klm3 = [0, 0, 0]
            phono3c.transpose_compact_fc3(fc3,
                                          permutations,
                                          s2pp_map,
                                          p2s_map,
                                          nsym_list,
                                          0)  # dim[0] <--> dim[1]
            for i, j, k, l, m in np.ndindex((num_patom, num_satom, 3, 3, 3)):
                val1 = fc3[i, :, j, k, l, m].sum()
                if abs(val1) > abs(maxval1):
                    maxval1 = val1
                    klm1 = [k, l, m]
            phono3c.transpose_compact_fc3(fc3,
                                          permutations,
                                          s2pp_map,
                                          p2s_map,
                                          nsym_list,
                                          0)  # dim[0] <--> dim[1]
            for i, j, k, l, m in np.ndindex((num_patom, num_satom, 3, 3, 3)):
                val2 = fc3[i, :, j, k, l, m].sum()
                val3 = fc3[i, j, :, k, l, m].sum()
                if abs(val2) > abs(maxval2):
                    maxval2 = val2
                    klm2 = [k, l, m]
                if abs(val3) > abs(maxval3):
                    maxval3 = val3
                    klm3 = [k, l, m]
        except ImportError:
            text = ("Import error at phono3c.tranpose_compact_fc3. "
                    "Corresponding python code is not implemented.")
            raise RuntimeError(text)

    text = "Max drift of %s: " % name
    text += "%f (%s%s%s) " % (maxval1,
                              "xyz"[klm1[0]], "xyz"[klm1[1]], "xyz"[klm1[2]])
    text += "%f (%s%s%s) " % (maxval2,
                              "xyz"[klm2[0]], "xyz"[klm2[1]], "xyz"[klm2[2]])
    text += "%f (%s%s%s)" % (maxval3,
                             "xyz"[klm3[0]], "xyz"[klm3[1]], "xyz"[klm3[2]])
    print(text)


def _set_permutation_symmetry_fc3_elem_with_cutoff(fc3, fc3_done, a, b, c):
    sum_done = (fc3_done[a, b, c] +
                fc3_done[c, a, b] +
                fc3_done[b, c, a] +
                fc3_done[b, a, c] +
                fc3_done[c, b, a] +
                fc3_done[a, c, b])
    tensor3 = np.zeros((3, 3, 3), dtype='double')
    if sum_done > 0:
        for (i, j, k) in list(np.ndindex(3, 3, 3)):
            tensor3[i, j, k] = (fc3[a, b, c, i, j, k] * fc3_done[a, b, c] +
                                fc3[c, a, b, k, i, j] * fc3_done[c, a, b] +
                                fc3[b, c, a, j, k, i] * fc3_done[b, c, a] +
                                fc3[a, c, b, i, k, j] * fc3_done[a, c, b] +
                                fc3[b, a, c, j, i, k] * fc3_done[b, a, c] +
                                fc3[c, b, a, k, j, i] * fc3_done[c, b, a])
            tensor3[i, j, k] /= sum_done
    return tensor3


def _get_fc3_least_atoms(supercell,
                         primitive,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_compact_fc=False,
                         verbose=True):
    symprec = symmetry.get_symmetry_tolerance()
    num_satom = supercell.get_number_of_atoms()
    unique_first_atom_nums = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])

    if is_compact_fc:
        num_patom = primitive.get_number_of_atoms()
        s2p_map = primitive.get_supercell_to_primitive_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        first_atom_nums = []
        for i in unique_first_atom_nums:
            if i != s2p_map[i]:
                print("Something wrong in disp_fc3.yaml")
                raise RuntimeError
            else:
                first_atom_nums.append(i)
        fc3 = np.zeros((num_patom, num_satom, num_satom, 3, 3, 3),
                       dtype='double', order='C')
    else:
        first_atom_nums = unique_first_atom_nums
        fc3 = np.zeros((num_satom, num_satom, num_satom, 3, 3, 3),
                       dtype='double', order='C')

    for first_atom_num in first_atom_nums:
        site_symmetry = symmetry.get_site_symmetry(first_atom_num)
        displacements_first = []
        delta_fc2s = []
        for dataset_first_atom in disp_dataset['first_atoms']:
            if first_atom_num != dataset_first_atom['number']:
                continue

            displacements_first.append(dataset_first_atom['displacement'])
            if 'delta_fc2' in dataset_first_atom:
                delta_fc2s.append(dataset_first_atom['delta_fc2'])
            else:
                direction = np.dot(dataset_first_atom['displacement'],
                                   np.linalg.inv(supercell.get_cell()))
                reduced_site_sym = get_reduced_site_symmetry(
                    site_symmetry, direction, symprec)
                delta_fc2s.append(get_delta_fc2(
                    dataset_first_atom['second_atoms'],
                    dataset_first_atom['number'],
                    fc2,
                    supercell,
                    reduced_site_sym,
                    symprec))

        fc3_first = solve_fc3(first_atom_num,
                              supercell,
                              site_symmetry,
                              displacements_first,
                              np.array(delta_fc2s, dtype='double', order='C'),
                              symprec,
                              verbose=verbose)
        if is_compact_fc:
            fc3[p2p_map[s2p_map[first_atom_num]]] = fc3_first
        else:
            fc3[first_atom_num] = fc3_first

    return fc3


def _get_rotated_fc2s(i, j, fc2s, rot_map_syms, site_sym_cart):
    rotated_fc2s = []
    for fc2 in fc2s:
        for sym, map_sym in zip(site_sym_cart, rot_map_syms):
            fc2_rot = fc2[map_sym[i], map_sym[j]]
            rotated_fc2s.append(similarity_transformation(sym, fc2_rot))
    return np.reshape(rotated_fc2s, (-1, 9))


def _third_rank_tensor_rotation_elem(rot, tensor, l, m, n):
    sum_elems = 0.
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                sum_elems += (rot[l, i] * rot[m, j] * rot[n, k]
                              * tensor[i, j, k])
    return sum_elems


def _get_fc3_done(supercell, disp_dataset, symmetry, array_shape):
    num_atom = supercell.get_number_of_atoms()
    fc3_done = np.zeros(array_shape, dtype='byte')
    symprec = symmetry.get_symmetry_tolerance()
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']

    atom_mapping = []
    for rot, trans in zip(rotations, translations):
        atom_indices = [
            _get_atom_by_symmetry(lattice,
                                  positions,
                                  rot,
                                  trans,
                                  i,
                                  symprec) for i in range(num_atom)]
        atom_mapping.append(atom_indices)

    for dataset_first_atom in disp_dataset['first_atoms']:
        first_atom_num = dataset_first_atom['number']
        site_symmetry = symmetry.get_site_symmetry(first_atom_num)
        direction = np.dot(dataset_first_atom['displacement'],
                           np.linalg.inv(supercell.get_cell()))
        reduced_site_sym = get_reduced_site_symmetry(
            site_symmetry, direction, symprec)
        least_second_atom_nums = []
        for second_atoms in dataset_first_atom['second_atoms']:
            if second_atoms['included']:
                least_second_atom_nums.append(second_atoms['number'])
        positions_shifted = positions - positions[first_atom_num]
        least_second_atom_nums = np.unique(least_second_atom_nums)

        for red_rot in reduced_site_sym:
            second_atom_nums = [
                _get_atom_by_symmetry(lattice,
                                      positions_shifted,
                                      red_rot,
                                      np.zeros(3, dtype='double'),
                                      i,
                                      symprec) for i in least_second_atom_nums]
        second_atom_nums = np.unique(second_atom_nums)

        for i in range(len(rotations)):
            rotated_atom1 = atom_mapping[i][first_atom_num]
            for j in second_atom_nums:
                fc3_done[rotated_atom1, atom_mapping[i][j]] = 1

    return fc3_done


def _get_atom_by_symmetry(lattice,
                          positions,
                          rotation,
                          trans,
                          atom_number,
                          symprec):
    rot_pos = np.dot(positions[atom_number], rotation.T) + trans
    diffs = positions - rot_pos
    diffs -= np.rint(diffs)
    dists = np.sqrt((np.dot(diffs, lattice.T) ** 2).sum(axis=1))
    rot_atoms = np.where(dists < symprec)[0]  # only one should be found
    if len(rot_atoms) > 0:
        return rot_atoms[0]
    else:
        print("Position or symmetry is wrong.")
        raise ValueError
