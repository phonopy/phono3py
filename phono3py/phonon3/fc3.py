"""Calculate fc3."""

# Copyright (C) 2020 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import logging
import sys

import numpy as np
from phonopy.harmonic.force_constants import (
    distribute_force_constants,
    get_fc2,
    get_nsym_list_and_s2pp,
    get_positions_sent_by_rot_inv,
    get_rotated_displacement,
    similarity_transformation,
    solve_force_constants,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, compute_all_sg_permutations
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon3.displacement_fc3 import (
    get_bond_symmetry,
    get_reduced_site_symmetry,
    get_smallest_vector_of_atom_pair,
)

logger = logging.getLogger(__name__)


def get_fc3(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    disp_dataset,
    symmetry: Symmetry,
    is_compact_fc=False,
    pinv_solver: str = "numpy",
    verbose=False,
):
    """Calculate fc3.

    Even when 'cutoff_distance' in dataset, all displacements are in the
    dataset, but force-sets out of cutoff-pair-distance are zero. fc3 is solved
    in exactly the same way. Then post-clean-up is performed.

    Returns
    -------
    tuple :
        (fc2, fc3) fc2 and fc3 can be compact or full array formats depending on
        `is_compact_fc`. See Phono3py.produce_fc3.

    """
    # fc2 has to be full matrix to compute delta-fc2
    # p2s_map elements are extracted if is_compact_fc=True at the last part.
    fc2 = get_fc2(supercell, symmetry, disp_dataset)
    fc3 = _get_fc3_least_atoms(
        supercell,
        primitive,
        disp_dataset,
        fc2,
        symmetry,
        is_compact_fc=(is_compact_fc and "cutoff_distance" not in disp_dataset),
        pinv_solver=pinv_solver,
        verbose=verbose,
    )
    if verbose:
        print("Expanding fc3.")

    first_disp_atoms = np.unique([x["number"] for x in disp_dataset["first_atoms"]])
    rotations = symmetry.symmetry_operations["rotations"]
    lattice = supercell.cell.T
    permutations = symmetry.atomic_permutations

    p2s_map = primitive.p2s_map
    for i in first_disp_atoms:
        assert i in p2s_map

    if is_compact_fc and "cutoff_distance" not in disp_dataset:
        s2p_map = primitive.s2p_map
        p2p_map = primitive.p2p_map
        s2compact = np.array([p2p_map[i] for i in s2p_map], dtype="int_")
        target_atoms = [i for i in p2s_map if i not in first_disp_atoms]
    else:
        # distribute_fc3 prefers pure translation operations in distributing
        # fc3. Below, fc3 already computed are distributed to the first index
        # atoms in primitive cell, and then distribute to all the other atoms.
        s2compact = np.arange(len(supercell), dtype="int_")
        target_atoms = [i for i in p2s_map if i not in first_disp_atoms]
        distribute_fc3(
            fc3,
            first_disp_atoms,
            target_atoms,
            lattice,
            rotations,
            permutations,
            s2compact,
            verbose=verbose,
        )
        first_disp_atoms = np.unique(np.concatenate((first_disp_atoms, p2s_map)))
        target_atoms = [i for i in s2compact if i not in first_disp_atoms]

    distribute_fc3(
        fc3,
        first_disp_atoms,
        target_atoms,
        lattice,
        rotations,
        permutations,
        s2compact,
        verbose=verbose,
    )

    if "cutoff_distance" in disp_dataset:
        if verbose:
            print(
                "Cutting-off fc3 (cut-off distance: %f)"
                % disp_dataset["cutoff_distance"]
            )
        _cutoff_fc3_for_cutoff_pairs(
            fc3, supercell, disp_dataset, symmetry, verbose=verbose
        )

    if is_compact_fc:
        p2s_map = primitive.p2s_map
        fc2 = np.array(fc2[p2s_map], dtype="double", order="C")
        if "cutoff_distance" in disp_dataset:
            fc3_shape = (len(p2s_map), fc3.shape[1], fc3.shape[2])
            fc3_cfc = np.zeros(fc3_shape, dtype="double", order="C")
            fc3_cfc = fc3[p2s_map]
            fc3 = fc3_cfc

    return fc2, fc3


def distribute_fc3(
    fc3,
    first_disp_atoms,
    target_atoms,
    lattice,
    rotations,
    permutations,
    s2compact,
    verbose=False,
):
    """Distribute fc3.

    fc3[i, :, :, 0:3, 0:3, 0:3] where i=indices done are distributed to
    symmetrically equivalent fc3 elements by tensor rotations.

    Search symmetry operation (R, t) that performs
        i_target -> i_done
    and
        atom_mapping[i_target] = i_done
        fc3[i_target, j_target, k_target] = R_inv[i_done, j, k]
    When multiple (R, t) can be found, the pure translation operation is
    preferred.

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
    identity = np.eye(3, dtype=int)
    pure_trans_indices = [i for i, r in enumerate(rotations) if (r == identity).all()]

    n_satom = fc3.shape[1]
    for i_target in target_atoms:
        for i_done in first_disp_atoms:
            rot_indices = np.where(permutations[:, i_target] == i_done)[0]
            if len(rot_indices) > 0:
                rot_index = rot_indices[0]
                for rot_i in rot_indices:
                    if rot_i in pure_trans_indices:
                        rot_index = rot_i
                        break
                atom_mapping = np.array(permutations[rot_index], dtype="int_")
                rot = rotations[rot_index]
                rot_cart_inv = np.array(
                    similarity_transformation(lattice, rot).T, dtype="double", order="C"
                )
                break

        if len(rot_indices) == 0:
            print("Position or symmetry may be wrong.")
            raise RuntimeError

        if verbose > 2:
            print("    [ %d, x, x ] to [ %d, x, x ]" % (i_done + 1, i_target + 1))
            sys.stdout.flush()

        try:
            import phono3py._phono3py as phono3c

            phono3c.distribute_fc3(
                fc3, s2compact[i_target], s2compact[i_done], atom_mapping, rot_cart_inv
            )
        except ImportError:
            print("Phono3py C-routine is not compiled correctly.")
            for j in range(n_satom):
                j_rot = atom_mapping[j]
                for k in range(n_satom):
                    k_rot = atom_mapping[k]
                    fc3[i_target, j, k] = _third_rank_tensor_rotation(
                        rot_cart_inv, fc3[i_done, j_rot, k_rot]
                    )


def set_permutation_symmetry_fc3(fc3):
    """Enforce permutation symmetry to full fc3."""
    try:
        import phono3py._phono3py as phono3c

        phono3c.permutation_symmetry_fc3(fc3)
    except ImportError:
        print("Phono3py C-routine is not compiled correctly.")
        num_atom = fc3.shape[0]
        for i in range(num_atom):
            for j in range(i, num_atom):
                for k in range(j, num_atom):
                    fc3_elem = _set_permutation_symmetry_fc3_elem(fc3, i, j, k)
                    _copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, i, j, k)


def set_permutation_symmetry_compact_fc3(fc3, primitive):
    """Enforce permulation symmetry to compact fc3."""
    try:
        import phono3py._phono3py as phono3c

        s2p_map = primitive.s2p_map
        p2s_map = primitive.p2s_map
        p2p_map = primitive.p2p_map
        permutations = primitive.atomic_permutations
        s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map, p2p_map, permutations)
        phono3c.permutation_symmetry_compact_fc3(
            fc3,
            np.array(permutations, dtype="int_", order="C"),
            np.array(s2pp_map, dtype="int_"),
            np.array(p2s_map, dtype="int_"),
            np.array(nsym_list, dtype="int_"),
        )
    except ImportError as exc:
        text = (
            "Import error at phono3c.permutation_symmetry_compact_fc3. "
            "Corresponding python code is not implemented."
        )
        raise RuntimeError(text) from exc


def _copy_permutation_symmetry_fc3_elem(fc3, fc3_elem, a, b, c):
    for i, j, k in list(np.ndindex(3, 3, 3)):
        fc3[a, b, c, i, j, k] = fc3_elem[i, j, k]
        fc3[c, a, b, k, i, j] = fc3_elem[i, j, k]
        fc3[b, c, a, j, k, i] = fc3_elem[i, j, k]
        fc3[a, c, b, i, k, j] = fc3_elem[i, j, k]
        fc3[b, a, c, j, i, k] = fc3_elem[i, j, k]
        fc3[c, b, a, k, j, i] = fc3_elem[i, j, k]


def _set_permutation_symmetry_fc3_elem(fc3, a, b, c, divisor=6):
    tensor3 = np.zeros((3, 3, 3), dtype="double")
    for i, j, k in list(np.ndindex(3, 3, 3)):
        tensor3[i, j, k] = (
            fc3[a, b, c, i, j, k]
            + fc3[c, a, b, k, i, j]
            + fc3[b, c, a, j, k, i]
            + fc3[a, c, b, i, k, j]
            + fc3[b, a, c, j, i, k]
            + fc3[c, b, a, k, j, i]
        ) / divisor
    return tensor3


def set_translational_invariance_fc3(fc3):
    """Enforce translational symmetry to fc3."""
    for i in range(3):
        _set_translational_invariance_fc3_per_index(fc3, index=i)


def set_translational_invariance_compact_fc3(fc3, primitive: Primitive):
    """Enforce translational symmetry to compact fc3."""
    try:
        import phono3py._phono3py as phono3c

        s2p_map = primitive.s2p_map
        p2s_map = primitive.p2s_map
        p2p_map = primitive.p2p_map
        permutations = primitive.atomic_permutations
        s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map, p2p_map, permutations)

        permutations = np.array(permutations, dtype="int_", order="C")
        s2pp_map = np.array(s2pp_map, dtype="int_")
        p2s_map = np.array(p2s_map, dtype="int_")
        nsym_list = np.array(nsym_list, dtype="int_")
        phono3c.transpose_compact_fc3(
            fc3, permutations, s2pp_map, p2s_map, nsym_list, 0
        )  # dim[0] <--> dim[1]
        _set_translational_invariance_fc3_per_index(fc3, index=1)
        phono3c.transpose_compact_fc3(
            fc3, permutations, s2pp_map, p2s_map, nsym_list, 0
        )  # dim[0] <--> dim[1]
        _set_translational_invariance_fc3_per_index(fc3, index=1)
        _set_translational_invariance_fc3_per_index(fc3, index=2)

    except ImportError as exc:
        text = (
            "Import error at phono3c.tranpose_compact_fc3. "
            "Corresponding python code is not implemented."
        )
        raise RuntimeError(text) from exc


def _set_translational_invariance_fc3_per_index(fc3, index=0):
    for i in range(fc3.shape[(1 + index) % 3]):
        for j in range(fc3.shape[(2 + index) % 3]):
            for k, ll, m in list(np.ndindex(3, 3, 3)):
                if index == 0:
                    fc3[:, i, j, k, ll, m] -= (
                        np.sum(fc3[:, i, j, k, ll, m]) / fc3.shape[0]
                    )
                elif index == 1:
                    fc3[j, :, i, k, ll, m] -= (
                        np.sum(fc3[j, :, i, k, ll, m]) / fc3.shape[1]
                    )
                elif index == 2:
                    fc3[i, j, :, k, ll, m] -= (
                        np.sum(fc3[i, j, :, k, ll, m]) / fc3.shape[2]
                    )


def _third_rank_tensor_rotation(rot_cart, tensor):
    rot_tensor = np.zeros((3, 3, 3), dtype="double")
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                rot_tensor[i, j, k] = _third_rank_tensor_rotation_elem(
                    rot_cart, tensor, i, j, k
                )
    return rot_tensor


def _get_delta_fc2(
    dataset_second_atoms, atom1, forces1, fc2, supercell, reduced_site_sym, symprec
):
    logger.debug("get_delta_fc2")
    disp_fc2 = _get_constrained_fc2(
        supercell, dataset_second_atoms, atom1, forces1, reduced_site_sym, symprec
    )
    return disp_fc2 - fc2


def _get_constrained_fc2(
    supercell: PhonopyAtoms,
    dataset_second_atoms,
    atom1,
    forces1,
    reduced_site_sym,
    symprec,
):
    """Return fc2 under reduced (broken) site symmetry by first displacement.

    dataset_second_atoms: [{'number': 7,
                            'displacement': [],
                            'forces': []}, ...]

    """
    lattice = supercell.cell.T
    positions = supercell.scaled_positions
    num_atom = len(supercell)

    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype="double")
    atom_list = np.unique([x["number"] for x in dataset_second_atoms])
    for atom2 in atom_list:
        disps2 = []
        sets_of_forces = []
        for disps_second in dataset_second_atoms:
            if atom2 != disps_second["number"]:
                continue
            bond_sym = get_bond_symmetry(
                reduced_site_sym, lattice, positions, atom1, atom2, symprec
            )

            disps2.append(disps_second["displacement"])
            sets_of_forces.append(disps_second["forces"] - forces1)

        solve_force_constants(
            fc2, atom2, disps2, sets_of_forces, supercell, bond_sym, symprec
        )

    # Shift positions according to set atom1 is at origin
    pos_center = positions[atom1].copy()
    positions -= pos_center
    rotations = np.array(reduced_site_sym, dtype="intc", order="C")
    translations = np.zeros((len(reduced_site_sym), 3), dtype="double", order="C")
    permutations = compute_all_sg_permutations(
        positions, rotations, translations, lattice, symprec
    )
    distribute_force_constants(fc2, atom_list, lattice, rotations, permutations)
    return fc2


def _solve_fc3(
    first_atom_num,
    supercell,
    site_symmetry,
    displacements_first,
    delta_fc2s,
    symprec,
    pinv_solver="numpy",
    verbose=False,
):
    logger.debug("solve_fc3")

    if pinv_solver == "numpy":
        solver = "numpy.linalg.pinv"
    else:
        try:
            import phono3py._phono3py as phono3c

            solver = "lapacke-dgesvd"
        except ImportError:
            print("Phono3py C-routine is not compiled correctly.")
            solver = "numpy.linalg.pinv"

    if verbose:
        print(f"Computing fc3[ {first_atom_num + 1}, x, x ] using {solver}.")
        if len(displacements_first) > 1:
            print("Displacements (in Angstrom):")
        else:
            print("One displacement (in Angstrom):")
        for v in displacements_first:
            print("    [%7.4f %7.4f %7.4f]" % tuple(v))
            sys.stdout.flush()
        if verbose > 2:
            print("  Site symmetry:")
            for i, v in enumerate(site_symmetry):
                print("    [%2d %2d %2d] #%2d" % tuple(list(v[0]) + [i + 1]))
                print("    [%2d %2d %2d]" % tuple(v[1]))
                print("    [%2d %2d %2d]\n" % tuple(v[2]))
                sys.stdout.flush()

    lattice = supercell.cell.T
    site_sym_cart = np.array(
        [similarity_transformation(lattice, sym) for sym in site_symmetry],
        dtype="double",
        order="C",
    )
    num_atom = len(supercell)
    positions = supercell.scaled_positions
    pos_center = positions[first_atom_num].copy()
    positions -= pos_center

    logger.debug("get_positions_sent_by_rot_inv")

    rot_map_syms = get_positions_sent_by_rot_inv(
        lattice, positions, site_symmetry, symprec
    )
    rot_map_syms = np.array(rot_map_syms, dtype="int_", order="C")
    rot_disps = get_rotated_displacement(displacements_first, site_sym_cart)

    logger.debug("pinv")

    if "numpy" in solver:
        inv_U = np.array(np.linalg.pinv(rot_disps), dtype="double", order="C")
    else:
        inv_U = np.zeros(
            (rot_disps.shape[1], rot_disps.shape[0]), dtype="double", order="C"
        )
        phono3c.lapacke_pinv(inv_U, rot_disps, 1e-13)

    fc3 = np.zeros((num_atom, num_atom, 3, 3, 3), dtype="double", order="C")

    logger.debug("rotate_delta_fc2s")

    try:
        import phono3py._phono3py as phono3c

        phono3c.rotate_delta_fc2s(fc3, delta_fc2s, inv_U, site_sym_cart, rot_map_syms)
    except ImportError:
        for i, j in np.ndindex(num_atom, num_atom):
            fc3[i, j] = np.dot(
                inv_U, _get_rotated_fc2s(i, j, delta_fc2s, rot_map_syms, site_sym_cart)
            ).reshape(3, 3, 3)

    return fc3


def _cutoff_fc3_for_cutoff_pairs(fc3, supercell, disp_dataset, symmetry, verbose=False):
    if verbose:
        print("Building atom mapping table...")
    fc3_done = _get_fc3_done(supercell, disp_dataset, symmetry, fc3.shape[:3])

    if verbose:
        print("Creating contracted fc3...")
    num_atom = len(supercell)
    for i in range(num_atom):
        for j in range(i, num_atom):
            for k in range(j, num_atom):
                ave_fc3 = _set_permutation_symmetry_fc3_elem_with_cutoff(
                    fc3, fc3_done, i, j, k
                )
                _copy_permutation_symmetry_fc3_elem(fc3, ave_fc3, i, j, k)


def cutoff_fc3_by_zero(fc3, supercell, cutoff_distance, p2s_map=None, symprec=1e-5):
    """Set zero in fc3 elements where pair distances are larger than cutoff."""
    num_atom = len(supercell)
    lattice = supercell.cell.T
    min_distances = np.zeros((num_atom, num_atom), dtype="double")
    for i in range(num_atom):  # run in supercell
        for j in range(num_atom):  # run in primitive
            min_distances[i, j] = np.linalg.norm(
                np.dot(
                    lattice, get_smallest_vector_of_atom_pair(i, j, supercell, symprec)
                )
            )

    if fc3.shape[0] == fc3.shape[1]:
        _p2s_map = np.arange(num_atom)
    elif p2s_map is None or len(p2s_map) != fc3.shape[0]:
        raise RuntimeError("Array shape of fc3 is incorrect.")
    else:
        _p2s_map = p2s_map

    for i_index, i in enumerate(_p2s_map):
        for j, k in np.ndindex(num_atom, num_atom):
            for pair in ((i, j), (j, k), (k, i)):
                if min_distances[pair] > cutoff_distance:
                    fc3[i_index, j, k] = 0
                    break


def show_drift_fc3(fc3, primitive=None, name="fc3"):
    """Show drift of fc3."""
    if fc3.shape[0] == fc3.shape[1]:
        num_atom = fc3.shape[0]
        maxval1 = 0
        maxval2 = 0
        maxval3 = 0
        klm1 = [0, 0, 0]
        klm2 = [0, 0, 0]
        klm3 = [0, 0, 0]
        for i, j, k, ll, m in list(np.ndindex((num_atom, num_atom, 3, 3, 3))):
            val1 = fc3[:, i, j, k, ll, m].sum()
            val2 = fc3[i, :, j, k, ll, m].sum()
            val3 = fc3[i, j, :, k, ll, m].sum()
            if abs(val1) > abs(maxval1):
                maxval1 = val1
                klm1 = [k, ll, m]
            if abs(val2) > abs(maxval2):
                maxval2 = val2
                klm2 = [k, ll, m]
            if abs(val3) > abs(maxval3):
                maxval3 = val3
                klm3 = [k, ll, m]
    else:
        try:
            import phono3py._phono3py as phono3c

            s2p_map = primitive.s2p_map
            p2s_map = primitive.p2s_map
            p2p_map = primitive.p2p_map
            permutations = primitive.atomic_permutations
            s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map, p2p_map, permutations)
            permutations = np.array(permutations, dtype="int_", order="C")
            s2pp_map = np.array(s2pp_map, dtype="int_")
            p2s_map = np.array(p2s_map, dtype="int_")
            nsym_list = np.array(nsym_list, dtype="int_")
            num_patom = fc3.shape[0]
            num_satom = fc3.shape[1]
            maxval1 = 0
            maxval2 = 0
            maxval3 = 0
            klm1 = [0, 0, 0]
            klm2 = [0, 0, 0]
            klm3 = [0, 0, 0]
            phono3c.transpose_compact_fc3(
                fc3, permutations, s2pp_map, p2s_map, nsym_list, 0
            )  # dim[0] <--> dim[1]
            for i, j, k, ll, m in np.ndindex((num_patom, num_satom, 3, 3, 3)):
                val1 = fc3[i, :, j, k, ll, m].sum()
                if abs(val1) > abs(maxval1):
                    maxval1 = val1
                    klm1 = [k, ll, m]
            phono3c.transpose_compact_fc3(
                fc3, permutations, s2pp_map, p2s_map, nsym_list, 0
            )  # dim[0] <--> dim[1]
            for i, j, k, ll, m in np.ndindex((num_patom, num_satom, 3, 3, 3)):
                val2 = fc3[i, :, j, k, ll, m].sum()
                val3 = fc3[i, j, :, k, ll, m].sum()
                if abs(val2) > abs(maxval2):
                    maxval2 = val2
                    klm2 = [k, ll, m]
                if abs(val3) > abs(maxval3):
                    maxval3 = val3
                    klm3 = [k, ll, m]
        except ImportError as exc:
            text = (
                "Import error at phono3c.tranpose_compact_fc3. "
                "Corresponding python code is not implemented."
            )
            raise RuntimeError(text) from exc

    text = "Max drift of %s: " % name
    text += "%f (%s%s%s) " % (maxval1, "xyz"[klm1[0]], "xyz"[klm1[1]], "xyz"[klm1[2]])
    text += "%f (%s%s%s) " % (maxval2, "xyz"[klm2[0]], "xyz"[klm2[1]], "xyz"[klm2[2]])
    text += "%f (%s%s%s)" % (maxval3, "xyz"[klm3[0]], "xyz"[klm3[1]], "xyz"[klm3[2]])
    print(text)


def _set_permutation_symmetry_fc3_elem_with_cutoff(fc3, fc3_done, a, b, c):
    sum_done = (
        fc3_done[a, b, c]
        + fc3_done[c, a, b]
        + fc3_done[b, c, a]
        + fc3_done[b, a, c]
        + fc3_done[c, b, a]
        + fc3_done[a, c, b]
    )
    tensor3 = np.zeros((3, 3, 3), dtype="double")
    if sum_done > 0:
        for i, j, k in list(np.ndindex(3, 3, 3)):
            tensor3[i, j, k] = (
                fc3[a, b, c, i, j, k] * fc3_done[a, b, c]
                + fc3[c, a, b, k, i, j] * fc3_done[c, a, b]
                + fc3[b, c, a, j, k, i] * fc3_done[b, c, a]
                + fc3[a, c, b, i, k, j] * fc3_done[a, c, b]
                + fc3[b, a, c, j, i, k] * fc3_done[b, a, c]
                + fc3[c, b, a, k, j, i] * fc3_done[c, b, a]
            )
            tensor3[i, j, k] /= sum_done
    return tensor3


def _get_fc3_least_atoms(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    disp_dataset,
    fc2,
    symmetry: Symmetry,
    is_compact_fc: bool = False,
    pinv_solver="numpy",
    verbose: bool = True,
):
    symprec = symmetry.tolerance
    num_satom = len(supercell)
    unique_first_atom_nums = np.unique(
        [x["number"] for x in disp_dataset["first_atoms"]]
    )

    if is_compact_fc:
        num_patom = len(primitive)
        s2p_map = primitive.s2p_map
        p2p_map = primitive.p2p_map
        first_atom_nums = []
        for i in unique_first_atom_nums:
            if i != s2p_map[i]:
                print("Something wrong in displacement dataset.")
                raise RuntimeError
            else:
                first_atom_nums.append(i)
        fc3 = np.zeros(
            (num_patom, num_satom, num_satom, 3, 3, 3), dtype="double", order="C"
        )
    else:
        first_atom_nums = unique_first_atom_nums
        fc3 = np.zeros(
            (num_satom, num_satom, num_satom, 3, 3, 3), dtype="double", order="C"
        )

    for first_atom_num in first_atom_nums:
        site_symmetry = symmetry.get_site_symmetry(first_atom_num)
        displacements_first = []
        delta_fc2s = []
        for dataset_first_atom in disp_dataset["first_atoms"]:
            if first_atom_num != dataset_first_atom["number"]:
                continue

            displacements_first.append(dataset_first_atom["displacement"])
            if "delta_fc2" in dataset_first_atom:
                delta_fc2s.append(dataset_first_atom["delta_fc2"])
            else:
                direction = np.dot(
                    dataset_first_atom["displacement"],
                    np.linalg.inv(supercell.cell),
                )
                reduced_site_sym = get_reduced_site_symmetry(
                    site_symmetry, direction, symprec
                )
                delta_fc2s.append(
                    _get_delta_fc2(
                        dataset_first_atom["second_atoms"],
                        dataset_first_atom["number"],
                        dataset_first_atom["forces"],
                        fc2,
                        supercell,
                        reduced_site_sym,
                        symprec,
                    )
                )

        fc3_first = _solve_fc3(
            first_atom_num,
            supercell,
            site_symmetry,
            displacements_first,
            np.array(delta_fc2s, dtype="double", order="C"),
            symprec,
            pinv_solver=pinv_solver,
            verbose=verbose,
        )
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


def _third_rank_tensor_rotation_elem(rot, tensor, ll, m, n):
    sum_elems = 0.0
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                sum_elems += rot[ll, i] * rot[m, j] * rot[n, k] * tensor[i, j, k]
    return sum_elems


def _get_fc3_done(
    supercell: PhonopyAtoms, disp_dataset, symmetry: Symmetry, array_shape
):
    num_atom = len(supercell)
    fc3_done = np.zeros(array_shape, dtype="byte")
    symprec = symmetry.tolerance
    lattice = supercell.cell.T
    positions = supercell.scaled_positions
    rotations = symmetry.symmetry_operations["rotations"]
    translations = symmetry.symmetry_operations["translations"]

    atom_mapping = []
    for rot, trans in zip(rotations, translations):
        atom_indices = [
            _get_atom_by_symmetry(lattice, positions, rot, trans, i, symprec)
            for i in range(num_atom)
        ]
        atom_mapping.append(atom_indices)

    for dataset_first_atom in disp_dataset["first_atoms"]:
        first_atom_num = dataset_first_atom["number"]
        site_symmetry = symmetry.get_site_symmetry(first_atom_num)
        direction = np.dot(
            dataset_first_atom["displacement"], np.linalg.inv(supercell.cell)
        )
        reduced_site_sym = get_reduced_site_symmetry(site_symmetry, direction, symprec)
        least_second_atom_nums = []
        for second_atoms in dataset_first_atom["second_atoms"]:
            if "included" in second_atoms:
                if second_atoms["included"]:
                    least_second_atom_nums.append(second_atoms["number"])
            elif "cutoff_distance" in disp_dataset:
                min_vec = get_smallest_vector_of_atom_pair(
                    first_atom_num, second_atoms["number"], supercell, symprec
                )
                min_distance = np.linalg.norm(np.dot(lattice, min_vec))
                if "pair_distance" in second_atoms:
                    assert abs(min_distance - second_atoms["pair_distance"]) < 1e-4
                if min_distance < disp_dataset["cutoff_distance"]:
                    least_second_atom_nums.append(second_atoms["number"])

        positions_shifted = positions - positions[first_atom_num]
        least_second_atom_nums = np.unique(least_second_atom_nums)

        for red_rot in reduced_site_sym:
            second_atom_nums = [
                _get_atom_by_symmetry(
                    lattice,
                    positions_shifted,
                    red_rot,
                    np.zeros(3, dtype="double"),
                    i,
                    symprec,
                )
                for i in least_second_atom_nums
            ]
        second_atom_nums = np.unique(second_atom_nums)

        for i in range(len(rotations)):
            rotated_atom1 = atom_mapping[i][first_atom_num]
            for j in second_atom_nums:
                fc3_done[rotated_atom1, atom_mapping[i][j]] = 1

    return fc3_done


def _get_atom_by_symmetry(lattice, positions, rotation, trans, atom_number, symprec):
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
