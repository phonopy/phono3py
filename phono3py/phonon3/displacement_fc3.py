"""Procedures to handle atomic displacements for fc3."""

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

import numpy as np
from phonopy.harmonic.displacement import (
    directions_axis,
    get_displacement,
    get_least_displacements,
    is_minus_displacement,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_smallest_vectors
from phonopy.structure.symmetry import Symmetry


def direction_to_displacement(
    direction_dataset,
    displacement_distance,
    supercell: PhonopyAtoms,
    cutoff_distance=None,
):
    """Convert displacement directions to those in Cartesian coordinates.

    Parameters
    ----------
    direction_dataset : Return value of get_third_order_displacements
    displacement_distance :


    Returns
    -------
    dict
        Data structure is like (see docstring of Phonopy.dataset):
        {'natom': 64,
         'cutoff_distance': 4.000000,
         'first_atoms':
          [{'number': atom1,
            'displacement': [0.03, 0., 0.],
            'id': 1,
            'second_atoms': [ {'number': atom2,
                               'displacement': [0., -0.03, 0.],
                               'distance': 2.353,
                               'id': 7},
                              {'number': ... }, ... ] },
           {'number': atom1, ... }, ... ]}

    """
    duplicates = _find_duplicates(direction_dataset)
    d3_count = len(direction_dataset) + 1

    lattice = supercell.cell.T
    new_dataset = {}
    new_dataset["natom"] = len(supercell)

    if duplicates:
        new_dataset["duplicates"] = duplicates

    if cutoff_distance is not None:
        new_dataset["cutoff_distance"] = cutoff_distance
    new_first_atoms = []
    for i, first_atoms in enumerate(direction_dataset):
        atom1 = first_atoms["number"]
        direction1 = first_atoms["direction"]
        disp_cart1 = np.dot(direction1, lattice.T)
        disp_cart1 *= displacement_distance / np.linalg.norm(disp_cart1)
        new_second_atoms = []
        for second_atom in first_atoms["second_atoms"]:
            atom2 = second_atom["number"]
            pair_distance = second_atom["distance"]
            included = cutoff_distance is None or pair_distance < cutoff_distance
            for direction2 in second_atom["directions"]:
                disp_cart2 = np.dot(direction2, lattice.T)
                norm = np.linalg.norm(disp_cart2)
                disp_cart2 *= displacement_distance / norm
                disp2_dict = {
                    "id": d3_count,
                    "number": atom2,
                    "direction": direction2,
                    "displacement": disp_cart2,
                    "pair_distance": pair_distance,
                }
                if cutoff_distance is not None:
                    disp2_dict["included"] = included
                new_second_atoms.append(disp2_dict)
                d3_count += 1
        new_first_atoms.append(
            {
                "number": atom1,
                "direction": direction1,
                "displacement": disp_cart1,
                "id": (i + 1),
                "second_atoms": new_second_atoms,
            }
        )
    new_dataset["first_atoms"] = new_first_atoms

    return new_dataset


def get_third_order_displacements(
    cell: PhonopyAtoms, symmetry: Symmetry, is_plusminus="auto", is_diagonal=False
):
    """Create dispalcement dataset.

    Note
    ----
    Atoms 1, 2, and 3 are defined as follows:

    Atom 1: The first displaced atom. Third order force constant
            between Atoms 1, 2, and 3 is calculated.
    Atom 2: The second displaced atom. Second order force constant
            between Atoms 2 and 3 is calculated.
    Atom 3: Force is mesuared on this atom.

    Parameters
    ----------
    cell : PhonopyAtoms
        Supercell
    symmetry : Symmetry
        Symmetry of supercell
    is_plusminus : str or bool, optional
        Type of displacements, plus only (False), always plus and minus (True),
        and plus and minus depending on site symmetry ('auto').
    is_diagonal : bool, optional
        Whether allow diagonal displacements of Atom 2 or not

    Returns
    -------
    [{'number': atom1,
      'direction': [1, 0, 0],  # int
      'second_atoms': [ {'number': atom2,
                         'directions': [ [1, 0, 0], [-1, 0, 0], ... ]
                         'distance': distance-between-atom1-and-atom2},
                        {'number': ... }, ... ] },
     {'number': atom1, ... } ]

    """
    positions = cell.scaled_positions
    lattice = cell.cell.T

    # Least displacements of first atoms (Atom 1) are searched by
    # using respective site symmetries of the original crystal.
    # 'is_diagonal=False' below is made intentionally to expect
    # better accuracy.
    disps_first = get_least_displacements(
        symmetry, is_plusminus=is_plusminus, is_diagonal=False
    )

    symprec = symmetry.tolerance

    dds = []
    for disp in disps_first:
        atom1 = disp[0]
        disp1 = disp[1:4]
        site_sym = symmetry.get_site_symmetry(atom1)

        dds_atom1 = {"number": atom1, "direction": disp1, "second_atoms": []}

        # Reduced site symmetry at the first atom with respect to
        # the displacement of the first atoms.
        reduced_site_sym = get_reduced_site_symmetry(site_sym, disp1, symprec)
        # Searching orbits (second atoms) with respect to
        # the first atom and its reduced site symmetry.
        second_atoms = get_least_orbits(atom1, cell, reduced_site_sym, symprec)

        for atom2 in second_atoms:
            dds_atom2 = _get_next_displacements(
                atom1, atom2, reduced_site_sym, lattice, positions, symprec, is_diagonal
            )

            min_vec = get_smallest_vector_of_atom_pair(atom1, atom2, cell, symprec)
            min_distance = np.linalg.norm(np.dot(lattice, min_vec))
            dds_atom2["distance"] = min_distance
            dds_atom1["second_atoms"].append(dds_atom2)
        dds.append(dds_atom1)

    return dds


def _get_next_displacements(
    atom1, atom2, reduced_site_sym, lattice, positions, symprec, is_diagonal
):
    """Find displacements of second atom."""
    # Bond symmetry between first and second atoms.
    reduced_bond_sym = get_bond_symmetry(
        reduced_site_sym, lattice, positions, atom1, atom2, symprec
    )

    # Since displacement of first atom breaks translation
    # symmetry, the crystal symmetry is reduced to point
    # symmetry and it is equivalent to the site symmetry
    # on the first atom. Therefore site symmetry on the
    # second atom with the displacement is equivalent to
    # this bond symmetry.
    if is_diagonal:
        disps_second = get_displacement(reduced_bond_sym)
    else:
        disps_second = get_displacement(reduced_bond_sym, directions_axis)
    dds_atom2 = {"number": atom2, "directions": []}
    for disp2 in disps_second:
        dds_atom2["directions"].append(list(disp2))
        if is_minus_displacement(disp2, reduced_bond_sym):
            dds_atom2["directions"].append(list(-disp2))

    return dds_atom2


def get_reduced_site_symmetry(site_sym, direction, symprec=1e-5):
    """Return site symmetry that may be broken by a displacement."""
    reduced_site_sym = []
    for rot in site_sym:
        if (abs(direction - np.dot(direction, rot.T)) < symprec).all():
            reduced_site_sym.append(rot)
    return np.array(reduced_site_sym, dtype="intc")


def get_bond_symmetry(
    site_symmetry, lattice, positions, atom_center, atom_disp, symprec=1e-5
):
    """Return bond symmetry.

    Bond symmetry is the symmetry operations that keep the symmetry
    of the cell containing two fixed atoms.

    """
    bond_sym = []
    pos = positions
    for rot in site_symmetry:
        rot_pos = np.dot(pos[atom_disp] - pos[atom_center], rot.T) + pos[atom_center]
        diff = pos[atom_disp] - rot_pos
        diff -= np.rint(diff)
        dist = np.linalg.norm(np.dot(lattice, diff))
        if dist < symprec:
            bond_sym.append(rot)

    return np.array(bond_sym)


def get_least_orbits(atom_index, cell, site_symmetry, symprec=1e-5):
    """Find least orbits for a centering atom."""
    orbits = _get_orbits(atom_index, cell, site_symmetry, symprec)
    mapping = np.arange(cell.get_number_of_atoms())

    for i, orb in enumerate(orbits):
        for num in np.unique(orb):
            if mapping[num] > mapping[i]:
                mapping[num] = mapping[i]

    return np.unique(mapping)


def get_smallest_vector_of_atom_pair(
    atom_number_supercell, atom_number_primitive, supercell: PhonopyAtoms, symprec
):
    """Return smallest vectors of an atom pair in supercell."""
    s_pos = supercell.scaled_positions
    svecs, _ = get_smallest_vectors(
        supercell.cell,
        [s_pos[atom_number_supercell]],
        [s_pos[atom_number_primitive]],
        store_dense_svecs=True,
        symprec=symprec,
    )
    return svecs[0]


def _get_orbits(atom_index, cell, site_symmetry, symprec=1e-5):
    lattice = cell.cell.T
    positions = cell.scaled_positions
    center = positions[atom_index]

    # orbits[num_atoms, num_site_sym]
    orbits = []
    for pos in positions:
        mapping = []

        for rot in site_symmetry:
            rot_pos = np.dot(pos - center, rot.T) + center

            for i, pos in enumerate(positions):
                diff = pos - rot_pos
                diff -= np.rint(diff)
                dist = np.linalg.norm(np.dot(lattice, diff))
                if dist < symprec:
                    mapping.append(i)
                    break

        if len(mapping) < len(site_symmetry):
            print("Site symmetry is broken.")
            raise ValueError
        else:
            orbits.append(mapping)

    return np.array(orbits)


def _find_duplicates(direction_dataset):
    direction_sets = {}
    idx = len(direction_dataset) + 1
    pair_idx = {}

    # (List index of direction_sets + id_offset + 1) gives the displacement id.
    # This id is stamped in direction_to_displacement by the sequence of
    # the loops. Therefore the same system of the loops should be used here.
    for direction1 in direction_dataset:
        n1 = direction1["number"]
        d1 = direction1["direction"]
        for directions2 in direction1["second_atoms"]:
            n2 = directions2["number"]
            if (n1, n2) not in direction_sets:
                direction_sets[(n1, n2)] = []
                pair_idx[(n1, n2)] = []
            for i, d2 in enumerate(directions2["directions"]):
                direction_sets[(n1, n2)].append(d1 + d2)
                pair_idx[(n1, n2)].append(idx + i)
            idx += len(directions2["directions"])

    duplucates = []
    done = []
    for direction1 in direction_dataset:
        n1 = direction1["number"]
        for directions2 in direction1["second_atoms"]:
            n2 = directions2["number"]
            if n2 > n1 and (n2, n1) not in done and (n2, n1) in direction_sets:  # noqa E129
                done.append((n2, n1))
                duplucates += _compare(
                    n1,
                    n2,
                    direction_sets[(n1, n2)],
                    direction_sets[(n2, n1)],
                    pair_idx[(n1, n2)],
                    pair_idx[(n2, n1)],
                )

    done = []
    for direction1 in direction_dataset:
        n1 = direction1["number"]
        for directions2 in direction1["second_atoms"]:
            n2 = directions2["number"]
            if n1 == n2 and n1 not in done:
                done.append(n1)
                duplucates += _compare_opposite(
                    direction_sets[(n1, n1)], pair_idx[(n1, n1)]
                )

    return duplucates


def _compare(n1, n2, dset1, dset2, pidx1, pidx2):
    flip_sets = np.array(dset2)[:, [3, 4, 5, 0, 1, 2]]
    duplucates = []

    for i, d1 in enumerate(dset1):
        eq_indices = np.where(np.abs(flip_sets - d1).sum(axis=1) == 0)[0]
        if len(eq_indices) > 0:
            duplucates += [[pidx2[j], pidx1[i]] for j in eq_indices]

    return [[i, j] for (i, j) in duplucates if i > j]


def _compare_opposite(dset1, pidx1):
    flip_sets = np.array(dset1)[:, [3, 4, 5, 0, 1, 2]]
    duplucates = []
    for d1 in dset1:
        eq_indices = np.where(np.abs(flip_sets + d1).sum(axis=1) == 0)[0]
        if len(eq_indices) > 0:
            duplucates += [[pidx1[j], 0] for j in eq_indices]

    return [[i, j] for (i, j) in duplucates if i > j]
