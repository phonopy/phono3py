"""Procedures to handle atomic displacements for fc4."""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.displacement import (
    directions_axis,
    get_displacement,
    get_least_displacements,
    is_minus_displacement,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon3.displacement_fc3 import (
    _get_next_displacements,
    get_bond_symmetry,
    get_least_orbits,
    get_reduced_site_symmetry,
)


class ThirdAtomDisplacement(TypedDict):
    """Displacement information of the third displaced atom."""

    number: int
    displacement: NDArray[np.double]


class ThirdAtomDisplacementWithForces(ThirdAtomDisplacement, total=False):
    """Third atom displacement entry optionally containing forces.

    Forces are measured for the supercell in which Atoms 1, 2, and 3 are
    simultaneously displaced.

    """

    id: int
    forces: NDArray[np.double]


class SecondAtomDisplacementFc4(TypedDict):
    """Displacement information of the second displaced atom for fc4."""

    number: int
    displacement: NDArray[np.double]
    third_atoms: list[ThirdAtomDisplacementWithForces]


class SecondAtomDisplacementFc4WithForces(SecondAtomDisplacementFc4, total=False):
    """Second atom entry optionally containing its id and forces.

    Forces are measured for the supercell in which Atoms 1 and 2 are
    simultaneously displaced.

    """

    id: int
    forces: NDArray[np.double]


class FirstAtomDisplacementFc4(TypedDict):
    """Displacement information of the first displaced atom for fc4."""

    number: int
    displacement: NDArray[np.double]
    second_atoms: list[SecondAtomDisplacementFc4WithForces]


class FirstAtomDisplacementFc4WithForces(FirstAtomDisplacementFc4, total=False):
    """First atom entry optionally containing its id and forces.

    Forces are measured for the supercell in which only Atom 1 is displaced;
    this is the baseline reference of the inner fc3 of the Atom-1-displaced
    structure.

    """

    id: int
    forces: NDArray[np.double]


class Fc4Type1DisplacementDataset(TypedDict):
    """Type-1 displacement dataset for fc4.

    The dataset is self-contained and mirrors the fc3 dataset one level
    deeper. Every node (first, second, and third atoms) is a supercell whose
    forces are to be computed: F(a1), F(a1+a2), and F(a1+a2+a3) respectively.

    """

    natom: int
    first_atoms: list[FirstAtomDisplacementFc4WithForces]


def get_fourth_order_displacements(
    cell: PhonopyAtoms,
    symmetry: Symmetry,
    displacement_distance: float,
    is_plusminus: bool | Literal["auto"] = "auto",
    is_diagonal: bool = False,
) -> Fc4Type1DisplacementDataset:
    """Create a displacement dataset for fc4.

    Note
    ----
    Atoms 1, 2, 3, and 4 are defined as follows:

    Atom 1: The first displaced atom. Fourth order force constant
            between Atoms 1, 2, 3, and 4 is calculated.
    Atom 2: The second displaced atom. Third order force constant
            between Atoms 2, 3, and 4 is calculated.
    Atom 3: The third displaced atom. Second order force constant
            between Atoms 3 and 4 is calculated.
    Atom 4: Force is measured on this atom.

    The independent displacements are reduced by, in this order, the site
    symmetry at Atom 1, the bond symmetry of the (Atom 1, Atom 2) pair, and
    the bond symmetry of the (Atom 2, Atom 3) pair, each further reduced by
    the displacement applied at the previous level.

    Parameters
    ----------
    cell : PhonopyAtoms
        Supercell.
    symmetry : Symmetry
        Symmetry of the supercell.
    displacement_distance : float
        Displacement distance in Cartesian coordinates.
    is_plusminus : str or bool, optional
        Type of displacements of Atom 1: plus only (False), always plus and
        minus (True), or plus and minus depending on site symmetry ('auto').
    is_diagonal : bool, optional
        Whether to allow diagonal displacements of Atoms 2 and 3.

    Returns
    -------
    Fc4Type1DisplacementDataset
        Nested type-1 displacement dataset (first -> second -> third atoms).

    Note
    ----
    Pair-distance cutoff is not yet implemented. For fc4 the number of
    supercells scales steeply, so a cutoff analogous to
    ``get_third_order_displacements`` is expected to be required for
    practical use.

    """
    warnings.warn(
        "fc4 (4th-order force constants) support in phono3py is an experimental "
        "feature under active development; results, defaults, and the API may "
        "change without notice.",
        stacklevel=2,
    )
    positions = cell.scaled_positions
    lattice = cell.cell.T
    symprec = symmetry.tolerance

    # Least displacements of the first atoms (Atom 1) are searched using
    # the respective site symmetries of the original crystal.
    # 'is_diagonal=False' is intentional to expect better accuracy.
    disps_first = get_least_displacements(
        symmetry, is_plusminus=is_plusminus, is_diagonal=False
    )

    dds = []
    for disp in disps_first:
        atom1 = disp[0]
        disp1 = disp[1:4]
        site_sym = symmetry.get_site_symmetry(atom1)

        dds_atom1: dict[str, Any] = {
            "number": atom1,
            "direction": disp1,
            "second_atoms": [],
        }

        # Reduced site symmetry at Atom 1 with respect to its displacement.
        reduced_site_sym = get_reduced_site_symmetry(site_sym, disp1, symprec)
        # Orbits of the second atoms under the reduced site symmetry.
        second_atoms = get_least_orbits(atom1, cell, reduced_site_sym, symprec)

        for atom2 in second_atoms:
            # Bond symmetry of the (Atom 1, Atom 2) pair.
            reduced_bond_sym = get_bond_symmetry(
                reduced_site_sym, lattice, positions, atom1, atom2, symprec
            )

            for disp2 in _get_displacements_second(reduced_bond_sym, is_diagonal):
                dds_atom2 = _get_third_atom_directions(
                    atom2,
                    disp2,
                    cell,
                    reduced_bond_sym,
                    lattice,
                    positions,
                    symprec,
                    is_diagonal,
                )
                dds_atom1["second_atoms"].append(dds_atom2)
        dds.append(dds_atom1)

    return _direction_to_displacement_fc4(dds, displacement_distance, cell)


def _get_displacements_second(
    reduced_bond_sym: NDArray[np.int64], is_diagonal: bool
) -> list[NDArray[np.double]]:
    """Return the displacement directions of the second atom.

    The directions are reduced by the bond symmetry of the (Atom 1, Atom 2)
    pair, and the minus counterpart is added when it is not generated by the
    symmetry.

    """
    if is_diagonal:
        disps_second = get_displacement(reduced_bond_sym)
    else:
        disps_second = get_displacement(reduced_bond_sym, directions_axis)

    disps_second_with_minus = []
    for disp2 in disps_second:
        disps_second_with_minus.append(disp2)
        if is_minus_displacement(disp2, reduced_bond_sym):
            disps_second_with_minus.append(-disp2)

    return disps_second_with_minus


def _get_third_atom_directions(
    atom2: int,
    disp2: NDArray[np.double],
    cell: PhonopyAtoms,
    reduced_bond_sym: NDArray[np.int64],
    lattice: NDArray[np.double],
    positions: NDArray[np.double],
    symprec: float,
    is_diagonal: bool,
) -> dict[str, Any]:
    """Return the second-atom entry holding its third-atom directions.

    For the given displacement ``disp2`` of Atom 2, the symmetry is reduced
    to the bond symmetry of the (Atom 2, Atom 3) pair (the "plane symmetry"),
    and the third-atom displacement directions are generated for each orbit
    of third atoms.

    """
    dds_atom2: dict[str, Any] = {
        "number": atom2,
        "direction": disp2,
        "third_atoms": [],
    }
    # Reduced bond symmetry with respect to the displacement of Atom 2.
    # This plays the role of the "site symmetry" of the center Atom 2 when
    # searching the third-atom displacements: _get_next_displacements derives
    # the (Atom 2, Atom 3) bond symmetry from it internally.
    reduced_bond_sym2 = get_reduced_site_symmetry(reduced_bond_sym, disp2, symprec)
    third_atoms = get_least_orbits(atom2, cell, reduced_bond_sym2, symprec)

    for atom3 in third_atoms:
        dds_atom3 = _get_next_displacements(
            atom2, atom3, reduced_bond_sym2, lattice, positions, symprec, is_diagonal
        )
        dds_atom2["third_atoms"].append(dds_atom3)

    return dds_atom2


def _direction_to_displacement_fc4(
    direction_dataset: list[dict],
    displacement_distance: float,
    supercell: PhonopyAtoms,
) -> Fc4Type1DisplacementDataset:
    """Convert displacement directions to Cartesian displacements.

    The direction dataset has three nesting levels (first, second, and third
    atoms). The first and second atoms carry a single direction each, while
    the third atom carries a list of directions. The dataset is self-contained:
    every node is a supercell whose forces are to be computed -- F(a1) at the
    first atoms, F(a1+a2) at the second atoms, and F(a1+a2+a3) at the third
    atoms.

    Running ids are assigned level by level (all first atoms, then all second
    atoms, then all third atoms), mirroring the fc3 convention one level
    deeper, so that the supercells are grouped by displacement level.

    """
    lattice = supercell.cell.T
    new_dataset: Fc4Type1DisplacementDataset = {
        "natom": len(supercell),
        "first_atoms": [],
    }

    n_first = len(direction_dataset)
    n_second = sum(len(first_atom["second_atoms"]) for first_atom in direction_dataset)
    first_id = 1
    second_id = n_first + 1
    third_id = n_first + n_second + 1

    for first_atom in direction_dataset:
        atom1 = first_atom["number"]
        disp_cart1 = _direction_to_cartesian(
            first_atom["direction"], lattice, displacement_distance
        )
        new_second_atoms: list[SecondAtomDisplacementFc4WithForces] = []
        for second_atom in first_atom["second_atoms"]:
            atom2 = second_atom["number"]
            disp_cart2 = _direction_to_cartesian(
                second_atom["direction"], lattice, displacement_distance
            )
            new_third_atoms: list[ThirdAtomDisplacementWithForces] = []
            for third_atom in second_atom["third_atoms"]:
                atom3 = third_atom["number"]
                for direction3 in third_atom["directions"]:
                    disp_cart3 = _direction_to_cartesian(
                        direction3, lattice, displacement_distance
                    )
                    new_third_atoms.append(
                        {
                            "number": atom3,
                            "displacement": disp_cart3,
                            "id": third_id,
                        }
                    )
                    third_id += 1
            new_second_atoms.append(
                {
                    "number": atom2,
                    "displacement": disp_cart2,
                    "id": second_id,
                    "third_atoms": new_third_atoms,
                }
            )
            second_id += 1
        new_dataset["first_atoms"].append(
            {
                "number": atom1,
                "displacement": disp_cart1,
                "id": first_id,
                "second_atoms": new_second_atoms,
            }
        )
        first_id += 1

    return new_dataset


def _direction_to_cartesian(
    direction: NDArray[np.double] | list,
    lattice: NDArray[np.double],
    displacement_distance: float,
) -> NDArray[np.double]:
    """Return a Cartesian displacement of the given length from a direction."""
    disp_cart = np.dot(direction, lattice.T)
    disp_cart *= displacement_distance / np.linalg.norm(disp_cart)
    return disp_cart
