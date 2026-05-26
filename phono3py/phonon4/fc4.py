"""4th-order force constants (fc4): tensor algebra and finite-difference solver.

The first half of this module collects the method-independent tensor algebra
for fc4 stored as a full array of shape ``(N, N, N, N, 3, 3, 3, 3)``:
permutation symmetrization, translational invariance (acoustic sum rule),
Cartesian rotation of a rank-4 tensor, and drift diagnostics. These operations
are shared by any fc4 producer (finite-difference or regression such as symfc)
and are used when comparing or post-processing fc4.

The second half is the finite-difference fc4 solver (``get_fc4`` and its
helpers). fc4 is the derivative of fc3 with respect to one more displacement,
so the solver wraps the fc3 machinery in ``phono3py.phonon3.fc3`` one level
deeper: for each first-atom displacement it rebuilds the fc3 of the displaced
reference structure under the reduced (broken) site symmetry, and the
difference from the equilibrium fc3, divided by the displacement, gives fc4.
Only the full-fc4 layout is supported for now; compact-fc4 and a pair-distance
cutoff are deferred.

"""

from __future__ import annotations

import itertools
import logging
import sys
import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.force_constants import (
    get_nsym_list_and_s2pp,
    get_positions_sent_by_rot_inv,
    get_rotated_displacement,
    similarity_transformation,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, compute_all_sg_permutations
from phonopy.structure.symmetry import Symmetry

from phono3py._lang import log_dispatch, resolve_lang
from phono3py.phonon3.displacement_fc3 import (
    get_bond_symmetry,
    get_reduced_site_symmetry,
)
from phono3py.phonon3.fc3 import (
    _get_constrained_fc2,
    _get_delta_fc2,
    _solve_fc3,
    _third_rank_tensor_rotation,
    distribute_fc3,
    set_permutation_symmetry_fc3,
    set_translational_invariance_fc3,
    show_drift_fc3,
)
from phono3py.phonon4.displacement_fc4 import (
    Fc4Type1DisplacementDataset,
    FirstAtomDisplacementFc4WithForces,
)

logger = logging.getLogger(__name__)


def set_permutation_symmetry_fc4(fc4: NDArray[np.double]) -> None:
    """Enforce permutation symmetry on full fc4 in place.

    fc4 is invariant under any simultaneous permutation of its four
    (atom, Cartesian) legs, i.e. of the pairs ``(a, i)``, ``(b, j)``,
    ``(c, k)``, ``(d, l)``. The symmetrized fc4 is the average over the 24
    permutations of these legs.

    """
    fc4_sym = np.zeros_like(fc4)
    for perm in itertools.permutations(range(4)):
        axes = list(perm) + [4 + p for p in perm]
        fc4_sym += fc4.transpose(axes)
    fc4_sym /= 24.0
    fc4[:] = fc4_sym


def set_translational_invariance_fc4(fc4: NDArray[np.double]) -> None:
    """Enforce translational invariance (acoustic sum rule) on fc4 in place.

    For each atom index, the sum of fc4 over that index is driven to zero by
    subtracting the mean along that axis. The four axes are treated
    sequentially, so the sum rules are satisfied approximately and
    simultaneously, consistent with the fc3 treatment.

    """
    for index in range(4):
        _set_translational_invariance_fc4_per_index(fc4, index=index)


def _set_translational_invariance_fc4_per_index(
    fc4: NDArray[np.double], index: int = 0
) -> None:
    fc4 -= fc4.mean(axis=index, keepdims=True)


def set_permutation_symmetry_compact_fc4(
    fc4: NDArray[np.double],
    primitive: Primitive,
    lang: Literal["C", "Rust"] = "Rust",
) -> None:
    """Enforce permutation symmetry on compact fc4 in place.

    Requires the ``phonors`` Rust extension: the 24 leg permutations mix the
    primitive-index and supercell-index axes, so there is no pure-Python path
    (as for compact fc3).
    """
    lang = resolve_lang(lang)
    permutations = primitive.atomic_permutations
    s2pp_map, nsym_list = get_nsym_list_and_s2pp(
        primitive.s2p_map, primitive.p2p_map, permutations
    )
    if lang != "Rust":
        raise RuntimeError(
            "Compact fc4 permutation symmetry requires the phonors Rust "
            "extension; no C or Python fallback is implemented."
        )
    import phonors  # type: ignore[import-untyped]

    log_dispatch(lang, "set_permutation_symmetry_compact_fc4")
    phonors.permutation_symmetry_compact_fc4(
        fc4, permutations, s2pp_map, primitive.p2s_map, nsym_list
    )


def set_translational_invariance_compact_fc4(
    fc4: NDArray[np.double],
    primitive: Primitive,
    lang: Literal["C", "Rust"] = "Rust",
) -> None:
    """Enforce translational invariance (acoustic sum rule) on compact fc4.

    The first axis is the compressed primitive-atom axis, so its sum rule is
    applied after a dim-0/dim-1 transpose (via ``phonors``); the remaining
    supercell axes (1, 2, 3) are handled directly by mean subtraction, mirroring
    ``set_translational_invariance_compact_fc3``.
    """
    lang = resolve_lang(lang)
    permutations = primitive.atomic_permutations
    s2pp_map, nsym_list = get_nsym_list_and_s2pp(
        primitive.s2p_map, primitive.p2p_map, permutations
    )
    if lang != "Rust":
        raise RuntimeError(
            "Compact fc4 translational invariance requires the phonors Rust "
            "extension; no C or Python fallback is implemented."
        )
    import phonors  # type: ignore[import-untyped]

    log_dispatch(lang, "set_translational_invariance_compact_fc4")
    transpose = phonors.transpose_compact_fc4
    p2s_map = primitive.p2s_map
    transpose(fc4, permutations, s2pp_map, p2s_map, nsym_list, 0)  # dim0 <-> dim1
    _set_translational_invariance_fc4_per_index(fc4, index=1)
    transpose(fc4, permutations, s2pp_map, p2s_map, nsym_list, 0)  # dim0 <-> dim1
    _set_translational_invariance_fc4_per_index(fc4, index=1)
    _set_translational_invariance_fc4_per_index(fc4, index=2)
    _set_translational_invariance_fc4_per_index(fc4, index=3)


def fourth_rank_tensor_rotation(
    rot_cart: NDArray[np.double], tensor: NDArray[np.double]
) -> NDArray[np.double]:
    """Rotate a rank-4 Cartesian tensor.

    Returns ``T'[a, b, c, d] = R[a, i] R[b, j] R[c, k] R[d, l] T[i, j, k, l]``.

    """
    return np.einsum(
        "ai,bj,ck,dl,ijkl->abcd", rot_cart, rot_cart, rot_cart, rot_cart, tensor
    )


def get_drift_fc4(fc4: NDArray[np.double]) -> list[float]:
    """Return the maximum absolute translational drift for each atom index.

    The d-th value is the largest absolute value of the sum of fc4 over the
    d-th atom index. Zero values indicate that the acoustic sum rule is
    satisfied.

    """
    drifts = []
    for index in range(4):
        drifts.append(float(np.abs(fc4.sum(axis=index)).max()))
    return drifts


def show_drift_fc4(fc4: NDArray[np.double], name: str = "fc4") -> None:
    """Print the translational drift of fc4 for each atom index."""
    drifts = get_drift_fc4(fc4)
    print(f"Max drift of {name}: " + " ".join(f"{d:f}" for d in drifts))


def get_fc4(
    supercell: PhonopyAtoms,
    disp_dataset: Fc4Type1DisplacementDataset,
    fc3: NDArray[np.double],
    symmetry: Symmetry,
    primitive: Primitive | None = None,
    is_compact_fc: bool = False,
    is_translational_symmetry: bool = False,
    is_permutation_symmetry: bool = False,
    verbose: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.double]:
    """Calculate fc4 by finite difference of fc3.

    fc4 is the derivative of fc3 with respect to one more atomic displacement.
    For each independent first-atom displacement, the fc3 of the displaced
    reference structure (the "constrained fc3") is rebuilt under the site
    symmetry broken by that displacement, and ``constrained_fc3 - fc3`` divided
    by the displacement gives a column of fc4. The result is then expanded to
    all atoms by the crystal symmetry.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell.
    disp_dataset : Fc4Type1DisplacementDataset
        Self-contained type-1 fc4 displacement dataset with forces computed at
        every level (first, second, and third atoms).
    fc3 : ndarray
        Equilibrium fc3 in the full layout, shape
        ``(N, N, N, 3, 3, 3)``. Compact fc3 is not supported yet.
    symmetry : Symmetry
        Symmetry of the supercell.
    primitive : Primitive, optional
        Primitive cell. Required when ``is_compact_fc`` is True. Default None.
    is_compact_fc : bool, optional
        fc4 shape::

            True:  (n_patom, N, N, N, 3, 3, 3, 3)
            False: (N, N, N, N, 3, 3, 3, 3)

        The compact layout requires the ``phonors`` Rust extension for its
        symmetrization. Default is False.
    is_translational_symmetry : bool, optional
        Enforce the acoustic sum rule. The same symmetrization is applied to
        the equilibrium and constrained fc3 before differencing, and to the
        final fc4. Default is False.
    is_permutation_symmetry : bool, optional
        Enforce permutation symmetry, applied as for
        ``is_translational_symmetry``. Default is False.
    verbose : bool, optional
        Print progress. Default is False.
    lang : {"C", "Rust"}, optional
        Backend for the fc3 sub-computations. Default is "Rust".

    Returns
    -------
    ndarray
        fc4, shape ``(N, N, N, N, 3, 3, 3, 3)`` (full) or
        ``(n_patom, N, N, N, 3, 3, 3, 3)`` (compact), dtype=double, order=C.

    """
    warnings.warn(
        "Finite-difference fc4 (4th-order force constants) is an experimental "
        "feature under active development; results, defaults, and the API may "
        "change without notice.",
        stacklevel=2,
    )
    lang = resolve_lang(lang)
    num_atom = len(supercell)
    if fc3.shape[0] != num_atom or fc3.shape[1] != num_atom:
        raise RuntimeError(
            "Finite-difference fc4 requires full fc3 of shape "
            "(N, N, N, 3, 3, 3); compact fc3 is not supported yet."
        )

    # Symmetrize the equilibrium fc3 with the same operations applied to each
    # constrained fc3, so that the difference isolates fc4 instead of carrying
    # symmetrization artifacts.
    fc3_ref = np.array(fc3, dtype="double", order="C")
    if is_translational_symmetry:
        set_translational_invariance_fc3(fc3_ref)
    if is_permutation_symmetry:
        set_permutation_symmetry_fc3(fc3_ref, lang=lang)

    if is_compact_fc:
        if primitive is None:
            raise RuntimeError("Primitive cell is required for compact fc4.")
        num_patom = len(primitive)
        fc4 = np.zeros(
            (num_patom, num_atom, num_atom, num_atom, 3, 3, 3, 3),
            dtype="double",
            order="C",
        )
    else:
        fc4 = np.zeros(
            (num_atom, num_atom, num_atom, num_atom, 3, 3, 3, 3),
            dtype="double",
            order="C",
        )
    _get_fc4_least_atoms(
        fc4,
        supercell,
        disp_dataset,
        fc3_ref,
        symmetry,
        is_translational_symmetry,
        is_permutation_symmetry,
        primitive=primitive,
        is_compact_fc=is_compact_fc,
        verbose=verbose,
        lang=lang,
    )

    if verbose:
        print("Expanding fc4.")

    first_disp_atoms = np.unique([x["number"] for x in disp_dataset["first_atoms"]])
    rotations = symmetry.symmetry_operations["rotations"]
    permutations = symmetry.atomic_permutations
    lattice = supercell.cell.T
    if is_compact_fc:
        assert primitive is not None
        s2p_map = primitive.s2p_map
        p2p_map = primitive.p2p_map
        s2compact = np.array([p2p_map[i] for i in s2p_map], dtype="int64")
        target_atoms = [i for i in primitive.p2s_map if i not in first_disp_atoms]
    else:
        s2compact = np.arange(num_atom, dtype="int64")
        target_atoms = [i for i in range(num_atom) if i not in first_disp_atoms]
    distribute_fc4(
        fc4,
        first_disp_atoms,
        target_atoms,
        lattice,
        rotations,
        permutations,
        s2compact,
        verbose=verbose,
        lang=lang,
    )

    if is_translational_symmetry:
        if is_compact_fc:
            assert primitive is not None
            set_translational_invariance_compact_fc4(fc4, primitive, lang=lang)
        else:
            set_translational_invariance_fc4(fc4)
    if is_permutation_symmetry:
        if is_compact_fc:
            assert primitive is not None
            set_permutation_symmetry_compact_fc4(fc4, primitive, lang=lang)
        else:
            set_permutation_symmetry_fc4(fc4)

    return fc4


def distribute_fc4(
    fc4: NDArray[np.double],
    first_disp_atoms: NDArray[np.int64],
    target_atoms: list[int],
    lattice: NDArray[np.double],
    rotations: NDArray[np.int64],
    permutations: NDArray[np.int64],
    s2compact: NDArray[np.int64],
    verbose: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
) -> None:
    """Distribute fc4 to symmetrically equivalent first-index atoms (in place).

    This mirrors :func:`phono3py.phonon3.fc3.distribute_fc3` one rank higher:
    for each ``i_target`` it finds an ``i_done`` in ``first_disp_atoms`` and a
    symmetry operation mapping ``i_target -> i_done`` (preferring a pure
    translation to minimise round-off), then writes::

        fc4[i_target, j, k, l]
            = R_inv . fc4[i_done, map[j], map[k], map[l]]

    The compact layout is supported via ``s2compact``: a supercell first-index
    ``i`` is redirected to row ``s2compact[i]``. For full fc4, pass
    ``s2compact = np.arange(n_satom)``. With ``lang="Rust"`` the per-target
    rank-4 rotation is done by the ``phonors`` kernel; otherwise it falls back
    to a Python ``np.einsum`` over the whole ``fc4[i_done]`` block.

    """
    lang = resolve_lang(lang)
    identity = np.eye(3, dtype=int)
    pure_trans_indices = [i for i, r in enumerate(rotations) if (r == identity).all()]

    for i_target in target_atoms:
        rot_indices: NDArray[np.int64] = np.array([], dtype="int64")
        for i_done in first_disp_atoms:
            rot_indices = np.where(permutations[:, i_target] == i_done)[0]
            if len(rot_indices) > 0:
                rot_index = rot_indices[0]
                for rot_i in rot_indices:
                    if rot_i in pure_trans_indices:
                        rot_index = rot_i
                        break
                atom_mapping = np.array(permutations[rot_index], dtype="int64")
                rot = rotations[rot_index]
                rot_cart_inv = np.array(
                    similarity_transformation(lattice, rot).T,
                    dtype="double",
                    order="C",
                )
                break

        if len(rot_indices) == 0:
            print("Position or symmetry may be wrong.")
            raise RuntimeError

        if verbose > 2:
            print("    [ %d, x, x, x ] to [ %d, x, x, x ]" % (i_done + 1, i_target + 1))
            sys.stdout.flush()

        if lang == "Rust":
            import phonors  # type: ignore[import-untyped]

            phonors.distribute_fc4(
                fc4,
                int(s2compact[i_target]),
                int(s2compact[i_done]),
                atom_mapping,
                rot_cart_inv,
            )
            continue

        src = fc4[s2compact[i_done]][np.ix_(atom_mapping, atom_mapping, atom_mapping)]
        fc4[s2compact[i_target]] = np.einsum(
            "ai,bj,ck,dl,xyzijkl->xyzabcd",
            rot_cart_inv,
            rot_cart_inv,
            rot_cart_inv,
            rot_cart_inv,
            src,
        )


def _get_fc4_least_atoms(
    fc4: NDArray[np.double],
    supercell: PhonopyAtoms,
    disp_dataset: Fc4Type1DisplacementDataset,
    fc3: NDArray[np.double],
    symmetry: Symmetry,
    is_translational_symmetry: bool,
    is_permutation_symmetry: bool,
    primitive: Primitive | None = None,
    is_compact_fc: bool = False,
    verbose: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
) -> None:
    symprec = symmetry.tolerance
    unique_first_atom_nums = np.unique(
        [x["number"] for x in disp_dataset["first_atoms"]]
    )
    for first_atom_num in unique_first_atom_nums:
        if is_compact_fc:
            assert primitive is not None
            if first_atom_num != primitive.s2p_map[first_atom_num]:
                raise RuntimeError(
                    "First displaced atoms must be in the primitive cell for "
                    "compact fc4."
                )
            fc4_index = primitive.p2p_map[primitive.s2p_map[first_atom_num]]
        else:
            fc4_index = first_atom_num
        _get_fc4_one_atom(
            fc4,
            fc4_index,
            supercell,
            disp_dataset,
            fc3,
            first_atom_num,
            symmetry.get_site_symmetry(first_atom_num),
            is_translational_symmetry,
            is_permutation_symmetry,
            symprec,
            verbose=verbose,
            lang=lang,
        )


def _get_fc4_one_atom(
    fc4: NDArray[np.double],
    fc4_index: int,
    supercell: PhonopyAtoms,
    disp_dataset: Fc4Type1DisplacementDataset,
    fc3: NDArray[np.double],
    first_atom_num: int,
    site_symmetry: NDArray[np.int64],
    is_translational_symmetry: bool,
    is_permutation_symmetry: bool,
    symprec: float,
    verbose: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
) -> None:
    displacements_first = []
    delta_fc3s = []
    for dataset_first_atom in disp_dataset["first_atoms"]:
        if first_atom_num != dataset_first_atom["number"]:
            continue
        displacements_first.append(dataset_first_atom["displacement"])
        direction = np.dot(
            dataset_first_atom["displacement"], np.linalg.inv(supercell.cell)
        )
        reduced_site_sym = get_reduced_site_symmetry(site_symmetry, direction, symprec)

        if verbose:
            print(
                "Solving fc4[ %d, x, x, x ] with %d displacement(s)."
                % (first_atom_num + 1, len(displacements_first))
            )
            sys.stdout.flush()

        delta_fc3s.append(
            _get_delta_fc3(
                dataset_first_atom,
                fc3,
                supercell,
                reduced_site_sym,
                is_translational_symmetry,
                is_permutation_symmetry,
                symprec,
                verbose=verbose,
                lang=lang,
            )
        )

    fc4[fc4_index] = _solve_fc4(
        first_atom_num,
        supercell,
        site_symmetry,
        displacements_first,
        np.array(delta_fc3s, dtype="double", order="C"),
        symprec,
        lang=lang,
    )


def _get_delta_fc3(
    dataset_first_atom: FirstAtomDisplacementFc4WithForces,
    fc3: NDArray[np.double],
    supercell: PhonopyAtoms,
    reduced_site_sym: NDArray[np.int64],
    is_translational_symmetry: bool,
    is_permutation_symmetry: bool,
    symprec: float,
    verbose: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.double]:
    constrained_fc3 = _get_constrained_fc3(
        supercell, dataset_first_atom, reduced_site_sym, symprec, lang=lang
    )
    # Apply the same symmetrization as the equilibrium fc3 (see get_fc4) before
    # differencing.
    if is_translational_symmetry:
        set_translational_invariance_fc3(constrained_fc3)
    if is_permutation_symmetry:
        set_permutation_symmetry_fc3(constrained_fc3, lang=lang)
    if verbose:
        show_drift_fc3(constrained_fc3, name="constrained fc3")
    return constrained_fc3 - fc3


def _get_constrained_fc3(
    supercell: PhonopyAtoms,
    dataset_first_atom: FirstAtomDisplacementFc4WithForces,
    reduced_site_sym: NDArray[np.int64],
    symprec: float,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.double]:
    """Return the fc3 of the structure with Atom 1 displaced.

    The Atom-1-displaced supercell is treated as the reference of an inner fc3
    calculation, evaluated under the site symmetry broken by the Atom-1
    displacement (``reduced_site_sym``). Its own equilibrium fc2 is the
    constrained fc2 of the Atom-1-displaced structure (baseline forces
    ``F(a1)``), and the second- and third-atom displacements provide the fc3
    finite differences just as in ``phono3py.phonon3.fc3``.

    """
    num_atom = len(supercell)
    lattice = supercell.cell.T
    positions = supercell.scaled_positions
    atom1 = dataset_first_atom["number"]
    forces1 = dataset_first_atom["forces"]
    second_atoms = dataset_first_atom["second_atoms"]

    # Inner equilibrium fc2: fc2 of the Atom-1-displaced reference structure.
    fc2_with_one_disp = _get_constrained_fc2(
        supercell,
        second_atoms,
        atom1,
        forces1,
        reduced_site_sym,
        symprec,
        lang=lang,
    )

    constrained_fc3 = np.zeros(
        (num_atom, num_atom, num_atom, 3, 3, 3), dtype="double", order="C"
    )
    atom_list = np.unique([x["number"] for x in second_atoms])
    for atom2 in atom_list:
        bond_sym = get_bond_symmetry(
            reduced_site_sym, lattice, positions, atom1, atom2, symprec
        )
        disps2 = []
        delta_fc2s = []
        for second_atom in second_atoms:
            if atom2 != second_atom["number"]:
                continue
            disps2.append(second_atom["displacement"])
            direction = np.dot(
                second_atom["displacement"], np.linalg.inv(supercell.cell)
            )
            reduced_bond_sym = get_reduced_site_symmetry(bond_sym, direction, symprec)
            delta_fc2s.append(
                _get_delta_fc2(
                    second_atom["third_atoms"],
                    atom2,
                    second_atom["forces"],
                    fc2_with_one_disp,
                    supercell,
                    reduced_bond_sym,
                    symprec,
                    lang=lang,
                )
            )
        constrained_fc3[atom2] = _solve_fc3(
            atom2,
            supercell,
            bond_sym,
            disps2,
            np.array(delta_fc2s, dtype="double", order="C"),
            symprec,
            lang=lang,
        )

    # Expand to all atoms under the reduced site symmetry (Atom 1 at origin).
    positions_shifted = supercell.scaled_positions - positions[atom1]
    rotations = np.array(reduced_site_sym, dtype="int64", order="C")
    translations = np.zeros((len(reduced_site_sym), 3), dtype="double", order="C")
    permutations = compute_all_sg_permutations(
        positions_shifted, rotations, translations, lattice, symprec, lang=lang
    )
    target_atoms = [i for i in range(num_atom) if i not in atom_list]
    distribute_fc3(
        constrained_fc3,
        atom_list,
        target_atoms,
        lattice,
        rotations,
        permutations,
        np.arange(num_atom, dtype="int64"),
        lang=lang,
    )
    return constrained_fc3


def _solve_fc4(
    first_atom_num: int,
    supercell: PhonopyAtoms,
    site_symmetry: NDArray[np.int64],
    displacements_first: list[NDArray[np.double]],
    delta_fc3s: NDArray[np.double],
    symprec: float,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.double]:
    """Solve fc4[first_atom_num] from the delta fc3 of each first displacement.

    The first-atom displacements are expanded by the site symmetry and a
    pseudo-inverse maps the rotated delta fc3s onto the displacement derivative,
    giving ``fc4[first_atom_num, i, j, k]`` for every atom triple.

    """
    lattice = supercell.cell.T
    site_sym_cart = np.array(
        [similarity_transformation(lattice, sym) for sym in site_symmetry],
        dtype="double",
        order="C",
    )
    num_atom = len(supercell)
    positions = supercell.scaled_positions
    positions = positions - positions[first_atom_num]
    rot_map_syms = get_positions_sent_by_rot_inv(
        lattice, positions, site_symmetry, symprec, lang=lang
    )
    rot_map_syms = np.array(rot_map_syms, dtype="int64", order="C")
    rot_disps = get_rotated_displacement(displacements_first, site_sym_cart)  # type: ignore[arg-type]
    inv_U = np.array(np.linalg.pinv(rot_disps), dtype="double", order="C")

    fc4_first = np.zeros(
        (num_atom, num_atom, num_atom, 3, 3, 3, 3), dtype="double", order="C"
    )

    if lang == "Rust":
        import phonors  # type: ignore[import-untyped]

        phonors.rotate_delta_fc3s(
            fc4_first, delta_fc3s, inv_U, site_sym_cart, rot_map_syms
        )
        return fc4_first

    for i, j, k in np.ndindex(num_atom, num_atom, num_atom):
        fc4_first[i, j, k] = np.dot(
            inv_U,
            _get_rotated_fc3s(i, j, k, delta_fc3s, rot_map_syms, site_sym_cart),
        ).reshape(3, 3, 3, 3)
    return fc4_first


def _get_rotated_fc3s(
    i: int,
    j: int,
    k: int,
    fc3s: NDArray[np.double],
    rot_map_syms: NDArray[np.int64],
    site_sym_cart: NDArray[np.double],
) -> NDArray[np.double]:
    rotated_fc3s = []
    for fc3 in fc3s:
        for sym, map_sym in zip(site_sym_cart, rot_map_syms, strict=True):
            fc3_rot = fc3[map_sym[i], map_sym[j], map_sym[k]]
            rotated_fc3s.append(_third_rank_tensor_rotation(sym, fc3_rot))
    return np.reshape(rotated_fc3s, (-1, 27))
