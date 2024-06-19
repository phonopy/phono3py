"""Tests of displacements.py."""

import numpy as np

import phono3py
from phono3py import Phono3py
from phono3py.phonon3.displacement_fc3 import get_smallest_vector_of_atom_pair

distances_NaCl = [
    0.0000000,
    5.6903015,
    5.6903015,
    8.0473015,
    5.6903015,
    8.0473015,
    8.0473015,
    9.8558913,
    4.0236508,
    6.9691675,
    4.0236508,
    6.9691675,
    4.0236508,
    6.9691675,
    4.0236508,
    6.9691675,
    4.0236508,
    4.0236508,
    6.9691675,
    6.9691675,
    4.0236508,
    4.0236508,
    6.9691675,
    6.9691675,
    4.0236508,
    4.0236508,
    4.0236508,
    4.0236508,
    6.9691675,
    6.9691675,
    6.9691675,
    6.9691675,
    4.9279456,
    4.9279456,
    4.9279456,
    4.9279456,
    4.9279456,
    4.9279456,
    4.9279456,
    4.9279456,
    2.8451507,
    2.8451507,
    6.3619505,
    6.3619505,
    6.3619505,
    6.3619505,
    8.5354522,
    8.5354522,
    2.8451507,
    6.3619505,
    2.8451507,
    6.3619505,
    6.3619505,
    8.5354522,
    6.3619505,
    8.5354522,
    2.8451507,
    6.3619505,
    6.3619505,
    8.5354522,
    2.8451507,
    6.3619505,
    6.3619505,
    8.5354522,
]


def test_duplicates_agno2(agno2_cell):
    """Test duplicated pairs of displacements."""
    ph3 = phono3py.load(unitcell=agno2_cell, supercell_matrix=[1, 1, 1])
    ph3.generate_displacements()
    duplicates_ref = [
        [106, 22],
        [220, 80],
        [252, 81],
        [221, 96],
        [253, 97],
        [290, 142],
        [348, 244],
        [364, 245],
        [349, 276],
        [365, 277],
        [119, 0],
        [261, 0],
        [229, 0],
        [260, 0],
        [228, 0],
    ]
    np.testing.assert_equal(duplicates_ref, ph3.dataset["duplicates"])


def test_nacl_pbe(nacl_pbe: Phono3py):
    """Test generated displacements and duplicates."""
    ph3 = Phono3py(
        nacl_pbe.unitcell,
        supercell_matrix=nacl_pbe.supercell_matrix,
        primitive_matrix=nacl_pbe.primitive_matrix,
    )
    ph3.generate_displacements()
    duplicates_ref = [[77, 41]]
    ph3.dataset["duplicates"]
    np.testing.assert_equal(duplicates_ref, ph3.dataset["duplicates"])

    pairs_ref = [
        0,
        0,
        0,
        1,
        0,
        2,
        0,
        3,
        0,
        6,
        0,
        7,
        0,
        8,
        0,
        9,
        0,
        16,
        0,
        17,
        0,
        18,
        0,
        19,
        0,
        32,
        0,
        33,
        0,
        40,
        0,
        41,
        0,
        42,
        0,
        43,
        0,
        46,
        0,
        47,
        0,
        48,
        0,
        49,
        0,
        52,
        0,
        53,
        32,
        0,
        32,
        1,
        32,
        8,
        32,
        9,
        32,
        10,
        32,
        11,
        32,
        14,
        32,
        15,
        32,
        16,
        32,
        17,
        32,
        20,
        32,
        21,
        32,
        32,
        32,
        33,
        32,
        34,
        32,
        35,
        32,
        38,
        32,
        39,
        32,
        40,
        32,
        41,
        32,
        48,
        32,
        49,
        32,
        50,
        32,
        51,
    ]
    pairs = []
    for first_atoms in ph3.dataset["first_atoms"]:
        n1 = first_atoms["number"]
        n2s = np.unique(
            [second_atoms["number"] for second_atoms in first_atoms["second_atoms"]]
        )
        pairs += [[n1, n2] for n2 in n2s]
    # print("".join(["%d, " % i for i in np.array(pairs).ravel()]))

    np.testing.assert_equal(pairs_ref, np.array(pairs).ravel())


def test_get_smallest_vector_of_atom_pair(nacl_pbe):
    """Test get_smallest_vector_of_atom_pair."""
    ph3 = nacl_pbe
    distances = []
    for i in range(len(ph3.supercell)):
        vec = get_smallest_vector_of_atom_pair(i, 0, ph3.supercell, 1e-5)
        if vec.ndim == 2:
            vec = vec[0]
        distances.append(np.linalg.norm(np.dot(vec, ph3.supercell.cell)))

    # _show(distances)
    np.testing.assert_allclose(distances_NaCl, distances, rtol=0, atol=1e-6)


def _show(vals):
    for i, v in enumerate(vals):
        print("%.7f, " % v, end="")
        if (i + 1) % 5 == 0:
            print("")
