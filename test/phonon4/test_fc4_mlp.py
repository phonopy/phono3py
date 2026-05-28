"""Physical regression test for fc4 against brute-force MLP-force derivatives.

Unlike ``test_fc4.py`` (self-contained, random forces), this drives the
finite-difference fc4 solver end to end with a pypolymlp potential on the 8-atom
conventional diamond cell and checks that the dominant fc4 elements match a
brute-force ground truth computed directly from the MLP forces
(``fc4[p,q,r,s] = -d^3 F_p / dx_q dx_r dx_s``). The ground truth shares no code
with the force-constants machinery, so it is an independent check.

Test data (compressed):
- ``phonopy_params-C-diamond.yaml.xz`` : 8-atom diamond cell + supercell matrix.
- ``polymlp-C-diamond.yaml.xz``        : carbon pypolymlp potential.

See ``tools/fc4_validate_mlp.py`` and ``tools/fc4_validation_report`` for the
full validation (the observed FD-vs-brute error is ~0.2%).
"""

from __future__ import annotations

import itertools
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms

from phono3py import Phono3py
from phono3py.phonon4.displacement_fc4 import (
    Fc4Type1DisplacementDataset,
    FirstAtomDisplacementFc4WithForces,
    SecondAtomDisplacementFc4WithForces,
    ThirdAtomDisplacementWithForces,
    get_fourth_order_displacements,
)
from phono3py.phonon4.fc4 import get_drift_fc4, get_fc4

if TYPE_CHECKING:
    from phonopy.interface.mlp import PhonopyMLP

pytest.importorskip("phonors")
pytest.importorskip("pypolymlp")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DATA_DIR = pathlib.Path(__file__).parent
DISTANCE = 0.03
BRUTE_H = 0.03

# Four (atom, Cartesian) legs of an fc4 element.
Legs = tuple[tuple[int, int], ...]
# A displacement node at any of the three levels (each carries a "forces" key).
_Node = (
    FirstAtomDisplacementFc4WithForces
    | SecondAtomDisplacementFc4WithForces
    | ThirdAtomDisplacementWithForces
)


def _evaluate_forces(mlp: PhonopyMLP, cells: list[PhonopyAtoms]) -> NDArray[np.double]:
    _, forces, _ = mlp.evaluate(cells)
    return np.array(forces, dtype="double")


def _displaced_cell(supercell: PhonopyAtoms, disp: NDArray[np.double]) -> PhonopyAtoms:
    return PhonopyAtoms(
        cell=supercell.cell,
        symbols=supercell.symbols,
        positions=supercell.positions + disp,
    )


def _fill_fc4_forces(
    dataset: Fc4Type1DisplacementDataset, supercell: PhonopyAtoms, mlp: PhonopyMLP
) -> None:
    """Evaluate MLP forces for every (cumulatively displaced) supercell, in place."""
    natom = len(supercell)
    cells: list[PhonopyAtoms] = []
    refs: list[_Node] = []
    for first in dataset["first_atoms"]:
        d1 = np.zeros((natom, 3))
        d1[first["number"]] += first["displacement"]
        cells.append(_displaced_cell(supercell, d1))
        refs.append(first)
        for second in first["second_atoms"]:
            d2 = d1.copy()
            d2[second["number"]] += second["displacement"]
            cells.append(_displaced_cell(supercell, d2))
            refs.append(second)
            for third in second["third_atoms"]:
                d3 = d2.copy()
                d3[third["number"]] += third["displacement"]
                cells.append(_displaced_cell(supercell, d3))
                refs.append(third)
    forces = _evaluate_forces(mlp, cells)
    for ref, force in zip(refs, forces, strict=True):
        ref["forces"] = force


def _brute_force_element(
    mlp: PhonopyMLP, supercell: PhonopyAtoms, legs: Legs, h: float
) -> float:
    """One fc4 element by direct 3rd finite difference of the MLP forces."""
    natom = len(supercell)
    out_leg, diff_legs = legs[0], legs[1:]
    cells: list[PhonopyAtoms] = []
    sign_sets: list[tuple[float, ...]] = []
    for signs in itertools.product((1.0, -1.0), repeat=3):
        disp = np.zeros((natom, 3))
        for (atom, cart), sign in zip(diff_legs, signs, strict=True):
            disp[atom, cart] += sign * h
        cells.append(_displaced_cell(supercell, disp))
        sign_sets.append(signs)
    forces = _evaluate_forces(mlp, cells)
    total = sum(
        s[0] * s[1] * s[2] * f[out_leg[0], out_leg[1]]
        for s, f in zip(sign_sets, forces, strict=True)
    )
    return -total / (2.0 * h) ** 3


def _dominant_legs(fc4: NDArray[np.double], n: int) -> list[Legs]:
    """Largest-|fc4| elements, deduplicated by value (permutation-equivalent)."""
    order = np.argsort(np.abs(fc4), axis=None)[::-1]
    seen: set[str] = set()
    picks: list[Legs] = []
    for flat in order:
        a1, a2, a3, a4, c1, c2, c3, c4 = (
            int(i) for i in np.unravel_index(flat, fc4.shape)
        )
        key = f"{float(fc4[a1, a2, a3, a4, c1, c2, c3, c4]):.4g}"
        if key in seen:
            continue
        seen.add(key)
        picks.append(((a1, c1), (a2, c2), (a3, c3), (a4, c4)))
        if len(picks) >= n:
            break
    return picks


def test_fc4_mlp_matches_brute_force() -> None:
    """Dominant fc4 elements agree with brute-force MLP derivatives (<1%)."""
    from phonopy.interface.mlp import PhonopyMLP
    from phonopy.interface.phonopy_yaml import PhonopyYaml

    pyaml = PhonopyYaml()
    pyaml.read(DATA_DIR / "phonopy_params-C-diamond.yaml.xz")
    assert pyaml.unitcell is not None
    mlp = PhonopyMLP(log_level=0)
    mlp.load(str(DATA_DIR / "polymlp-C-diamond.yaml.xz"))

    ph3 = Phono3py(
        pyaml.unitcell,
        supercell_matrix=pyaml.supercell_matrix,
        primitive_matrix=pyaml.primitive_matrix,
        log_level=0,
    )
    supercell, symmetry = ph3.supercell, ph3.symmetry

    # Equilibrium fc3 (traditional finite difference + MLP forces, full layout).
    ph3.generate_displacements(distance=DISTANCE)
    scells = [c for c in ph3.supercells_with_displacements if c is not None]
    ph3.forces = _evaluate_forces(mlp, scells)
    ph3.produce_fc3(is_compact_fc=False, fc_calculator="traditional")
    fc3 = ph3.fc3
    assert fc3 is not None

    # Finite-difference fc4.
    dataset = get_fourth_order_displacements(supercell, symmetry, DISTANCE)
    _fill_fc4_forces(dataset, supercell, mlp)
    fc4 = get_fc4(
        supercell,
        dataset,
        fc3,
        symmetry,
        is_translational_symmetry=True,
        is_permutation_symmetry=True,
    )

    assert fc4.shape == (8, 8, 8, 8, 3, 3, 3, 3)
    assert max(get_drift_fc4(fc4)) < 1e-6  # acoustic sum rule

    # Dominant elements vs brute-force ground truth.
    for legs in _dominant_legs(fc4, 3):
        idx = tuple(a for a, _ in legs) + tuple(c for _, c in legs)
        brute = _brute_force_element(mlp, supercell, legs, BRUTE_H)
        fd = float(fc4[idx])
        rel = abs(fd - brute) / abs(brute)
        assert rel < 0.01, f"legs={legs} fd={fd:.3f} brute={brute:.3f} rel={rel:.3%}"
