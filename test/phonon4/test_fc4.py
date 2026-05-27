"""Tests for the experimental fc4 (phono3py.phonon4).

These are self-contained (no MLP / no external force data): the finite-difference
solver is exercised with random forces and a random equilibrium fc3, which is
enough to check the machinery (compact == full on primitive rows, permutation
symmetry, acoustic sum rule) since the solver is linear in the forces. Physical
validation against brute-force / symfc lives in ``tools/fc4_validate_mlp.py``.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy.api_phonopy import set_data_to_phonopy_yaml
from phonopy.structure.atoms import PhonopyAtoms

from phono3py import Phono3py
from phono3py.phonon4.dataset import (
    count_supercells_fc4,
    get_displacements_and_forces_fc4,
    set_forces_in_dataset_fc4,
)
from phono3py.phonon4.displacement_fc4 import (
    Fc4Type1DisplacementDataset,
    get_fourth_order_displacements,
)
from phono3py.phonon4.fc4 import (
    fourth_rank_tensor_rotation,
    get_drift_fc4,
    get_fc4,
    set_permutation_symmetry_fc4,
    set_translational_invariance_fc4,
)
from phono3py.phonon4.file_IO import (
    parse_FORCES_FC4,
    read_fc4_from_hdf5,
    write_fc4_to_hdf5,
    write_FORCES_FC4,
)
from phono3py.phonon4.phono4py_yaml import Phono4pyYaml
from phono3py.phonon4.real_to_reciprocal import RealToReciprocalFc4

pytest.importorskip("phonors")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _diamond() -> PhonopyAtoms:
    """Return the 8-atom conventional cubic cell of diamond (Fd-3m)."""
    a = 3.5727
    scaled_positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
    return PhonopyAtoms(
        cell=np.eye(3) * a, symbols=["C"] * 8, scaled_positions=scaled_positions
    )


@pytest.fixture(scope="module")
def ph3() -> Phono3py:
    """Phono3py for the 8-atom diamond (supercell = unit cell)."""
    return Phono3py(
        _diamond(),
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="F",
        log_level=0,
    )


@pytest.fixture(scope="module")
def dataset(ph3: Phono3py) -> Fc4Type1DisplacementDataset:
    """fc4 displacement dataset with reproducible random forces."""
    ds = get_fourth_order_displacements(ph3.supercell, ph3.symmetry, 0.03)
    rng = np.random.default_rng(0)
    ncells = count_supercells_fc4(ds)
    forces = rng.standard_normal((ncells, len(ph3.supercell), 3))
    set_forces_in_dataset_fc4(ds, forces)
    return ds


def _random_full_fc3(natom: int) -> NDArray[np.double]:
    rng = np.random.default_rng(1)
    return rng.standard_normal((natom, natom, natom, 3, 3, 3))


# --- tensor algebra (pure, no forces) ------------------------------------


def test_permutation_symmetry_fc4_idempotent_and_symmetric() -> None:
    """set_permutation_symmetry_fc4 is idempotent and enforces all 24 perms."""
    n = 2
    rng = np.random.default_rng(0)
    fc4 = rng.standard_normal((n, n, n, n, 3, 3, 3, 3))
    set_permutation_symmetry_fc4(fc4)
    snapshot = fc4.copy()
    set_permutation_symmetry_fc4(fc4)
    np.testing.assert_allclose(fc4, snapshot, atol=1e-13)
    # Check a representative leg permutation (swap legs 0 and 1).
    np.testing.assert_allclose(fc4, fc4.transpose(1, 0, 2, 3, 5, 4, 6, 7), atol=1e-13)


def test_translational_invariance_fc4_zeros_drift() -> None:
    """set_translational_invariance_fc4 drives the per-index drift to zero."""
    n = 3
    rng = np.random.default_rng(0)
    fc4 = rng.standard_normal((n, n, n, n, 3, 3, 3, 3))
    set_translational_invariance_fc4(fc4)
    # After sequential mean subtraction, the last index sum is ~0.
    assert np.abs(fc4.sum(axis=3)).max() < 1e-12


def test_fourth_rank_tensor_rotation_identity() -> None:
    """Rotation by the identity leaves a rank-4 tensor unchanged."""
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((3, 3, 3, 3))
    rotated = fourth_rank_tensor_rotation(np.eye(3), tensor)
    np.testing.assert_allclose(rotated, tensor, atol=1e-13)


# --- displacement dataset ------------------------------------------------


def test_displacements_structure(ph3: Phono3py) -> None:
    """The fc4 dataset has level-grouped, contiguous ids over its 3 levels."""
    ds = get_fourth_order_displacements(ph3.supercell, ph3.symmetry, 0.03)
    assert ds["natom"] == len(ph3.supercell)
    ids: list[int] = []
    for first in ds["first_atoms"]:
        ids.append(first["id"])
        for second in first["second_atoms"]:
            ids.append(second["id"])
            for third in second["third_atoms"]:
                ids.append(third["id"])
    assert sorted(ids) == list(range(1, count_supercells_fc4(ds) + 1))


def test_dataset_forces_roundtrip(dataset: Fc4Type1DisplacementDataset) -> None:
    """Forces set into the dataset come back unchanged (id order)."""
    _, forces = get_displacements_and_forces_fc4(dataset)
    assert forces is not None
    assert forces.shape[0] == count_supercells_fc4(dataset)


# --- solver machinery (random forces + random fc3) -----------------------


def test_solver_compact_equals_full_on_primitive_rows(
    ph3: Phono3py, dataset: Fc4Type1DisplacementDataset
) -> None:
    """Without symmetrization, compact fc4 equals full fc4 on primitive rows."""
    fc3 = _random_full_fc3(len(ph3.supercell))
    full = get_fc4(
        ph3.supercell,
        dataset,
        fc3,
        ph3.symmetry,
        primitive=ph3.primitive,
        is_compact_fc=False,
        is_translational_symmetry=False,
        is_permutation_symmetry=False,
    )
    compact = get_fc4(
        ph3.supercell,
        dataset,
        fc3,
        ph3.symmetry,
        primitive=ph3.primitive,
        is_compact_fc=True,
        is_translational_symmetry=False,
        is_permutation_symmetry=False,
    )
    assert full.shape == (8, 8, 8, 8, 3, 3, 3, 3)
    assert compact.shape == (2, 8, 8, 8, 3, 3, 3, 3)
    np.testing.assert_allclose(compact, full[ph3.primitive.p2s_map], atol=1e-10)


def test_solver_full_symmetrization(
    ph3: Phono3py, dataset: Fc4Type1DisplacementDataset
) -> None:
    """Symmetrized full fc4 is permutation symmetric with near-zero drift."""
    fc3 = _random_full_fc3(len(ph3.supercell))
    fc4 = get_fc4(
        ph3.supercell,
        dataset,
        fc3,
        ph3.symmetry,
        primitive=ph3.primitive,
        is_compact_fc=False,
        is_translational_symmetry=True,
        is_permutation_symmetry=True,
    )
    np.testing.assert_allclose(fc4, fc4.transpose(1, 0, 2, 3, 5, 4, 6, 7), atol=1e-10)
    assert max(get_drift_fc4(fc4)) < 1e-8


# --- file I/O round-trips ------------------------------------------------


def test_fc4_hdf5_roundtrip(tmp_path: Path) -> None:
    """write/read fc4 hdf5 round-trips for full and compact layouts."""
    rng = np.random.default_rng(0)
    full = rng.standard_normal((4, 4, 4, 4, 3, 3, 3, 3))
    fn = tmp_path / "fc4.hdf5"
    write_fc4_to_hdf5(full, filename=fn)
    np.testing.assert_array_equal(read_fc4_from_hdf5(fn), full)

    compact = rng.standard_normal((2, 8, 8, 8, 3, 3, 3, 3))
    p2s = np.array([0, 4], dtype="int64")
    fn2 = tmp_path / "fc4c.hdf5"
    write_fc4_to_hdf5(compact, filename=fn2, p2s_map=p2s)
    np.testing.assert_array_equal(read_fc4_from_hdf5(fn2, p2s_map=p2s), compact)
    with pytest.raises(RuntimeError):
        read_fc4_from_hdf5(fn2, p2s_map=np.array([0, 5], dtype="int64"))


def test_FORCES_FC4_roundtrip(
    ph3: Phono3py, dataset: Fc4Type1DisplacementDataset, tmp_path: Path
) -> None:
    """write/parse FORCES_FC4 preserves the forces (id order)."""
    _, forces = get_displacements_and_forces_fc4(dataset)
    fn = tmp_path / "FORCES_FC4"
    write_FORCES_FC4(dataset, filename=fn)
    ds2 = get_fourth_order_displacements(ph3.supercell, ph3.symmetry, 0.03)
    parse_FORCES_FC4(ds2, filename=fn)
    _, forces2 = get_displacements_and_forces_fc4(ds2)
    assert forces is not None
    assert forces2 is not None
    np.testing.assert_allclose(forces2, forces, atol=1e-8)


def test_phono4py_disp_yaml_roundtrip(ph3: Phono3py, tmp_path: Path) -> None:
    """phono4py_disp.yaml round-trips the cell and the fc4 dataset."""
    ds = get_fourth_order_displacements(ph3.supercell, ph3.symmetry, 0.03)
    yaml = Phono4pyYaml(settings={"force_sets": False})
    set_data_to_phonopy_yaml(yaml, ph3)  # type: ignore[arg-type]  # Phono3py, not Phonopy
    yaml.dataset_fc4 = ds
    fn = tmp_path / "phono4py_disp.yaml"
    with open(fn, "w") as w:
        w.write(str(yaml))

    loaded = Phono4pyYaml()
    loaded.read(fn)
    assert loaded.dataset_fc4 is not None
    assert count_supercells_fc4(loaded.dataset_fc4) == count_supercells_fc4(ds)
    disp1, _ = get_displacements_and_forces_fc4(ds)
    disp2, _ = get_displacements_and_forces_fc4(loaded.dataset_fc4)
    np.testing.assert_allclose(disp2, disp1, atol=1e-12)


# --- reciprocal space ----------------------------------------------------


def test_real_to_reciprocal_gamma(ph3: Phono3py) -> None:
    """At Gamma the reciprocal fc4 is real and equals the supercell sum."""
    n_satom = len(ph3.supercell)
    rng = np.random.default_rng(0)
    fc4 = rng.standard_normal((n_satom, n_satom, n_satom, n_satom, 3, 3, 3, 3))
    r2r = RealToReciprocalFc4(fc4, ph3.primitive, np.array([2, 2, 2]))
    rec = r2r.run(np.zeros((4, 3), dtype=int))

    n_patom = len(ph3.primitive)
    assert rec.shape == (n_patom, n_patom, n_patom, n_patom, 3, 3, 3, 3)
    assert np.abs(rec.imag).max() < 1e-10

    p2s = ph3.primitive.p2s_map
    s2p = ph3.primitive.s2p_map
    manual = np.zeros((3, 3, 3, 3))
    for j in range(n_satom):
        if s2p[j] != p2s[0]:
            continue
        for k in range(n_satom):
            if s2p[k] != p2s[0]:
                continue
            for ll in range(n_satom):
                if s2p[ll] != p2s[0]:
                    continue
                manual += fc4[p2s[0], j, k, ll]
    np.testing.assert_allclose(rec[0, 0, 0, 0].real, manual, atol=1e-10)
