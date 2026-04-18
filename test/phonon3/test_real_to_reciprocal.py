"""Regression tests comparing the Rust real_to_reciprocal transform
to a pure-Python reference with C-matching pre-phase semantics."""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.real_to_reciprocal import run_real_to_reciprocal_rust

pytest.importorskip("phono3py_rs")


def _make_itr(ph3: Phono3py, mesh=(4, 4, 4)) -> Interaction:
    ph3.mesh_numbers = mesh
    assert ph3.grid is not None
    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        make_r0_average=False,
        cutoff_frequency=1e-4,
    )
    itr.init_dynamical_matrix(ph3.fc2, ph3.phonon_supercell, ph3.phonon_primitive)
    return itr


def _python_reference(fc3, primitive, q_vecs):
    """Pure-Python reference mirroring c/real_to_reciprocal.c semantics.

    Computes fc3_reciprocal in atom-first layout
    (num_patom, num_patom, num_patom, 3, 3, 3) for ``make_r0_average=False``
    and no fc3_nonzero / all_shortest skipping (full fc3 assumed)."""
    svecs, multi = primitive.get_smallest_vectors()
    p2s = np.asarray(primitive.p2s_map, dtype="int64")
    s2p = np.asarray(primitive.s2p_map, dtype="int64")
    num_patom = len(p2s)
    num_satom = len(s2p)

    def phase_factor(q, satom, patom):
        count = multi[satom, patom, 0]
        start = multi[satom, patom, 1]
        vs = svecs[start : start + count]
        return np.exp(2j * np.pi * (vs @ q)).sum() / count

    pre_phase = np.empty(num_patom, dtype="complex128")
    for i in range(num_patom):
        s = svecs[multi[p2s[i], 0, 1]]
        pre_phase[i] = np.exp(2j * np.pi * (s @ (q_vecs[0] + q_vecs[1] + q_vecs[2])))

    pf1 = np.empty((num_patom, num_satom), dtype="complex128")
    pf2 = np.empty((num_patom, num_satom), dtype="complex128")
    for i in range(num_patom):
        for j in range(num_satom):
            pf1[i, j] = phase_factor(q_vecs[1], j, i)
            pf2[i, j] = phase_factor(q_vecs[2], j, i)

    out = np.zeros((num_patom, num_patom, num_patom, 3, 3, 3), dtype="complex128")
    for pi0 in range(num_patom):
        i = p2s[pi0]
        for pi1 in range(num_patom):
            for pi2 in range(num_patom):
                block = np.zeros((3, 3, 3), dtype="complex128")
                for j in range(num_satom):
                    if s2p[j] != p2s[pi1]:
                        continue
                    for k in range(num_satom):
                        if s2p[k] != p2s[pi2]:
                            continue
                        block += fc3[i, j, k] * pf1[pi0, j] * pf2[pi0, k]
                out[pi0, pi1, pi2] = block * pre_phase[pi0]
    return out


def test_real_to_reciprocal_rust_vs_python_si(si_pbesol_no_r0avg: Phono3py):
    """Rust fc3_reciprocal matches a C-semantics Python reference."""
    itr = _make_itr(si_pbesol_no_r0avg, mesh=(4, 4, 4))
    itr.set_grid_point(1)
    triplets, _, _, _ = itr.get_triplets_at_q()
    triplet = itr.bz_grid.addresses[triplets[1]]  # pick a non-trivial triplet

    mesh = np.asarray(itr.mesh_numbers, dtype="int64")
    q_vecs = triplet.astype("double") / mesh
    fc3 = itr.fc3
    primitive = itr.primitive

    ref = _python_reference(fc3, primitive, q_vecs)

    num_patom = len(primitive)
    num_satom = len(primitive.s2p_map)
    num_rows = fc3.shape[0]
    all_shortest = np.zeros((num_patom, num_satom, num_satom), dtype="byte")
    nonzero_indices = np.ones((num_rows, num_satom, num_satom), dtype="byte")

    out = run_real_to_reciprocal_rust(
        fc3,
        primitive,
        mesh,
        triplet,
        is_compact_fc3=(num_rows == num_patom),
        make_r0_average=False,
        all_shortest=all_shortest,
        nonzero_indices=nonzero_indices,
    )

    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)
