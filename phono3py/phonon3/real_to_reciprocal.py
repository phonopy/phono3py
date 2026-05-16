"""Transform fc3 in real space to reciprocal space."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.cells import Primitive

from phono3py.phonon3.fc3 import compact_fc3_to_full_fc3


def run_real_to_reciprocal_rust(
    fc3: NDArray[np.double],
    primitive: Primitive,
    mesh: NDArray[np.int64],
    triplet: NDArray[np.int64],
    *,
    is_compact_fc3: bool,
    make_r0_average: bool,
    all_shortest: NDArray[np.byte],
    nonzero_indices: NDArray[np.byte],
) -> NDArray[np.cdouble]:
    """Transform fc3 to reciprocal space at a q-triplet using the Rust backend.

    Returns fc3_reciprocal with shape (num_patom, num_patom, num_patom, 3, 3, 3),
    matching the layout used by ``RealToReciprocal``.

    ``triplet`` is the grid-address triplet (3, 3); the fractional q-vectors
    are ``triplet / mesh``.  ``all_shortest`` has shape
    ``(num_patom, num_satom, num_satom)`` and ``nonzero_indices`` has shape
    ``fc3.shape[:3]``, both ``byte``.

    """
    import phonors

    svecs, multi = primitive.get_smallest_vectors()
    p2s = np.asarray(primitive.p2s_map, dtype="int64")
    s2p = np.asarray(primitive.s2p_map, dtype="int64")
    num_patom = len(p2s)

    q_vecs = np.ascontiguousarray(triplet.astype("double") / mesh)
    fc3_reciprocal = np.zeros(
        (num_patom, num_patom, num_patom, 3, 3, 3), dtype="complex128"
    )

    phonors.real_to_reciprocal(
        fc3_reciprocal,
        q_vecs,
        np.ascontiguousarray(fc3, dtype="double"),
        bool(is_compact_fc3),
        np.ascontiguousarray(svecs, dtype="double"),
        np.ascontiguousarray(multi, dtype="int64"),
        p2s,
        s2p,
        bool(make_r0_average),
        np.ascontiguousarray(all_shortest, dtype="byte"),
        np.ascontiguousarray(nonzero_indices, dtype="byte"),
    )
    return fc3_reciprocal


class RealToReciprocal:
    """Transform fc3 in real space to reciprocal space."""

    def __init__(
        self,
        fc3: NDArray[np.double],
        primitive: Primitive,
        mesh: NDArray[np.int64],
        symprec: float = 1e-5,
        make_r0_average: bool = False,
        all_shortest: NDArray[np.byte] | None = None,
    ):
        """Init method.

        ``make_r0_average`` enables the three-leg r0 averaging path.
        ``all_shortest`` (shape ``(num_patom, num_satom, num_satom)``,
        ``byte``) is required when ``make_r0_average`` is True.

        """
        # Only full fc3 is supported in this python implementation.
        if fc3.shape[0] != fc3.shape[1]:
            self._fc3 = compact_fc3_to_full_fc3(primitive, fc3)
        else:
            self._fc3 = fc3
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec
        self._make_r0_average = make_r0_average
        if make_r0_average and all_shortest is None:
            raise ValueError("all_shortest is required when make_r0_average=True")
        self._all_shortest = all_shortest

        self._p2s_map = primitive.p2s_map
        self._s2p_map = primitive.s2p_map
        # Reduce supercell atom index to primitive index
        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        self._fc3_reciprocal: NDArray[np.cdouble] | None = None

    def run(self, triplet: NDArray[np.int64]) -> None:
        """Run at each triplet of q-vectors."""
        self._triplet = triplet
        num_patom = len(self._primitive)
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_reciprocal = np.zeros(
            (num_patom, num_patom, num_patom, 3, 3, 3), dtype=dtype
        )
        if self._make_r0_average:
            self._real_to_reciprocal_r0_average()
        else:
            self._real_to_reciprocal()

    def get_fc3_reciprocal(self) -> NDArray[np.cdouble] | None:
        """Return fc3 in reciprocal space."""
        return self._fc3_reciprocal

    def _real_to_reciprocal(self) -> None:
        assert self._fc3_reciprocal is not None
        num_patom = len(self._primitive)
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0
        )
        sum_q = sum_triplets.astype("double") / self._mesh
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._fc3_reciprocal[i, j, k] = self._real_to_reciprocal_elements(
                        (i, j, k),
                        q_indices=(1, 2),
                        leg_index=0,
                    )

            prephase = self._get_prephase(sum_q, i)
            self._fc3_reciprocal[i] *= prephase

    def _real_to_reciprocal_r0_average(self) -> None:
        """Three-leg r0 averaging (mirrors c/real_to_reciprocal.c)."""
        assert self._fc3_reciprocal is not None
        num_patom = len(self._primitive)
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0
        )
        sum_q = sum_triplets.astype("double") / self._mesh
        prephase = np.array([self._get_prephase(sum_q, i) for i in range(num_patom)])

        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    # Leg 1: anchor = i, phase from (q1, q2), no element swap.
                    elem = self._real_to_reciprocal_elements(
                        (i, j, k), q_indices=(1, 2), leg_index=1
                    )
                    block = elem * prephase[i]
                    # Leg 2: anchor = j, phase from (q0, q2), swap l <-> m.
                    elem = self._real_to_reciprocal_elements(
                        (j, i, k), q_indices=(0, 2), leg_index=2
                    )
                    block += np.transpose(elem, (1, 0, 2)) * prephase[j]
                    # Leg 3: anchor = k, phase from (q1, q0), swap l <-> n.
                    elem = self._real_to_reciprocal_elements(
                        (k, j, i), q_indices=(1, 0), leg_index=3
                    )
                    block += np.transpose(elem, (2, 1, 0)) * prephase[k]
                    self._fc3_reciprocal[i, j, k] = block / 3.0

    def _real_to_reciprocal_elements(
        self,
        patom_indices: tuple[int, int, int],
        q_indices: tuple[int, int] = (1, 2),
        leg_index: int = 0,
    ) -> NDArray[np.cdouble]:
        """Accumulate the 3x3x3 block at primitive triplet ``patom_indices``.

        ``q_indices`` selects which two triplet q's drive the phase
        factors at the inner (satom) indices.  ``leg_index`` controls
        the all_shortest interaction:

        - 0: legacy single-leg (no all_shortest).
        - 1: r0-average leg 1 — multiply fc3 by 3 where all_shortest[pi0, j, k].
        - 2, 3: r0-average legs 2 and 3 — skip where all_shortest[pi0, j, k].

        When ``all_shortest[pi0, j, k] == 1`` the three atoms sit at a
        unique shortest-vector configuration, so the three legs produce
        physically identical contributions.  Instead of summing three
        equal terms and then dividing by 3, leg 1 contributes 3x and
        legs 2 and 3 are skipped; the final ``/ 3`` in
        ``_real_to_reciprocal_r0_average`` then recovers the same value
        with a third of the work.

        """
        num_satom = len(self._s2p_map)
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        fc3_reciprocal = np.zeros((3, 3, 3), dtype=dtype)
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                if leg_index > 0:
                    assert self._all_shortest is not None
                    flag = self._all_shortest[pi[0], j, k]
                    if leg_index > 1 and flag != 0:
                        continue
                    mult = 3.0 if (leg_index == 1 and flag != 0) else 1.0
                else:
                    mult = 1.0
                phase = self._get_phase((j, k), pi[0], q_indices=q_indices)
                fc3_reciprocal += mult * self._fc3[i, j, k] * phase
        return fc3_reciprocal

    def _get_prephase(self, sum_q: NDArray[np.double], patom_index: int) -> complex:
        r = self._primitive.scaled_positions[patom_index]
        return np.exp(2j * np.pi * np.dot(sum_q, r))

    def _get_phase(
        self,
        satom_indices: tuple[int, int],
        patom0_index: int,
        q_indices: tuple[int, int] = (1, 2),
    ) -> complex:
        si = satom_indices
        p0 = patom0_index
        phase = 1 + 0j
        for i, q_idx in enumerate(q_indices):
            svecs_adrs = self._multi[si[i], p0, 1]
            multi = self._multi[si[i], p0, 0]
            vs = self._svecs[svecs_adrs : (svecs_adrs + multi)]
            phase *= (
                np.exp(
                    2j
                    * np.pi
                    * np.dot(vs, self._triplet[q_idx].astype("double") / self._mesh)
                ).sum()
                / multi
            )
        return phase
