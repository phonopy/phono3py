"""Fourier transform of fc4 to reciprocal space (experimental).

Port of the 2015 phono4py ``RealToReciprocal`` (Python path) onto modern
phonopy, using the dense smallest-vector format
(``primitive.get_smallest_vectors()`` returns ``svecs`` of shape ``(n, 3)`` in
primitive-fractional coordinates and ``multiplicity`` of shape
``(n_satom, n_patom, 2)`` holding ``[count, start]``).

The phase convention is kept identical to the 2015 implementation so that it
pairs consistently with the reciprocal-to-normal projection in
``phono3py.phonon4.frequency_shift``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.cells import Primitive

from phono3py._lang import resolve_lang


class RealToReciprocalFc4:
    """Transform a real-space fc4 to reciprocal space at a q-point quartet."""

    def __init__(
        self,
        fc4: NDArray[np.double],
        primitive: Primitive,
        mesh: NDArray[np.int64],
        is_compact_fc: bool = False,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        fc4 : ndarray
            fc4 in the full ``(N, N, N, N, 3, 3, 3, 3)`` or compact
            ``(n_patom, N, N, N, 3, 3, 3, 3)`` layout.
        primitive : Primitive
            Primitive cell (provides p2s/s2p maps and smallest vectors).
        mesh : array_like
            Reciprocal sampling mesh, shape ``(3,)``.
        is_compact_fc : bool, optional
            Whether ``fc4`` is in the compact layout. Default is False.
        lang : {"C", "Rust"}, optional
            With ``"Rust"`` (default) the phonors kernel is used; otherwise a
            pure-Python fallback.

        """
        self._fc4 = np.array(fc4, dtype="double", order="C")
        self._primitive = primitive
        self._mesh = np.array(mesh, dtype="int64")
        self._is_compact_fc = is_compact_fc
        self._lang = resolve_lang(lang)
        self._p2s_map = np.array(primitive.p2s_map, dtype="int64")
        self._s2p_map = np.array(primitive.s2p_map, dtype="int64")
        svecs, multiplicity = primitive.get_smallest_vectors()
        self._svecs = np.array(svecs, dtype="double", order="C")
        self._multiplicity = np.array(multiplicity, dtype="int64", order="C")
        self._num_satom = len(self._s2p_map)
        self._num_patom = len(self._p2s_map)

    def run(self, quartet: NDArray[np.int64]) -> NDArray[np.complex128]:
        """Return fc4 in reciprocal space at the given q-point quartet.

        Parameters
        ----------
        quartet : ndarray
            Four grid addresses (integers), shape ``(4, 3)``. The fractional
            q-points are ``quartet / mesh``; the first q-point is the reference
            (only the last three enter the phase factors), mirroring the 2015
            convention.

        Returns
        -------
        ndarray
            fc4 in reciprocal space, shape
            ``(n_patom, n_patom, n_patom, n_patom, 3, 3, 3, 3)``,
            dtype=complex128.

        """
        q = np.array(quartet, dtype="double", order="C") / self._mesh
        n_patom = self._num_patom
        fc4_reciprocal = np.zeros((n_patom,) * 4 + (3,) * 4, dtype="complex128")
        if self._lang == "Rust":
            import phonors  # type: ignore[import-untyped]

            phonors.real_to_reciprocal_fc4(
                fc4_reciprocal,
                np.array(q, dtype="double", order="C"),
                self._fc4,
                self._is_compact_fc,
                self._svecs,
                self._multiplicity,
                self._p2s_map,
                self._s2p_map,
            )
            return fc4_reciprocal
        for i, j, k, ll in np.ndindex(n_patom, n_patom, n_patom, n_patom):
            fc4_reciprocal[i, j, k, ll] = self._reciprocal_element((i, j, k, ll), q)
        return fc4_reciprocal

    def _reciprocal_element(
        self, patom_indices: tuple[int, int, int, int], q: NDArray[np.double]
    ) -> NDArray[np.complex128]:
        pi = patom_indices
        i = pi[0] if self._is_compact_fc else self._p2s_map[pi[0]]
        s2p_pi1 = self._p2s_map[pi[1]]
        s2p_pi2 = self._p2s_map[pi[2]]
        s2p_pi3 = self._p2s_map[pi[3]]
        elem = np.zeros((3, 3, 3, 3), dtype="complex128")
        for j in range(self._num_satom):
            if self._s2p_map[j] != s2p_pi1:
                continue
            phase_j = self._phase(j, pi[0], q[1])
            for k in range(self._num_satom):
                if self._s2p_map[k] != s2p_pi2:
                    continue
                phase_jk = phase_j * self._phase(k, pi[0], q[2])
                for ll in range(self._num_satom):
                    if self._s2p_map[ll] != s2p_pi3:
                        continue
                    phase = phase_jk * self._phase(ll, pi[0], q[3])
                    elem += self._fc4[i, j, k, ll] * phase
        return elem

    def _phase(self, satom: int, patom0: int, q: NDArray[np.double]) -> complex:
        count, start = self._multiplicity[satom, patom0]
        svecs = self._svecs[start : start + count]
        return np.exp(2j * np.pi * (svecs @ q)).sum() / count
