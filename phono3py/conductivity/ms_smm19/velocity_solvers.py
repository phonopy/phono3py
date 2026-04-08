"""Velocity solver for the Wigner transport equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets

from phono3py.conductivity.grid_point_data import VelocityResult
from phono3py.conductivity.ms_smm19.velocity_operator import VelocityOperator
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS, get_kappa_star_operations
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction


class VelocityOperatorSolver:
    """Compute velocity operator and its symmetrised outer product at a grid point.

    This solver implements the ``VelocitySolver`` protocol for the Wigner
    transport equation.  It wraps phonopy's ``VelocityOperator`` and computes
    the k-star-averaged outer product of the velocity operator matrix.

    The returned ``VelocityResult`` contains:
    - ``group_velocities`` (num_band0, 3): diagonal (standard) group velocities,
      with degenerate-subspace diagonalisation applied.
    - ``gv_by_gv`` (num_band0, 6): real diagonal of the outer product (standard
      BTE-compatible).
    - ``vm_by_vm`` (num_band0, num_band, 6): complex k-star-averaged outer
      product  V(s,s') x V*(s,s') packed into 6 Voigt components (xx, yy, zz,
      yz, xz, xy).
    - ``num_sampling_grid_points``: k-star order for this irreducible point.
    - ``extra["velocity_operator"]``: raw velocity operator matrix.

    Notes
    -----
    ``num_band0 == num_band`` is assumed (Wigner requires all phonon branches).
    The velocity operator is Hermitian, so ``V(s,s') = V*(s',s)``.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix()`` must have been called.
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for velocity operator. Default None.
    log_level : int, optional
        Verbosity level. Default 0.

    """

    produces_gv_by_gv: bool = False
    produces_vm_by_vm: bool = True

    def __init__(
        self,
        pp: Interaction,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        if pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() must be called.")
        self._pp = pp
        self._is_kappa_star = is_kappa_star
        self._log_level = log_level
        self._point_operations, self._rotations_cartesian = get_kappa_star_operations(
            pp.bz_grid, is_kappa_star
        )
        self._velocity_obj = VelocityOperator(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=pp.primitive_symmetry,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )

    def compute(self, grid_point: int) -> VelocityResult:
        """Compute velocity operator quantities at a grid point.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_band0, 3), ``gv_by_gv``
            (num_band0, 6) real, ``vm_by_vm``
            (num_band0, num_band, 6) complex, and
            ``num_sampling_grid_points`` are set.
            ``extra["velocity_operator"]`` contains the raw operator for HDF5.

        """
        q_point = get_qpoints_from_bz_grid_points(grid_point, self._pp.bz_grid)
        self._velocity_obj.run([q_point])
        assert self._velocity_obj.velocity_operators is not None
        # Full operator at this q: (nat3, nat3, 3)
        gv_op_full = self._velocity_obj.velocity_operators[0]
        # Filter to selected bands: (num_band0, nat3, 3)
        band_indices = self._pp.band_indices
        gv_op = gv_op_full[band_indices, :, :]

        gv = self._get_group_velocities(grid_point, gv_op_full)
        vm_by_vm, kstar_order = self._get_gv_by_gv_operator(grid_point, gv_op)
        return VelocityResult(
            group_velocities=gv,
            vm_by_vm=vm_by_vm,
            num_sampling_grid_points=kstar_order,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_group_velocities(
        self,
        grid_point: int,
        gv_op_full: NDArray[np.cdouble],
    ) -> NDArray[np.double]:
        """Return diagonal group velocities with degenerate-subspace correction.

        For degenerate phonon branches the velocity operator restricted to the
        degenerate subspace is diagonalised so that the group velocities are
        well-defined.  This follows the same procedure as
        ``ConductivityWignerComponents._set_gv_operator``.

        """
        nat3 = gv_op_full.shape[0]
        # Diagonal of operator gives standard group velocities
        gv = np.einsum("iij->ij", gv_op_full).real  # (nat3, 3)
        frequencies = self._pp.get_phonons()[0][grid_point]
        deg_sets = degenerate_sets(frequencies)
        pos = 0
        for deg in deg_sets:
            if len(deg) > 1:
                for id_dir in range(3):
                    sl = slice(pos, pos + len(deg))
                    block = gv_op_full[sl, sl, id_dir]
                    gv[sl, id_dir] = np.linalg.eigvalsh(block)
            pos += len(deg)
        assert pos == nat3
        return gv[self._pp.band_indices, :]  # (num_band0, 3)

    def _get_gv_by_gv_operator(
        self,
        grid_point: int,
        gv_op: NDArray[np.cdouble],
    ) -> tuple[NDArray[np.cdouble], int]:
        """Return k-star-averaged velocity-operator outer product.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.
        gv_op : (num_band0, nat3, 3) complex
            Velocity operator filtered to selected bands.

        Returns
        -------
        gv_by_gv_op : (num_band0, nat3, 6) complex
            Symmetry-averaged outer product packed in Voigt order.
        kstar_order : int
            Number of arms in the k-star.

        """
        if self._is_kappa_star:
            rotation_map = get_grid_points_by_rotations(grid_point, self._pp.bz_grid)
        else:
            rotation_map = get_grid_points_by_rotations(
                grid_point,
                self._pp.bz_grid,
                reciprocal_rotations=self._point_operations,
            )

        num_band0, nat3, _ = gv_op.shape
        gv_by_gv_op = np.zeros((num_band0, nat3, 6), dtype="complex128")

        for r in self._rotations_cartesian:
            # gv_rot[s, s', i] = sum_j gv_op[s, s', j] * r.T[j, i]
            gv_rot = np.dot(gv_op, r.T)  # (num_band0, nat3, 3)
            for j, (a, b) in enumerate(VOIGT_INDEX_PAIRS):
                gv_by_gv_op[:, :, j] += gv_rot[:, :, a] * np.conj(gv_rot[:, :, b])

        # Divide by site multiplicity (number of rotations per k-star arm)
        order_kstar = len(np.unique(rotation_map))
        site_multi = len(rotation_map) // order_kstar
        gv_by_gv_op /= site_multi

        return gv_by_gv_op, order_kstar
