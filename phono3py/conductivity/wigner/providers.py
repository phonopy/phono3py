"""Velocity provider for the Wigner transport equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.conductivity.wigner.velocity_operator import VelocityOperator
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction


class VelocityOperatorProvider:
    """Compute velocity operator and its symmetrised outer product at a grid point.

    This provider implements the ``VelocityProvider`` protocol for the Wigner
    transport equation.  It wraps phonopy's ``VelocityOperator`` and computes
    the k-star-averaged outer product of the velocity operator matrix.

    The returned ``GridPointResult`` fields:
    - ``group_velocities`` (num_band0, 3): diagonal (standard) group velocities,
      with degenerate-subspace diagonalisation applied.
    - ``velocity_product`` (num_band0, num_band, 6): complex k-star-averaged outer
      product  V(s,s') x V*(s,s') packed into 6 Voigt components (xx, yy, zz,
      yz, xz, xy).
    - ``num_sampling_grid_points``: k-star order for this irreducible point.

    Notes
    -----
    ``num_band0 == num_band`` is assumed (Wigner requires all phonon branches).
    The velocity operator is Hermitian, so ``V(s,s') = V*(s',s)``.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix()`` must have been called.
    point_operations : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations (integer representation).
    rotations_cartesian : ndarray of double, shape (num_ops, 3, 3)
        Corresponding Cartesian rotation matrices.
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for velocity operator. Default None.
    log_level : int, optional
        Verbosity level. Default 0.
    """

    def __init__(
        self,
        pp: Interaction,
        point_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.double],
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        if pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() must be called.")
        self._pp = pp
        self._point_operations = point_operations
        self._rotations_cartesian = rotations_cartesian
        self._is_kappa_star = is_kappa_star
        self._log_level = log_level
        self._velocity_obj = VelocityOperator(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=pp.primitive_symmetry,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )

    def compute(self, gp: GridPointInput) -> GridPointResult:
        """Compute velocity operator quantities at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.  ``frequencies`` must cover all bands
            (num_band,) so that degenerate sets can be determined.

        Returns
        -------
        GridPointResult
            ``group_velocities`` (num_band0, 3), ``velocity_product``
            (num_band0, num_band, 6) complex, and
            ``num_sampling_grid_points`` are set.
        """
        result = GridPointResult(input=gp)

        q_point = get_qpoints_from_bz_grid_points(gp.grid_point, self._pp.bz_grid)
        self._velocity_obj.run([q_point])
        assert self._velocity_obj.velocity_operators is not None
        # Full operator at this q: (nat3, nat3, 3)
        gv_op_full = self._velocity_obj.velocity_operators[0]
        # Filter to selected bands: (num_band0, nat3, 3)
        gv_op = gv_op_full[gp.band_indices, :, :]

        result.group_velocities = self._get_group_velocities(gp, gv_op_full)
        gv_by_gv_op, kstar_order = self._get_gv_by_gv_operator(gp, gv_op)
        result.velocity_product = gv_by_gv_op
        result.num_sampling_grid_points = kstar_order
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_group_velocities(
        self,
        gp: GridPointInput,
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
        deg_sets = degenerate_sets(gp.frequencies)
        pos = 0
        for deg in deg_sets:
            if len(deg) > 1:
                for id_dir in range(3):
                    sl = slice(pos, pos + len(deg))
                    block = gv_op_full[sl, sl, id_dir]
                    gv[sl, id_dir] = np.linalg.eigvalsh(block)
            pos += len(deg)
        assert pos == nat3
        return gv[gp.band_indices, :]  # (num_band0, 3)

    def _get_gv_by_gv_operator(
        self,
        gp: GridPointInput,
        gv_op: NDArray[np.cdouble],
    ) -> tuple[NDArray[np.cdouble], int]:
        """Return k-star-averaged velocity-operator outer product.

        Parameters
        ----------
        gp : GridPointInput
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
            rotation_map = get_grid_points_by_rotations(gp.grid_point, self._pp.bz_grid)
        else:
            rotation_map = get_grid_points_by_rotations(
                gp.grid_point,
                self._pp.bz_grid,
                reciprocal_rotations=self._point_operations,
            )

        num_band0, nat3, _ = gv_op.shape
        gv_by_gv_op = np.zeros((num_band0, nat3, 6), dtype="complex128")

        for r in self._rotations_cartesian:
            # gv_rot[s, s', i] = sum_j gv_op[s, s', j] * r.T[j, i]
            gv_rot = np.dot(gv_op, r.T)  # (num_band0, nat3, 3)
            voigt = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
            for j, (a, b) in enumerate(voigt):
                gv_by_gv_op[:, :, j] += gv_rot[:, :, a] * np.conj(gv_rot[:, :, b])

        # Divide by site multiplicity (number of rotations per k-star arm)
        order_kstar = len(np.unique(rotation_map))
        site_multi = len(rotation_map) // order_kstar
        gv_by_gv_op /= site_multi

        return gv_by_gv_op, order_kstar
