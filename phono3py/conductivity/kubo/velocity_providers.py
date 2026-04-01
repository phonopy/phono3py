"""Velocity provider for the Green-Kubo formula."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS
from phono3py.conductivity.velocity_providers import (
    get_kstar_order,
    get_multiplicity_at_q,
)
from phono3py.phonon.grid import get_qpoints_from_bz_grid_points
from phono3py.phonon.group_velocity_matrix import GroupVelocityMatrix
from phono3py.phonon3.interaction import Interaction


class VelocityMatrixProvider:
    """Compute group velocity matrix and its k-star-averaged outer product.

    This provider implements the ``VelocityProvider`` protocol for the
    Green-Kubo formula.  It wraps phono3py's ``GroupVelocityMatrix`` and
    computes the k-star-averaged outer product of velocity matrix elements.

    The returned ``GridPointResult`` fields:
    - ``group_velocities`` (num_band0, 3): diagonal (standard) group velocities,
      real part of the velocity matrix diagonal.
    - ``velocity_product`` (num_band0, num_band, 6): complex k-star-averaged outer
      product  V(s,s') * V*(s,s') packed into 6 Voigt components (xx, yy, zz,
      yz, xz, xy).
    - ``num_sampling_grid_points``: k-star order for this irreducible point.

    Notes
    -----
    The k-star average is computed by evaluating ``GroupVelocityMatrix`` at all
    rotated q-points and summing the outer products, divided by the site
    multiplicity.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix()`` must have been called.
    point_operations : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations (integer representation).
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for velocity matrix. Default None.
    log_level : int, optional
        Verbosity level. Default 0.

    """

    def __init__(
        self,
        pp: Interaction,
        point_operations: NDArray[np.int64],
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        if pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() must be called.")
        self._pp = pp
        self._point_operations = point_operations
        self._is_kappa_star = is_kappa_star
        self._log_level = log_level
        self._velocity_obj = GroupVelocityMatrix(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=pp.primitive_symmetry,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )

    def compute(self, gp: GridPointInput) -> GridPointResult:
        """Compute velocity matrix quantities at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.

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
        assert self._velocity_obj.group_velocity_matrices is not None
        # gvm_full: (3, nat3, nat3) at the irreducible q
        gvm_full = self._velocity_obj.group_velocity_matrices[0]

        result.group_velocities = self._get_group_velocities(gp, gvm_full)
        gvm_by_gvm, kstar_order = self._get_gvm_by_gvm(gp)
        result.velocity_product = gvm_by_gvm
        result.num_sampling_grid_points = kstar_order
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_group_velocities(
        self,
        gp: GridPointInput,
        gvm_full: NDArray[np.cdouble],
    ) -> NDArray[np.double]:
        """Return diagonal group velocities from velocity matrix.

        Parameters
        ----------
        gp : GridPointInput
        gvm_full : (3, nat3, nat3) complex
            Full velocity matrix at the irreducible q-point.

        Returns
        -------
        ndarray of double, shape (num_band0, 3)

        """
        nat3 = gvm_full.shape[1]
        gv = np.zeros((nat3, 3), dtype="double")
        for i in range(3):
            gv[:, i] = np.diag(gvm_full[i]).real
        return gv[gp.band_indices, :]

    def _get_gvm_by_gvm(
        self,
        gp: GridPointInput,
    ) -> tuple[NDArray[np.cdouble], int]:
        r"""Compute k-star-averaged outer product of velocity matrix elements.

        For irreducible q-point q and k-star arms {Rq}:

            sum_R V^alpha_{s s'}(Rq) * conj(V^beta_{s s'}(Rq)) / site_multi

        where site_multi is the number of rotations per k-star arm.

        Returns
        -------
        gvm_by_gvm : (num_band0, num_band, 6) complex
            k-star-averaged outer product in Voigt order (xx, yy, zz, yz, xz, xy).
        kstar_order : int
            Number of arms in the k-star.

        """
        if self._is_kappa_star:
            multi = get_multiplicity_at_q(
                gp.grid_point, self._pp, self._point_operations
            )
        else:
            multi = 1

        q = get_qpoints_from_bz_grid_points(gp.grid_point, self._pp.bz_grid)
        qpoints = [np.dot(r, q) for r in self._point_operations]
        self._velocity_obj.run(qpoints)
        assert self._velocity_obj.group_velocity_matrices is not None

        nat3 = len(self._pp.primitive) * 3
        num_band0 = len(gp.band_indices)
        gvm_by_gvm = np.zeros((num_band0, nat3, 6), dtype="complex128")

        for gvm in self._velocity_obj.group_velocity_matrices:
            # gvm: (3, nat3, nat3) complex at one rotated q-point
            for i_pair, (a, b) in enumerate(VOIGT_INDEX_PAIRS):
                # V^a_{s s'} * conj(V^b_{s s'}) for selected band0 vs all bands
                gvm_by_gvm[:, :, i_pair] += gvm[a][gp.band_indices, :] * np.conj(
                    gvm[b][gp.band_indices, :]
                )
        gvm_by_gvm /= multi

        kstar_order = get_kstar_order(
            gp.grid_weight,
            multi,
            self._point_operations,
            verbose=self._log_level > 0,
        )
        return gvm_by_gvm, kstar_order
