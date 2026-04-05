"""Velocity provider for the Green-Kubo formula."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointInput, VelocityResult
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS
from phono3py.conductivity.velocity_providers import (
    get_kstar_order,
    get_multiplicity_at_q,
)
from phono3py.phonon.grid import get_qpoints_from_bz_grid_points
from phono3py.phonon.velocity_matrix import VelocityMatrix
from phono3py.phonon3.interaction import Interaction


class VelocityMatrixProvider:
    """Compute group velocity matrix and its k-star-averaged outer product.

    This provider implements the ``VelocityProvider`` protocol for the
    Green-Kubo formula.  It wraps phono3py's ``VelocityMatrix`` and
    computes the k-star-averaged outer product of velocity matrix elements.

    The returned ``VelocityResult`` contains:
    - ``group_velocities`` (num_band0, 3): diagonal (standard) group velocities,
      real part of the velocity matrix diagonal.
    - ``gv_by_gv`` (num_band0, 6): real diagonal of the outer product (standard
      BTE-compatible).
    - ``vm_by_vm`` (num_band0, num_band, 6): complex k-star-averaged outer
      product  V(s,s') * V*(s,s') packed into 6 Voigt components (xx, yy, zz,
      yz, xz, xy).
    - ``num_sampling_grid_points``: k-star order for this irreducible point.

    Notes
    -----
    The k-star average is computed by evaluating ``VelocityMatrix`` at all
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
        reciprocal_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.double],
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
        self._velocity_obj = VelocityMatrix(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            rotations_cartesian=rotations_cartesian,
            reciprocal_operations=reciprocal_operations,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )
        self._reciprocal_operations = reciprocal_operations
        self._rotations_cartesian = rotations_cartesian

    def compute(self, gp: GridPointInput) -> VelocityResult:
        """Compute velocity matrix quantities at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_band0, 3), ``gv_by_gv``
            (num_band0, 6) real, ``vm_by_vm``
            (num_band0, num_band, 6) complex, and
            ``num_sampling_grid_points`` are set.

        """
        q_point = get_qpoints_from_bz_grid_points(gp.grid_point, self._pp.bz_grid)
        self._velocity_obj.run([q_point])
        assert self._velocity_obj.velocity_matrices is not None
        gv = self._velocity_obj.group_velocities[0]
        vm_by_vm, kstar_order = self._get_vm_by_vm(
            gp, self._velocity_obj.velocity_matrices[0]
        )
        return VelocityResult(
            group_velocities=gv,
            vm_by_vm=vm_by_vm,
            num_sampling_grid_points=kstar_order,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_vm_by_vm(
        self,
        gp: GridPointInput,
        vm: NDArray[np.cdouble],
    ) -> tuple[NDArray[np.cdouble], int]:
        r"""Compute k-star-averaged outer product of velocity matrix elements.

        For irreducible q-point q and k-star arms {Rq}:

            sum_R V^alpha_{s s'}(Rq) * conj(V^beta_{s s'}(Rq)) / site_multi

        where site_multi is the number of rotations per k-star arm.

        Returns
        -------
        gv : (num_band0, 3) real
            Diagonal of the velocity matrix (standard group velocities).
        vm_by_vm : (num_band0, num_band, 6) complex
            k-star-averaged outer product in Voigt order (xx, yy, zz, yz, xz, xy).
        kstar_order : int
            Number of arms in the k-star.

        """
        if self._is_kappa_star:
            multi = get_multiplicity_at_q(
                gp.grid_point, self._pp, self._reciprocal_operations
            )
        else:
            multi = 1

        vm_by_vm = np.zeros(vm.shape[1:] + (6,), order="C", dtype="complex128")

        for r in self._rotations_cartesian:
            # vm: (3, nat3, nat3) complex
            _vm = np.einsum("ab, bcd -> acd", r, vm)
            for i_pair, (a, b) in enumerate(VOIGT_INDEX_PAIRS):
                # V^a_{s s'} * conj(V^b_{s s'}) for selected band0 vs all bands
                vm_by_vm[:, :, i_pair] += _vm[a] * _vm[b].conj()
        vm_by_vm /= multi

        kstar_order = get_kstar_order(
            gp.grid_weight,
            multi,
            self._reciprocal_operations,
            verbose=self._log_level > 0,
        )

        return vm_by_vm, kstar_order
