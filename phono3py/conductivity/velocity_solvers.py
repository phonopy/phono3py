"""Velocity solver building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.group_velocity import GroupVelocity

from phono3py.conductivity.grid_point_data import VelocityResult
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS, get_kappa_star_operations
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon.velocity_matrix import VelocityMatrix
from phono3py.phonon3.interaction import Interaction


class GroupVelocitySolver:
    """Compute group velocities and their symmetrised outer product at a grid point.

    This solver implements the ``VelocitySolver`` protocol and corresponds
    to the velocity computation previously embedded in
    ``ConductivityComponents``.

    The returned ``VelocityResult`` contains ``group_velocities``
    (num_band0, 3) and ``gv_by_gv`` (num_band0, 6) for the standard-BTE
    velocity quantities.  ``gv_by_gv`` stores the six independent
    components of the symmetry-averaged outer product
    ``v x v``: xx, yy, zz, yz, xz, xy.

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  ``init_dynamical_matrix()`` must have been
        called beforehand.
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
    average_gv_over_kstar : bool, optional
        When True average group velocities over the k-star arms before
        computing the outer product. Default False.
    gv_delta_q : float or None, optional
        Finite-difference step in 1/Angstrom for group velocity with NAC.
        Default None.
    log_level : int, optional
        Verbosity level. Default 0.
    """

    produces_gv_by_gv: bool = True
    produces_vm_by_vm: bool = False

    def __init__(
        self,
        pp: Interaction,
        is_kappa_star: bool = True,
        average_gv_over_kstar: bool = False,
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        if pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() must be called.")
        self._pp = pp
        self._is_kappa_star = is_kappa_star
        self._average_gv_over_kstar = average_gv_over_kstar
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._point_operations, self._rotations_cartesian = get_kappa_star_operations(
            pp.bz_grid, is_kappa_star
        )
        self._velocity_obj = GroupVelocity(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=pp.primitive_symmetry,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )

    @property
    def gv_delta_q(self) -> float | None:
        """Return delta-q used for finite-difference group velocity."""
        return self._gv_delta_q

    def compute(self, grid_point: int) -> VelocityResult:
        """Compute group velocity and v x v product at a grid point.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_band0, 3) and ``gv_by_gv``
            (num_band0, 6) are set.
        """
        gv = self._get_gv(grid_point)
        gv_by_gv = self._get_gv_by_gv(grid_point, gv)
        return VelocityResult(
            group_velocities=gv,
            gv_by_gv=gv_by_gv,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_gv(self, grid_point: int) -> NDArray[np.double]:
        if self._average_gv_over_kstar and len(self._point_operations) > 1:
            return self._get_averaged_gv_over_kstar(grid_point)
        return self._get_gv_at_bz_grid_point(grid_point)

    def _get_gv_at_bz_grid_point(self, bz_gp: int) -> NDArray[np.double]:
        self._velocity_obj.run(
            [get_qpoints_from_bz_grid_points(bz_gp, self._pp.bz_grid)]
        )
        assert self._velocity_obj.group_velocities is not None
        return self._velocity_obj.group_velocities[0, self._pp.band_indices, :]

    def _get_averaged_gv_over_kstar(self, grid_point: int) -> NDArray[np.double]:
        gps_rotated = get_grid_points_by_rotations(
            grid_point, self._pp.bz_grid, with_surface=True
        )
        assert len(gps_rotated) == len(self._point_operations)
        gvs = {
            bz_gp: self._get_gv_at_bz_grid_point(int(bz_gp))
            for bz_gp in np.unique(gps_rotated)
        }
        gv = np.zeros_like(gvs[grid_point])
        for bz_gp, r in zip(gps_rotated, self._rotations_cartesian, strict=True):
            gv += np.dot(gvs[bz_gp], r)
        return gv / len(self._point_operations)

    def _get_gv_by_gv(
        self, grid_point: int, gv: NDArray[np.double]
    ) -> NDArray[np.double]:
        if self._is_kappa_star:
            gps_rotated = get_grid_points_by_rotations(
                grid_point, self._pp.bz_grid, with_surface=False
            )
            multi = len(np.where(gps_rotated[0] == gps_rotated)[0])
        else:
            multi = 1

        gv_by_gv_3x3 = np.zeros((len(gv), 3, 3), dtype="double")
        for r in self._rotations_cartesian:
            gvs_rot = np.dot(gv, r.T)
            gv_by_gv_3x3 += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv_3x3 /= multi

        # Convert (num_band0, 3, 3) to (num_band0, 6) Voigt notation
        gv_by_gv = np.zeros((len(gv), 6), dtype="double")
        for j, (a, b) in enumerate(VOIGT_INDEX_PAIRS):
            gv_by_gv[:, j] = gv_by_gv_3x3[:, a, b]
        return gv_by_gv


class VelocityMatrixSolver:
    """Compute group velocity matrix and its k-star-averaged outer product.

    This solver implements the ``VelocitySolver`` protocol for the
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

    Notes
    -----
    The k-star average is computed by evaluating ``VelocityMatrix`` at all
    rotated q-points and summing the outer products, divided by the site
    multiplicity.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix()`` must have been called.
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for velocity matrix. Default None.
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
        self._reciprocal_operations, self._rotations_cartesian = (
            get_kappa_star_operations(pp.bz_grid, is_kappa_star)
        )
        self._velocity_obj = VelocityMatrix(
            pp.dynamical_matrix,
            q_length=gv_delta_q,
            rotations_cartesian=self._rotations_cartesian,
            reciprocal_operations=self._reciprocal_operations,
            frequency_factor_to_THz=pp.frequency_factor_to_THz,
        )

    def compute(self, grid_point: int) -> VelocityResult:
        """Compute velocity matrix quantities at a grid point.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_band0, 3), ``gv_by_gv``
            (num_band0, 6) real, and ``vm_by_vm``
            (num_band0, num_band, 6) complex are set.

        """
        q_point = get_qpoints_from_bz_grid_points(grid_point, self._pp.bz_grid)
        self._velocity_obj.run([q_point])
        assert self._velocity_obj.velocity_matrices is not None
        gv = self._velocity_obj.group_velocities[0]
        vm_by_vm = self._get_vm_by_vm(
            grid_point, self._velocity_obj.velocity_matrices[0]
        )
        return VelocityResult(
            group_velocities=gv,
            vm_by_vm=vm_by_vm,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_vm_by_vm(
        self,
        grid_point: int,
        vm: NDArray[np.cdouble],
    ) -> NDArray[np.cdouble]:
        r"""Compute k-star-averaged outer product of velocity matrix elements.

        For irreducible q-point q and k-star arms {Rq}:

            sum_R V^alpha_{s s'}(Rq) * conj(V^beta_{s s'}(Rq)) / site_multi

        where site_multi is the number of rotations per k-star arm.

        Returns
        -------
        vm_by_vm : (num_band0, num_band, 6) complex
            k-star-averaged outer product in Voigt order (xx, yy, zz, yz, xz, xy).

        """
        if self._is_kappa_star:
            gps_rotated = get_grid_points_by_rotations(
                grid_point, self._pp.bz_grid, with_surface=False
            )
            multi = len(np.where(gps_rotated[0] == gps_rotated)[0])
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

        return vm_by_vm
