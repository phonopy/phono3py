"""Velocity solver building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.velocity_matrix import VelocityMatrix

from phono3py.conductivity.grid_point_data import VelocityResult
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS, get_kappa_star_operations
from phono3py.phonon3.interaction import Interaction


class GroupVelocitySolver:
    """Compute group velocities and their symmetrised outer product at grid points.

    This solver implements the ``VelocitySolver`` protocol and corresponds
    to the velocity computation previously embedded in
    ``ConductivityComponents``.

    The returned ``VelocityResult`` contains ``group_velocities``
    (num_gp, num_band0, 3) and ``gv_by_gv`` (num_gp, num_band0, 6) for the
    standard-BTE velocity quantities.  ``gv_by_gv`` stores the six independent
    components of the symmetry-averaged outer product
    ``v x v``: xx, yy, zz, yz, xz, xy.

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  ``init_dynamical_matrix()`` must have been
        called beforehand.
    is_kappa_star : bool, optional
        When True use the full k-star for symmetry averaging. Default True.
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
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        if pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() must be called.")
        self._pp = pp
        self._is_kappa_star = is_kappa_star
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        _, self._rotations_cartesian = get_kappa_star_operations(
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

    def compute(self, grid_points: NDArray[np.int64]) -> VelocityResult:
        """Compute group velocities and v x v products at grid points.

        Group velocities at all grid points are computed by one call of
        ``GroupVelocity.run`` of phonopy.

        Parameters
        ----------
        grid_points : ndarray of int64, shape (num_gp,)
            BZ grid point indices.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_gp, num_band0, 3) and ``gv_by_gv``
            (num_gp, num_band0, 6) are set.

        """
        grid_points = np.asarray(grid_points, dtype="int64")
        num_band0 = len(self._pp.band_indices)
        gv = np.zeros((len(grid_points), num_band0, 3), dtype="double", order="C")
        gv_by_gv = np.zeros((len(grid_points), num_band0, 6), dtype="double", order="C")
        if len(grid_points) > 0:
            gv[:] = self._get_gv_at_bz_grid_points(grid_points)
            for i_gp, grid_point in enumerate(grid_points):
                gv_by_gv[i_gp] = self._get_gv_by_gv(int(grid_point), gv[i_gp])
        return VelocityResult(group_velocities=gv, gv_by_gv=gv_by_gv)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_gv_at_bz_grid_points(
        self, bz_gps: NDArray[np.int64]
    ) -> NDArray[np.double]:
        self._velocity_obj.run(
            get_qpoints_from_bz_grid_points(bz_gps, self._pp.bz_grid)
        )
        assert self._velocity_obj.group_velocities is not None
        return self._velocity_obj.group_velocities[:, self._pp.band_indices, :]

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
    Green-Kubo formula.  It wraps phonopy's ``VelocityMatrix`` and
    computes the k-star-averaged outer product of velocity matrix elements.

    The returned ``VelocityResult`` contains:
    - ``group_velocities`` (num_gp, num_band, 3): diagonal (standard) group
      velocities, real part of the velocity matrix diagonal.
    - ``vm_by_vm`` (num_gp, num_band, num_band, 6): complex k-star-averaged
      outer product  V(s,s') * V*(s,s') packed into 6 Voigt components (xx,
      yy, zz, yz, xz, xy).

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

    def compute(self, grid_points: NDArray[np.int64]) -> VelocityResult:
        """Compute velocity matrix quantities at grid points.

        Velocity matrices at all grid points are computed by one call of
        ``VelocityMatrix.run`` of phonopy.

        Parameters
        ----------
        grid_points : ndarray of int64, shape (num_gp,)
            BZ grid point indices.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_gp, num_band, 3) real and ``vm_by_vm``
            (num_gp, num_band, num_band, 6) complex are set.

        """
        grid_points = np.asarray(grid_points, dtype="int64")
        num_band = len(self._pp.primitive) * 3
        gv = np.zeros((len(grid_points), num_band, 3), dtype="double", order="C")
        vm_by_vm = np.zeros(
            (len(grid_points), num_band, num_band, 6), dtype="cdouble", order="C"
        )
        if len(grid_points) > 0:
            self._velocity_obj.run(
                get_qpoints_from_bz_grid_points(grid_points, self._pp.bz_grid)
            )
            assert self._velocity_obj.velocity_matrices is not None
            assert self._velocity_obj.group_velocities is not None
            gv[:] = self._velocity_obj.group_velocities
            for i_gp, grid_point in enumerate(grid_points):
                vm_by_vm[i_gp] = self._get_vm_by_vm(
                    int(grid_point), self._velocity_obj.velocity_matrices[i_gp]
                )
        return VelocityResult(group_velocities=gv, vm_by_vm=vm_by_vm)

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
            unique_gps = np.unique(gps_rotated)
            multi = len(gps_rotated) // len(unique_gps)
            assert multi * len(unique_gps) == len(gps_rotated)
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
