"""Velocity provider building blocks for conductivity calculations."""

from __future__ import annotations

import textwrap

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.group_velocity import GroupVelocity

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon.group_velocity_matrix import GroupVelocityMatrix
from phono3py.phonon3.interaction import Interaction


def get_multiplicity_at_q(
    gp: int,
    pp: Interaction,
    point_operations: NDArray[np.int64],
) -> int:
    """Return multiplicity (order of site-symmetry) of q-point."""
    q = get_qpoints_from_bz_grid_points(gp, pp.bz_grid)
    reclat = np.linalg.inv(pp.primitive.cell)
    multi = 0
    for q_rot in [np.dot(r, q) for r in point_operations]:
        diff = q - q_rot
        diff -= np.rint(diff)
        dist = np.linalg.norm(np.dot(reclat, diff))
        if dist < pp.primitive_symmetry.tolerance:
            multi += 1
    return multi


def get_kstar_order(
    grid_weight: int,
    multi: int,
    point_operations: NDArray[np.int64],
    verbose: bool = False,
) -> int:
    """Return order (number of arms) of kstar.

    multi : int
        Multiplicity of grid point.

    """
    order_kstar = len(point_operations) // multi
    if order_kstar != grid_weight:
        if verbose:
            text = (
                "Number of elements in k* is unequal "
                "to number of equivalent grid-points. "
                "This means that the mesh sampling grids break "
                "symmetry. Please check carefully "
                "the convergence over grid point densities."
            )
            msg = textwrap.fill(
                text, initial_indent=" ", subsequent_indent=" ", width=70
            )
            print("*" * 30 + "Warning" + "*" * 30)
            print(msg)
            print("*" * 67)

    return order_kstar


class GroupVelocityProvider:
    """Compute group velocities and their symmetrised outer product at a grid point.

    This provider implements the ``VelocityProvider`` protocol and corresponds
    to the velocity computation previously embedded in
    ``ConductivityComponents``.

    The returned ``GridPointResult`` fields
    ``group_velocities`` (num_band0, 3) and ``velocity_product`` (num_band0, 6)
    contain the standard-BTE velocity quantities.  ``velocity_product`` stores
    the six independent components of the symmetry-averaged outer product
    ``v x v``: xx, yy, zz, yz, xz, xy.

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  ``init_dynamical_matrix()`` must have been
        called beforehand.
    point_operations : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations (integer representation).
    rotations_cartesian : ndarray of double, shape (num_ops, 3, 3)
        Corresponding Cartesian rotation matrices.
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

    def __init__(
        self,
        pp: Interaction,
        point_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.double],
        is_kappa_star: bool = True,
        average_gv_over_kstar: bool = False,
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
        self._average_gv_over_kstar = average_gv_over_kstar
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
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

    def compute(self, gp: GridPointInput) -> GridPointResult:
        """Compute group velocity and v x v product at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.

        Returns
        -------
        GridPointResult
            ``group_velocities`` (num_band0, 3), ``velocity_product``
            (num_band0, 6), and ``num_sampling_grid_points`` are set.
        """
        result = GridPointResult(input=gp)
        gv = self._get_gv(gp)
        gv_by_gv, kstar_order = self._get_gv_by_gv(gp, gv)
        result.group_velocities = gv
        result.velocity_product = gv_by_gv
        result.num_sampling_grid_points = kstar_order
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_gv(self, gp: GridPointInput) -> NDArray[np.double]:
        if self._average_gv_over_kstar and len(self._point_operations) > 1:
            return self._get_averaged_gv_over_kstar(gp)
        return self._get_gv_at_bz_grid_point(gp.grid_point)

    def _get_gv_at_bz_grid_point(self, bz_gp: int) -> NDArray[np.double]:
        self._velocity_obj.run(
            [get_qpoints_from_bz_grid_points(bz_gp, self._pp.bz_grid)]
        )
        assert self._velocity_obj.group_velocities is not None
        return self._velocity_obj.group_velocities[0, self._pp.band_indices, :]

    def _get_averaged_gv_over_kstar(self, gp: GridPointInput) -> NDArray[np.double]:
        gps_rotated = get_grid_points_by_rotations(
            gp.grid_point, self._pp.bz_grid, with_surface=True
        )
        assert len(gps_rotated) == len(self._point_operations)
        gvs = {
            bz_gp: self._get_gv_at_bz_grid_point(int(bz_gp))
            for bz_gp in np.unique(gps_rotated)
        }
        gv = np.zeros_like(gvs[gp.grid_point])
        for bz_gp, r in zip(gps_rotated, self._rotations_cartesian, strict=True):
            gv += np.dot(gvs[bz_gp], r)
        return gv / len(self._point_operations)

    def _get_gv_by_gv(
        self, gp: GridPointInput, gv: NDArray[np.double]
    ) -> tuple[NDArray[np.double], int]:
        if self._is_kappa_star:
            multi = get_multiplicity_at_q(
                gp.grid_point, self._pp, self._point_operations
            )
        else:
            multi = 1

        gv_by_gv_3x3 = np.zeros((len(gv), 3, 3), dtype="double")
        for r in self._rotations_cartesian:
            gvs_rot = np.dot(gv, r.T)
            gv_by_gv_3x3 += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv_3x3 /= multi

        kstar_order = get_kstar_order(
            gp.grid_weight,
            multi,
            self._point_operations,
            verbose=self._log_level > 0,
        )

        # Convert (num_band0, 3, 3) to (num_band0, 6) Voigt notation
        gv_by_gv = np.zeros((len(gv), 6), dtype="double")
        for j, (a, b) in enumerate([[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]):
            gv_by_gv[:, j] = gv_by_gv_3x3[:, a, b]
        return gv_by_gv, kstar_order


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

        voigt = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
        for gvm in self._velocity_obj.group_velocity_matrices:
            # gvm: (3, nat3, nat3) complex at one rotated q-point
            for i_pair, (a, b) in enumerate(voigt):
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
