"""Velocity provider building blocks for conductivity calculations."""

from __future__ import annotations

import textwrap

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.group_velocity import GroupVelocity

from phono3py.conductivity.grid_point_data import GridPointInput, VelocityResult
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
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

    def compute(self, gp: GridPointInput) -> VelocityResult:
        """Compute group velocity and v x v product at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.

        Returns
        -------
        VelocityResult
            ``group_velocities`` (num_band0, 3), ``gv_by_gv``
            (num_band0, 6), and ``num_sampling_grid_points`` are set.
        """
        gv = self._get_gv(gp)
        gv_by_gv, kstar_order = self._get_gv_by_gv(gp, gv)
        return VelocityResult(
            group_velocities=gv,
            gv_by_gv=gv_by_gv,
            num_sampling_grid_points=kstar_order,
        )

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
        for j, (a, b) in enumerate(VOIGT_INDEX_PAIRS):
            gv_by_gv[:, j] = gv_by_gv_3x3[:, a, b]
        return gv_by_gv, kstar_order
