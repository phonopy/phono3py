"""Thermal conductivity base class."""

# Copyright (C) 2020 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.physical_units import get_physical_units

from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.imag_self_energy import ImagSelfEnergy
from phono3py.phonon3.interaction import Interaction


def get_unit_to_WmK() -> float:
    """Return conversion factor to WmK."""
    unit_to_WmK = (
        (get_physical_units().THz * get_physical_units().Angstrom) ** 2
        / (get_physical_units().Angstrom ** 3)
        * get_physical_units().EV
        / get_physical_units().THz
        / (2 * np.pi)
    )  # 2pi comes from definition of lifetime.
    return unit_to_WmK


def get_multiplicity_at_q(
    gp: int,
    pp: Interaction,
    point_operations: np.ndarray,
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
    grid_weight: int, multi: int, point_operations: np.ndarray, verbose: bool = False
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


def get_heat_capacities(
    grid_point: int,
    pp: Interaction,
    temperatures: NDArray[np.float64],
):
    """Return mode heat capacity.

    cv returned should be given to self._cv by

        self._cv[:, i_data, :] = cv

    """
    if not pp.phonon_all_done:
        raise RuntimeError(
            "Phonon calculation has not been done yet. "
            "Run phono3py.run_phonon_solver() before this method."
        )

    frequencies, _, _ = pp.get_phonons()
    freqs = (
        frequencies[grid_point][pp.band_indices]  # type: ignore
        * get_physical_units().THzToEv
    )
    cutoff = pp.cutoff_frequency * get_physical_units().THzToEv
    cv = np.zeros((len(temperatures), len(freqs)), dtype="double")
    # x=freq/T has to be small enough to avoid overflow of exp(x).
    # x < 100 is the hard-corded criterion.
    # Otherwise just set 0.
    for i, f in enumerate(freqs):
        if f > cutoff:
            condition = f < 100 * temperatures * get_physical_units().KB
            cv[:, i] = np.where(
                condition,
                mode_cv(np.where(condition, temperatures, 10000), f),  # type: ignore
                0,
            )
    return cv


class ConductivityComponentsBase(ABC):
    """Base class of ConductivityComponents."""

    def __init__(
        self,
        pp: Interaction,
        grid_points: NDArray[np.int64],
        grid_weights: NDArray[np.int64],
        point_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.int64],
        temperatures: NDArray[np.float64] | None = None,
        average_gv_over_kstar: bool = False,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        gv_delta_q : float, optional, default is None,  # for group velocity
            With non-analytical correction, group velocity is calculated
            by central finite difference method. This value gives the distance
            in both directions in 1/Angstrom. The default value will be 1e-5.

        """
        self._pp = pp
        self._grid_points = grid_points
        self._grid_weights = grid_weights
        self._point_operations = point_operations
        self._rotations_cartesian = rotations_cartesian
        self._temperatures = temperatures
        self._average_gv_over_kstar = average_gv_over_kstar
        self._gv_delta_q = gv_delta_q
        self._is_kappa_star = is_kappa_star
        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        self._log_level = log_level

        self._gv: np.ndarray
        self._cv: np.ndarray

        self._num_sampling_grid_points = 0

    @property
    def mode_heat_capacities(self) -> NDArray:
        """Return mode heat capacity at constant volume at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._cv

    @property
    def group_velocities(self) -> NDArray:
        """Return group velocities at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._gv

    @property
    def gv_delta_q(self):
        """Return delta q for group velocity."""
        return self._gv_delta_q

    @property
    def number_of_sampling_grid_points(self):
        """Return number of grid points.

        This is calculated by the sum of numbers of arms of k-start.

        """
        return self._num_sampling_grid_points

    def set_heat_capacities(self, i_gp, i_data):
        """Set heat capacity at grid point and at data location."""
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        cv = get_heat_capacities(self._grid_points[i_gp], self._pp, self._temperatures)
        self._cv[:, i_data, :] = cv

    @abstractmethod
    def set_velocities(self, i_gp, i_data):
        """Set velocities at grid point and at data location."""
        raise NotImplementedError()

    def _allocate_values(self):
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        num_band0 = len(self._pp.band_indices)
        if self._is_reducible_collision_matrix:
            num_grid_points = np.prod(self._pp.mesh_numbers)
        else:
            num_grid_points = len(self._grid_points)
        num_temp = len(self._temperatures)

        self._cv = np.zeros(
            (num_temp, num_grid_points, num_band0), order="C", dtype="double"
        )
        self._gv = np.zeros((num_grid_points, num_band0, 3), order="C", dtype="double")


class ConductivityComponents(ConductivityComponentsBase):
    """Thermal conductivity components.

    Used by ConductivityRTA and ConductivityLBTE.

    """

    def __init__(
        self,
        pp: Interaction,
        grid_points: NDArray[np.int64],
        grid_weights: NDArray[np.int64],
        point_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.int64],
        temperatures: NDArray[np.float64] | None = None,
        average_gv_over_kstar: bool = False,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(
            pp,
            grid_points,
            grid_weights,
            point_operations,
            rotations_cartesian,
            temperatures=temperatures,
            average_gv_over_kstar=average_gv_over_kstar,
            is_kappa_star=is_kappa_star,
            gv_delta_q=gv_delta_q,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            log_level=log_level,
        )

        self._gv_by_gv: NDArray

        if self._pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() has to be called.")
        self._velocity_obj = GroupVelocity(
            self._pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=self._pp.primitive_symmetry,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
        )

        if self._temperatures is not None:
            self._allocate_values()

    @property
    def gv_by_gv(self) -> NDArray:
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        return self._gv_by_gv

    def set_velocities(self, i_gp, i_data):
        """Set group velocities at grid point and at data location."""
        self._gv[i_data] = self._get_gv(i_gp)
        self._set_gv_by_gv(i_gp, i_data)

    def _get_gv(self, i_gp):
        """Get group velocity."""
        irgp = self._grid_points[i_gp]

        if self._average_gv_over_kstar and len(self._point_operations) > 1:
            gps_rotated = get_grid_points_by_rotations(
                irgp, self._pp.bz_grid, with_surface=True
            )
            assert len(gps_rotated) == len(self._point_operations)

            unique_gps = np.unique(gps_rotated)
            gvs = {}
            for bz_gp in unique_gps:
                self._velocity_obj.run(
                    [get_qpoints_from_bz_grid_points(bz_gp, self._pp.bz_grid)]
                )
                assert self._velocity_obj.group_velocities is not None
                gvs[bz_gp] = self._velocity_obj.group_velocities[
                    0, self._pp.band_indices, :
                ]
            gv = np.zeros_like(gvs[irgp])
            for bz_gp, r in zip(gps_rotated, self._rotations_cartesian, strict=True):
                gv += np.dot(gvs[bz_gp], r)  # = dot(r_inv, gv)
            return gv / len(self._point_operations)
        else:
            self._velocity_obj.run(
                [get_qpoints_from_bz_grid_points(irgp, self._pp.bz_grid)]
            )
            assert self._velocity_obj.group_velocities is not None
            return self._velocity_obj.group_velocities[0, self._pp.band_indices, :]

    def _set_gv_by_gv(self, i_gp, i_data):
        """Outer product of group velocities.

        (v x v) [num_k*, num_freqs, 3, 3]

        """
        gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i_gp, i_data)
        self._num_sampling_grid_points += order_kstar

        # Sum all vxv at k*
        for j, vxv in enumerate(([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            self._gv_by_gv[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

    def _get_gv_by_gv(self, i_gp, i_data):
        if self._is_kappa_star:
            multi = get_multiplicity_at_q(
                self._grid_points[i_gp],  # type: ignore
                self._pp,
                self._point_operations,
            )
        else:
            multi = 1
        gv = self._gv[i_data]
        gv_by_gv = np.zeros((len(gv), 3, 3), dtype="double")
        for r in self._rotations_cartesian:
            gvs_rot = np.dot(gv, r.T)
            gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv /= multi
        kstar_order = get_kstar_order(
            self._grid_weights[i_gp],  # type: ignore
            multi,
            self._point_operations,
            verbose=self._log_level > 0,
        )
        return gv_by_gv, kstar_order

    def _allocate_values(self):
        super()._allocate_values()

        num_band0 = len(self._pp.band_indices)
        if self._is_reducible_collision_matrix:
            num_grid_points = np.prod(self._pp.mesh_numbers)
        else:
            num_grid_points = len(self._grid_points)
        self._gv_by_gv = np.zeros(
            (num_grid_points, num_band0, 6), order="C", dtype="double"
        )


class ConductivityBase(ABC):
    """Base class of Conductivity classes.

    All Conductivity* classes have to inherit this base class.

    """

    _average_gv_over_kstar = False

    def __init__(
        self,
        interaction: Interaction,
        grid_points: ArrayLike | None = None,
        temperatures: ArrayLike | None = None,
        sigmas: Sequence[float | None] | None = None,
        sigma_cutoff: float | None = None,
        is_isotope=False,
        mass_variances: ArrayLike | None = None,
        boundary_mfp: float | None = None,
        is_kappa_star: bool = True,
        is_full_pp: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        interaction : Interaction
            Interaction class instance.
        grid_points : array_like or None, optional
            Grid point indices in BZgrid. When None, ir-grid points are searched
            internally. Default is None.
            shape=(grid_points, ), dtype='int64'.
        temperatures : array_like, optional, default is None
            Temperatures at which thermal conductivity is calculated.
            shape=(temperature_points, ), dtype='double'.
        sigmas : Sequence[float | None], optional, default is None
            The float values are given as the standard deviations of Gaussian
            function. If None is given as an element of this list, linear
            tetrahedron method is used instead of smearing method.
        sigma_cutoff : float, optional, default is None
            This is given as a multiple of the standard deviation. For example,
            if this value is 5, the tail of the Gaussian function is cut at 5 sigma.
        is_isotope : bool, optional, default is False
            With or without isotope scattering.
        mass_variances : array_like, optional, default is None
            Mass variances for isotope scattering calculation. When None,
            the values stored in phono3py are used with `is_isotope=True`.
            shape(atoms_in_primitive, ), dtype='double'.
        boundary_mfp : float, optional, default is None
            Mean free path in micrometer to calculate simple boundary
            scattering contribution to thermal conductivity.
            None ignores this contribution.
        is_kappa_star : bool, optional
            When True, reciprocal space symmetry is used to calculate
            lattice thermal conductivity. This calculation is performed
            iterating over specific grid points. With `is_kappa_star=True`
            and `grid_points=None`, ir-grid points are used for the iteration.
            Default is True.
        is_full_pp : bool, optional, default is False
            With True, full elements of phonon-phonon interaction strength
            are computed. However with tetrahedron method, part of them are
            known to be zero and unnecessary to calculation. With False,
            those elements are not calculated, by which considerable
            improve of efficiency is expected.
            With smearing method, even if this is set False, full elements
            are computed unless `sigma_cutoff` is specified.
        log_level : int, optional
            Verbosity control. Default is 0.

        """
        self._pp: Interaction = interaction
        self._is_kappa_star = is_kappa_star
        self._is_full_pp = is_full_pp
        self._log_level = log_level
        self._complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)

        self._point_operations, self._rotations_cartesian = self._get_point_operations()
        (
            self._grid_points,
            self._ir_grid_points,
            self._grid_weights,
        ) = self._get_grid_info(grid_points)
        self._grid_point_count = 0

        if sigmas is None:
            self._sigmas = []
        else:
            self._sigmas = list(sigmas)
        self._sigma_cutoff = sigma_cutoff
        self._collision: ImagSelfEnergy | CollisionMatrix
        if temperatures is None:
            self._temperatures = None
        else:
            self._temperatures = np.array(temperatures, dtype="double")
        self._boundary_mfp = boundary_mfp

        self._pp.nac_q_direction = None
        (
            self._frequencies,
            self._eigenvectors,
            self._phonon_done,
        ) = self._pp.get_phonons()
        if not self._pp.phonon_all_done:
            self._pp.run_phonon_solver()

        self._is_isotope = is_isotope
        if mass_variances is not None:
            self._is_isotope = True
        self._isotope: Isotope
        self._mass_variances: np.ndarray
        if self._is_isotope:
            self._set_isotope(mass_variances)

        self._read_gamma = False
        self._read_gamma_iso = False

        # Allocated in self._allocate_values.
        self._gamma: NDArray
        self._gamma_iso: NDArray | None = None

        self._conversion_factor = get_unit_to_WmK() / self._pp.primitive.volume
        self._averaged_pp_interaction: NDArray | None = None

        self._conductivity_components: ConductivityComponentsBase

    def __iter__(self):
        """Calculate mode kappa at each grid point."""
        return self

    def __next__(self):
        """Return grid point count for mode kappa."""
        if self._grid_point_count == len(self._grid_points):
            if self._log_level:
                print(
                    "=================== End of collection of collisions "
                    "==================="
                )
            raise StopIteration
        else:
            self._run_at_grid_point()
            self._grid_point_count += 1
            return self._grid_point_count - 1

    @property
    def mode_heat_capacities(self) -> NDArray:
        """Return mode heat capacity at constant volume at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._conductivity_components.mode_heat_capacities

    @property
    def group_velocities(self) -> NDArray:
        """Return group velocities at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._conductivity_components.group_velocities

    @property
    def mesh_numbers(self) -> NDArray:
        """Return mesh numbers of GR-grid."""
        return self._pp.mesh_numbers

    @property
    def bz_grid(self) -> BZGrid:
        """Return GR-grid."""
        return self._pp.bz_grid

    @property
    def frequencies(self) -> NDArray:
        """Return frequencies at grid points.

        Grid points are those at mode kappa are calculated.

        """
        assert self._frequencies is not None
        return self._frequencies[self._grid_points]

    @property
    def qpoints(self) -> NDArray:
        """Return q-points where mode kappa are calculated."""
        return np.array(
            get_qpoints_from_bz_grid_points(self._grid_points, self._pp.bz_grid),
            dtype="double",
            order="C",
        )

    @property
    def grid_points(self) -> NDArray:
        """Return grid point indices where mode kappa are calculated.

        Grid point indices are given in BZ-grid.

        """
        return self._grid_points

    @property
    def grid_weights(self) -> NDArray:
        """Return grid point weights where mode kappa are calculated."""
        return self._grid_weights

    @property
    def temperatures(self) -> NDArray | None:
        """Setter and getter of temperatures."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        self._temperatures = np.array(temperatures, dtype="double")
        self._allocate_values()

    @property
    def gamma(self) -> NDArray:
        """Setter and getter of gamma."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True

    @property
    def gamma_isotope(self) -> NDArray | None:
        """Setter and getter of gamma from isotope."""
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, gamma_iso):
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = True

    @property
    def sigmas(self) -> Sequence[float | None]:
        """Return sigmas."""
        return self._sigmas

    @property
    def sigma_cutoff_width(self) -> float | None:
        """Return smearing width cutoff."""
        return self._sigma_cutoff

    @property
    def grid_point_count(self) -> int:
        """Return iterator count of self."""
        return self._grid_point_count

    @property
    def averaged_pp_interaction(self) -> NDArray | None:
        """Return averaged pp strength."""
        return self._averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary MFP."""
        return self._boundary_mfp

    @property
    def number_of_sampling_grid_points(self) -> int:
        """Return number of grid points.

        This is calculated by the sum of numbers of arms of k-start in
        `Conductivity._set_gv_by_gv`.

        """
        return self._conductivity_components.number_of_sampling_grid_points

    def _get_point_operations(self) -> tuple[np.ndarray, np.ndarray]:
        """Return reciprocal point group operations.

        Returns
        -------
        point_operations : ndarray
            Operations in reduced coordinates.
            shape=(num_operations, 3, 3), dtype='int64'
        rotations_cartesian : ndarray
            Operations in Cartesian coordinates.
            shape=(num_operations, 3, 3), dtype='double'

        """
        if not self._is_kappa_star:
            point_operations = np.array(
                [np.eye(3, dtype="int64")], dtype="int64", order="C"
            )
            rotations_cartesian = np.array(
                [np.eye(3, dtype="double")], dtype="double", order="C"
            )
        else:
            point_operations = self._pp.bz_grid.reciprocal_operations
            rotations_cartesian = self._pp.bz_grid.rotations_cartesian

        return point_operations, rotations_cartesian

    def _get_grid_info(self, grid_points) -> tuple[NDArray, NDArray, NDArray]:
        """Return grid point information in BZGrid.

        Returns
        -------
        grid_points : ndarray
            Grid point indices in BZ-grid to be iterated over.
            shape=(len(grid_points),), dtype='int64'
        ir_grid_points : ndarray
            Irreducible grid points in BZ-grid on regular grid.
            shape=(len(ir_grid_points),), dtype='int64'
        grid_weights : ndarray
            Grid weights of `grid_points`. If grid symmetry is not broken,
            these values are equivalent to numbers of k-star arms.

        """
        ir_grid_points, grid_weights = self._get_ir_grid_points(grid_points)
        if grid_points is not None:  # Specify grid points
            _grid_points = np.array(grid_points, dtype="int64")
            _ir_grid_points = ir_grid_points
            _grid_weights = grid_weights
        elif not self._is_kappa_star:  # All grid points
            _grid_points = self._pp.bz_grid.grg2bzg
            _ir_grid_points = _grid_points
            _grid_weights = np.ones(len(_grid_points), dtype="int64")
        else:  # Automatic sampling
            _grid_points = ir_grid_points
            _ir_grid_points = ir_grid_points
            _grid_weights = grid_weights
        return _grid_points, _ir_grid_points, _grid_weights

    @abstractmethod
    def _run_at_grid_point(self):
        """Run at conductivity calculation at specified grid point."""
        raise NotImplementedError()

    @abstractmethod
    def _allocate_values(self):
        """Allocate necessary data arrays."""
        raise NotImplementedError()

    @abstractmethod
    def _set_velocities(self, i_gp, i_data):
        """Set velocities at grid point and at data location."""
        raise NotImplementedError()

    @abstractmethod
    def _set_cv(self, i_gp, i_data):
        """Set heat capacity at grid point and at data location."""
        raise NotImplementedError()

    def _get_ir_grid_points(self, grid_points):
        """Return ir-grid-points and grid weights in BZGrid."""
        ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(
            self._pp.bz_grid
        )
        ir_grid_points = np.array(
            self._pp.bz_grid.grg2bzg[ir_grid_points], dtype="int64"
        )
        if grid_points is None:
            grid_weights = ir_grid_weights
        else:
            weights = np.zeros_like(ir_grid_map)
            for gp in ir_grid_map:
                weights[gp] += 1
            grid_weights = np.array(
                weights[ir_grid_map[self._pp.bz_grid.bzg2grg[grid_points]]],
                dtype="int64",
            )

        return ir_grid_points, grid_weights

    def _get_gamma_isotope_at_sigmas(self, i):
        gamma_iso = []
        for sigma in self._sigmas:
            if self._log_level:
                text = "Calculating Gamma of ph-isotope with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)

            self._isotope.sigma = sigma
            self._isotope.set_phonons(
                self._frequencies,
                self._eigenvectors,
                self._phonon_done,
                dm=self._pp.dynamical_matrix,
            )
            gp = self._grid_points[i]
            self._isotope.set_grid_point(gp)
            self._isotope.run()
            gamma_iso.append(self._isotope.gamma)

        return np.array(gamma_iso, dtype="double", order="C")

    def _set_isotope(self, mass_variances):
        self._isotope = Isotope(
            self._pp.mesh_numbers,
            self._pp.primitive,
            mass_variances=mass_variances,
            bz_grid=self._pp.bz_grid,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
            symprec=self._pp.primitive_symmetry.tolerance,
            cutoff_frequency=self._pp.cutoff_frequency,
            lapack_zheev_uplo=self._pp.lapack_zheev_uplo,
        )
        self._mass_variances = self._isotope.mass_variances

    def _get_main_diagonal(self, i, j, k):
        main_diagonal = self._gamma[j, k, i].copy()
        if self._is_isotope:
            main_diagonal += self._gamma_iso[j, i]
        if self._boundary_mfp is not None:
            main_diagonal += self._get_boundary_scattering(i)
        return main_diagonal

    def _get_boundary_scattering(self, i_gp):
        num_band = len(self._pp.primitive) * 3
        g_boundary = np.zeros(num_band, dtype="double")
        try:
            gv = self._conductivity_components.group_velocities
        except AttributeError:
            print("(_get_boundary_scattering) _gv has to be implemented.")
            return g_boundary

        for ll in range(num_band):
            g_boundary[ll] = (
                np.linalg.norm(gv[i_gp, ll])
                * get_physical_units().Angstrom
                * 1e6
                / (4 * np.pi * self._boundary_mfp)
            )
        return g_boundary

    def _show_log_header(self, i_gp):
        if self._log_level:
            bzgp = self._grid_points[i_gp]
            print(
                "======================= Grid point %d (%d/%d) "
                "=======================" % (bzgp, i_gp + 1, len(self._grid_points))
            )
            qpoint = get_qpoints_from_bz_grid_points(bzgp, self._pp.bz_grid)
            print("q-point: (%5.2f %5.2f %5.2f)" % tuple(qpoint))
            if self._boundary_mfp is not None:
                if self._boundary_mfp > 1000:
                    print(
                        "Boundary mean free path (millimeter): %.3f"
                        % (self._boundary_mfp / 1000.0)
                    )
                else:
                    print(
                        "Boundary mean free path (micrometer): %.5f"
                        % self._boundary_mfp
                    )
            if self._is_isotope:
                print(
                    (
                        "Mass variance parameters: "
                        + "%5.2e " * len(self._mass_variances)
                    )
                    % tuple(self._mass_variances)
                )
            print(end="", flush=True)
