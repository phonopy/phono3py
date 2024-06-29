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

import textwrap
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.units import EV, Angstrom, Kb, THz, THzToEv

from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import get_grid_points_by_rotations, get_ir_grid_points
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.imag_self_energy import ImagSelfEnergy
from phono3py.phonon3.interaction import Interaction

unit_to_WmK = (
    (THz * Angstrom) ** 2 / (Angstrom**3) * EV / THz / (2 * np.pi)
)  # 2pi comes from definition of lifetime.


class HeatCapacityMixIn:
    """Heat capacity mix-in.

    Used by other mix-in.

    """

    @property
    def mode_heat_capacities(self):
        """Return mode heat capacity at constant volume at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._cv

    def get_mode_heat_capacities(self):
        """Return mode heat capacity at constant volume at grid points.

        Grid points are those at mode kappa are calculated.

        """
        warnings.warn(
            "Use attribute, Conductivity.mode_heat_capacities "
            "instead of Conductivity.get_mode_heat_capacities().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_heat_capacities

    def _set_cv(self, i_gp, i_data):
        """Set mode heat capacity.

        The array has to be allocated somewhere out of the mix-in.

        self._cv = np.zeros(
            (num_temp, num_grid_points, num_band0), order="C", dtype="double"
        )

        """
        grid_point = self._grid_points[i_gp]
        freqs = self._frequencies[grid_point][self._pp.band_indices] * THzToEv
        cutoff = self._pp.cutoff_frequency * THzToEv
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype="double")
        # x=freq/T has to be small enough to avoid overflow of exp(x).
        # x < 100 is the hard-corded criterion.
        # Otherwise just set 0.
        for i, f in enumerate(freqs):
            if f > cutoff:
                condition = f < 100 * self._temperatures * Kb
                cv[:, i] = np.where(
                    condition,
                    mode_cv(np.where(condition, self._temperatures, 10000), f),
                    0,
                )
        self._cv[:, i_data, :] = cv


class ConductivityMixIn(HeatCapacityMixIn):
    """Thermal conductivity mix-in.

    Used by ConductivityRTA and ConductivityLBTE.

    """

    @property
    def kappa(self):
        """Return kappa."""
        return self._kappa

    def get_kappa(self):
        """Return kappa."""
        warnings.warn(
            "Use attribute, Conductivity.kappa " "instead of Conductivity.get_kappa().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.kappa

    @property
    def mode_kappa(self):
        """Return mode_kappa."""
        return self._mode_kappa

    def get_mode_kappa(self):
        """Return mode_kappa."""
        warnings.warn(
            "Use attribute, Conductivity.mode_kappa "
            "instead of Conductivity.get_mode_kappa().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_kappa

    @property
    def group_velocities(self):
        """Return group velocities at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._gv

    def get_group_velocities(self):
        """Return group velocities at grid points.

        Grid points are those at mode kappa are calculated.

        """
        warnings.warn(
            "Use attribute, Conductivity.group_velocities "
            "instead of Conductivity.get_group_velocities().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.group_velocities

    @property
    def gv_by_gv(self):
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        return self._gv_sum2

    def get_gv_by_gv(self):
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        warnings.warn(
            "Use attribute, Conductivity.gv_by_gv "
            "instead of Conductivity.get_gv_by_gv().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gv_by_gv

    def _init_velocity(self, gv_delta_q):
        self._velocity_obj = GroupVelocity(
            self._pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=self._pp.primitive_symmetry,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
        )

    def _set_velocities(self, i_gp, i_data):
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
            for bz_gp in unique_gps.tolist():  # To conver to int type.
                self._velocity_obj.run([self._get_qpoint_from_gp_index(bz_gp)])
                gvs[bz_gp] = self._velocity_obj.group_velocities[
                    0, self._pp.band_indices, :
                ]
            gv = np.zeros_like(gvs[irgp])
            for bz_gp, r in zip(gps_rotated, self._rotations_cartesian):
                gv += np.dot(gvs[bz_gp], r)  # = dot(r_inv, gv)
            return gv / len(self._point_operations)
        else:
            self._velocity_obj.run([self._get_qpoint_from_gp_index(irgp)])
            return self._velocity_obj.group_velocities[0, self._pp.band_indices, :]

    def _set_gv_by_gv(self, i_gp, i_data):
        """Outer product of group velocities.

        (v x v) [num_k*, num_freqs, 3, 3]

        """
        gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i_gp, i_data)
        self._num_sampling_grid_points += order_kstar

        # Sum all vxv at k*
        for j, vxv in enumerate(([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            self._gv_sum2[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

    def _get_gv_by_gv(self, i_gp, i_data):
        multi = self._get_multiplicity_at_q(i_gp)
        gv = self._gv[i_data]
        gv_by_gv = np.zeros((len(gv), 3, 3), dtype="double")
        for r in self._rotations_cartesian:
            gvs_rot = np.dot(gv, r.T)
            gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv /= multi
        return gv_by_gv, self._get_kstar_order(i_gp, multi)


class ConductivityBase(ABC):
    """Base class of Conductivity classes.

    All Conductivity* classes have to inherit this base class.

    self._gv has to be allocated in the inherited classes.

    """

    _average_gv_over_kstar = False

    def __init__(
        self,
        interaction: Interaction,
        grid_points=None,
        temperatures: Optional[Union[List, np.ndarray]] = None,
        sigmas: Optional[Union[List, np.ndarray]] = None,
        sigma_cutoff: Optional[float] = None,
        is_isotope=False,
        mass_variances: Optional[Union[List, np.ndarray]] = None,
        boundary_mfp: Optional[float] = None,
        is_kappa_star=True,
        gv_delta_q=None,
        is_full_pp=False,
        log_level=0,
    ):
        """Init method.

        interaction : Interaction
            Interaction class instance.
        grid_points : array_like or None, optional
            Grid point indices in BZgrid. When None, ir-grid points are searched
            internally. Default is None.
            shape=(grid_points, ), dtype='int_'.
        temperatures : array_like, optional, default is None
            Temperatures at which thermal conductivity is calculated.
            shape=(temperature_points, ), dtype='double'.
        sigmas : array_like, optional, default is None
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
        boundary_mfp : float, optiona, default is None
            Mean free path in micrometre to calculate simple boundary
            scattering contribution to thermal conductivity.
            None ignores this contribution.
        is_kappa_star : bool, optional
            When True, reciprocal space symmetry is used to calculate
            lattice thermal conductivity. This calculation is peformed
            iterating over specific grid points. With `is_kappa_star=True`
            and `grid_points=None`, ir-grid points are used for the iteration.
            Default is True.
        gv_delta_q : float, optional, default is None,  # for group velocity
            With non-analytical correction, group velocity is calculated
            by central finite difference method. This value gives the distance
            in both directions in 1/Angstrom. The default value will be 1e-5.
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
        self._grid_point_count: int = 0
        self._num_sampling_grid_points: int = 0

        self._sigmas: List
        if sigmas is None:
            self._sigmas = []
        else:
            self._sigmas = list(sigmas)
        self._sigma_cutoff = sigma_cutoff
        self._collision: Union[ImagSelfEnergy, CollisionMatrix]
        self._temperatures: Optional[np.ndarray]
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
        if (self._phonon_done == 0).any():
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
        self._gv: np.ndarray
        self._gamma: np.ndarray
        self._gamma_iso: Optional[np.ndarray] = None

        volume = self._pp.primitive.volume
        self._conversion_factor = unit_to_WmK / volume

        self._averaged_pp_interaction = None

        # `self._velocity_obj` is the instance of an inherited class of
        # `GroupVelocity`. `self._init_velocity()` is the method setup the instance,
        # which must be implmented in the inherited class of `ConductivityBase`.
        self._velocity_obj: GroupVelocity
        self._init_velocity(gv_delta_q)

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
    def mesh_numbers(self):
        """Return mesh numbers of GR-grid."""
        return self._pp.mesh_numbers

    def get_mesh_numbers(self):
        """Return mesh numbers of GR-grid."""
        warnings.warn(
            "Use attribute, Conductivity.mesh_numbers "
            "instead of Conductivity.get_mesh_numbers().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mesh_numbers

    @property
    def bz_grid(self):
        """Return GR-grid."""
        return self._pp.bz_grid

    @property
    def frequencies(self):
        """Return frequencies at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._frequencies[self._grid_points]

    def get_frequencies(self):
        """Return frequencies at grid points.

        Grid points are those at mode kappa are calculated.

        """
        warnings.warn(
            "Use attribute, Conductivity.frequencies "
            "instead of Conductivity.get_frequencies().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frequencies

    @property
    def qpoints(self):
        """Return q-points where mode kappa are calculated."""
        return np.array(
            self._get_qpoint_from_gp_index(self._grid_points),
            dtype="double",
            order="C",
        )

    def get_qpoints(self):
        """Return q-points where mode kappa are calculated."""
        warnings.warn(
            "Use attribute, Conductivity.qpoints "
            "instead of Conductivity.get_qpoints().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.qpoints

    @property
    def grid_points(self):
        """Return grid point indices where mode kappa are calculated.

        Grid point indices are given in BZ-grid.

        """
        return self._grid_points

    def get_grid_points(self):
        """Return grid point indices where mode kappa are calculated.

        Grid point indices are given in BZ-grid.

        """
        warnings.warn(
            "Use attribute, Conductivity.grid_points "
            "instead of Conductivity.get_grid_points().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.grid_points

    @property
    def grid_weights(self):
        """Return grid point weights where mode kappa are calculated."""
        return self._grid_weights

    def get_grid_weights(self):
        """Return grid point weights where mode kappa are calculated."""
        warnings.warn(
            "Use attribute, Conductivity.grid_weights "
            "instead of Conductivity.get_grid_weights().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.grid_weights

    @property
    def temperatures(self):
        """Setter and getter of temperatures."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        self._temperatures = np.array(temperatures, dtype="double")
        self._allocate_values()

    def get_temperatures(self):
        """Return temperatures."""
        warnings.warn(
            "Use attribute, Conductivity.temperatures "
            "instead of Conductivity.get_temperatures().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.temperatures

    def set_temperatures(self, temperatures):
        """Set temperatures."""
        warnings.warn(
            "Use attribute, Conductivity.temperatures "
            "instead of Conductivity.set_temperatures().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.temperatures = temperatures

    @property
    def gamma(self):
        """Setter and getter of gamma."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True

    def get_gamma(self):
        """Return gamma."""
        warnings.warn(
            "Use attribute, Conductivity.gamma " "instead of Conductivity.get_gamma().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gamma

    def set_gamma(self, gamma):
        """Set gamma."""
        warnings.warn(
            "Use attribute, Conductivity.gamma " "instead of Conductivity.set_gamma().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.gamma = gamma

    @property
    def gamma_isotope(self):
        """Setter and getter of gamma from isotope."""
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, gamma_iso):
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = True

    def get_gamma_isotope(self):
        """Return gamma from isotope."""
        warnings.warn(
            "Use attribute, Conductivity.gamma_isotope "
            "instead of Conductivity.get_gamma_isotope().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gamma_isotope

    def set_gamma_isotope(self, gamma_iso):
        """Set gamma from isotope."""
        warnings.warn(
            "Use attribute, Conductivity.gamma_isotope "
            "instead of Conductivity.set_gamma_isotope().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.gamma_isotope = gamma_iso

    @property
    def sigmas(self):
        """Return sigmas."""
        return self._sigmas

    def get_sigmas(self):
        """Return sigmas."""
        warnings.warn(
            "Use attribute, Conductivity.sigmas "
            "instead of Conductivity.get_sigmas().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sigmas

    @property
    def sigma_cutoff_width(self):
        """Return smearing width cutoff."""
        return self._sigma_cutoff

    def get_sigma_cutoff_width(self):
        """Return smearing width cutoff."""
        warnings.warn(
            "Use attribute, Conductivity.sigma_cutoff_width "
            "instead of Conductivity.get_sigma_cutoff_width().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sigma_cutoff_width

    @property
    def grid_point_count(self):
        """Return interator count of self."""
        return self._grid_point_count

    def get_grid_point_count(self):
        """Return interator count of self."""
        warnings.warn(
            "Use attribute, Conductivity.grid_point_count "
            "instead of Conductivity.get_grid_point_count().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.grid_point_count

    @property
    def averaged_pp_interaction(self):
        """Return averaged pp strength."""
        return self._averaged_pp_interaction

    def get_averaged_pp_interaction(self):
        """Return averaged pp interaction strength."""
        warnings.warn(
            "Use attribute, Conductivity.averaged_pp_interaction "
            "instead of Conductivity.get_averaged_pp_interaction().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float:
        """Return boundary MFP."""
        return self._boundary_mfp

    def get_number_of_sampling_grid_points(self):
        """Return number of grid points.

        This is calculated by the sum of numbers of arms of k-start in
        `Conductivity._set_gv_by_gv`.

        """
        return self._num_sampling_grid_points

    def _get_point_operations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return reciprocal point group operations.

        Returns
        -------
        point_operations : ndarray
            Operations in reduced coordinates.
            shape=(num_operations, 3, 3), dtype='int_'
        rotations_cartesian : ndarray
            Operations in Cartesian coordinates.
            shape=(num_operations, 3, 3), dtype='double'

        """
        if not self._is_kappa_star:
            point_operations = np.array(
                [np.eye(3, dtype="int_")], dtype="int_", order="C"
            )
            rotations_cartesian = np.array(
                [np.eye(3, dtype="double")], dtype="double", order="C"
            )
        else:
            point_operations = self._pp.bz_grid.reciprocal_operations
            rotations_cartesian = self._pp.bz_grid.rotations_cartesian

        return point_operations, rotations_cartesian

    def _get_grid_info(self, grid_points) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return grid point information in BZGrid.

        Returns
        -------
        grid_points : ndarray
            Grid point indices in BZ-grid to be iterated over.
            shape=(len(grid_points),), dtype='int_'
        ir_grid_points : ndarray
            Irreducible grid points in BZ-grid on regular grid.
            shape=(len(ir_grid_points),), dtype='int_'
        grid_weights : ndarray
            Grid weights of `grid_points`. If grid symmetry is not broken,
            these values are equivalent to numbers of k-star arms.

        """
        ir_grid_points, grid_weights = self._get_ir_grid_points(grid_points)
        if grid_points is not None:  # Specify grid points
            _grid_points = np.array(grid_points, dtype="int_")
            _ir_grid_points = ir_grid_points
            _grid_weights = grid_weights
        elif not self._is_kappa_star:  # All grid points
            _grid_points = self._pp.bz_grid.grg2bzg
            _ir_grid_points = _grid_points
            _grid_weights = np.ones(len(_grid_points), dtype="int_")
        else:  # Automatic sampling
            _grid_points = ir_grid_points
            _ir_grid_points = ir_grid_points
            _grid_weights = grid_weights
        return _grid_points, _ir_grid_points, _grid_weights

    @abstractmethod
    def _run_at_grid_point(self):
        """Run at conductivity calculation at specified grid point.

        Should be implementated in Conductivity* class.

        """
        raise NotImplementedError()

    @abstractmethod
    def _allocate_values(self):
        """Allocate necessary data arrays.

        Should be implementated in Conductivity* class.

        """
        raise NotImplementedError()

    @abstractmethod
    def _set_velocities(self, i_gp, i_data):
        """Set velocities at grid point and at data location.

        Should be implementated in Conductivity*MixIn.

        """
        raise NotImplementedError()

    @abstractmethod
    def _init_velocity(self, gv_delta_q):
        """Initialize velocitiy class instance.

        Should be implementated in Conductivity*MixIn.

        """
        raise NotImplementedError()

    @abstractmethod
    def _set_cv(self, i_gp, i_data):
        """Set heat capacity at grid point and at data location.

        Should be implementated in Conductivity*MixIn.

        """
        raise NotImplementedError()

    def _get_ir_grid_points(self, grid_points):
        """Return ir-grid-points and grid weights in BZGrid."""
        ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(
            self._pp.bz_grid
        )
        ir_grid_points = np.array(
            self._pp.bz_grid.grg2bzg[ir_grid_points], dtype="int_"
        )
        if grid_points is None:
            grid_weights = ir_grid_weights
        else:
            weights = np.zeros_like(ir_grid_map)
            for gp in ir_grid_map:
                weights[gp] += 1
            grid_weights = np.array(
                weights[ir_grid_map[self._pp.bz_grid.bzg2grg[grid_points]]],
                dtype="int_",
            )

        return ir_grid_points, grid_weights

    def _get_qpoint_from_gp_index(self, i_gps):
        """Return q-point(s) in reduced coordinates of grid point(s).

        Parameters
        ----------
        i_gps : int or ndarray
            BZ-grid index (int) or indices (ndarray).

        """
        return np.dot(self._pp.bz_grid.addresses[i_gps], self._pp.bz_grid.QDinv.T)

    def _get_multiplicity_at_q(self, i_gp):
        """Return multiplicity (order of site-symmetry) of q-point."""
        if self._is_kappa_star:
            q = self._get_qpoint_from_gp_index(self._grid_points[i_gp])
            reclat = np.linalg.inv(self._pp.primitive.cell)
            multi = 0
            for q_rot in [np.dot(r, q) for r in self._point_operations]:
                diff = q - q_rot
                diff -= np.rint(diff)
                dist = np.linalg.norm(np.dot(reclat, diff))
                if dist < self._pp.primitive_symmetry.tolerance:
                    multi += 1
        else:
            multi = 1
        return multi

    def _get_kstar_order(self, i_gp, multi):
        """Return order (number of arms) of kstar.

        multi : int
            Multiplicity of q-point of `i_gp`, which can be obtained by
            `self._get_multiplicity_at_q(i_gp)`.

        """
        order_kstar = len(self._point_operations) // multi
        if order_kstar != self._grid_weights[i_gp]:
            if self._log_level:
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
            gv = self._gv
        except AttributeError:
            print("(_get_boundary_scattering) _gv has to be implemented.")
            return g_boundary

        for ll in range(num_band):
            g_boundary[ll] = (
                np.linalg.norm(gv[i_gp, ll])
                * Angstrom
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
            qpoint = self._get_qpoint_from_gp_index(bzgp)
            print("q-point: (%5.2f %5.2f %5.2f)" % tuple(qpoint))
            if self._boundary_mfp is not None:
                if self._boundary_mfp > 1000:
                    print(
                        "Boundary mean free path (millimetre): %.3f"
                        % (self._boundary_mfp / 1000.0)
                    )
                else:
                    print(
                        "Boundary mean free path (micrometre): %.5f"
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
