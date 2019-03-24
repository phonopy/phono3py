# Copyright (C) 2016 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
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

import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import (get_supercell, get_primitive,
                                     guess_primitive_matrix)
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import (
    get_fc2,
    symmetrize_force_constants,
    symmetrize_compact_force_constants,
    set_translational_invariance,
    set_permutation_symmetry)
from phonopy.harmonic.displacement import get_least_displacements
from phonopy.harmonic.displacement import directions_to_displacement_dataset
from phonopy.phonon.mesh import length2mesh
from phono3py.version import __version__
from phono3py.phonon3.imag_self_energy import (get_imag_self_energy,
                                               write_imag_self_energy)
from phono3py.phonon3.frequency_shift import get_frequency_shift
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.conductivity_RTA import get_thermal_conductivity_RTA
from phono3py.phonon3.conductivity_LBTE import get_thermal_conductivity_LBTE
from phono3py.phonon3.joint_dos import JointDos
from phono3py.phonon3.displacement_fc3 import (get_third_order_displacements,
                                               direction_to_displacement)
from phono3py.file_IO import write_joint_dos
from phono3py.other.isotope import Isotope
from phono3py.phonon3.fc3 import (get_fc3,
                                  set_permutation_symmetry_fc3,
                                  set_permutation_symmetry_compact_fc3,
                                  set_translational_invariance_fc3,
                                  set_translational_invariance_compact_fc3,
                                  cutoff_fc3_by_zero)


class Phono3py(object):
    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 primitive_matrix=None,
                 phonon_supercell_matrix=None,
                 masses=None,
                 mesh=None,
                 band_indices=None,
                 sigmas=None,
                 sigma_cutoff=None,
                 cutoff_frequency=1e-4,
                 frequency_factor_to_THz=VaspToTHz,
                 is_symmetry=True,
                 is_mesh_symmetry=True,
                 symmetrize_fc3q=False,
                 symprec=1e-5,
                 log_level=0,
                 lapack_zheev_uplo='L'):
        if sigmas is None:
            self._sigmas = [None]
        else:
            self._sigmas = sigmas
        self._sigma_cutoff = sigma_cutoff
        self._symprec = symprec
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_mesh_symmetry = is_mesh_symmetry
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._symmetrize_fc3q = symmetrize_fc3q
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        if type(primitive_matrix) is str and primitive_matrix == 'auto':
            self._primitive_matrix = self._guess_primitive_matrix()
        else:
            self._primitive_matrix = primitive_matrix
        self._phonon_supercell_matrix = phonon_supercell_matrix  # optional
        self._supercell = None
        self._primitive = None
        self._phonon_supercell = None
        self._phonon_primitive = None
        self._build_supercell()
        self._build_primitive_cell()
        self._build_phonon_supercell()
        self._build_phonon_primitive_cell()

        if masses is not None:
            self._set_masses(masses)

        # Set supercell, primitive, and phonon supercell symmetries
        self._symmetry = None
        self._primitive_symmetry = None
        self._phonon_supercell_symmetry = None
        self._search_symmetry()
        self._search_primitive_symmetry()
        self._search_phonon_supercell_symmetry()

        # Displacements and supercells
        self._supercells_with_displacements = None
        self._displacement_dataset = None
        self._phonon_displacement_dataset = None
        self._phonon_supercells_with_displacements = None

        # Thermal conductivity
        self._thermal_conductivity = None  # conductivity_RTA object

        # Imaginary part of self energy at frequency points
        self._imag_self_energy = None
        self._scattering_event_class = None

        self._grid_points = None
        self._frequency_points = None
        self._temperatures = None

        # Other variables
        self._fc2 = None
        self._fc3 = None
        self._nac_params = None

        # Setup interaction
        self._interaction = None
        self._mesh_numbers = None
        self._band_indices = None
        self._band_indices_flatten = None
        if mesh is not None:
            _mesh = np.array(mesh)
            mesh_nums = None
            if _mesh.shape:
                if _mesh.shape == (3,):
                    mesh_nums = mesh
            else:
                mesh_nums = length2mesh(mesh, self._primitive.get_cell())
            if mesh_nums is None:
                msg = "mesh has inappropriate type."
                raise TypeError(msg)
            self._mesh_numbers = mesh_nums
        self.set_band_indices(band_indices)

    def set_band_indices(self, band_indices):
        if band_indices is None:
            num_band = self._primitive.get_number_of_atoms() * 3
            self._band_indices = [np.arange(num_band, dtype='intc')]
        else:
            self._band_indices = band_indices
        self._band_indices_flatten = np.hstack(
            self._band_indices).astype('intc')

    def set_phph_interaction(self,
                             nac_params=None,
                             nac_q_direction=None,
                             constant_averaged_interaction=None,
                             frequency_scale_factor=None,
                             unit_conversion=None,
                             solve_dynamical_matrices=True):
        if self._mesh_numbers is None:
            print("'mesh' has to be set in Phono3py instantiation.")
            raise RuntimeError

        self._nac_params = nac_params
        self._interaction = Interaction(
            self._supercell,
            self._primitive,
            self._mesh_numbers,
            self._primitive_symmetry,
            fc3=self._fc3,
            band_indices=self._band_indices_flatten,
            constant_averaged_interaction=constant_averaged_interaction,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            frequency_scale_factor=frequency_scale_factor,
            unit_conversion=unit_conversion,
            cutoff_frequency=self._cutoff_frequency,
            is_mesh_symmetry=self._is_mesh_symmetry,
            symmetrize_fc3q=self._symmetrize_fc3q,
            lapack_zheev_uplo=self._lapack_zheev_uplo)
        self._interaction.set_nac_q_direction(nac_q_direction=nac_q_direction)
        self._interaction.set_dynamical_matrix(
            self._fc2,
            self._phonon_supercell,
            self._phonon_primitive,
            nac_params=self._nac_params,
            solve_dynamical_matrices=solve_dynamical_matrices,
            verbose=self._log_level)

    def set_phonon_data(self, frequencies, eigenvectors, grid_address):
        if self._interaction is not None:
            return self._interaction.set_phonon_data(frequencies,
                                                     eigenvectors,
                                                     grid_address)
        else:
            return False

    def get_phonon_data(self):
        if self._interaction is not None:
            grid_address = self._interaction.get_grid_address()
            freqs, eigvecs, _ = self._interaction.get_phonons()
            return freqs, eigvecs, grid_address
        else:
            msg = "set_phph_interaction has to be done."
            raise RuntimeError(msg)

    def generate_displacements(self,
                               distance=0.03,
                               cutoff_pair_distance=None,
                               is_plusminus='auto',
                               is_diagonal=True):
        direction_dataset = get_third_order_displacements(
            self._supercell,
            self._symmetry,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal)
        self._displacement_dataset = direction_to_displacement(
            direction_dataset,
            distance,
            self._supercell,
            cutoff_distance=cutoff_pair_distance)

        if self._phonon_supercell_matrix is not None:
            # 'is_diagonal=False' below is made intentionally. For
            # third-order force constants, we need better accuracy,
            # and I expect this choice is better for it, but not very
            # sure.
            # In phono3py, two atoms are displaced for each
            # configuration and the displacements are chosen, first
            # displacement from the perfect supercell, then second
            # displacement, considering symmetry. If I choose
            # is_diagonal=False for the first displacement, the
            # symmetry is less broken and the number of second
            # displacements can be smaller than in the case of
            # is_diagonal=True for the first displacement.  This is
            # done in the call get_least_displacements() in
            # phonon3.displacement_fc3.get_third_order_displacements().
            #
            # The call get_least_displacements() is only for the
            # second order force constants, but 'is_diagonal=False' to
            # be consistent with the above function call, and also for
            # the accuracy when calculating ph-ph interaction
            # strength because displacement directions are better to be
            # close to perpendicular each other to fit force constants.
            #
            # On the discussion of the accuracy, these are just my
            # expectation when I designed phono3py in the early time,
            # and in fact now I guess not very different. If these are
            # little different, then I should not surprise users to
            # change the default behaviour. At this moment, this is
            # open question and we will have more advance and should
            # have better specificy external software on this.
            phonon_displacement_directions = get_least_displacements(
                self._phonon_supercell_symmetry,
                is_plusminus=is_plusminus,
                is_diagonal=False)
            self._phonon_displacement_dataset = directions_to_displacement_dataset(
                phonon_displacement_directions,
                distance,
                self._phonon_supercell)

    def produce_fc2(self,
                    forces_fc2,
                    displacement_dataset=None,
                    symmetrize_fc2=False,
                    is_compact_fc=False,
                    use_alm=False,
                    alm_options=None):
        if displacement_dataset is None:
            if self._phonon_displacement_dataset is None:
                disp_dataset = self._displacement_dataset
            else:
                disp_dataset = self._phonon_displacement_dataset
        else:
            disp_dataset = displacement_dataset

        for forces, disp1 in zip(forces_fc2, disp_dataset['first_atoms']):
            disp1['forces'] = forces

        if is_compact_fc:
            p2s_map = self._phonon_primitive.p2s_map
        else:
            p2s_map = None

        if use_alm:
            from phonopy.interface.alm import get_fc2 as get_fc2_alm
            self._fc2 = get_fc2_alm(self._phonon_supercell,
                                    self._phonon_primitive,
                                    disp_dataset,
                                    atom_list=p2s_map,
                                    alm_options=alm_options,
                                    log_level=self._log_level)
        else:
            self._fc2 = get_fc2(self._phonon_supercell,
                                self._phonon_supercell_symmetry,
                                disp_dataset,
                                atom_list=p2s_map)
            if symmetrize_fc2:
                if is_compact_fc:
                    symmetrize_compact_force_constants(
                        self._fc2, self._phonon_primitive)
                else:
                    symmetrize_force_constants(self._fc2)

    def produce_fc3(self,
                    forces_fc3,
                    displacement_dataset=None,
                    cutoff_distance=None,  # set fc3 zero
                    symmetrize_fc3r=False,
                    is_compact_fc=False,
                    use_alm=False,
                    alm_options=None):
        if displacement_dataset is None:
            disp_dataset = self._displacement_dataset
        else:
            disp_dataset = displacement_dataset

        if use_alm:
            from phono3py.other.alm_wrapper import get_fc3 as get_fc3_alm
            fc2, fc3 = get_fc3_alm(self._supercell,
                                   self._primitive,
                                   forces_fc3,
                                   disp_dataset,
                                   self._symmetry,
                                   alm_options=alm_options,
                                   is_compact_fc=is_compact_fc,
                                   log_level=self._log_level)
        else:
            fc2, fc3 = self._get_fc3(forces_fc3,
                                     disp_dataset,
                                     is_compact_fc=is_compact_fc)
            if symmetrize_fc3r:
                if is_compact_fc:
                    set_translational_invariance_compact_fc3(
                        fc3, self._primitive)
                    set_permutation_symmetry_compact_fc3(fc3, self._primitive)
                    if self._fc2 is None:
                        symmetrize_compact_force_constants(fc2,
                                                           self._primitive)
                else:
                    set_translational_invariance_fc3(fc3)
                    set_permutation_symmetry_fc3(fc3)
                    if self._fc2 is None:
                        symmetrize_force_constants(fc2)

        # Set fc2 and fc3
        self._fc3 = fc3

        # Normally self._fc2 is overwritten in produce_fc2
        if self._fc2 is None:
            self._fc2 = fc2

    def cutoff_fc3_by_zero(self, cutoff_distance, fc3=None):
        if fc3 is None:
            _fc3 = self._fc3
        else:
            _fc3 = fc3
        cutoff_fc3_by_zero(_fc3,  # overwritten
                           self._supercell,
                           cutoff_distance,
                           self._symprec)

    def set_permutation_symmetry(self):
        if self._fc2 is not None:
            set_permutation_symmetry(self._fc2)
        if self._fc3 is not None:
            set_permutation_symmetry_fc3(self._fc3)

    def set_translational_invariance(self):
        if self._fc2 is not None:
            set_translational_invariance(self._fc2)
        if self._fc3 is not None:
            set_translational_invariance_fc3(self._fc3)

    @property
    def version(self):
        return __version__

    def get_version(self):
        return self.version

    def get_interaction_strength(self):
        return self._interaction

    def get_fc2(self):
        return self._fc2

    def set_fc2(self, fc2):
        self._fc2 = fc2

    def get_fc3(self):
        return self._fc3

    def set_fc3(self, fc3):
        self._fc3 = fc3

    @property
    def nac_params(self):
        return self._nac_params

    def get_nac_params(self):
        return self.nac_params

    @property
    def primitive(self):
        return self._primitive

    def get_primitive(self):
        return self.primitive

    @property
    def unitcell(self):
        return self._unitcell

    def get_unitcell(self):
        return self.unitcell

    @property
    def supercell(self):
        return self._supercell

    def get_supercell(self):
        return self.supercell

    @property
    def phonon_supercell(self):
        return self._phonon_supercell

    def get_phonon_supercell(self):
        return self.phonon_supercell

    @property
    def phonon_primitive(self):
        return self._phonon_primitive

    def get_phonon_primitive(self):
        return self.phonon_primitive

    @property
    def symmetry(self):
        """return symmetry of supercell"""
        return self._symmetry

    def get_symmetry(self):
        return self.symmetry

    @property
    def primitive_symmetry(self):
        """return symmetry of primitive cell"""
        return self._primitive_symmetry

    def get_primitive_symmetry(self):
        """return symmetry of primitive cell"""
        return self.primitive_symmetry

    def get_phonon_supercell_symmetry(self):
        return self._phonon_supercell_symmetry

    @property
    def supercell_matrix(self):
        return self._supercell_matrix

    def get_supercell_matrix(self):
        return self.supercell_matrix

    @property
    def phonon_supercell_matrix(self):
        return self._phonon_supercell_matrix

    def get_phonon_supercell_matrix(self):
        return self.phonon_supercell_matrix

    @property
    def primitive_matrix(self):
        return self._primitive_matrix

    def get_primitive_matrix(self):
        return self.primitive_matrix

    @property
    def unit_conversion_factor(self):
        return self._frequency_factor_to_THz

    def set_displacement_dataset(self, dataset):
        self._displacement_dataset = dataset

    @property
    def displacement_dataset(self):
        return self._displacement_dataset

    def get_displacement_dataset(self):
        return self.displacement_dataset

    def get_phonon_displacement_dataset(self):
        return self._phonon_displacement_dataset

    def get_supercells_with_displacements(self):
        if self._supercells_with_displacements is None:
            self._build_supercells_with_displacements()
        return self._supercells_with_displacements

    def get_phonon_supercells_with_displacements(self):
        if self._phonon_supercells_with_displacements is None:
            if self._phonon_displacement_dataset is not None:
                self._phonon_supercells_with_displacements = \
                  self._build_phonon_supercells_with_displacements(
                      self._phonon_supercell,
                      self._phonon_displacement_dataset)
        return self._phonon_supercells_with_displacements

    @property
    def mesh_numbers(self):
        return self._mesh_numbers

    def run_imag_self_energy(self,
                             grid_points,
                             frequency_step=None,
                             num_frequency_points=None,
                             temperatures=None,
                             scattering_event_class=None,
                             write_gamma_detail=False,
                             output_filename=None):
        if self._interaction is None:
            self.set_phph_interaction()
        if temperatures is None:
            temperatures = [0.0, 300.0]
        self._grid_points = grid_points
        self._temperatures = temperatures
        self._scattering_event_class = scattering_event_class
        self._imag_self_energy, self._frequency_points = get_imag_self_energy(
            self._interaction,
            grid_points,
            self._sigmas,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
            temperatures=temperatures,
            scattering_event_class=scattering_event_class,
            write_detail=write_gamma_detail,
            output_filename=output_filename,
            log_level=self._log_level)

    def write_imag_self_energy(self, filename=None):
        write_imag_self_energy(
            self._imag_self_energy,
            self._mesh_numbers,
            self._grid_points,
            self._band_indices,
            self._frequency_points,
            self._temperatures,
            self._sigmas,
            scattering_event_class=self._scattering_event_class,
            filename=filename,
            is_mesh_symmetry=self._is_mesh_symmetry)

    def run_thermal_conductivity(
            self,
            is_LBTE=False,
            temperatures=np.arange(0, 1001, 10, dtype='double'),
            is_isotope=False,
            mass_variances=None,
            grid_points=None,
            boundary_mfp=None,  # in micrometre
            solve_collective_phonon=False,
            use_ave_pp=False,
            gamma_unit_conversion=None,
            mesh_divisors=None,
            coarse_mesh_shifts=None,
            is_reducible_collision_matrix=False,
            is_kappa_star=True,
            gv_delta_q=None,  # for group velocity
            is_full_pp=False,
            pinv_cutoff=1.0e-8,  # for pseudo-inversion of collision matrix
            pinv_solver=0,  # solver of pseudo-inversion of collision matrix
            write_gamma=False,
            read_gamma=False,
            is_N_U=False,
            write_kappa=False,
            write_gamma_detail=False,
            write_collision=False,
            read_collision=False,
            write_pp=False,
            read_pp=False,
            write_LBTE_solution=False,
            compression=None,
            input_filename=None,
            output_filename=None):
        if self._interaction is None:
            self.set_phph_interaction()
        if is_LBTE:
            self._thermal_conductivity = get_thermal_conductivity_LBTE(
                self._interaction,
                self._primitive_symmetry,
                temperatures=temperatures,
                sigmas=self._sigmas,
                sigma_cutoff=self._sigma_cutoff,
                is_isotope=is_isotope,
                mass_variances=mass_variances,
                grid_points=grid_points,
                boundary_mfp=boundary_mfp,
                solve_collective_phonon=solve_collective_phonon,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                is_kappa_star=is_kappa_star,
                gv_delta_q=gv_delta_q,
                is_full_pp=is_full_pp,
                pinv_cutoff=pinv_cutoff,
                pinv_solver=pinv_solver,
                write_collision=write_collision,
                read_collision=read_collision,
                write_kappa=write_kappa,
                write_pp=write_pp,
                read_pp=read_pp,
                write_LBTE_solution=write_LBTE_solution,
                compression=compression,
                input_filename=input_filename,
                output_filename=output_filename,
                log_level=self._log_level)
        else:
            self._thermal_conductivity = get_thermal_conductivity_RTA(
                self._interaction,
                self._primitive_symmetry,
                temperatures=temperatures,
                sigmas=self._sigmas,
                sigma_cutoff=self._sigma_cutoff,
                is_isotope=is_isotope,
                mass_variances=mass_variances,
                grid_points=grid_points,
                boundary_mfp=boundary_mfp,
                use_ave_pp=use_ave_pp,
                gamma_unit_conversion=gamma_unit_conversion,
                mesh_divisors=mesh_divisors,
                coarse_mesh_shifts=coarse_mesh_shifts,
                is_kappa_star=is_kappa_star,
                gv_delta_q=gv_delta_q,
                is_full_pp=is_full_pp,
                write_gamma=write_gamma,
                read_gamma=read_gamma,
                is_N_U=is_N_U,
                write_kappa=write_kappa,
                write_pp=write_pp,
                read_pp=read_pp,
                write_gamma_detail=write_gamma_detail,
                compression=compression,
                input_filename=input_filename,
                output_filename=output_filename,
                log_level=self._log_level)

    def get_thermal_conductivity(self):
        return self._thermal_conductivity

    def get_frequency_shift(
            self,
            grid_points,
            temperatures=np.arange(0, 1001, 10, dtype='double'),
            epsilons=None,
            output_filename=None):
        """Frequency shift from lowest order diagram is calculated.

        Args:
            epslins(list of float):
               The value to avoid divergence. When multiple values are given
               frequency shifts for those values are returned.

        """

        if self._interaction is None:
            self.set_phph_interaction()
        if epsilons is None:
            _epsilons = [0.1]
        else:
            _epsilons = epsilons
        self._grid_points = grid_points
        get_frequency_shift(self._interaction,
                            self._grid_points,
                            self._band_indices,
                            _epsilons,
                            temperatures,
                            output_filename=output_filename,
                            log_level=self._log_level)

    def _search_symmetry(self):
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)

    def _search_primitive_symmetry(self):
        self._primitive_symmetry = Symmetry(self._primitive,
                                            self._symprec,
                                            self._is_symmetry)
        if (len(self._symmetry.get_pointgroup_operations()) !=
            len(self._primitive_symmetry.get_pointgroup_operations())):
            print("Warning: point group symmetries of supercell and primitive"
                  "cell are different.")

    def _search_phonon_supercell_symmetry(self):
        if self._phonon_supercell_matrix is None:
            self._phonon_supercell_symmetry = self._symmetry
        else:
            self._phonon_supercell_symmetry = Symmetry(self._phonon_supercell,
                                                       self._symprec,
                                                       self._is_symmetry)

    def _build_supercell(self):
        self._supercell = get_supercell(self._unitcell,
                                        self._supercell_matrix,
                                        self._symprec)

    def _build_primitive_cell(self):
        """
        primitive_matrix:
          Relative axes of primitive cell to the input unit cell.
          Relative axes to the supercell is calculated by:
             supercell_matrix^-1 * primitive_matrix
          Therefore primitive cell lattice is finally calculated by:
             (supercell_lattice * (supercell_matrix)^-1 * primitive_matrix)^T
        """
        self._primitive = self._get_primitive_cell(
            self._supercell, self._supercell_matrix, self._primitive_matrix)

    def _build_phonon_supercell(self):
        """
        phonon_supercell:
          This supercell is used for harmonic phonons (frequencies,
          eigenvectors, group velocities, ...)
        phonon_supercell_matrix:
          Different supercell size can be specified.
        """
        if self._phonon_supercell_matrix is None:
            self._phonon_supercell = self._supercell
        else:
            self._phonon_supercell = get_supercell(
                self._unitcell, self._phonon_supercell_matrix, self._symprec)

    def _build_phonon_primitive_cell(self):
        if self._phonon_supercell_matrix is None:
            self._phonon_primitive = self._primitive
        else:
            self._phonon_primitive = self._get_primitive_cell(
                self._phonon_supercell,
                self._phonon_supercell_matrix,
                self._primitive_matrix)
            if (self._primitive is not None and
                (self._primitive.get_atomic_numbers() !=
                 self._phonon_primitive.get_atomic_numbers()).any()):
                print(" Primitive cells for fc2 and fc3 can be different.")
                raise RuntimeError

    def _build_phonon_supercells_with_displacements(self,
                                                    supercell,
                                                    displacement_dataset):
        supercells = []
        magmoms = supercell.get_magnetic_moments()
        masses = supercell.get_masses()
        numbers = supercell.get_atomic_numbers()
        lattice = supercell.get_cell()

        for disp1 in displacement_dataset['first_atoms']:
            disp_cart1 = disp1['displacement']
            positions = supercell.get_positions()
            positions[disp1['number']] += disp_cart1
            supercells.append(
                Atoms(numbers=numbers,
                      masses=masses,
                      magmoms=magmoms,
                      positions=positions,
                      cell=lattice,
                      pbc=True))

        return supercells

    def _build_supercells_with_displacements(self):
        supercells = []
        magmoms = self._supercell.get_magnetic_moments()
        masses = self._supercell.get_masses()
        numbers = self._supercell.get_atomic_numbers()
        lattice = self._supercell.get_cell()

        supercells = self._build_phonon_supercells_with_displacements(
            self._supercell,
            self._displacement_dataset)

        for disp1 in self._displacement_dataset['first_atoms']:
            disp_cart1 = disp1['displacement']
            for disp2 in disp1['second_atoms']:
                if 'included' in disp2:
                    included = disp2['included']
                else:
                    included = True
                if included:
                    positions = self._supercell.get_positions()
                    positions[disp1['number']] += disp_cart1
                    positions[disp2['number']] += disp2['displacement']
                    supercells.append(Atoms(numbers=numbers,
                                            masses=masses,
                                            magmoms=magmoms,
                                            positions=positions,
                                            cell=lattice,
                                            pbc=True))
                else:
                    supercells.append(None)

        self._supercells_with_displacements = supercells

    def _get_primitive_cell(self,
                            supercell,
                            supercell_matrix,
                            primitive_matrix):
        inv_supercell_matrix = np.linalg.inv(supercell_matrix)
        if primitive_matrix is None:
            t_mat = inv_supercell_matrix
        else:
            t_mat = np.dot(inv_supercell_matrix, primitive_matrix)

        return get_primitive(supercell, t_mat, self._symprec)

    def _guess_primitive_matrix(self):
        return guess_primitive_matrix(self._unitcell, symprec=self._symprec)

    def _set_masses(self, masses):
        p_masses = np.array(masses)
        self._primitive.set_masses(p_masses)
        p2p_map = self._primitive.get_primitive_to_primitive_map()
        s_masses = p_masses[[p2p_map[x] for x in
                             self._primitive.get_supercell_to_primitive_map()]]
        self._supercell.set_masses(s_masses)
        u2s_map = self._supercell.get_unitcell_to_supercell_map()
        u_masses = s_masses[u2s_map]
        self._unitcell.set_masses(u_masses)

        self._phonon_primitive.set_masses(p_masses)
        p2p_map = self._phonon_primitive.get_primitive_to_primitive_map()
        s_masses = p_masses[
            [p2p_map[x] for x in
             self._phonon_primitive.get_supercell_to_primitive_map()]]
        self._phonon_supercell.set_masses(s_masses)

    def _get_fc3(self,
                 forces_fc3,
                 disp_dataset,
                 is_compact_fc=False):
        count = 0
        for disp1 in disp_dataset['first_atoms']:
            disp1['forces'] = forces_fc3[count]
            count += 1
        for disp1 in disp_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                disp2['delta_forces'] = forces_fc3[count] - disp1['forces']
                count += 1

        fc2, fc3 = get_fc3(self._supercell,
                           self._primitive,
                           disp_dataset,
                           self._symmetry,
                           is_compact_fc=is_compact_fc,
                           verbose=self._log_level)

        return fc2, fc3


class Phono3pyIsotope(object):
    def __init__(self,
                 mesh,
                 primitive,
                 mass_variances=None,  # length of list is num_atom.
                 band_indices=None,
                 sigmas=None,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        if sigmas is None:
            self._sigmas = [None]
        else:
            self._sigmas = sigmas
        self._mesh_numbers = mesh
        self._iso = Isotope(mesh,
                            primitive,
                            mass_variances=mass_variances,
                            band_indices=band_indices,
                            frequency_factor_to_THz=frequency_factor_to_THz,
                            symprec=symprec,
                            cutoff_frequency=cutoff_frequency,
                            lapack_zheev_uplo=lapack_zheev_uplo)

    def run(self, grid_points):
        for gp in grid_points:
            self._iso.set_grid_point(gp)

            print("--------------- Isotope scattering ---------------")
            print("Grid point: %d" % gp)
            adrs = self._iso.get_grid_address()[gp]
            q = adrs.astype('double') / self._mesh_numbers
            print("q-point: %s" % q)

            if self._sigmas:
                for sigma in self._sigmas:
                    if sigma is None:
                        print("Tetrahedron method")
                    else:
                        print("Sigma: %s" % sigma)
                    self._iso.set_sigma(sigma)
                    self._iso.run()

                    frequencies = self._iso.get_phonons()[0]
                    print('')
                    print("Phonon-isotope scattering rate in THz (1/4pi-tau)")
                    print(" Frequency     Rate")
                    for g, f in zip(self._iso.get_gamma(), frequencies[gp]):
                        print("%8.3f     %5.3e" % (f, g))
            else:
                print("sigma or tetrahedron method has to be set.")

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._primitive = primitive
        self._iso.set_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals)

    def set_sigma(self, sigma):
        self._iso.set_sigma(sigma)


class Phono3pyJointDos(object):
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 fc2,
                 nac_params=None,
                 nac_q_direction=None,
                 sigmas=None,
                 cutoff_frequency=1e-4,
                 frequency_step=None,
                 num_frequency_points=None,
                 temperatures=None,
                 frequency_factor_to_THz=VaspToTHz,
                 frequency_scale_factor=None,
                 is_mesh_symmetry=True,
                 symprec=1e-5,
                 output_filename=None,
                 log_level=0):
        if sigmas is None:
            self._sigmas = [None]
        else:
            self._sigmas = sigmas
        self._supercell = supercell
        self._primitive = primitive
        self._mesh_numbers = mesh
        self._fc2 = fc2
        self._nac_params = nac_params
        self._nac_q_direction = nac_q_direction
        self._cutoff_frequency = cutoff_frequency
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
        self._temperatures = temperatures
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor
        self._is_mesh_symmetry = is_mesh_symmetry
        self._symprec = symprec
        self._filename = output_filename
        self._log_level = log_level

        self._jdos = JointDos(
            self._mesh_numbers,
            self._primitive,
            self._supercell,
            self._fc2,
            nac_params=self._nac_params,
            nac_q_direction=self._nac_q_direction,
            cutoff_frequency=self._cutoff_frequency,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points,
            temperatures=self._temperatures,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            frequency_scale_factor=self._frequency_scale_factor,
            is_mesh_symmetry=self._is_mesh_symmetry,
            symprec=self._symprec,
            filename=output_filename,
            log_level=self._log_level)

    def run(self, grid_points):
        if self._log_level:
            print("--------------------------------- Joint DOS "
                  "---------------------------------")
            print("Sampling mesh: [ %d %d %d ]" % tuple(self._mesh_numbers))

        for i, gp in enumerate(grid_points):
            self._jdos.set_grid_point(gp)

            if self._log_level:
                weights = self._jdos.get_triplets_at_q()[1]
                print("======================= "
                      "Grid point %d (%d/%d) "
                      "=======================" % (gp, i + 1, len(grid_points)))
                adrs = self._jdos.get_grid_address()[gp]
                q = adrs.astype('double') / self._mesh_numbers
                print("q-point: (%5.2f %5.2f %5.2f)" % tuple(q))
                print("Number of triplets: %d" % len(weights))
                print("Frequency")
                for f in self._jdos.get_phonons()[0][gp]:
                    print("%8.3f" % f)

            if self._sigmas:
                for sigma in self._sigmas:
                    if self._log_level:
                        if sigma is None:
                            print("Tetrahedron method is used.")
                        else:
                            print("Smearing method with sigma=%s is used." % sigma)
                    self._jdos.set_sigma(sigma)
                    self._jdos.run()
                    filename = self._write(gp, sigma=sigma)
                    if self._log_level:
                        print("JDOS is written into \"%s\"." % filename)
            else:
                if self._log_level:
                    print("sigma or tetrahedron method has to be set.")

    def _write(self, gp, sigma=None):
        return write_joint_dos(gp,
                               self._mesh_numbers,
                               self._jdos.get_frequency_points(),
                               self._jdos.get_joint_dos(),
                               sigma=sigma,
                               temperatures=self._temperatures,
                               filename=self._filename,
                               is_mesh_symmetry=self._is_mesh_symmetry)
