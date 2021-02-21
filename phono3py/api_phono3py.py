# Copyright (C) 2016 Atsushi Togo
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

import warnings
import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import (
    get_supercell, get_primitive, guess_primitive_matrix,
    shape_supercell_matrix)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.dataset import get_displacements_and_forces
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import (
    symmetrize_force_constants,
    symmetrize_compact_force_constants,
    set_translational_invariance,
    set_permutation_symmetry)
from phonopy.harmonic.force_constants import get_fc2 as get_phonopy_fc2
from phonopy.interface.fc_calculator import get_fc2
from phonopy.harmonic.displacement import (
    get_least_displacements, directions_to_displacement_dataset)
from phonopy.structure.grid_points import length2mesh
from phono3py.version import __version__
from phono3py.phonon3.imag_self_energy import (get_imag_self_energy,
                                               write_imag_self_energy)
from phono3py.phonon3.real_self_energy import (
    get_real_self_energy, write_real_self_energy)
from phono3py.phonon3.spectral_function import run_spectral_function
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.conductivity_RTA import get_thermal_conductivity_RTA
from phono3py.phonon3.conductivity_LBTE import get_thermal_conductivity_LBTE
from phono3py.phonon3.displacement_fc3 import (get_third_order_displacements,
                                               direction_to_displacement)
from phono3py.phonon3.fc3 import (
    set_permutation_symmetry_fc3,
    set_permutation_symmetry_compact_fc3,
    set_translational_invariance_fc3,
    set_translational_invariance_compact_fc3,
    cutoff_fc3_by_zero)
from phono3py.phonon3.fc3 import get_fc3 as get_phono3py_fc3
from phono3py.phonon3.dataset import get_displacements_and_forces_fc3
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.interface.fc_calculator import get_fc3


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
                 calculator=None,
                 log_level=0,
                 lapack_zheev_uplo='L'):
        self.sigmas = sigmas
        self.sigma_cutoff = sigma_cutoff
        self._symprec = symprec
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_mesh_symmetry = is_mesh_symmetry
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._symmetrize_fc3q = symmetrize_fc3q
        self._cutoff_frequency = cutoff_frequency
        self._calculator = calculator
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = unitcell
        self._supercell_matrix = shape_supercell_matrix(supercell_matrix)
        if type(primitive_matrix) is str and primitive_matrix == 'auto':
            self._primitive_matrix = self._guess_primitive_matrix()
        else:
            self._primitive_matrix = primitive_matrix
        self._nac_params = None
        if phonon_supercell_matrix is not None:
            self._phonon_supercell_matrix = shape_supercell_matrix(
                phonon_supercell_matrix)
        else:
            self._phonon_supercell_matrix = None
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
        self._dataset = None
        self._phonon_dataset = None
        self._phonon_supercells_with_displacements = None

        # Thermal conductivity
        # conductivity_RTA or conductivity_LBTE class instance
        self._thermal_conductivity = None

        # Imaginary part of self energy at frequency points
        self._gammas = None
        self._scattering_event_class = None

        # Frequency shift (real part of bubble diagram)
        self._real_self_energy = None

        self._grid_points = None
        self._frequency_points = None
        self._temperatures = None

        # Other variables
        self._fc2 = None
        self._fc3 = None

        # Setup interaction
        self._interaction = None
        self._mesh_numbers = None
        self._band_indices = None
        self._band_indices_flatten = None
        if mesh is not None:
            warnings.warn("Phono3py(mesh) is deprecated."
                          "Use Phono3py.mesh_number to set sampling mesh.",
                          DeprecationWarning)
            self._set_mesh_numbers(mesh)
        self.band_indices = band_indices

    @property
    def version(self):
        """Phono3py release version number

        str
            Phono3py release version number

        """

        return __version__

    def get_version(self):
        return self.version

    @property
    def calculator(self):
        """Return calculator name

        str
            Calculator name such as 'vasp', 'qe', etc.

        """
        return self._calculator

    @property
    def fc3(self):
        """

        """

        return self._fc3

    def get_fc3(self):
        return self.fc3

    @fc3.setter
    def fc3(self, fc3):
        """Third order force constants (fc3).

        ndarray
            fc3 shape is either (supercell, supecell, supercell, 3, 3, 3) or
            (primitive, supercell, supecell, 3, 3, 3),
            where 'supercell' and 'primitive' indicate number of atoms in
            these cells.

        """

        self._fc3 = fc3

    def set_fc3(self, fc3):
        self.fc3 = fc3

    @property
    def fc2(self):
        """Second order force constants (fc2).

        ndarray
            fc2 shape is either (supercell, supecell, 3, 3) or
            (primitive, supecell, 3, 3),
            where 'supercell' and 'primitive' indicate number of atoms in
            these cells.

        """

        return self._fc2

    def get_fc2(self):
        return self.fc2

    @fc2.setter
    def fc2(self, fc2):
        self._fc2 = fc2

    def set_fc2(self, fc2):
        self.fc2 = fc2

    @property
    def force_constants(self):
        """Alias to fc2"""
        return self.fc2

    @property
    def sigmas(self):
        return self._sigmas

    @sigmas.setter
    def sigmas(self, sigmas):
        if sigmas is None:
            self._sigmas = [None, ]
        elif isinstance(sigmas, float) or isinstance(sigmas, int):
            self._sigmas = [float(sigmas), ]
        else:
            self._sigmas = []
            for s in sigmas:
                if isinstance(s, float) or isinstance(s, int):
                    self._sigmas.append(float(s))
                elif s is None:
                    self._sigmas.append(None)

    @property
    def sigma_cutoff(self):
        return self._sigma_cutoff

    @sigma_cutoff.setter
    def sigma_cutoff(self, sigma_cutoff):
        self._sigma_cutoff = sigma_cutoff

    @property
    def nac_params(self):
        """Parameters for non-analytical term correction

        dict
            Parameters used for non-analytical term correction
            'born': ndarray
                Born effective charges
                shape=(primitive cell atoms, 3, 3), dtype='double', order='C'
            'factor': float
                Unit conversion factor
            'dielectric': ndarray
                Dielectric constant tensor
                shape=(3, 3), dtype='double', order='C'

        """
        return self._nac_params

    def get_nac_params(self):
        return self.nac_params

    @nac_params.setter
    def nac_params(self, nac_params):
        self._nac_params = nac_params
        self._init_dynamical_matrix()

    def set_nac_params(self, nac_params):
        self.nac_params = nac_params

    @property
    def dynamical_matrix(self):
        """DynamicalMatrix instance

        This is not dynamical matrices but the instance of DynamicalMatrix
        class.

        """
        if self._interaction is None:
            return None
        else:
            return self._interaction.dynamical_matrix

    @property
    def primitive(self):
        """Primitive cell

        Primitive
            Primitive cell.

        """

        return self._primitive

    def get_primitive(self):
        return self.primitive

    @property
    def unitcell(self):
        """Unit cell

        PhonopyAtoms
            Unit cell.

        """

        return self._unitcell

    def get_unitcell(self):
        return self.unitcell

    @property
    def supercell(self):
        """Supercell

        Supercell
            Supercell.

        """

        return self._supercell

    def get_supercell(self):
        return self.supercell

    @property
    def phonon_supercell(self):
        """Supercell for fc2.

        Supercell
            Supercell for fc2.

        """

        return self._phonon_supercell

    def get_phonon_supercell(self):
        return self.phonon_supercell

    @property
    def phonon_primitive(self):
        """Primitive cell for fc2.

        Primitive
            Primitive cell for fc2. This should be the same as the primitive
            cell for fc3, but this is created from supercell for fc2 and
            can be not numerically perfectly identical.

        """

        return self._phonon_primitive

    def get_phonon_primitive(self):
        return self.phonon_primitive

    @property
    def symmetry(self):
        """Symmetry of supercell

        Symmetry
            Symmetry of supercell

        """

        return self._symmetry

    def get_symmetry(self):
        return self.symmetry

    @property
    def primitive_symmetry(self):
        """Symmetry of primitive cell

        Symmetry
            Symmetry of primitive cell.

        """

        return self._primitive_symmetry

    def get_primitive_symmetry(self):
        return self.primitive_symmetry

    @property
    def phonon_supercell_symmetry(self):
        """Symmetry of supercell for fc2.

        Symmetry
            Symmetry of supercell for fc2 (phonon_supercell).

        """

        return self._phonon_supercell_symmetry

    def get_phonon_supercell_symmetry(self):
        return self.phonon_supercell_symmetry

    @property
    def supercell_matrix(self):
        """Transformation matrix to supercell cell from unit cell

        ndarray
            Supercell matrix with respect to unit cell.
            shape=(3, 3), dtype='intc', order='C'

        """

        return self._supercell_matrix

    def get_supercell_matrix(self):
        return self.supercell_matrix

    @property
    def phonon_supercell_matrix(self):
        """Transformation matrix to supercell cell from unit cell for fc2

        ndarray
            Supercell matrix with respect to unit cell.
            shape=(3, 3), dtype='intc', order='C'

        """

        return self._phonon_supercell_matrix

    def get_phonon_supercell_matrix(self):
        return self.phonon_supercell_matrix

    @property
    def primitive_matrix(self):
        """Transformation matrix to primitive cell from unit cell

        ndarray
            Primitive matrix with respect to unit cell.
            shape=(3, 3), dtype='double', order='C'

        """

        return self._primitive_matrix

    def get_primitive_matrix(self):
        return self.primitive_matrix

    @property
    def unit_conversion_factor(self):
        """Phonon frequency unit conversion factor.

        float
            Phonon frequency unit conversion factor. This factor
            converts sqrt(<force>/<distance>/<AMU>)/2pi/1e12 to THz
            (ordinary frequency).

        """

        return self._frequency_factor_to_THz

    def set_displacement_dataset(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        """Displacement dataset

        dict
            Displacements in supercells. There are two types of formats.
            Type 1. Two atomic displacement in each supercell:
                {'natom': number of atoms in supercell,
                 'first_atoms': [
                   {'number': atom index of first displaced atom,
                    'displacement': displacement in Cartesian coordinates,
                    'forces': forces on atoms in supercell,
                    'id': displacement id (1, 2,...,n_first_atoms)
                    'second_atoms': [
                      {'number': atom index of second displaced atom,
                       'displacement': displacement in Cartesian coordinates},
                       'forces': forces on atoms in supercell,
                       'pair_distance': distance between paired atoms,
                       'included': with cutoff pair distance in displacement
                                   pair generation, this indicates if this
                                   pair displacements is included to compute
                                   fc3 or not,
                       'id': displacement id. (n_first_atoms + 1, ...)
                      ... ] }, ... ] }
            Type 2. All atomic displacements in each supercell:
                {'displacements': ndarray, dtype='double', order='C',
                                  shape=(supercells, atoms in supercell, 3)
                 'forces': ndarray, dtype='double',, order='C',
                                  shape=(supercells, atoms in supercell, 3)}
            In type 2, displacements and forces can be given by numpy array
            with different shape but that can be reshaped to
            (supercells, natom, 3).

            In addition, 'duplicates' and 'cutoff_distance' can exist in this
            dataset in displacement pair generation. 'duplicates' gives
            duplicated supercell ids as pairs.

        """

        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def phonon_dataset(self):
        """Displacement dataset for fc2

        dict
            Displacements in supercells. There are two types of formats.
            Type 1. Two atomic displacement in each supercell:
                {'natom': number of atoms in supercell,
                 'first_atoms': [
                   {'number': atom index of first displaced atom,
                    'displacement': displacement in Cartesian coordinates,
                    'forces': forces on atoms in supercell} ... ]}
            Type 2. All atomic displacements in each supercell:
                {'displacements': ndarray, dtype='double', order='C',
                                  shape=(supercells, atoms in supercell, 3)
                 'forces': ndarray, dtype='double',, order='C',
                                  shape=(supercells, atoms in supercell, 3)}
            In type 2, displacements and forces can be given by numpy array
            with different shape but that can be reshaped to
            (supercells, natom, 3).

        """

        return self._phonon_dataset

    @phonon_dataset.setter
    def phonon_dataset(self, dataset):
        self._phonon_dataset = dataset

    @property
    def band_indices(self):
        """Band index

        array_like
            List of band indices specified to select specific bands
            to computer ph-ph interaction related properties.

        """

        return self._band_indices

    @band_indices.setter
    def band_indices(self, band_indices):
        if band_indices is None:
            num_band = len(self._primitive) * 3
            self._band_indices = [np.arange(num_band, dtype='intc')]
        else:
            self._band_indices = band_indices
        self._band_indices_flatten = np.hstack(
            self._band_indices).astype('intc')

    def set_band_indices(self, band_indices):
        self.band_indices = band_indices

    @property
    def displacement_dataset(self):
        warnings.warn("Phono3py.displacement_dataset is deprecated."
                      "Use Phono3py.dataset.",
                      DeprecationWarning)
        return self.dataset

    def get_displacement_dataset(self):
        return self.displacement_dataset

    @property
    def phonon_displacement_dataset(self):
        warnings.warn("Phono3py.phonon_displacement_dataset is deprecated."
                      "Use Phono3py.phonon_dataset.",
                      DeprecationWarning)
        return self._phonon_dataset

    def get_phonon_displacement_dataset(self):
        return self.phonon_displacement_dataset

    @property
    def supercells_with_displacements(self):
        """Supercells with displacements

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phono3py.generate_displacements.

        """

        if self._supercells_with_displacements is None:
            self._build_supercells_with_displacements()
        return self._supercells_with_displacements

    def get_supercells_with_displacements(self):
        return self.supercells_with_displacements

    @property
    def phonon_supercells_with_displacements(self):
        """Supercells with displacements for fc2

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phono3py.generate_displacements.

        """

        if self._phonon_supercells_with_displacements is None:
            if self._phonon_dataset is not None:
                self._phonon_supercells_with_displacements = \
                  self._build_phonon_supercells_with_displacements(
                      self._phonon_supercell,
                      self._phonon_dataset)
        return self._phonon_supercells_with_displacements

    def get_phonon_supercells_with_displacements(self):
        return self.phonon_supercells_with_displacements

    @property
    def mesh_numbers(self):
        """Sampling mesh numbers in reciprocal space"""
        return self._mesh_numbers

    @mesh_numbers.setter
    def mesh_numbers(self, mesh_numbers):
        self._set_mesh_numbers(mesh_numbers)

    @property
    def thermal_conductivity(self):
        """Thermal conductivity class instance"""
        return self._thermal_conductivity

    def get_thermal_conductivity(self):
        return self.thermal_conductivity

    @property
    def displacements(self):
        """Displacements in supercells

        getter : ndarray
            Displacements of all atoms of all supercells in Cartesian
            coordinates.
            shape=(supercells, natom, 3), dtype='double', order='C'

        setter : array_like
            Atomic displacements of all atoms of all supercells.
            shape=(supercells, natom, 3).

            If type-1 displacement dataset for fc2 exists already, type-2
            displacement dataset is newly created and information of
            forces is abandoned. If type-1 displacement dataset for fc2
            exists already information of forces is preserved.

        """

        dataset = self._dataset

        if 'first_atoms' in dataset:
            num_scells = len(dataset['first_atoms'])
            for disp1 in dataset['first_atoms']:
                num_scells += len(disp1['second_atoms'])
            displacements = np.zeros(
                (num_scells, self._supercell.get_number_of_atoms(), 3),
                dtype='double', order='C')
            i = 0
            for disp1 in dataset['first_atoms']:
                displacements[i, disp1['number']] = disp1['displacement']
                i += 1
            for disp1 in dataset['first_atoms']:
                for disp2 in disp1['second_atoms']:
                    displacements[i, disp2['number']] = disp2['displacement']
                    i += 1
        elif 'forces' in dataset or 'displacements' in dataset:
            displacements = dataset['displacements']
        else:
            raise RuntimeError("displacement dataset has wrong format.")

        return displacements

    @displacements.setter
    def displacements(self, displacements):
        dataset = self._dataset
        disps = np.array(displacements, dtype='double', order='C')
        natom = self._supercell.get_number_of_atoms()
        if (disps.ndim != 3 or disps.shape[1:] != (natom, 3)):
            raise RuntimeError("Array shape of displacements is incorrect.")

        if 'first_atoms' in dataset:
            dataset = {'displacements': disps}
        elif 'displacements' in dataset or 'forces' in dataset:
            dataset['displacements'] = disps

    @property
    def forces(self):
        """Set forces in displacement dataset

        A set of atomic forces in displaced supercells. The order of
        displaced supercells has to match with that in displacement dataset.
        shape=(displaced supercells, atoms in supercell, 3)

        getter : ndarray

        setter : array_like
            The order of supercells used for calculating forces has to
            be the same order of supercells_with_displacements.

        """

        dataset = self._dataset
        if 'forces' in dataset:
            return dataset['forces']
        elif 'first_atoms' in dataset:
            num_scells = len(dataset['first_atoms'])
            for disp1 in dataset['first_atoms']:
                num_scells += len(disp1['second_atoms'])
            forces = np.zeros(
                (num_scells, self._supercell.get_number_of_atoms(), 3),
                dtype='double', order='C')
            i = 0
            for disp1 in dataset['first_atoms']:
                forces[i] = disp1['forces']
                i += 1
            for disp1 in dataset['first_atoms']:
                for disp2 in disp1['second_atoms']:
                    forces[i] = disp2['forces']
                    i += 1
            return forces
        else:
            raise RuntimeError("displacement dataset has wrong format.")

    @forces.setter
    def forces(self, forces_fc3):
        forces = np.array(forces_fc3, dtype='double', order='C')
        dataset = self._dataset
        if 'first_atoms' in dataset:
            i = 0
            for disp1 in dataset['first_atoms']:
                disp1['forces'] = forces[i]
                i += 1
            for disp1 in dataset['first_atoms']:
                for disp2 in disp1['second_atoms']:
                    disp2['forces'] = forces[i]
                    i += 1
        elif 'displacements' in dataset or 'forces' in dataset:
            dataset['forces'] = forces

    @property
    def phonon_displacements(self):
        """Displacements in supercells for fc2

        Displacements of all atoms of all supercells in Cartesian
        coordinates.
        shape=(supercells, natom, 3), dtype='double', order='C'

        getter : ndarray

        setter : array_like
            If type-1 displacement dataset for fc2 exists already, type-2
            displacement dataset is newly created and information of
            forces is abandoned. If type-1 displacement dataset for fc2
            exists already information of forces is preserved.

        """

        if self._phonon_dataset is None:
            raise RuntimeError("phonon_displacement_dataset does not exist.")

        dataset = self._phonon_dataset
        if 'first_atoms' in dataset:
            num_scells = len(dataset['first_atoms'])
            natom = self._phonon_supercell.get_number_of_atoms()
            displacements = np.zeros(
                (num_scells, natom, 3), dtype='double', order='C')
            for i, disp1 in enumerate(dataset['first_atoms']):
                displacements[i, disp1['number']] = disp1['displacement']
        elif 'forces' in dataset or 'displacements' in dataset:
            displacements = dataset['displacements']
        else:
            raise RuntimeError("displacement dataset has wrong format.")

        return displacements

    @phonon_displacements.setter
    def phonno_displacements(self, displacements):
        if self._phonon_dataset is None:
            raise RuntimeError("phonon_displacement_dataset does not exist.")

        dataset = self._phonon_dataset
        disps = np.array(displacements, dtype='double', order='C')
        natom = self._phonon_supercell.get_number_of_atoms()
        if (disps.ndim != 3 or disps.shape[1:] != (natom, 3)):
            raise RuntimeError("Array shape of displacements is incorrect.")

        if 'first_atoms' in dataset:
            dataset = {'displacements': disps}
        elif 'displacements' in dataset or 'forces' in dataset:
            dataset['displacements'] = disps

    @property
    def phonon_forces(self):
        """Set forces in displacement dataset for fc2

        A set of atomic forces in displaced supercells. The order of
        displaced supercells has to match with that in phonon displacement
        dataset.
        shape=(displaced supercells, atoms in supercell, 3)

        getter : ndarray

        setter : array_like
            The order of supercells used for calculating forces has to
            be the same order of phonon_supercells_with_displacements.

        """

        if self._phonon_dataset is None:
            raise RuntimeError("phonon_displacement_dataset does not exist.")

        dataset = self._phonon_dataset
        if 'forces' in dataset:
            return dataset['forces']
        elif 'first_atoms' in dataset:
            num_scells = len(dataset['first_atoms'])
            forces = np.zeros(
                (num_scells, self._phonon_supercell.get_number_of_atoms(), 3),
                dtype='double', order='C')
            for i, disp1 in enumerate(dataset['first_atoms']):
                forces[i] = disp1['forces']
            return forces
        else:
            raise RuntimeError("displacement dataset has wrong format.")

    @phonon_forces.setter
    def phonon_forces(self, forces_fc2):
        if self._phonon_dataset is None:
            raise RuntimeError("phonon_displacement_dataset does not exist.")

        forces = np.array(forces_fc2, dtype='double', order='C')
        dataset = self._phonon_dataset
        if 'first_atoms' in dataset:
            i = 0
            for i, disp1 in enumerate(dataset['first_atoms']):
                disp1['forces'] = forces[i]
                i += 1
        elif 'displacements' in dataset or 'forces' in dataset:
            dataset['forces'] = forces

    @property
    def phph_interaction(self):
        return self._interaction

    def get_phph_interaction(self):
        return self.phph_interaction

    @property
    def detailed_gammas(self):
        return self._detailed_gammas

    def init_phph_interaction(self,
                              nac_q_direction=None,
                              constant_averaged_interaction=None,
                              frequency_scale_factor=None,
                              solve_dynamical_matrices=True):
        """Initialize ph-ph interaction calculation

        This method creates an instance of Interaction class, which
        is necessary to run ph-ph interaction calculation.
        The input data such as grids, force constants, etc, are
        stored to be ready for the calculation.
        ``solve_dynamical_matrices=True`` runs harmonic phonon solver
        immediately to store phonons on all regular mesh grids.

        Note
        ----
        fc3 and fc2, and optionally nac_params have to be set before calling
        this method. fc3 and fc2 can be made either from sets of forces
        and displacements of supercells or be set simply via attributes.

        Parameters
        ----------
        nac_q_direction : array_like, optional
            Direction of q-vector watching from Gamma point used for
            non-analytical term correction. This is effective only at q=0
            (physically q->0). The direction is given in crystallographic
            (fractional) coordinates.
            shape=(3,), dtype='double'.
            Default value is None, which means this feature is not used.
        constant_averaged_interaction : float, optional
            Ph-ph interaction strength array is replaced by a scalar value.
            Default is None, which means this feature is not used.
        frequency_scale_factor : float, optional
            All phonon frequences are scaled by this value. Default is None,
            which means phonon frequencies are not scaled.
        solve_dynamical_matrices : Bool, optional
            When True, harmonic phonon solver is immediately executed and
            the phonon data on all regular mesh grids are store phonons.
            Default is True.

        """

        if self._mesh_numbers is None:
            msg = "Phono3py.mesh_numbers of instance has to be set."
            raise RuntimeError(msg)

        if self._fc2 is None:
            msg = "Phono3py.fc2 of instance is not found."
            raise RuntimeError(msg)

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
            cutoff_frequency=self._cutoff_frequency,
            is_mesh_symmetry=self._is_mesh_symmetry,
            symmetrize_fc3q=self._symmetrize_fc3q,
            lapack_zheev_uplo=self._lapack_zheev_uplo)
        self._interaction.set_nac_q_direction(nac_q_direction=nac_q_direction)
        self._init_dynamical_matrix()
        if solve_dynamical_matrices:
            self.run_phonon_solver(verbose=self._log_level)

    def set_phph_interaction(self,
                             nac_params=None,
                             nac_q_direction=None,
                             constant_averaged_interaction=None,
                             frequency_scale_factor=None,
                             solve_dynamical_matrices=True):
        """Initialize ph-ph interaction calculation

        This method is deprecated at v2.0. Phono3py.init_phph_interaction
        should be used instead of this method.

        Parameters
        ----------
        Most of parameters are given at docstring of
        Phono3py.init_phph_interaction.

        nac_params : dict, Deprecated at v2.0
            Parameters used for non-analytical term correction
            'born': ndarray
                Born effective charges
                shape=(primitive cell atoms, 3, 3), dtype='double', order='C'
            'factor': float
                Unit conversion factor
            'dielectric': ndarray
                Dielectric constant tensor
                shape=(3, 3), dtype='double', order='C'

        """

        msg = ("Phono3py.init_phph_interaction is deprecated at v2.0. "
               "Use Phono3py.prepare_interaction instead.")
        warnings.warn(msg, DeprecationWarning)

        if nac_params is not None:
            self._nac_params = nac_params
            msg = ("nac_params will be set by Phono3py.nac_params attributes.")
            warnings.warn(msg, DeprecationWarning)

        self.init_phph_interaction(
            nac_q_direction=nac_q_direction,
            constant_averaged_interaction=constant_averaged_interaction,
            frequency_scale_factor=frequency_scale_factor,
            solve_dynamical_matrices=solve_dynamical_matrices)

    def set_phonon_data(self, frequencies, eigenvectors, grid_address):
        """Set phonon frequencies and eigenvectors in Interaction instance

        Harmonic phonon information is stored in Interaction instance. For
        example, this information store in a file is read and passed to
        Phono3py instance by using this method. The grid_address is used
        for the consistency check.

        Parameters
        ----------
        frequencies : array_like
            Phonon frequencies.
            shape=(num_grid_points, num_band), dtype='double', order='C'
        eigenvectors : array_like
            Phonon eigenvectors
            shape=(num_grid_points, num_band, num_band)
            dtype='complex128', order='C'
        grid_address : array_like
            Grid point addresses by integers. The first dimension may not be
            prod(mesh) because it includes Brillouin zone boundary. The detail
            is found in the docstring of
            phono3py.phonon3.triplets.get_triplets_at_q.
            shape=(num_grid_points, 3), dtype='intc', order='C'

        """

        if self._interaction is not None:
            self._interaction.set_phonon_data(
                frequencies, eigenvectors, grid_address)

    def get_phonon_data(self):
        """Get phonon frequencies and eigenvectors in Interaction instance

        Harmonic phonon information is stored in Interaction instance. This
        information can be obtained. The grid_address returned give the
        q-points locations with respect to reciprocal basis vectors by
        integers in the way that
            q_points = grid_address / np.array(mesh, dtype='double').

        Returns
        -------
        tuple
            (frequencies, eigenvectors, grid_address)
            See more details at the docstring of set_phonon_data.

        """

        if self._interaction is not None:
            grid_address = self._interaction.get_grid_address()
            freqs, eigvecs, _ = self._interaction.get_phonons()
            return freqs, eigvecs, grid_address
        else:
            msg = ("Phono3py.init_phph_interaction has to be called "
                   "before running this method.")
            raise RuntimeError(msg)

    def run_phonon_solver(self, verbose=False):
        if self._interaction is not None:
            self._interaction.run_phonon_solver(verbose=verbose)
        else:
            msg = ("Phono3py.init_phph_interaction has to be called "
                   "before running this method.")
            raise RuntimeError(msg)

    def generate_displacements(self,
                               distance=0.03,
                               cutoff_pair_distance=None,
                               is_plusminus='auto',
                               is_diagonal=True):
        """Generate displacement dataset in supercell for fc3

        This systematically generates single and pair atomic displacements
        in supercells to calculate fc3 considering crystal symmetry.

        For fc3, two atoms are displaced for each configuration
        considering crystal symmetry. The first displacement is chosen
        in the perfect supercell, and the second displacement in the
        displaced supercell. The first displacements are taken along
        the basis vectors of the supercell. This is because the
        symmetry is expected to be less broken by the introduced first
        displacement, and as the result, the number of second
        displacements may become smaller than the case that the first
        atom is displaced not along the basis vectors.

        Note
        ----
        When phonon_supercell_matrix is not given, fc2 is also
        computed from the same set of the displacements for fc3 and
        respective supercell forces. When phonon_supercell_matrix is
        set, the displacements in phonon_supercell are generated.

        Parameters
        ----------
        distance : float, optional
            Constant displacement Euclidean distance. Default is 0.03.
        cutoff_pair_distance : float, optional
            This is used as a cutoff Euclidean distance to determine if
            each pair of displacements is considered to calculate fc3 or not.
            Default is None, which means cutoff is not used.
        is_plusminus : True, False, or 'auto', optional
            With True, atomis are displaced in both positive and negative
            directions. With False, only one direction. With 'auto',
            mostly equivalent to is_plusminus=True, but only one direction
            is chosen when the displacements in both directions are
            symmetrically equivalent. Default is 'auto'.
        is_diagonal : Bool, optional
            With False, the second displacements are made along the basis
            vectors of the supercell. With True, direction not along the basis
            vectors can be chosen when the number of the displacements
            may be reduced.

        """

        direction_dataset = get_third_order_displacements(
            self._supercell,
            self._symmetry,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal)
        self._dataset = direction_to_displacement(
            direction_dataset,
            distance,
            self._supercell,
            cutoff_distance=cutoff_pair_distance)

        if self._phonon_supercell_matrix is not None:
            self.generate_fc2_displacements(distance=distance,
                                            is_plusminus=is_plusminus,
                                            is_diagonal=False)

    def generate_fc2_displacements(self,
                                   distance=0.03,
                                   is_plusminus='auto',
                                   is_diagonal=False):
        """Generate displacement dataset in phonon supercell for fc2

        This systematically generates single atomic displacements
        in supercells to calculate phonon_fc2 considering crystal symmetry.


        Note
        ----
        is_diagonal=False is chosen as the default setting intentionally
        to be consistent to the first displacements of the fc3 pair
        displacemets in supercell.

        Parameters
        ----------
        distance : float, optional
            Constant displacement Euclidean distance. Default is 0.03.
        is_plusminus : True, False, or 'auto', optional
            With True, atomis are displaced in both positive and negative
            directions. With False, only one direction. With 'auto',
            mostly equivalent to is_plusminus=True, but only one direction
            is chosen when the displacements in both directions are
            symmetrically equivalent. Default is 'auto'.
        is_diagonal : Bool, optional
            With False, the displacements are made along the basis
            vectors of the supercell. With True, direction not along the basis
            vectors can be chosen when the number of the displacements
            may be reduced. Default is False.

        """

        if self._phonon_supercell_matrix is None:
            msg = ("phonon_supercell_matrix is not set. "
                   "This method is used to generate displacements to "
                   "calculate phonon_fc2.")
            raise RuntimeError(msg)

        phonon_displacement_directions = get_least_displacements(
            self._phonon_supercell_symmetry,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal)
        self._phonon_dataset = directions_to_displacement_dataset(
            phonon_displacement_directions,
            distance,
            self._phonon_supercell)

    def produce_fc3(self,
                    forces_fc3=None,
                    displacement_dataset=None,
                    cutoff_distance=None,  # set fc3 zero
                    symmetrize_fc3r=False,
                    is_compact_fc=False,
                    fc_calculator=None,
                    fc_calculator_options=None):
        """Calculate fc3 from displacements and forces

        Parameters
        ----------
        forces_fc3 :
            Dummy argument. Deprecated at v2.0.
        displacement_dataset : dict
            See docstring of Phono3py.dataset. Deprecated at v2.0.
        cutoff_distance : float
            After creating force constants, fc elements where any pair
            distance in atom triplets larger than cutoff_distance are set zero.
        symmetrize_fc3r : bool
            Only for type 1 displacement_dataset, translational and
            permutation symmetries are applied after creating fc3. This
            symmetrization is not very sophisticated and can break space
            group symmetry, but often useful. If better symmetrization is
            expected, it is recommended to use external force constants
            calculator such as ALM. Default is False.
        is_compact_fc : bool
            fc3 shape is
                False: (supercell, supercell, supecell, 3, 3, 3)
                True: (primitive, supercell, supecell, 3, 3, 3)
            where 'supercell' and 'primitive' indicate number of atoms in these
            cells. Default is False.
        fc_calculator : str or None
            Force constants calculator given by str.
        fc_calculator_options : dict
            Options for external force constants calculator.

        """

        if displacement_dataset is None:
            disp_dataset = self._dataset
        else:
            msg = ("Displacement dataset has to set by Phono3py.dataset.")
            warnings.warn(msg, DeprecationWarning)
            disp_dataset = displacement_dataset

        if forces_fc3 is not None:
            self.forces = forces_fc3
            msg = ("Forces have to be set by Phono3py.forces or via "
                   "Phono3py.dataset.")
            warnings.warn(msg, DeprecationWarning)

        if fc_calculator is not None:
            disps, forces = get_displacements_and_forces_fc3(disp_dataset)
            fc2, fc3 = get_fc3(self._supercell,
                               self._primitive,
                               disps,
                               forces,
                               fc_calculator=fc_calculator,
                               fc_calculator_options=fc_calculator_options,
                               is_compact_fc=is_compact_fc,
                               log_level=self._log_level)
        else:
            if 'displacements' in disp_dataset:
                msg = ("fc_calculator has to be set to produce force "
                       "constans from this dataset.")
                raise RuntimeError(msg)
            fc2, fc3 = get_phono3py_fc3(self._supercell,
                                        self._primitive,
                                        disp_dataset,
                                        self._symmetry,
                                        is_compact_fc=is_compact_fc,
                                        verbose=self._log_level)
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

    def produce_fc2(self,
                    forces_fc2=None,
                    displacement_dataset=None,
                    symmetrize_fc2=False,
                    is_compact_fc=False,
                    fc_calculator=None,
                    fc_calculator_options=None):
        """Calculate fc2 from displacements and forces

        Parameters
        ----------
        forces_fc2 :
            Dummy argument. Deprecated at v2.0
        displacement_dataset : dict
            See docstring of Phono3py.phonon_dataset. Deprecated at v2.0.
        symmetrize_fc2 : bool
            Only for type 1 displacement_dataset, translational and
            permutation symmetries are applied after creating fc3. This
            symmetrization is not very sophisticated and can break space
            group symmetry, but often useful. If better symmetrization is
            expected, it is recommended to use external force constants
            calculator such as ALM. Default is False.
        is_compact_fc : bool
            fc2 shape is
                False: (supercell, supecell, 3, 3)
                True: (primitive, supecell, 3, 3)
            where 'supercell' and 'primitive' indicate number of atoms in these
            cells. Default is False.
        fc_calculator : str or None
            Force constants calculator given by str.
        fc_calculator_options : dict
            Options for external force constants calculator.

        """

        if displacement_dataset is None:
            if self._phonon_dataset is None:
                disp_dataset = self._dataset
            else:
                disp_dataset = self._phonon_dataset
        else:
            disp_dataset = displacement_dataset
            msg = ("Displacement dataset for fc2 has to set by "
                   "Phono3py.phonon_dataset.")
            warnings.warn(msg, DeprecationWarning)

        if forces_fc2 is not None:
            self.phonon_forces = forces_fc2
            msg = ("Forces for fc2 have to be set by Phono3py.phonon_forces "
                   "or via Phono3py.phonon_dataset.")
            warnings.warn(msg, DeprecationWarning)

        if is_compact_fc:
            p2s_map = self._phonon_primitive.p2s_map
        else:
            p2s_map = None

        if fc_calculator is not None:
            disps, forces = get_displacements_and_forces(disp_dataset)
            self._fc2 = get_fc2(self._phonon_supercell,
                                self._phonon_primitive,
                                disps,
                                forces,
                                fc_calculator=fc_calculator,
                                fc_calculator_options=fc_calculator_options,
                                atom_list=p2s_map,
                                log_level=self._log_level)
        else:
            if 'displacements' in disp_dataset:
                msg = ("fc_calculator has to be set to produce force "
                       "constans from this dataset for fc2.")
                raise RuntimeError(msg)
            self._fc2 = get_phonopy_fc2(self._phonon_supercell,
                                        self._phonon_supercell_symmetry,
                                        disp_dataset,
                                        atom_list=p2s_map)
            if symmetrize_fc2:
                if is_compact_fc:
                    symmetrize_compact_force_constants(
                        self._fc2, self._phonon_primitive)
                else:
                    symmetrize_force_constants(self._fc2)

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

    def run_imag_self_energy(self,
                             grid_points,
                             temperatures,
                             frequency_points=None,
                             frequency_step=None,
                             num_frequency_points=None,
                             scattering_event_class=None,
                             write_txt=False,
                             write_gamma_detail=False,
                             keep_gamma_detail=False,
                             output_filename=None):
        """Calculate imaginary part of self-energy of bubble diagram (Gamma)

        Pi = Delta - i Gamma.

        Parameters
        ----------
        grid_points : array_like
            Grid-point indices where imaginary part of self-energies are
            caclculated.
            dtype=int, shape=(grid_points,)
        temperatures : array_like
            Temperatures where imaginary part of self-energies are calculated.
            dtype=float, shape=(temperatures,)
        frequency_points : array_like, optional
            Frequency sampling points. Default is None. In this case,
            num_frequency_points or frequency_step is used to generate uniform
            frequency sampling points.
            dtype=float, shape=(frequency_points,)
        frequency_step : float, optional
            Uniform pitch of frequency sampling points. Default is None. This
            results in using num_frequency_points.
        num_frequency_points: Int, optional
            Number of sampling sampling points to be used instead of
            frequency_step. This number includes end points. Default is None,
            which gives 201.
        scattering_event_class : int, optional
            Specific choice of scattering event class, 1 or 2 that is specified
            1 or 2, respectively. The result is stored in gammas. Therefore
            usual gammas are not stored in the variable. Default is None, which
            doesn't specify scattering_event_class.
        write_txt : bool, optional
            Frequency points and imaginary part of self-energies are written
            into text files.
        write_gamma_detail : bool, optional
            Detailed gammas are written into a file in hdf5. Default is False.
        keep_gamma_detail : bool, optional
            With True, detailed gammas are stored. Default is False.
        output_filename : str
            This string is inserted in the output file names.

        """

        if self._interaction is None:
            msg = ("Phono3py.init_phph_interaction has to be called "
                   "before running this method.")
            raise RuntimeError(msg)

        if temperatures is None:
            self._temperatures = [300.0, ]
        else:
            self._temperatures = temperatures
        self._grid_points = grid_points
        self._scattering_event_class = scattering_event_class
        vals = get_imag_self_energy(
            self._interaction,
            grid_points,
            temperatures,
            sigmas=self._sigmas,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
            scattering_event_class=scattering_event_class,
            write_gamma_detail=write_gamma_detail,
            return_gamma_detail=keep_gamma_detail,
            output_filename=output_filename,
            log_level=self._log_level)
        if keep_gamma_detail:
            (self._frequency_points,
             self._gammas,
             self._detailed_gammas) = vals
        else:
            self._frequency_points, self._gammas = vals

        if write_txt:
            self._write_imag_self_energy(output_filename=output_filename)

        return vals

    def write_imag_self_energy(self, filename=None):
        warnings.warn("Phono3py.write_imag_self_energy is deprecated."
                      "Use Phono3py.run_imag_self_energy with write_txt=True.",
                      DeprecationWarning)
        self._write_imag_self_energy(output_filename=filename)

    def _write_imag_self_energy(self, output_filename=None):
        write_imag_self_energy(
            self._gammas,
            self._mesh_numbers,
            self._grid_points,
            self._band_indices,
            self._frequency_points,
            self._temperatures,
            self._sigmas,
            scattering_event_class=self._scattering_event_class,
            output_filename=output_filename,
            is_mesh_symmetry=self._is_mesh_symmetry,
            log_level=self._log_level)

    def run_real_self_energy(
            self,
            grid_points,
            temperatures,
            run_on_bands=False,
            frequency_points=None,
            frequency_step=None,
            num_frequency_points=None,
            epsilons=None,
            write_txt=False,
            write_hdf5=False,
            output_filename=None):
        """Calculate real-part of self-energy of bubble diagram (Delta)

        Pi = Delta - i Gamma.

        Parameters
        ----------
        grid_points : array_like
            Grid-point indices where real part of self-energies are
            caclculated.
            dtype=int, shape=(grid_points,)
        temperatures : array_like
            Temperatures where real part of self-energies  are calculated.
            dtype=float, shape=(temperatures,)
        run_on_bands : bool, optional
            With False, frequency shifts are calculated at frquency sampling
            points. When True, they are done at the phonon frequencies.
            Default is False.
        frequency_points : array_like, optional
            Frequency sampling points. Default is None. In this case,
            num_frequency_points or frequency_step is used to generate uniform
            frequency sampling points.
            dtype=float, shape=(frequency_points,)
        frequency_step : float, optional
            Uniform pitch of frequency sampling points. Default is None. This
            results in using num_frequency_points.
        num_frequency_points: Int, optional
            Number of sampling sampling points to be used instead of
            frequency_step. This number includes end points. Default is None,
            which gives 201.
        epsilons : array_like
            Smearing widths to computer principal part. When multiple values
            are given frequency shifts for those values are returned.
            dtype=float, shape=(epsilons,)
        write_txt : bool, optional
            Frequency points and real part of self-energies are written
            into text files.
        write_hdf5 : bool
            Results are stored in hdf5 files independently at grid points,
            epsilons, and temperatures.
        output_filename : str
            This string is inserted in the output file names.

        """

        if self._interaction is None:
            msg = ("Phono3py.init_phph_interaction has to be called "
                   "before running this method.")
            raise RuntimeError(msg)

        if epsilons is not None:
            _epsilons = epsilons
        else:
            if len(self._sigmas) == 1 and self._sigmas[0] is None:
                _epsilons = None
            elif self._sigmas[0] is None:
                _epsilons = self._sigmas[1:]
            else:
                _epsilons = self._sigmas

        # (epsilon, grid_point, temperature, band)
        frequency_points, deltas = get_real_self_energy(
            self._interaction,
            grid_points,
            temperatures,
            run_on_bands=run_on_bands,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
            epsilons=_epsilons,
            write_hdf5=write_hdf5,
            output_filename=output_filename,
            log_level=self._log_level)

        if write_txt:
            write_real_self_energy(
                deltas,
                self._mesh_numbers,
                grid_points,
                self._band_indices,
                frequency_points,
                temperatures,
                _epsilons,
                output_filename=output_filename,
                is_mesh_symmetry=self._is_mesh_symmetry,
                log_level=self._log_level)

        return frequency_points, deltas

    def run_spectral_function(
            self,
            grid_points,
            temperatures,
            frequency_points=None,
            frequency_step=None,
            num_frequency_points=None,
            num_points_in_batch=None,
            write_txt=False,
            write_hdf5=False,
            output_filename=None):
        """Frequency shift from lowest order diagram is calculated.

        Parameters
        ----------
        grid_points : array_like
            Grid-point indices where imag-self-energeis are caclculated.
            dtype=int, shape=(grid_points,)
        temperatures : array_like
            Temperatures where imag-self-energies are calculated.
            dtype=float, shape=(temperatures,)
        frequency_points : array_like, optional
            Frequency sampling points. Default is None. In this case,
            num_frequency_points or frequency_step is used to generate uniform
            frequency sampling points.
            dtype=float, shape=(frequency_points,)
        frequency_step : float, optional
            Uniform pitch of frequency sampling points. Default is None. This
            results in using num_frequency_points.
        num_frequency_points: Int, optional
            Number of sampling sampling points to be used instead of
            frequency_step. This number includes end points. Default is None,
            which gives 201.
        num_points_in_batch: int, optional
            Number of sampling points in one batch. This is for the frequency
            sampling mode and the sampling points are divided into batches.
            Lager number provides efficient use of multi-cores but more
            memory demanding. Default is None, which give the number of 10.
        write_txt : bool, optional
            Frequency points and spectral functions are written
            into text files. Default is False.
        write_hdf5 : bool
            Results are stored in hdf5 files independently at grid points,
            epsilons. Default is False.
        output_filename : str
            This string is inserted in the output file names.

        """

        if self._interaction is None:
            msg = ("Phono3py.init_phph_interaction has to be called "
                   "before running this method.")
            raise RuntimeError(msg)

        self._spectral_function = run_spectral_function(
            self._interaction,
            grid_points,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
            num_points_in_batch=num_points_in_batch,
            temperatures=temperatures,
            band_indices=self._band_indices,
            sigmas=self._sigmas,
            write_txt=write_txt,
            write_hdf5=write_hdf5,
            log_level=self._log_level)

    def run_thermal_conductivity(
            self,
            is_LBTE=False,
            temperatures=None,
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
            compression="gzip",
            input_filename=None,
            output_filename=None):
        if self._interaction is None:
            msg = ("Phono3py.init_phph_interaction has to be called "
                   "before running this method.")
            raise RuntimeError(msg)

        if is_LBTE:
            if temperatures is None:
                _temperatures = [300, ]
            else:
                _temperatures = temperatures
            self._thermal_conductivity = get_thermal_conductivity_LBTE(
                self._interaction,
                self._primitive_symmetry,
                temperatures=_temperatures,
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
            if temperatures is None:
                _temperatures = np.arange(0, 1001, 10, dtype='double')
            else:
                _temperatures = temperatures
            self._thermal_conductivity = get_thermal_conductivity_RTA(
                self._interaction,
                self._primitive_symmetry,
                temperatures=_temperatures,
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

    def save(self,
             filename="phono3py_params.yaml",
             settings=None):
        """Save parameters in Phono3py instants into file.

        Parameters
        ----------
        filename: str, optional
            File name. Default is "phono3py_params.yaml"
        settings: dict, optional
            It is described which parameters are written out. Only
            the settings expected to be updated from the following
            default settings are needed to be set in the dictionary.
            The possible parameters and their default settings are:
                {'force_sets': False,
                 'displacements': True,
                 'force_constants': False,
                 'born_effective_charge': True,
                 'dielectric_constant': True}

        """
        ph3py_yaml = Phono3pyYaml(settings=settings)
        ph3py_yaml.set_phonon_info(self)
        with open(filename, 'w') as w:
            w.write(str(ph3py_yaml))

    ###################
    # private methods #
    ###################
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
        magmoms = supercell.magnetic_moments
        masses = supercell.masses
        numbers = supercell.numbers
        lattice = supercell.cell

        for disp1 in displacement_dataset['first_atoms']:
            disp_cart1 = disp1['displacement']
            positions = supercell.get_positions()
            positions[disp1['number']] += disp_cart1
            supercells.append(PhonopyAtoms(numbers=numbers,
                                           masses=masses,
                                           magmoms=magmoms,
                                           positions=positions,
                                           cell=lattice,
                                           pbc=True))

        return supercells

    def _build_supercells_with_displacements(self):
        supercells = []
        magmoms = self._supercell.magnetic_moments
        masses = self._supercell.masses
        numbers = self._supercell.numbers
        lattice = self._supercell.cell

        supercells = self._build_phonon_supercells_with_displacements(
            self._supercell,
            self._dataset)

        for disp1 in self._dataset['first_atoms']:
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
                    supercells.append(PhonopyAtoms(numbers=numbers,
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

    def _set_mesh_numbers(self, mesh):
        # initialization related to mesh
        self._interaction = None

        _mesh = np.array(mesh)
        mesh_nums = None
        if _mesh.shape:
            if _mesh.shape == (3,):
                mesh_nums = mesh
        elif self._primitive_symmetry is None:
            mesh_nums = length2mesh(mesh, self._primitive.get_cell())
        else:
            rotations = self._primitive_symmetry.get_pointgroup_operations()
            mesh_nums = length2mesh(mesh, self._primitive.cell,
                                    rotations=rotations)
        if mesh_nums is None:
            msg = "mesh has inappropriate type."
            raise TypeError(msg)
        self._mesh_numbers = mesh_nums

    def _init_dynamical_matrix(self):
        if self._interaction is not None:
            self._interaction.init_dynamical_matrix(
                self._fc2,
                self._phonon_supercell,
                self._phonon_primitive,
                nac_params=self._nac_params,
                solve_dynamical_matrices=False,
                verbose=self._log_level)
