import numpy as np
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.units import (VaspToTHz, Hbar, EV, Angstrom, THz, AMU,
                           PlanckConstant)
from phono3py.phonon.solver import set_phonon_c, set_phonon_py
from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phono3py.phonon3.reciprocal_to_normal import ReciprocalToNormal
from phono3py.phonon3.triplets import (get_triplets_at_q,
                                       get_nosym_triplets_at_q,
                                       get_bz_grid_address)

class Interaction(object):
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 symmetry,
                 fc3=None,
                 band_indices=None,
                 constant_averaged_interaction=None,
                 frequency_factor_to_THz=VaspToTHz,
                 unit_conversion=None,
                 is_mesh_symmetry=True,
                 symmetrize_fc3_q=False,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._fc3 = fc3 
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = np.array(mesh, dtype='intc')
        self._symmetry = symmetry

        self._band_indices = None
        self._set_band_indices(band_indices)
        self._constant_averaged_interaction = constant_averaged_interaction
        self._frequency_factor_to_THz = frequency_factor_to_THz

        # Unit to eV^2
        if unit_conversion is None:
            num_grid = np.prod(self._mesh)
            self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                     * EV ** 2 / Angstrom ** 6
                                     / (2 * np.pi * THz) ** 3
                                     / AMU ** 3 / num_grid
                                     / EV ** 2)
        else:
            self._unit_conversion = unit_conversion
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._is_mesh_symmetry = is_mesh_symmetry
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._symprec = symmetry.get_symmetry_tolerance()

        self._grid_point = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._triplets_map_at_q = None
        self._ir_map_at_q = None
        self._grid_address = None
        self._bz_map = None
        self._interaction_strength = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._nac_q_direction = None

        self._band_index_count = 0

        try:
            svecs, multiplicity = self._primitive.get_smallest_vectors()
        except AttributeError:
            from phonopy.harmonic.dynamical_matrix import get_smallest_vectors
            svecs, multiplicity = get_smallest_vectors(self._supercell,
                                                       self._primitive,
                                                       self._symprec)
        self._smallest_vectors = svecs
        self._multiplicity = multiplicity
        self._masses = np.array(self._primitive.get_masses(), dtype='double')
        self._p2s = self._primitive.get_primitive_to_supercell_map()
        self._s2p = self._primitive.get_supercell_to_primitive_map()
        
        self._allocate_phonon()
        
    def run(self, lang='C', g_zero=None):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_triplets = len(self._triplets_at_q)

        self._interaction_strength = np.empty(
            (num_triplets, len(self._band_indices), num_band, num_band),
            dtype='double')
        if self._constant_averaged_interaction is None:
            self._interaction_strength[:] = 0
            if lang == 'C':
                self._run_c(g_zero)
            else:
                self._run_py()
        else:
            num_grid = np.prod(self._mesh)
            self._interaction_strength[:] = (
                self._constant_averaged_interaction / num_grid)

    def get_interaction_strength(self):
        return self._interaction_strength

    def get_mesh_numbers(self):
        return self._mesh
    
    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done

    def get_fc3(self):
        return self._fc3

    def get_dynamical_matrix(self):
        return self._dm

    def get_primitive(self):
        return self._primitive

    def get_supercell(self):
        return self._supercell

    def get_triplets_at_q(self):
        return (self._triplets_at_q,
                self._weights_at_q,
                self._triplets_map_at_q,
                self._ir_map_at_q)

    def get_grid_address(self):
        return self._grid_address

    def get_bz_map(self):
        return self._bz_map
    
    def get_band_indices(self):
        return self._band_indices

    def get_frequency_factor_to_THz(self):
        return self._frequency_factor_to_THz

    def get_lapack_zheev_uplo(self):
        return self._lapack_zheev_uplo

    def get_cutoff_frequency(self):
        return self._cutoff_frequency
        
    def get_averaged_interaction(self):
        v = self._interaction_strength
        w = self._weights_at_q
        v_sum = v.sum(axis=2).sum(axis=2)
        num_band = self._primitive.get_number_of_atoms() * 3
        return np.dot(w, v_sum) / num_band ** 2

    def get_primitive_and_supercell_correspondence(self):
        return (self._smallest_vectors,
                self._multiplicity,
                self._p2s,
                self._s2p,
                self._masses)

    def get_nac_q_direction(self):
        return self._nac_q_direction

    def get_unit_conversion_factor(self):
        return self._unit_conversion

    def set_grid_point(self, grid_point, stores_triplets_map=False):
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        if not self._is_mesh_symmetry:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map,
             triplets_map_at_q,
             ir_map_at_q) = get_nosym_triplets_at_q(
                 grid_point,
                 self._mesh,
                 reciprocal_lattice,
                 stores_triplets_map=stores_triplets_map)
        else:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map,
             triplets_map_at_q,
             ir_map_at_q)= get_triplets_at_q(
                 grid_point,
                 self._mesh,
                 self._symmetry.get_pointgroup_operations(),
                 reciprocal_lattice,
                 stores_triplets_map=stores_triplets_map)

        # Special treatment of symmetry is applied when q_direction is used.
        if (grid_address[grid_point] == 0).all():
            self._phonon_done[grid_point] = 0
            self.set_phonons(np.array([0], dtype='intc'))
            if self._nac_q_direction is not None:
                rotations = []
                for r in self._symmetry.get_pointgroup_operations():
                    dq = self._nac_q_direction
                    dq /= np.linalg.norm(dq)
                    diff = np.dot(dq, r) - dq
                    if (abs(diff) < 1e-5).all():
                        rotations.append(r)
                (triplets_at_q,
                 weights_at_q,
                 grid_address,
                 bz_map,
                 triplets_map_at_q,
                 ir_map_at_q) = get_triplets_at_q(
                     grid_point,
                     self._mesh,
                     np.array(rotations, dtype='intc', order='C'),
                     reciprocal_lattice,
                     is_time_reversal=False,
                     stores_triplets_map=stores_triplets_map)

        for triplet in triplets_at_q:
            sum_q = (grid_address[triplet]).sum(axis=0)
            if (sum_q % self._mesh != 0).any():
                print("============= Warning ==================")
                print("%s" % triplet)
                for tp in triplet:
                    print("%s %s" %
                          (grid_address[tp],
                           np.linalg.norm(
                               np.dot(reciprocal_lattice,
                                      grid_address[tp] /
                                      self._mesh.astype('double')))))
                print("%s" % sum_q)
                print("============= Warning ==================")

        self._grid_point = grid_point
        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._triplets_map_at_q = triplets_map_at_q
        # self._grid_address = grid_address
        # self._bz_map = bz_map
        self._ir_map_at_q = ir_map_at_q

        # set_phonons is unnecessary now because all phonons are calculated in
        # set_dynamical_matrix, though Gamma-point is an exception.
        # self.set_phonons(self._triplets_at_q.ravel())

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=self._symprec)
        self.set_phonons(np.arange(len(self._grid_address), dtype='intc'))
        if (self._grid_address[0] == 0).all():
            if np.sum(self._frequencies[0] < self._cutoff_frequency) < 3:
                for i, f in enumerate(self._frequencies[0, :3]):
                    if not (f < self._cutoff_frequency):
                        self._frequencies[0, i] = 0
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(" Phonon frequency of band index %d at Gamma "
                              "is calculated to be %f." % (i + 1, f))
                        print(" But this frequency is forced to be zero.")
                        print("=" * 61)

    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def set_phonon_data(self, frequencies, eigenvectors, grid_address):
        if grid_address.shape != self._grid_address.shape:
            print("=" * 26 + " Warning " + "=" * 26)
            print("Input grid address size is inconsistent. "
                  "Setting phonons faild.")
            print("=" * 26 + " Warning " + "=" * 26)
            return False

        if (self._grid_address - grid_address).all():
            print("=" * 26 + " Warning " + "=" * 26)
            print("Input grid addresses are inconsistent. "
                  "Setting phonons faild.")
            print("=" * 26 + " Warning " + "=" * 26)
            return False
        else:
            self._phonon_done[:] = 1
            self._frequencies[:] = frequencies
            self._eigenvectors[:] = eigenvectors
            return True

    def set_phonons(self, grid_points):
        # for i, grid_triplet in enumerate(self._triplets_at_q):
        #     for gp in grid_triplet:
        #         self._set_phonon_py(gp)
        self._set_phonon_c(grid_points)

    def delete_interaction_strength(self):
        self._interaction_strength = None

    def _set_band_indices(self, band_indices):
        num_band = self._primitive.get_number_of_atoms() * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.array(band_indices, dtype='intc')

    def _run_c(self, g_zero):
        import phono3py._phono3py as phono3c
        
        num_band = self._primitive.get_number_of_atoms() * 3

        if g_zero is None or self._symmetrize_fc3_q:
            _g_zero = np.zeros(self._interaction_strength.shape,
                               dtype='byte', order='C')
        else:
            _g_zero = g_zero

        phono3c.interaction(self._interaction_strength,
                            _g_zero,
                            self._frequencies,
                            self._eigenvectors,
                            self._triplets_at_q,
                            self._grid_address,
                            self._mesh,
                            self._fc3,
                            self._smallest_vectors,
                            self._multiplicity,
                            self._masses,
                            self._p2s,
                            self._s2p,
                            self._band_indices,
                            self._symmetrize_fc3_q,
                            self._cutoff_frequency)
        self._interaction_strength *= self._unit_conversion

    def _set_phonon_c(self, grid_points):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
                     self._phonon_done,
                     grid_points,
                     self._grid_address,
                     self._mesh,
                     self._frequency_factor_to_THz,
                     self._nac_q_direction,
                     self._lapack_zheev_uplo)
        
    def _run_py(self):
        r2r = RealToReciprocal(self._fc3,
                               self._supercell,
                               self._primitive,
                               self._mesh,
                               symprec=self._symprec)
        r2n = ReciprocalToNormal(self._primitive,
                                 self._frequencies,
                                 self._eigenvectors,
                                 self._band_indices,
                                 cutoff_frequency=self._cutoff_frequency)

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print("%d / %d" % (i + 1, len(self._triplets_at_q)))
            r2r.run(self._grid_address[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            for gp in grid_triplet:
                self._set_phonon_py(gp)
            r2n.run(fc3_reciprocal, grid_triplet)
            self._interaction_strength[i] = np.abs(
                r2n.get_reciprocal_to_normal()) ** 2 * self._unit_conversion

    def _set_phonon_py(self, grid_point):
        set_phonon_py(grid_point,
                      self._phonon_done,
                      self._frequencies,
                      self._eigenvectors,
                      self._grid_address,
                      self._mesh,
                      self._dm,
                      self._frequency_factor_to_THz,                  
                      self._lapack_zheev_uplo)

    def _allocate_phonon(self):
        primitive_lattice = np.linalg.inv(self._primitive.get_cell())
        self._grid_address, self._bz_map = get_bz_grid_address(
            self._mesh, primitive_lattice, with_boundary=True)
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid = len(self._grid_address)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        itemsize = self._frequencies.itemsize
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype=("c%d" % (itemsize * 2)))
