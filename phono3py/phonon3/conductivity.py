import textwrap
import numpy as np
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.thermal_properties import mode_cv as get_mode_cv
from phonopy.units import THzToEv, EV, THz, Angstrom
from phono3py.file_IO import write_pp_to_hdf5
from phono3py.phonon3.triplets import (get_grid_address, reduce_grid_points,
                                       get_ir_grid_points,
                                       from_coarse_to_dense_grid_points,
                                       get_grid_points_by_rotations,
                                       get_all_triplets)
from phono3py.other.isotope import Isotope

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi))  # 2pi comes from definition of lifetime.


def all_bands_exist(interaction):
    band_indices = interaction.get_band_indices()
    num_band = interaction.get_primitive().get_number_of_atoms() * 3
    if len(band_indices) == num_band:
        if (band_indices - np.arange(num_band) == 0).all():
            return True
    return False


def write_pp(conductivity,
             pp,
             i,
             filename=None,
             compression=None):
    grid_point = conductivity.get_grid_points()[i]
    sigmas = conductivity.get_sigmas()
    sigma_cutoff = conductivity.get_sigma_cutoff_width()
    mesh = conductivity.get_mesh_numbers()
    triplets, weights, map_triplets, _ = pp.get_triplets_at_q()
    grid_address = pp.get_grid_address()
    bz_map = pp.get_bz_map()
    if map_triplets is None:
        all_triplets = None
    else:
        all_triplets = get_all_triplets(grid_point,
                                        grid_address,
                                        bz_map,
                                        mesh)

    if len(sigmas) > 1:
        print("Multiple smearing parameters were given. The last one in ")
        print("ph-ph interaction calculations was written in the file.")

    write_pp_to_hdf5(mesh,
                     pp=pp.get_interaction_strength(),
                     g_zero=pp.get_zero_value_positions(),
                     grid_point=grid_point,
                     triplet=triplets,
                     weight=weights,
                     triplet_map=map_triplets,
                     triplet_all=all_triplets,
                     sigma=sigmas[-1],
                     sigma_cutoff=sigma_cutoff,
                     filename=filename,
                     compression=compression)


class Conductivity(object):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=None,
                 sigmas=None,
                 sigma_cutoff=None,
                 is_isotope=False,
                 mass_variances=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 boundary_mfp=None,  # in micrometre
                 is_kappa_star=True,
                 gv_delta_q=None,  # finite difference for group veolocity
                 is_full_pp=False,
                 log_level=0):
        if sigmas is None:
            self._sigmas = []
        else:
            self._sigmas = sigmas
        self._sigma_cutoff = sigma_cutoff
        self._pp = interaction
        self._is_full_pp = is_full_pp
        self._collision = None  # has to be set derived class
        if temperatures is None:
            self._temperatures = None
        else:
            self._temperatures = np.array(temperatures, dtype='double')
        self._is_kappa_star = is_kappa_star
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._primitive = self._pp.get_primitive()
        self._dm = self._pp.get_dynamical_matrix()
        self._frequency_factor_to_THz = self._pp.get_frequency_factor_to_THz()
        self._cutoff_frequency = self._pp.get_cutoff_frequency()
        self._boundary_mfp = boundary_mfp

        self._symmetry = symmetry

        if not self._is_kappa_star:
            self._point_operations = np.array([np.eye(3, dtype='intc')],
                                              dtype='intc')
        else:
            self._point_operations = symmetry.get_reciprocal_operations()
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = np.array(
            [similarity_transformation(rec_lat, r)
             for r in self._point_operations], dtype='double')

        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None
        self._ir_grid_points = None
        self._ir_grid_weights = None

        self._read_gamma = False
        self._read_gamma_iso = False

        self._kappa = None
        self._mode_kappa = None

        self._frequencies = None
        self._cv = None
        self._gv = None
        self._gv_sum2 = None
        self._gamma = None
        self._gamma_iso = None
        self._num_sampling_grid_points = 0

        self._mesh = None
        self._mesh_divisors = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._set_mesh_numbers(mesh_divisors=mesh_divisors,
                               coarse_mesh_shifts=coarse_mesh_shifts)
        volume = self._primitive.get_volume()
        self._conversion_factor = unit_to_WmK / volume

        self._isotope = None
        self._mass_variances = None
        self._is_isotope = is_isotope
        if mass_variances is not None:
            self._is_isotope = True
        if self._is_isotope:
            self._set_isotope(mass_variances)

        self._grid_point_count = None
        self._set_grid_properties(grid_points)

        if (self._dm.is_nac() and
            self._dm.get_nac_method() == 'gonze' and
            self._gv_delta_q is None):
            self._gv_delta_q = 1e-5
            if self._log_level:
                msg = "Group velocity calculation:\n"
                text = ("Analytical derivative of dynamical matrix is not "
                        "implemented for NAC by Gonze et al. Instead "
                        "numerical derivative of it is used with dq=1e-5 "
                        "for group velocity calculation.")
                msg += textwrap.fill(text,
                                     initial_indent="  ",
                                     subsequent_indent="  ",
                                     width=70)
                print(msg)
        self._gv_obj = GroupVelocity(
            self._dm,
            q_length=self._gv_delta_q,
            symmetry=self._symmetry,
            frequency_factor_to_THz=self._frequency_factor_to_THz)
        # gv_delta_q may be changed.
        self._gv_delta_q = self._gv_obj.get_q_length()

    def __iter__(self):
        return self

    def __next__(self):
        if self._grid_point_count == len(self._grid_points):
            if self._log_level:
                print("=================== End of collection of collisions "
                      "===================")
            raise StopIteration
        else:
            self._run_at_grid_point()
            self._grid_point_count += 1
            return self._grid_point_count - 1

    def next(self):
        return self.__next__()

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def get_mesh_numbers(self):
        return self._mesh

    def get_mode_heat_capacities(self):
        return self._cv

    def get_group_velocities(self):
        return self._gv

    def get_gv_by_gv(self):
        return self._gv_sum2

    def get_frequencies(self):
        return self._frequencies[self._grid_points]

    def get_qpoints(self):
        return self._qpoints

    def get_grid_points(self):
        return self._grid_points

    def get_grid_weights(self):
        return self._grid_weights

    def get_temperatures(self):
        return self._temperatures

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures
        self._allocate_values()

    def set_gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True

    def set_gamma_isotope(self, gamma_iso):
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = True

    def get_gamma(self):
        return self._gamma

    def get_gamma_isotope(self):
        return self._gamma_iso

    def get_kappa(self):
        return self._kappa

    def get_mode_kappa(self):
        return self._mode_kappa

    def get_sigmas(self):
        return self._sigmas

    def get_sigma_cutoff_width(self):
        return self._sigma_cutoff

    def get_grid_point_count(self):
        return self._grid_point_count

    def get_averaged_pp_interaction(self):
        return self._averaged_pp_interaction

    def _run_at_grid_point(self):
        """This has to be implementated in the derived class"""
        pass

    def _allocate_values(self):
        """This has to be implementated in the derived class"""
        pass

    def _set_grid_properties(self, grid_points):
        self._grid_address = self._pp.get_grid_address()
        self._pp.set_nac_q_direction(nac_q_direction=None)

        if grid_points is not None:  # Specify grid points
            self._grid_points = reduce_grid_points(
                self._mesh_divisors,
                self._grid_address,
                grid_points,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            (self._ir_grid_points,
             self._ir_grid_weights) = self._get_ir_grid_points()
        elif not self._is_kappa_star:  # All grid points
            coarse_grid_address = get_grid_address(self._coarse_mesh)
            coarse_grid_points = np.arange(np.prod(self._coarse_mesh),
                                           dtype='uintp')
            self._grid_points = from_coarse_to_dense_grid_points(
                self._mesh,
                self._mesh_divisors,
                coarse_grid_points,
                coarse_grid_address,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            self._grid_weights = np.ones(len(self._grid_points), dtype='intc')
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights
        else:  # Automatic sampling
            self._grid_points, self._grid_weights = self._get_ir_grid_points()
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights

        self._qpoints = np.array(self._grid_address[self._grid_points] /
                                 self._mesh.astype('double'),
                                 dtype='double', order='C')

        self._grid_point_count = 0
        # set_phonons is unnecessary now because all phonons are calculated in
        # self._pp.set_dynamical_matrix, though Gamma-point is an exception,
        # which is treatd at self._pp.set_grid_point.
        # self._pp.set_phonons(self._grid_points)
        self._frequencies, self._eigenvectors, _ = self._pp.get_phonons()

    def _get_gamma_isotope_at_sigmas(self, i):
        gamma_iso = []
        bz_map = self._pp.get_bz_map()
        pp_freqs, pp_eigvecs, pp_phonon_done = self._pp.get_phonons()

        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating Gamma of ph-isotope with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)

            self._isotope.set_sigma(sigma)
            self._isotope.set_phonons(self._grid_address,
                                      bz_map,
                                      pp_freqs,
                                      pp_eigvecs,
                                      pp_phonon_done,
                                      dm=self._dm)
            gp = self._grid_points[i]
            self._isotope.set_grid_point(gp)
            self._isotope.run()
            gamma_iso.append(self._isotope.get_gamma())

        return np.array(gamma_iso, dtype='double', order='C')

    def _set_mesh_numbers(self, mesh_divisors=None, coarse_mesh_shifts=None):
        self._mesh = self._pp.get_mesh_numbers()

        if mesh_divisors is None:
            self._mesh_divisors = np.array([1, 1, 1], dtype='intc')
        else:
            self._mesh_divisors = []
            for i, (m, n) in enumerate(zip(self._mesh, mesh_divisors)):
                if m % n == 0:
                    self._mesh_divisors.append(n)
                else:
                    self._mesh_divisors.append(1)
                    print(("Mesh number %d for the " +
                           ["first", "second", "third"][i] +
                           " axis is not dividable by divisor %d.") % (m, n))
            self._mesh_divisors = np.array(self._mesh_divisors, dtype='intc')
            if coarse_mesh_shifts is None:
                self._coarse_mesh_shifts = [False, False, False]
            else:
                self._coarse_mesh_shifts = coarse_mesh_shifts
            for i in range(3):
                if (self._coarse_mesh_shifts[i] and
                    (self._mesh_divisors[i] % 2 != 0)):
                    print("Coarse grid along " +
                          ["first", "second", "third"][i] +
                          " axis can not be shifted. Set False.")
                    self._coarse_mesh_shifts[i] = False

        self._coarse_mesh = self._mesh // self._mesh_divisors

        if self._log_level:
            print("Lifetime sampling mesh: [ %d %d %d ]" %
                  tuple(self._mesh // self._mesh_divisors))

    def _get_ir_grid_points(self):
        if self._coarse_mesh_shifts is None:
            mesh_shifts = [False, False, False]
        else:
            mesh_shifts = self._coarse_mesh_shifts
        (coarse_grid_points,
         coarse_grid_weights,
         coarse_grid_address, _) = get_ir_grid_points(
             self._coarse_mesh,
             self._symmetry.get_pointgroup_operations(),
             mesh_shifts=mesh_shifts)
        grid_points = from_coarse_to_dense_grid_points(
            self._mesh,
            self._mesh_divisors,
            coarse_grid_points,
            coarse_grid_address,
            coarse_mesh_shifts=self._coarse_mesh_shifts)
        grid_weights = coarse_grid_weights

        assert grid_weights.sum() == np.prod(self._mesh // self._mesh_divisors)

        return grid_points, grid_weights

    def _set_isotope(self, mass_variances):
        if mass_variances is True:
            mv = None
        else:
            mv = mass_variances
        self._isotope = Isotope(
            self._mesh,
            self._primitive,
            mass_variances=mv,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._symmetry.get_symmetry_tolerance(),
            cutoff_frequency=self._cutoff_frequency,
            lapack_zheev_uplo=self._pp.get_lapack_zheev_uplo())
        self._mass_variances = self._isotope.get_mass_variances()

    def _set_harmonic_properties(self, i_irgp, i_data):
        grid_point = self._grid_points[i_irgp]
        freqs = self._frequencies[grid_point][self._pp.get_band_indices()]
        self._cv[:, i_data, :] = self._get_cv(freqs)
        gv = self._get_gv(self._qpoints[i_irgp])
        self._gv[i_data] = gv[self._pp.get_band_indices(), :]

        # Outer product of group velocities (v x v) [num_k*, num_freqs, 3, 3]
        gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i_irgp, i_data)
        self._num_sampling_grid_points += order_kstar

        # Sum all vxv at k*
        for j, vxv in enumerate(
            ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            self._gv_sum2[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

    def _get_gv(self, q):
        self._gv_obj.set_q_points([q])
        return self._gv_obj.get_group_velocity()[0]

    def _get_gv_by_gv(self, i_irgp, i_data):
        rotation_map = get_grid_points_by_rotations(
            self._grid_address[self._grid_points[i_irgp]],
            self._point_operations,
            self._mesh)
        gv = self._gv[i_data]
        gv_by_gv = np.zeros((len(gv), 3, 3), dtype='double')

        for r in self._rotations_cartesian:
            gvs_rot = np.dot(gv, r.T)
            gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv /= len(rotation_map) // len(np.unique(rotation_map))
        order_kstar = len(np.unique(rotation_map))

        if self._grid_weights is not None:
            if order_kstar != self._grid_weights[i_irgp]:
                if self._log_level:
                    print("*" * 33 + "Warning" + "*" * 33)
                    print(" Number of elements in k* is unequal "
                          "to number of equivalent grid-points.")
                    print("*" * 73)

        return gv_by_gv, order_kstar

    def _get_cv(self, freqs):
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype='double')
        # T/freq has to be large enough to avoid divergence.
        # Otherwise just set 0.
        for i, f in enumerate(freqs):
            finite_t = (self._temperatures > f / 100)
            if f > self._cutoff_frequency:
                cv[:, i] = np.where(
                    finite_t, get_mode_cv(
                        np.where(finite_t, self._temperatures, 10000),
                        f * THzToEv), 0)
        return cv

    def _get_main_diagonal(self, i, j, k):
        num_band = self._primitive.get_number_of_atoms() * 3
        main_diagonal = self._gamma[j, k, i].copy()
        if self._gamma_iso is not None:
            main_diagonal += self._gamma_iso[j, i]
        if self._boundary_mfp is not None:
            main_diagonal += self._get_boundary_scattering(i)

        # if self._boundary_mfp is not None:
        #     for l in range(num_band):
        #         # Acoustic modes at Gamma are avoided.
        #         if i == 0 and l < 3:
        #             continue
        #         gv_norm = np.linalg.norm(self._gv[i, l])
        #         mean_free_path = (gv_norm * Angstrom * 1e6 /
        #                           (4 * np.pi * main_diagonal[l]))
        #         if mean_free_path > self._boundary_mfp:
        #             main_diagonal[l] = (
        #                 gv_norm / (4 * np.pi * self._boundary_mfp))

        return main_diagonal

    def _get_boundary_scattering(self, i):
        num_band = self._primitive.get_number_of_atoms() * 3
        g_boundary = np.zeros(num_band, dtype='double')
        for l in range(num_band):
            g_boundary[l] = (np.linalg.norm(self._gv[i, l]) * Angstrom * 1e6 /
                             (4 * np.pi * self._boundary_mfp))
        return g_boundary

    def _show_log_header(self, i):
        if self._log_level:
            gp = self._grid_points[i]
            print("======================= Grid point %d (%d/%d) "
                  "=======================" %
                  (gp, i + 1, len(self._grid_points)))
            print("q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i]))
            if self._boundary_mfp is not None:
                if self._boundary_mfp > 1000:
                    print("Boundary mean free path (millimetre): %.3f" %
                          (self._boundary_mfp / 1000.0))
                else:
                    print("Boundary mean free path (micrometre): %.5f" %
                          self._boundary_mfp)
            if self._is_isotope:
                print(("Mass variance parameters: " +
                       "%5.2e " * len(self._mass_variances)) %
                      tuple(self._mass_variances))
