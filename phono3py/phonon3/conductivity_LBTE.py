import sys
import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.units import THz, Angstrom
from phono3py.phonon3.conductivity import Conductivity, unit_to_WmK
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.triplets import (get_grid_points_by_rotations,
                                       get_BZ_grid_points_by_rotations)
from phono3py.file_IO import (write_kappa_to_hdf5,
                              write_collision_to_hdf5,
                              read_collision_from_hdf5,
                              write_collision_eigenvalues_to_hdf5)
from phonopy.units import THzToEv, Kb

def get_thermal_conductivity_LBTE(
        interaction,
        symmetry,
        temperatures=np.arange(0, 1001, 10, dtype='double'),
        sigmas=None,
        is_isotope=False,
        mass_variances=None,
        grid_points=None,
        boundary_mfp=None, # in micrometre
        is_reducible_collision_matrix=False,
        is_kappa_star=True,
        gv_delta_q=1e-4, # for group velocity
        is_full_pp=False,
        pinv_cutoff=1.0e-8,
        pinv_solver=0, # default: dsyev in lapacke
        write_collision=False,
        read_collision=False,
        write_kappa=False,
        input_filename=None,
        output_filename=None,
        log_level=0):

    if sigmas is None:
        sigmas = []
    if log_level:
        print("-------------------- Lattice thermal conducitivity (LBTE) "
              "--------------------")
        print("Cutoff frequency of pseudo inversion of collision matrix: %s" %
              pinv_cutoff)

    if read_collision:
        temps = None
    else:
        temps = temperatures

    lbte = Conductivity_LBTE(
        interaction,
        symmetry,
        grid_points=grid_points,
        temperatures=temps,
        sigmas=sigmas,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        log_level=log_level)

    if read_collision:
        read_from = _set_collision_from_file(
            lbte,
            indices=read_collision,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            filename=input_filename,
            log_level=log_level)
        if not read_from:
            print("Reading gamma failed.")
            return False
        if log_level:
            temperatures = lbte.get_temperatures()
            if len(temperatures) > 5:
                text = (" %.1f " * 5 + "...") % tuple(temperatures[:5])
                text += " %.1f" % temperatures[-1]
            else:
                text = (" %.1f " * len(temperatures)) % tuple(temperatures)
            print("Temperature: " + text)

    for i in lbte:
        if write_collision:
            _write_collision(
                lbte,
                i=i,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                filename=output_filename)

    if (not read_collision) or (read_collision and read_from == "grid_points"):
        if grid_points is None:
            _write_collision(lbte, filename=output_filename)

    if write_kappa and grid_points is None:
        lbte.set_kappa_at_sigmas()
        _write_kappa(
            lbte,
            interaction.get_primitive().get_volume(),
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            filename=output_filename,
            log_level=log_level)

    return lbte

def _write_collision(lbte,
                     i=None,
                     is_reducible_collision_matrix=False,
                     filename=None):
    temperatures = lbte.get_temperatures()
    sigmas = lbte.get_sigmas()
    gamma = lbte.get_gamma()
    gamma_isotope = lbte.get_gamma_isotope()
    collision_matrix = lbte.get_collision_matrix()
    mesh = lbte.get_mesh_numbers()

    if i is not None:
        gp = lbte.get_grid_points()[i]
        if is_reducible_collision_matrix:
            igp = gp
        else:
            igp = i
        for j, sigma in enumerate(sigmas):
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[j, igp]
            else:
                gamma_isotope_at_sigma = None
                write_collision_to_hdf5(
                    temperatures,
                    mesh,
                    gamma=gamma[j, :, igp],
                    gamma_isotope=gamma_isotope_at_sigma,
                    collision_matrix=collision_matrix[j, :, igp],
                    grid_point=gp,
                    sigma=sigma,
                    filename=filename)
    else:
        for j, sigma in enumerate(sigmas):
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[j]
            else:
                gamma_isotope_at_sigma = None
            write_collision_to_hdf5(temperatures,
                                    mesh,
                                    gamma=gamma[j],
                                    gamma_isotope=gamma_isotope_at_sigma,
                                    collision_matrix=collision_matrix[j],
                                    sigma=sigma,
                                    filename=filename)

def _write_kappa(lbte,
                 volume,
                 is_reducible_collision_matrix=False,
                 filename=None,
                 log_level=0):
    temperatures = lbte.get_temperatures()
    sigmas = lbte.get_sigmas()
    mesh = lbte.get_mesh_numbers()
    weights = lbte.get_grid_weights()
    frequencies = lbte.get_frequencies()
    ave_pp = lbte.get_averaged_pp_interaction()
    qpoints = lbte.get_qpoints()
    kappa = lbte.get_kappa()
    kappa_RTA = lbte.get_kappa_RTA()
    coleigs = lbte.get_collision_eigenvalues()

    if is_reducible_collision_matrix:
        ir_gp = lbte.get_grid_points()
        # frequencies = lbte.get_frequencies_all()
        gamma = lbte.get_gamma()[:, :, ir_gp, :]
        gamma_isotope = lbte.get_gamma_isotope()
        if gamma_isotope is not None:
            gamma_isotope = gamma_isotope[:, ir_gp, :]
        gv = lbte.get_group_velocities()[ir_gp]
        gv_by_gv = lbte.get_gv_by_gv()[ir_gp]
        mode_cv = lbte.get_mode_heat_capacities()[:, ir_gp, :]
        mode_kappa = lbte.get_mode_kappa()[:, :, ir_gp, :, :]
        mode_kappa_RTA = lbte.get_mode_kappa_RTA()[:, :, ir_gp, :, :]
        for i, w in enumerate(weights):
            mode_kappa[:, :, i, :, :] *= w
        mfp = lbte.get_mean_free_path()[:, :, ir_gp, :, :]
    else:
        frequencies = lbte.get_frequencies()
        gamma = lbte.get_gamma()
        gamma_isotope = lbte.get_gamma_isotope()
        gv = lbte.get_group_velocities()
        gv_by_gv = lbte.get_gv_by_gv()
        mode_cv = lbte.get_mode_heat_capacities()
        mode_kappa = lbte.get_mode_kappa()
        mode_kappa_RTA = lbte.get_mode_kappa_RTA()
        mfp = lbte.get_mean_free_path()

    for i, sigma in enumerate(sigmas):
        if gamma_isotope is not None:
            gamma_isotope_at_sigma = gamma_isotope[i]
        else:
            gamma_isotope_at_sigma = None
        write_kappa_to_hdf5(temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            gv_by_gv=gv_by_gv,
                            mean_free_path=mfp[i],
                            heat_capacity=mode_cv,
                            kappa=kappa[i],
                            mode_kappa=mode_kappa[i],
                            kappa_RTA=kappa_RTA[i],
                            mode_kappa_RTA=mode_kappa_RTA[i],
                            gamma=gamma[i],
                            gamma_isotope=gamma_isotope_at_sigma,
                            averaged_pp_interaction=ave_pp,
                            qpoint=qpoints,
                            weight=weights,
                            sigma=sigma,
                            kappa_unit_conversion=unit_to_WmK / volume,
                            filename=filename,
                            verbose=log_level)
        
        if coleigs is not None:
            write_collision_eigenvalues_to_hdf5(temperatures,
                                                mesh,
                                                coleigs[i],
                                                sigma=sigma,
                                                filename=filename,
                                                verbose=log_level)


def _set_collision_from_file(lbte,
                             indices='all',
                             is_reducible_collision_matrix=False,
                             filename=None,
                             log_level=0):
    sigmas = lbte.get_sigmas()
    mesh = lbte.get_mesh_numbers()
    grid_points = lbte.get_grid_points()
    num_mesh_points = np.prod(mesh)

    if len(sigmas) > 1:
        gamma = []
        collision_matrix = []

    read_from = None

    if log_level:
        print("---------------------- Reading collision data from file "
              "----------------------")
        sys.stdout.flush()

    for j, sigma in enumerate(sigmas):
        collisions = read_collision_from_hdf5(mesh,
                                              indices=indices,
                                              sigma=sigma,
                                              filename=filename,
                                              verbose=(log_level > 0))
        if log_level:
            sys.stdout.flush()
        if collisions is None:
            gamma_of_gps = []
            collision_matrix_of_gps = []
            collision_gp = read_collision_from_hdf5(
                mesh,
                indices=indices,
                grid_point=grid_points[0],
                sigma=sigma,
                filename=filename,
                verbose=(log_level > 0))
            if collision_gp is None:
                if log_level:
                    print("Collision at grid point %d doesn't exist." %
                          grid_points[0])
                return False

            num_band = collision_gp[0].shape[2]
            num_temp = len(collision_gp[2])
            if is_reducible_collision_matrix:
                gamma_at_sigma = np.zeros(
                    (1, num_temp, num_mesh_points, num_band),
                    dtype='double', order='C')
                collision_matrix_at_sigma = np.zeros(
                    (1, num_temp,
                     num_mesh_points, num_band,
                     num_mesh_points, num_band),
                    dtype='double', order='C')
            else:
                gamma_at_sigma = np.zeros(
                    (1, num_temp, len(grid_points), num_band),
                    dtype='double', order='C')
                collision_matrix_at_sigma = np.zeros(
                    (1, num_temp,
                     len(grid_points), num_band, 3,
                     len(grid_points), num_band, 3),
                    dtype='double', order='C')
            temperatures = np.zeros(num_temp, dtype='double', order='C')
            for i, gp in enumerate(grid_points):
                collision_gp = read_collision_from_hdf5(
                    mesh,
                    indices=indices,
                    grid_point=gp,
                    sigma=sigma,
                    filename=filename,
                    verbose=(log_level > 0))

                if log_level:
                    sys.stdout.flush()

                if collision_gp is False:
                    if log_level:
                        print("Gamma at grid point %d doesn't exist." % gp)
                    return False

                (collision_matrix_at_gp,
                 gamma_at_gp,
                 temperatures_at_gp) = collision_gp
                if is_reducible_collision_matrix:
                    igp = gp
                else:
                    igp = i
                gamma_at_sigma[0, :, igp] = gamma_at_gp
                collision_matrix_at_sigma[0, :, igp] = collision_matrix_at_gp[0]
                temperatures[:] = temperatures_at_gp

            if len(sigmas) == 1:
                gamma = gamma_at_sigma
                collision_matrix = collision_matrix_at_sigma
            else:
                gamma.append(gamma_at_sigma[0])
                collision_matrix.append(collision_matrix_at_sigma[0])

            read_from = "grid_points"
        else:
            (collision_matrix_at_sigma,
             gamma_at_sigma,
             temperatures) = collisions

            if len(sigmas) == 1:
                collision_matrix = collision_matrix_at_sigma
                gamma = np.zeros((1,) + gamma_at_sigma.shape,
                                 dtype='double', order='C')
                gamma[0] = gamma_at_sigma
            else:                
                collision_matrix.append(collision_matrix_at_sigma)
                gamma.append(gamma_at_sigma)

            read_from = "full_matrix"

    if len(sigmas) > 1:
        temperatures = np.array(temperatures, dtype='double', order='C')
        gamma = np.array(gamma, dtype='double', order='C')
        collision_matrix = np.array(collision_matrix, dtype='double', order='C')

    lbte.set_gamma(gamma)
    lbte.set_collision_matrix(collision_matrix)
    # lbte.set_temperatures invokes allocation of arrays. So this must
    # be called after setting collision_matrix for saving memory
    # space.
    lbte.set_temperatures(temperatures)

    return read_from

class Conductivity_LBTE(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=None,
                 sigmas=None,
                 is_isotope=False,
                 mass_variances=None,
                 boundary_mfp=None, # in micrometre
                 is_reducible_collision_matrix=False,
                 is_kappa_star=True,
                 gv_delta_q=None, # finite difference for group veolocity
                 is_full_pp=False,
                 pinv_cutoff=1.0e-8,
                 pinv_solver=0,
                 log_level=0):
        self._pp = None
        self._temperatures = None
        self._sigmas = None
        self._is_kappa_star = None
        self._gv_delta_q = None
        self._is_full_pp = None
        self._log_level = None
        self._primitive = None
        self._dm = None
        self._frequency_factor_to_THz = None
        self._cutoff_frequency = None
        self._boundary_mfp = None

        self._symmetry = None
        self._point_operations = None
        self._rotations_cartesian = None

        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None
        self._ir_grid_points = None
        self._ir_grid_weights = None

        self._read_gamma = False
        self._read_gamma_iso = False

        self._frequencies = None
        self._cv = None
        self._gv = None
        self._gv_sum2 = None
        self._mfp = None
        self._gamma = None
        self._gamma_iso = None
        self._averaged_pp_interaction = None

        self._mesh = None
        self._conversion_factor = None

        self._is_isotope = None
        self._isotope = None
        self._mass_variances = None
        self._grid_point_count = None

        self._collision_eigenvalues = None

        Conductivity.__init__(self,
                              interaction,
                              symmetry,
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              is_isotope=is_isotope,
                              mass_variances=mass_variances,
                              boundary_mfp=boundary_mfp,
                              is_kappa_star=is_kappa_star,
                              gv_delta_q=gv_delta_q,
                              is_full_pp=is_full_pp,
                              log_level=log_level)

        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        if not self._is_kappa_star:
            self._is_reducible_collision_matrix = True
        self._collision_matrix = None
        self._pinv_cutoff = pinv_cutoff
        self._pinv_solver = pinv_solver

        if self._temperatures is not None:
            self._allocate_values()

    def set_kappa_at_sigmas(self):
        if len(self._grid_points) != len(self._ir_grid_points):
            print("Collision matrix is not well created.")
            import sys
            sys.exit(1)
        else:
            self._set_kappa_at_sigmas()

    def set_collision_matrix(self, collision_matrix):
        self._collision_matrix = collision_matrix

    def get_collision_matrix(self):
        return self._collision_matrix

    def get_collision_eigenvalues(self):
        return self._collision_eigenvalues

    def get_mean_free_path(self):
        return self._mfp

    def get_frequencies_all(self):
        return self._frequencies[:np.prod(self._mesh)]

    def get_kappa_RTA(self):
        return self._kappa_RTA

    def get_mode_kappa_RTA(self):
        return self._mode_kappa_RTA

    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        gp = self._grid_points[i]

        if not self._read_gamma:
            self._collision.set_grid_point(gp)

            if self._log_level:
                print("Number of triplets: %d" %
                      len(self._pp.get_triplets_at_q()[0]))
                print("Calculating ph-ph interaction...")

            self._set_collision_matrix_at_sigmas(i)

        if self._is_reducible_collision_matrix:
            self._set_harmonic_properties(i, gp)
            if self._isotope is not None:
                self._gamma_iso[:, gp, :] = self._get_gamma_isotope_at_sigmas(i)
        else:
            self._set_harmonic_properties(i, i)
            if self._isotope is not None:
                self._gamma_iso[:, i, :] = self._get_gamma_isotope_at_sigmas(i)

        if self._log_level:
            self._show_log(i)

    def _allocate_values(self):
        num_band0 = len(self._pp.get_band_indices())
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        num_temp = len(self._temperatures)
        num_mesh_points = np.prod(self._mesh)

        if self._is_reducible_collision_matrix:
            num_grid_points = num_mesh_points
        else:
            num_grid_points = len(self._grid_points)

        self._kappa = np.zeros((len(self._sigmas), num_temp, 6),
                               dtype='double', order='C')
        self._kappa_RTA = np.zeros((len(self._sigmas), num_temp, 6),
                                   dtype='double', order='C')
        self._gv = np.zeros((num_grid_points, num_band0, 3),
                            dtype='double', order='C')
        self._gv_sum2 = np.zeros((num_grid_points, num_band0, 6),
                                 dtype='double', order='C')
        self._mfp = np.zeros((len(self._sigmas),
                              num_temp,
                              num_grid_points,
                              num_band0,
                              3), dtype='double', order='C')
        self._cv = np.zeros((num_temp, num_grid_points, num_band0),
                            dtype='double', order='C')
        if self._is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_grid_points, num_band0), dtype='double', order='C')

        if self._gamma is None:
            self._gamma = np.zeros((len(self._sigmas),
                                    num_temp,
                                    num_grid_points,
                                    num_band0), dtype='double', order='C')
        if self._isotope is not None:
            self._gamma_iso = np.zeros((len(self._sigmas),
                                        num_grid_points,
                                        num_band0), dtype='double', order='C')

        if self._is_reducible_collision_matrix:
            self._mode_kappa = np.zeros((len(self._sigmas),
                                         num_temp,
                                         num_mesh_points,
                                         num_band0,
                                         6), dtype='double', order='C')
            self._mode_kappa_RTA = np.zeros((len(self._sigmas),
                                             num_temp,
                                             num_mesh_points,
                                             num_band0,
                                             6), dtype='double', order='C')
            self._collision = CollisionMatrix(
                self._pp,
                is_reducible_collision_matrix=True)
            if self._collision_matrix is None:
                self._collision_matrix = np.empty(
                    (len(self._sigmas), num_temp,
                     num_mesh_points, num_band, num_mesh_points, num_band),
                    dtype='double', order='C')
                self._collision_matrix[:] = 0
            self._collision_eigenvalues = np.zeros(
                (len(self._sigmas), num_temp, num_mesh_points * num_band),
                dtype='double', order='C')
        else:
            self._mode_kappa = np.zeros((len(self._sigmas),
                                         num_temp,
                                         num_grid_points,
                                         num_band0,
                                         6), dtype='double')
            self._mode_kappa_RTA = np.zeros((len(self._sigmas),
                                             num_temp,
                                             num_grid_points,
                                             num_band0,
                                             6), dtype='double')
            self._rot_grid_points = np.zeros(
                (len(self._ir_grid_points), len(self._point_operations)),
                dtype='intc')
            self._rot_BZ_grid_points = np.zeros(
                (len(self._ir_grid_points), len(self._point_operations)),
                dtype='intc')
            for i, ir_gp in enumerate(self._ir_grid_points):
                self._rot_grid_points[i] = get_grid_points_by_rotations(
                    self._grid_address[ir_gp],
                    self._point_operations,
                    self._mesh)
                self._rot_BZ_grid_points[i] = get_BZ_grid_points_by_rotations(
                    self._grid_address[ir_gp],
                    self._point_operations,
                    self._mesh,
                    self._pp.get_bz_map())
            self._collision = CollisionMatrix(
                self._pp,
                point_operations=self._point_operations,
                ir_grid_points=self._ir_grid_points,
                rotated_grid_points=self._rot_BZ_grid_points)
            if self._collision_matrix is None:
                self._collision_matrix = np.empty(
                    (len(self._sigmas),
                     num_temp,
                     num_grid_points, num_band, 3,
                     num_ir_grid_points, num_band, 3),
                    dtype='double', order='C')
                self._collision_matrix[:] = 0
            self._collision_eigenvalues = np.zeros(
                (len(self._sigmas),
                 num_temp,
                 num_ir_grid_points * num_band * 3),
                dtype='double', order='C')

    def _set_collision_matrix_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating collision matrix with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)
            self._collision.set_sigma(sigma)
            self._collision.set_integration_weights()

            if self._is_full_pp and j != 0:
                pass
            else:
                self._collision.run_interaction(is_full_pp=self._is_full_pp)
            if self._is_full_pp and j == 0:
                self._averaged_pp_interaction[i] = (
                    self._pp.get_averaged_interaction())

            for k, t in enumerate(self._temperatures):
                self._collision.set_temperature(t)
                self._collision.run()
                if self._is_reducible_collision_matrix:
                    i_data = self._grid_points[i]
                else:
                    i_data = i
                self._gamma[j, k, i_data] = (
                    self._collision.get_imag_self_energy())
                self._collision_matrix[j, k, i_data] = (
                    self._collision.get_collision_matrix())

        self._collision.delete_integration_weights()
        self._pp.delete_interaction_strength()

    def _set_kappa_at_sigmas(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        if self._is_reducible_collision_matrix:
            if self._is_kappa_star:
                self._average_collision_matrix_by_degeneracy()
                self._expand_collisions()
            self._combine_reducible_collisions()
            weights = np.ones(np.prod(self._mesh), dtype='intc')
            self._symmetrize_collision_matrix()
            size = np.prod(self._mesh) * num_band
        else:
            self._combine_collisions()
            weights = self._get_weights()
            for i, w_i in enumerate(weights):
                for j, w_j in enumerate(weights):
                    self._collision_matrix[:, :, i, :, :, j, :, :] *= w_i * w_j
            self._average_collision_matrix_by_degeneracy()
            self._symmetrize_collision_matrix()
            size = len(self._ir_grid_points) * num_band * 3

        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "----------- Thermal conductivity (W/m-k) "
                if sigma:
                    text += "for sigma=%s -----------" % sigma
                else:
                    text += "with tetrahedron method -----------"
                print(text)
                sys.stdout.flush()

            for k, t in enumerate(self._temperatures):
                if t > 0:
                    self._set_kappa_RTA(j, k, weights)

                    self._set_inv_collision_matrix(j, k, size)
                    self._set_kappa(j, k, weights)
                    # print("spectra")
                    # print(self._get_spectra(j, k, weights))

                    if self._log_level:
                        print(("#%6s       " + " %-10s" * 6) %
                              ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy"))
                        print(("%7.1f " + " %10.3f" * 6) %
                              ((t,) + tuple(self._kappa[j, k])))
                        print((" %6s " + " %10.3f" * 6) %
                              (("(RTA)",) + tuple(self._kappa_RTA[j, k])))
                        print("-" * 76)
                        sys.stdout.flush()

                        sys.stdout.flush()

        if self._log_level:
            print('')

    def _combine_collisions(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        for j, k in list(np.ndindex((len(self._sigmas),
                                     len(self._temperatures)))):
            for i, ir_gp in enumerate(self._ir_grid_points):
                multi = ((self._rot_grid_points == ir_gp).sum() //
                         (self._rot_BZ_grid_points == ir_gp).sum())
                for r, r_BZ_gp in zip(self._rotations_cartesian,
                                      self._rot_BZ_grid_points[i]):
                    if ir_gp != r_BZ_gp:
                        continue

                    main_diagonal = self._get_main_diagonal(i, j, k)
                    main_diagonal *= multi
                    for l in range(num_band):
                        self._collision_matrix[
                            j, k, i, l, :, i, l, :] += main_diagonal[l] * r

    def _combine_reducible_collisions(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_mesh_points = np.prod(self._mesh)

        for j, k in list(
                np.ndindex((len(self._sigmas), len(self._temperatures)))):
            for i in range(num_mesh_points):
                main_diagonal = self._get_main_diagonal(i, j, k)
                for l in range(num_band):
                    self._collision_matrix[
                        j, k, i, l, i, l] += main_diagonal[l]

    def _expand_collisions(self):
        num_mesh_points = np.prod(self._mesh)
        num_rot = len(self._point_operations)
        num_band = self._primitive.get_number_of_atoms() * 3
        rot_grid_points = np.zeros(
            (num_rot, num_mesh_points), dtype='intc')

        for i in range(num_mesh_points):
            rot_grid_points[:, i] = get_grid_points_by_rotations(
                self._grid_address[i],
                self._point_operations,
                self._mesh)

        for i, ir_gp in enumerate(self._ir_grid_points):
            colmat_irgp = self._collision_matrix[:, :, ir_gp, :, :, :].copy()
            self._collision_matrix[:, :, ir_gp, :, :, :] = 0
            gv_irgp = self._gv[ir_gp].copy()
            self._gv[ir_gp] = 0
            cv_irgp = self._cv[:, ir_gp, :].copy()
            self._cv[:, ir_gp, :] = 0
            gamma_irgp = self._gamma[:, :, ir_gp, :].copy()
            self._gamma[:, :, ir_gp, :] = 0
            multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
            if self._gamma_iso is not None:
                gamma_iso_irgp = self._gamma_iso[:, ir_gp, :].copy()
                self._gamma_iso[:, ir_gp, :] = 0
            for j, r in enumerate(self._rotations_cartesian):
                gp_r = rot_grid_points[j, ir_gp]
                self._gamma[:, :, gp_r, :] += gamma_irgp / multi
                if self._gamma_iso is not None:
                    self._gamma_iso[:, gp_r, :] += gamma_iso_irgp / multi
                for k in range(num_mesh_points):
                    gp_c = rot_grid_points[j, k]
                    self._collision_matrix[:, :, gp_r, :, gp_c, :] += (
                        colmat_irgp[:, :, :, k, :] / multi)
                self._gv[gp_r] += np.dot(gv_irgp, r.T) / multi
                self._cv[:, gp_r, :] += cv_irgp / multi

    def _get_weights(self):
        """Weights used for collision matrix and |X> and |f>

           r_gps: grid points on k-star with duplicates
                  len(r_gps) == order of crystallographic point group
                  len(unique(r_gps)) == # of arms of the k-star
           self._rot_grid_points contains r_gps for ir-grid points

        """
        weights = []
        for r_gps in self._rot_grid_points:
            weights.append(np.sqrt(len(np.unique(r_gps))) / np.sqrt(len(r_gps)))
        return weights

    def _symmetrize_collision_matrix(self):
        if self._log_level:
            print("- Making collision matrix symmetric...")
            sys.stdout.flush()

        # C-API implementation causes segmentation fault in specific
        # cases. The reason is not clear, but it seems to happen when
        # using large memory space. Passing data to C-API is doubted.
        # 
        # import phono3py._phono3py as phono3c
        # print(self._collision_matrix.flags)
        # print(self._collision_matrix.shape)
        # sys.stdout.flush()
        # phono3c.symmetrize_collision_matrix(self._collision_matrix)

        if self._is_reducible_collision_matrix:
            size = np.prod(self._collision_matrix.shape[2:4])
        else:
            size = np.prod(self._collision_matrix.shape[2:5])
        for i in range(self._collision_matrix.shape[0]):
            for j in range(self._collision_matrix.shape[1]):
                col_mat = self._collision_matrix[i, j].reshape(size, size)
                col_mat += col_mat.T
                col_mat /= 2

    def _average_collision_matrix_by_degeneracy(self):
        # Average matrix elements belonging to degenerate bands
        if self._log_level:
            print("- Averaging collision matrix elements "
                  "by phonon degeneracy ...")
            sys.stdout.flush()

        col_mat = self._collision_matrix
        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)

                if self._is_reducible_collision_matrix:
                    sum_col = (col_mat[:, :, gp, bi_set, :, :].sum(axis=2) /
                               len(bi_set))
                    for j in bi_set:
                        col_mat[:, :, gp, j, :, :] = sum_col
                else:
                    sum_col = (
                        col_mat[:, :, i, bi_set, :, :, :, :].sum(axis=2) /
                        len(bi_set))
                    for j in bi_set:
                        col_mat[:, :, i, j, :, :, :, :] = sum_col

        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)
                if self._is_reducible_collision_matrix:
                    sum_col = (col_mat[:, :, :, :, gp, bi_set].sum(axis=4) /
                               len(bi_set))
                    for j in bi_set:
                        col_mat[:, :, :, :, gp, j] = sum_col
                else:
                    sum_col = (
                        col_mat[:, :, :, :, :, i, bi_set, :].sum(axis=5) /
                        len(bi_set))
                    for j in bi_set:
                        col_mat[:, :, :, :, :, i, j, :] = sum_col

    def _get_X(self, i_temp, weights, gv):
        num_band = self._primitive.get_number_of_atoms() * 3
        X = gv.copy()
        if self._is_reducible_collision_matrix:
            num_mesh_points = np.prod(self._mesh)
            freqs = self._frequencies[:num_mesh_points]
        else:
            freqs = self._frequencies[self._ir_grid_points]

        t = self._temperatures[i_temp]
        sinh = np.where(freqs > self._cutoff_frequency,
                        np.sinh(freqs * THzToEv / (2 * Kb * t)),
                        -1.0)
        inv_sinh = np.where(sinh > 0, 1.0 / sinh, 0)
        freqs_sinh = freqs * THzToEv * inv_sinh / (4 * Kb * t ** 2)

        for i, f in enumerate(freqs_sinh):
            X[i] *= weights[i]
            for j in range(num_band):
                X[i, j] *= f[j]

        if t > 0:
            return X.reshape(-1, 3)
        else:
            return np.zeros_like(X.reshape(-1, 3))

    def _get_Y(self, i_sigma, i_temp, X):
        num_band = self._primitive.get_number_of_atoms() * 3

        if self._is_reducible_collision_matrix:
            num_mesh_points = np.prod(self._mesh)
            size = num_mesh_points * num_band
        else:
            num_ir_grid_points = len(self._ir_grid_points)
            size = num_ir_grid_points * num_band * 3
        v = self._collision_matrix[i_sigma, i_temp].reshape(size, size)

        if self._pinv_solver % 10 in [0, 6]:
            if self._log_level:
                print("Calculating pseudo-inv of collision matrix by np.dot "
                      "(cutoff=%-.1e)" % self._pinv_cutoff)
                sys.stdout.flush()

            w = self._collision_eigenvalues[i_sigma, i_temp]
            e = np.zeros_like(w)
            pinv_method = self._pinv_solver // 10
            if pinv_method not in [0, 1]:
                pinv_method = 0
            for l, val in enumerate(w):
                if (val > self._pinv_cutoff and pinv_method == 1 or
                    abs(val) > self._pinv_cutoff and pinv_method == 0):
                    e[l] = 1 / val
            if self._is_reducible_collision_matrix:
                X1 = np.dot(v.T, X)
                for i in range(3):
                    X1[:, i] *= e
                Y = np.dot(v, X1)
            else:
                Y = np.dot(v, e * np.dot(v.T, X.ravel())).reshape(-1, 3)
        else:
            if self._is_reducible_collision_matrix:
                Y = np.dot(v, X)
            else:
                Y = np.dot(v, X.ravel()).reshape(-1, 3)

        return Y

    def _get_I(self, a, b, size):
        r_sum = np.zeros((3, 3), dtype='double')
        for r in self._rotations_cartesian:
            for i in range(3):
                for j in range(3):
                    r_sum[i, j] += r[a, i] * r[b, j]
        I = np.zeros((size * 3, size * 3), dtype='double')
        for i in range(size):
            I[(i * 3):((i + 1) * 3), (i * 3):((i + 1) * 3)] = r_sum

        return I

    def _set_inv_collision_matrix(self, i_sigma, i_temp, size):
        """Compute pinv of collision matrix
        Args:
           size: The following value is expected:
              ir-colmat:  num_ir_grid_points * num_band * 3
              red-colmat: num_mesh_points * num_band
        """
        solver = self._pinv_solver % 10
        pinv_method = self._pinv_solver // 10
        # Safest choice
        if solver == 9:
            print("*** Attention: This isn't the calculation, but the test. ***")
        elif solver == 0: # default solver
            try:
                import scipy.linalg
            except ImportError:
                solver = 1
            else:
                solver = 6
        elif solver not in range(1, 7):
            solver = 1
        if pinv_method not in [0, 1]:
            pinv_method = 0

        if solver == 1: # dsyev: safer and slower than dsyevd and smallest
                        #        memory usage
            if self._log_level:
                print("Pseudo-inversion (cutoff=%-.1e) by lapacke dsyev..." %
                      self._pinv_cutoff)
                sys.stdout.flush()
            import phono3py._phono3py as phono3c
            w = np.zeros(size, dtype='double')
            phono3c.inverse_collision_matrix(self._collision_matrix,
                                             w,
                                             i_sigma,
                                             i_temp,
                                             self._pinv_cutoff,
                                             solver - 1,
                                             pinv_method)
        elif solver == 2: # dsyevd: faster than dsyev and lagest memory usage
            if self._log_level:
                print("Pseudo-inversion (cutoff=%-.1e) by lapacke dsyevd..." %
                      self._pinv_cutoff)
                sys.stdout.flush()
            import phono3py._phono3py as phono3c
            w = np.zeros(size, dtype='double')
            phono3c.inverse_collision_matrix(self._collision_matrix,
                                             w,
                                             i_sigma,
                                             i_temp,
                                             self._pinv_cutoff,
                                             solver - 1,
                                             pinv_method) # 1 or 11
        elif solver == 3: # np.linalg.eigh depends on dsyevd.
                          # Dangeraous to use this because
                          # the result becomes different.
            if self._log_level:
                print("Diagonalizing by np.linalg.eigh...")
                sys.stdout.flush()
            col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
                size, size)
            w, col_mat[:] = np.linalg.eigh(col_mat)

            if self._log_level:
                print("Calculating pseudo-inv of collision matrix "
                      "(cutoff=%-.1e)" % self._pinv_cutoff)
                sys.stdout.flush()
            import phono3py._phono3py as phono3c
            phono3c.pinv_from_eigensolution(self._collision_matrix,
                                            np.array(w, dtype='double'),
                                            i_sigma,
                                            i_temp,
                                            self._pinv_cutoff,
                                            pinv_method)
        elif solver == 4: # scipy.linalg.lapack.dsyev
            if self._log_level:
                print("Diagonalizing by scipy.linalg.lapack.dsyev...")
                sys.stdout.flush()
            import scipy.linalg
            col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
                size, size)
            w, col_mat[:], info = scipy.linalg.lapack.dsyev(col_mat)

            if self._log_level:
                print("Calculating pseudo-inv of collision matrix "
                      "(cutoff=%-.1e)" % self._pinv_cutoff)
                sys.stdout.flush()
            import phono3py._phono3py as phono3c
            phono3c.pinv_from_eigensolution(self._collision_matrix,
                                            np.array(w, dtype='double'),
                                            i_sigma,
                                            i_temp,
                                            self._pinv_cutoff,
                                            pinv_method)
        elif solver == 5: # scipy.linalg.lapack.dsyevd
                          # This is more stable than numpy.linalg.eigh.
            if self._log_level:
                print("Diagonalizing by scipy.linalg.lapack.dsyevd...")
                sys.stdout.flush()
            import scipy.linalg
            col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
                size, size)
            w, col_mat[:], info = scipy.linalg.lapack.dsyevd(col_mat)
            if self._log_level:
                print("Calculating pseudo-inv of collision matrix "
                      "(cutoff=%-.1e)" % self._pinv_cutoff)
                sys.stdout.flush()
            import phono3py._phono3py as phono3c
            phono3c.pinv_from_eigensolution(self._collision_matrix,
                                            np.array(w, dtype='double'),
                                            i_sigma,
                                            i_temp,
                                            self._pinv_cutoff,
                                            pinv_method)
        elif solver == 6: # fully scipy & numpy
            if self._log_level:
                print("Diagonalizing by scipy.linalg.lapack.dsyev...")
                sys.stdout.flush()
            import scipy.linalg
            col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
                size, size)
            w, col_mat[:], info = scipy.linalg.lapack.dsyev(col_mat)

            # Pseudo inversion is done together with with multiplying
            # X in _set_kappa to save memory space.

        elif solver == 9: # Test
            w = np.ones(size, dtype='double')
            if self._log_level:
                print("Calculating pseudo-inv of collision matrix "
                      "(cutoff=%-.1e)" % self._pinv_cutoff)
                sys.stdout.flush()
            import phono3py._phono3py as phono3c
            phono3c.pinv_from_eigensolution(self._collision_matrix,
                                            np.array(w, dtype='double'),
                                            i_sigma,
                                            i_temp,
                                            self._pinv_cutoff,
                                            pinv_method)

        self._collision_eigenvalues[i_sigma, i_temp] = w

    def _set_kappa(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            X = self._get_X(i_temp, weights, self._gv)
            num_mesh_points = np.prod(self._mesh)
            Y = self._get_Y(i_sigma, i_temp, X)
            self._set_mean_free_path(i_sigma, i_temp, weights, Y)
            # Putting self._rotations_cartesian is to symmetrize kappa.
            # None can be put instead for watching pure information.
            self._set_mode_kappa(self._mode_kappa,
                                 X,
                                 Y,
                                 num_mesh_points,
                                 self._rotations_cartesian,
                                 i_sigma,
                                 i_temp)
            self._mode_kappa[i_sigma, i_temp] /= len(self._rotations_cartesian)
            self._kappa[i_sigma, i_temp] = (
                self._mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0))
        else:
            X = self._get_X(i_temp, weights, self._gv)
            num_ir_grid_points = len(self._ir_grid_points)
            Y = self._get_Y(i_sigma, i_temp, X)
            self._set_mean_free_path(i_sigma, i_temp, weights, Y)
            self._set_mode_kappa(self._mode_kappa,
                                 X,
                                 Y,
                                 num_ir_grid_points,
                                 self._rotations_cartesian,
                                 i_sigma,
                                 i_temp)
            # self._set_mode_kappa_from_mfp(weights,
            #                               num_ir_grid_points,
            #                               self._rotations_cartesian,
            #                               i_sigma,
            #                               i_temp)

            self._kappa[i_sigma, i_temp] = (
                self._mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0))

    def _set_kappa_RTA(self, i_sigma, i_temp, weights):
        num_band = self._primitive.get_number_of_atoms() * 3
        X = self._get_X(i_temp, weights, self._gv)
        Y = np.zeros_like(X)

        if self._is_reducible_collision_matrix:
            # This RTA is not equivalent to conductivity_RTA.
            # The lifetime is defined from the diagonal part of
            # collision matrix.
            num_mesh_points = np.prod(self._mesh)
            size = num_mesh_points * num_band
            v_diag = np.diagonal(
                self._collision_matrix[i_sigma, i_temp].reshape(size, size))
                     
            for gp in range(num_mesh_points):
                frequencies = self._frequencies[gp]
                for j, f in enumerate(frequencies):
                    if f > self._cutoff_frequency:
                        i_mode = gp * num_band + j
                        Y[i_mode, :] = X[i_mode, :] / v_diag[i_mode]
            # Putting self._rotations_cartesian is to symmetrize kappa.
            # None can be put instead for watching pure information.
            self._set_mode_kappa(self._mode_kappa_RTA,
                                 X,
                                 Y,
                                 num_mesh_points,
                                 self._rotations_cartesian,
                                 i_sigma,
                                 i_temp)
            g = len(self._rotations_cartesian)
            self._mode_kappa_RTA[i_sigma, i_temp] /= g
            self._kappa_RTA[i_sigma, i_temp] = (
                self._mode_kappa_RTA[i_sigma, i_temp].sum(axis=0).sum(axis=0))
        else:
            # This RTA is supposed to be the same as conductivity_RTA.
            num_ir_grid_points = len(self._ir_grid_points)
            size = num_ir_grid_points * num_band * 3
            for i, gp in enumerate(self._ir_grid_points):
                g = self._get_main_diagonal(i, i_sigma, i_temp)
                frequencies = self._frequencies[gp]
                for j, f in enumerate(frequencies):
                    if f > self._cutoff_frequency:
                        i_mode = i * num_band + j
                        old_settings = np.seterr(all='raise')
                        try:
                            Y[i_mode, :] = X[i_mode, :] / g[j]
                        except:
                            print("=" * 26 + " Warning " + "=" * 26)
                            print(" Unexpected physical condition of ph-ph "
                                  "interaction calculation was found.")
                            print(" g[j]=%f at gp=%d, band=%d, freq=%f" %
                                  (g[j], gp, j + 1, f))
                            print("=" * 61)
                        np.seterr(**old_settings) 

            self._set_mode_kappa(self._mode_kappa_RTA,
                                 X,
                                 Y,
                                 num_ir_grid_points,
                                 self._rotations_cartesian,
                                 i_sigma,
                                 i_temp)
            self._kappa_RTA[i_sigma, i_temp] = (
                self._mode_kappa_RTA[i_sigma, i_temp].sum(axis=0).sum(axis=0))

    def _set_mode_kappa(self,
                        mode_kappa,
                        X,
                        Y,
                        num_grid_points,
                        rotations_cartesian,
                        i_sigma,
                        i_temp):
        num_band = self._primitive.get_number_of_atoms() * 3
        for i, (v_gp, f_gp) in enumerate(zip(X.reshape(num_grid_points,
                                                       num_band, 3),
                                             Y.reshape(num_grid_points,
                                                       num_band, 3))):
            for j, (v, f) in enumerate(zip(v_gp, f_gp)):
                # Do not consider three lowest modes at Gamma-point
                # It is assumed that there are no imaginary modes.
                if (self._grid_address[i] == 0).all() and j < 3:
                    continue

                if rotations_cartesian is None:
                    sum_k = np.outer(v, f)
                else:
                    sum_k = np.zeros((3, 3), dtype='double')
                    for r in rotations_cartesian:
                        sum_k += np.outer(np.dot(r, v), np.dot(r, f))
                sum_k = sum_k + sum_k.T
                for k, vxf in enumerate(
                        ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
                    mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]

        t = self._temperatures[i_temp]
        # Collision matrix is half of that defined in Chaput's paper.
        # Therefore here 2 is not necessary multiplied.
        # sum_k = sum_k + sum_k.T is equivalent to I(a,b) + I(b,a).
        mode_kappa[i_sigma, i_temp] *= (
            self._conversion_factor * Kb * t ** 2 / np.prod(self._mesh))

    def _set_mode_kappa_from_mfp(self,
                                 weights,
                                 num_grid_points,
                                 rotations_cartesian,
                                 i_sigma,
                                 i_temp):
        for i, (v_gp, mfp_gp, cv_gp) in enumerate(
                zip(self._gv, self._mfp[i_sigma, i_temp], self._cv[i_temp])):
            for j, (v, mfp, cv) in enumerate(zip(v_gp, mfp_gp, cv_gp)):
                sum_k = np.zeros((3, 3), dtype='double')
                for r in rotations_cartesian:
                    sum_k += np.outer(np.dot(r, v), np.dot(r, mfp))
                sum_k = (sum_k + sum_k.T) / 2 * cv * weights[i] ** 2 * 2 * np.pi
                for k, vxf in enumerate(
                        ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
                    self._mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]
        self._mode_kappa *= - self._conversion_factor / np.prod(self._mesh)

    def _set_mean_free_path(self, i_sigma, i_temp, weights, Y):
        if self._is_reducible_collision_matrix:
            num_grid_points = np.prod(self._mesh)
        else:
            num_grid_points = len(self._ir_grid_points)
        t = self._temperatures[i_temp]
        num_band = self._primitive.get_number_of_atoms() * 3
        # Collision matrix is half of that defined in Chaput's paper.
        # Therefore Y is divided by 2.
        for i, f_gp in enumerate((Y / 2).reshape(num_grid_points, num_band, 3)):
            for j, f in enumerate(f_gp):
                cv = self._cv[i_temp, i, j]
                if cv < 1e-10:
                    continue
                self._mfp[i_sigma, i_temp, i, j] = (
                    - 2 * t * np.sqrt(Kb / cv) / weights[i] * f / (2 * np.pi))

    def _get_spectra(self, i_sigma, i_temp, weights):
        import scipy.linalg
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        inv_col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
            num_ir_grid_points * num_band * 3,
            num_ir_grid_points * num_band * 3)
        X = self._get_X(i_temp, weights, self._gv)
        num_band = self._primitive.get_number_of_atoms() * 3
        I = self._get_I(2, 2, num_ir_grid_points * num_band)
        inv_col_mat[:] = np.dot(inv_col_mat, I.T)
        w, inv_col_mat[:], info = scipy.linalg.lapack.dsyev(inv_col_mat)
        for i, x in enumerate(w):
            if i % 8 == 0:
                print("")
            sys.stdout.write("%f " % x)

    def _show_log(self, i):
        q = self._qpoints[i]
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp]
        if self._is_reducible_collision_matrix:
            gv = self._gv[gp]
        else:
            gv = self._gv[i]
        if self._is_full_pp:
            ave_pp = self._averaged_pp_interaction[i]
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"

        if self._gv_delta_q is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._gv_delta_q
        print(text)
        if self._is_full_pp:
            for f, v, pp in zip(frequencies, gv, ave_pp):
                print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e" %
                      (f, v[0], v[1], v[2], np.linalg.norm(v), pp))
        else:
            for f, v in zip(frequencies, gv):
                print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f" %
                      (f, v[0], v[1], v[2], np.linalg.norm(v)))

        sys.stdout.flush()

    def _py_symmetrize_collision_matrix(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(3):
                    for l in range(num_ir_grid_points):
                        for m in range(num_band):
                            for n in range(3):
                                self._py_set_symmetrized_element(
                                    i, j, k, l, m, n)

    def _py_set_symmetrized_element(self, i, j, k, l, m, n):
        sym_val = (self._collision_matrix[:, :, i, j, k, l, m, n] +
                   self._collision_matrix[:, :, l, m, n, i, j, k]) / 2
        self._collision_matrix[:, :, i, j, k, l, m, n] = sym_val
        self._collision_matrix[:, :, l, m, n, i, j, k] = sym_val

    def _py_symmetrize_collision_matrix_no_kappa_stars(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(num_ir_grid_points):
                    for l in range(num_band):
                        self._py_set_symmetrized_element_no_kappa_stars(
                            i, j, k, l)

    def _py_set_symmetrized_element_no_kappa_stars(self, i, j, k, l):
        sym_val = (self._collision_matrix[:, :, i, j, k, l] +
                   self._collision_matrix[:, :, k, l, i, j]) / 2
        self._collision_matrix[:, :, i, j, k, l] = sym_val
        self._collision_matrix[:, :, k, l, i, j] = sym_val
