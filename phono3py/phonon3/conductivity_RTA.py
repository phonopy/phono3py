import sys
import numpy as np
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phono3py.file_IO import (write_kappa_to_hdf5, read_gamma_from_hdf5,
                              write_gamma_detail_to_hdf5, read_pp_from_hdf5)
from phono3py.phonon3.conductivity import (Conductivity, all_bands_exist,
                                           unit_to_WmK)
from phono3py.phonon3.conductivity import write_pp as _write_pp
from phono3py.phonon3.imag_self_energy import (ImagSelfEnergy,
                                               average_by_degeneracy)
from phono3py.phonon3.triplets import (get_grid_points_by_rotations,
                                       get_all_triplets)


def get_thermal_conductivity_RTA(
        interaction,
        symmetry,
        temperatures=np.arange(0, 1001, 10, dtype='double'),
        sigmas=None,
        sigma_cutoff=None,
        mass_variances=None,
        grid_points=None,
        is_isotope=False,
        boundary_mfp=None,  # in micrometre
        use_ave_pp=False,
        gamma_unit_conversion=None,
        mesh_divisors=None,
        coarse_mesh_shifts=None,
        is_kappa_star=True,
        gv_delta_q=1e-4,
        is_full_pp=False,
        write_gamma=False,
        read_gamma=False,
        is_N_U=False,
        write_kappa=False,
        write_pp=False,
        read_pp=False,
        write_gamma_detail=False,
        compression=None,
        input_filename=None,
        output_filename=None,
        log_level=0):

    if log_level:
        print("-------------------- Lattice thermal conducitivity (RTA) "
              "--------------------")
    br = Conductivity_RTA(
        interaction,
        symmetry,
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        gamma_unit_conversion=gamma_unit_conversion,
        mesh_divisors=mesh_divisors,
        coarse_mesh_shifts=coarse_mesh_shifts,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        store_pp=write_pp,
        pp_filename=input_filename,
        is_N_U=is_N_U,
        is_gamma_detail=write_gamma_detail,
        log_level=log_level)

    if read_gamma:
        if not _set_gamma_from_file(br, filename=input_filename):
            print("Reading collisions failed.")
            return False

    for i in br:
        if write_pp:
            _write_pp(br,
                      interaction,
                      i,
                      compression=compression,
                      filename=output_filename)
        if write_gamma:
            _write_gamma(br,
                         interaction,
                         i,
                         compression=compression,
                         filename=output_filename,
                         verbose=log_level)
        if write_gamma_detail:
            _write_gamma_detail(br,
                                interaction,
                                i,
                                compression=compression,
                                filename=output_filename,
                                verbose=log_level)

    if write_kappa:
        if grid_points is None and all_bands_exist(interaction):
            br.set_kappa_at_sigmas()
            _write_kappa(br,
                         interaction.get_primitive().get_volume(),
                         compression=compression,
                         filename=output_filename,
                         log_level=log_level)

    return br


def _write_gamma_detail(br, interaction, i, compression=None, filename=None,
                        verbose=True):
    gamma_detail = br.get_gamma_detail_at_q()
    temperatures = br.get_temperatures()
    mesh = br.get_mesh_numbers()
    grid_points = br.get_grid_points()
    gp = grid_points[i]
    sigmas = br.get_sigmas()
    sigma_cutoff = br.get_sigma_cutoff_width()
    triplets, weights, map_triplets, _ = interaction.get_triplets_at_q()
    grid_address = interaction.get_grid_address()
    bz_map = interaction.get_bz_map()
    if map_triplets is None:
        all_triplets = None
    else:
        all_triplets = get_all_triplets(gp,
                                        grid_address,
                                        bz_map,
                                        mesh)

    if all_bands_exist(interaction):
        for j, sigma in enumerate(sigmas):
            write_gamma_detail_to_hdf5(
                temperatures,
                mesh,
                gamma_detail=gamma_detail,
                grid_point=gp,
                triplet=triplets,
                weight=weights,
                triplet_map=map_triplets,
                triplet_all=all_triplets,
                sigma=sigma,
                sigma_cutoff=sigma_cutoff,
                compression=compression,
                filename=filename,
                verbose=verbose)
    else:
        for j, sigma in enumerate(sigmas):
            for k, bi in enumerate(interaction.get_band_indices()):
                write_gamma_detail_to_hdf5(
                    temperatures,
                    mesh,
                    gamma_detail=gamma_detail[:, :, k, :, :],
                    grid_point=gp,
                    triplet=triplets,
                    weight=weights,
                    band_index=bi,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    compression=compression,
                    filename=filename,
                    verbose=verbose)


def _write_gamma(br, interaction, i, compression=None, filename=None,
                 verbose=True):
    grid_points = br.get_grid_points()
    group_velocities = br.get_group_velocities()
    gv_by_gv = br.get_gv_by_gv()
    mode_heat_capacities = br.get_mode_heat_capacities()
    ave_pp = br.get_averaged_pp_interaction()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    temperatures = br.get_temperatures()
    gamma = br.get_gamma()
    gamma_isotope = br.get_gamma_isotope()
    sigmas = br.get_sigmas()
    sigma_cutoff = br.get_sigma_cutoff_width()
    volume = interaction.get_primitive().get_volume()
    gamma_N, gamma_U = br.get_gamma_N_U()

    gp = grid_points[i]
    if all_bands_exist(interaction):
        if ave_pp is None:
            ave_pp_i = None
        else:
            ave_pp_i = ave_pp[i]
        frequencies = interaction.get_phonons()[0][gp]
        for j, sigma in enumerate(sigmas):
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[j, i]
            else:
                gamma_isotope_at_sigma = None
            if gamma_N is None:
                gamma_N_at_sigma = None
            else:
                gamma_N_at_sigma = gamma_N[j, :, i]
            if gamma_U is None:
                gamma_U_at_sigma = None
            else:
                gamma_U_at_sigma = gamma_U[j, :, i]

            write_kappa_to_hdf5(temperatures,
                                mesh,
                                frequency=frequencies,
                                group_velocity=group_velocities[i],
                                gv_by_gv=gv_by_gv[i],
                                heat_capacity=mode_heat_capacities[:, i],
                                gamma=gamma[j, :, i],
                                gamma_isotope=gamma_isotope_at_sigma,
                                gamma_N=gamma_N_at_sigma,
                                gamma_U=gamma_U_at_sigma,
                                averaged_pp_interaction=ave_pp_i,
                                mesh_divisors=mesh_divisors,
                                grid_point=gp,
                                sigma=sigma,
                                sigma_cutoff=sigma_cutoff,
                                kappa_unit_conversion=unit_to_WmK / volume,
                                compression=compression,
                                filename=filename,
                                verbose=verbose)
    else:
        for j, sigma in enumerate(sigmas):
            for k, bi in enumerate(interaction.get_band_indices()):
                if ave_pp is None:
                    ave_pp_ik = None
                else:
                    ave_pp_ik = ave_pp[i, k]
                frequencies = interaction.get_phonons()[0][gp, bi]
                if gamma_isotope is not None:
                    gamma_isotope_at_sigma = gamma_isotope[j, i, k]
                else:
                    gamma_isotope_at_sigma = None
                if gamma_N is None:
                    gamma_N_at_sigma = None
                else:
                    gamma_N_at_sigma = gamma_N[j, :, i, k]
                if gamma_U is None:
                    gamma_U_at_sigma = None
                else:
                    gamma_U_at_sigma = gamma_U[j, :, i, k]
                write_kappa_to_hdf5(
                    temperatures,
                    mesh,
                    frequency=frequencies,
                    group_velocity=group_velocities[i, k],
                    gv_by_gv=gv_by_gv[i, k],
                    heat_capacity=mode_heat_capacities[:, i, k],
                    gamma=gamma[j, :, i, k],
                    gamma_isotope=gamma_isotope_at_sigma,
                    gamma_N=gamma_N_at_sigma,
                    gamma_U=gamma_U_at_sigma,
                    averaged_pp_interaction=ave_pp_ik,
                    mesh_divisors=mesh_divisors,
                    grid_point=gp,
                    band_index=bi,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    kappa_unit_conversion=unit_to_WmK / volume,
                    filename=filename,
                    verbose=verbose)


def _write_kappa(br, volume, compression=None, filename=None, log_level=0):
    temperatures = br.get_temperatures()
    sigmas = br.get_sigmas()
    sigma_cutoff = br.get_sigma_cutoff_width()
    gamma = br.get_gamma()
    gamma_isotope = br.get_gamma_isotope()
    gamma_N, gamma_U = br.get_gamma_N_U()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    frequencies = br.get_frequencies()
    gv = br.get_group_velocities()
    gv_by_gv = br.get_gv_by_gv()
    mode_cv = br.get_mode_heat_capacities()
    ave_pp = br.get_averaged_pp_interaction()
    qpoints = br.get_qpoints()
    weights = br.get_grid_weights()
    kappa = br.get_kappa()
    mode_kappa = br.get_mode_kappa()
    num_ignored_phonon_modes = br.get_number_of_ignored_phonon_modes()
    num_band = br.get_frequencies().shape[1]
    num_phonon_modes = br.get_number_of_sampling_grid_points() * num_band

    for i, sigma in enumerate(sigmas):
        kappa_at_sigma = kappa[i]
        if gamma_isotope is not None:
            gamma_isotope_at_sigma = gamma_isotope[i]
        else:
            gamma_isotope_at_sigma = None
        if gamma_N is None:
            gamma_N_at_sigma = None
        else:
            gamma_N_at_sigma = gamma_N[i]
        if gamma_U is None:
            gamma_U_at_sigma = None
        else:
            gamma_U_at_sigma = gamma_U[i]
        if log_level:
            text = "----------- Thermal conductivity (W/m-k) "
            if sigma:
                text += "for sigma=%s -----------" % sigma
            else:
                text += "with tetrahedron method -----------"
            print(text)
            if log_level > 1:
                print(("#%6s       " + " %-10s" * 6 + "#ipm") %
                      ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy"))
                for j, (t, k) in enumerate(zip(temperatures, kappa_at_sigma)):
                    print(("%7.1f" + " %10.3f" * 6 + " %d/%d") %
                          ((t,) + tuple(k) +
                           (num_ignored_phonon_modes[i, j], num_phonon_modes)))
            else:
                print(("#%6s       " + " %-10s" * 6) %
                      ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy"))
                for j, (t, k) in enumerate(zip(temperatures, kappa_at_sigma)):
                    print(("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print('')

        write_kappa_to_hdf5(temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            gv_by_gv=gv_by_gv,
                            heat_capacity=mode_cv,
                            kappa=kappa_at_sigma,
                            mode_kappa=mode_kappa[i],
                            gamma=gamma[i],
                            gamma_isotope=gamma_isotope_at_sigma,
                            gamma_N=gamma_N_at_sigma,
                            gamma_U=gamma_U_at_sigma,
                            averaged_pp_interaction=ave_pp,
                            qpoint=qpoints,
                            weight=weights,
                            mesh_divisors=mesh_divisors,
                            sigma=sigma,
                            sigma_cutoff=sigma_cutoff,
                            kappa_unit_conversion=unit_to_WmK / volume,
                            compression=compression,
                            filename=filename,
                            verbose=log_level)


def _set_gamma_from_file(br, filename=None, verbose=True):
    sigmas = br.get_sigmas()
    sigma_cutoff = br.get_sigma_cutoff_width()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    grid_points = br.get_grid_points()
    temperatures = br.get_temperatures()
    num_band = br.get_frequencies().shape[1]

    gamma = np.zeros((len(sigmas),
                      len(temperatures),
                      len(grid_points),
                      num_band), dtype='double')
    gamma_N = np.zeros_like(gamma)
    gamma_U = np.zeros_like(gamma)
    gamma_iso = np.zeros((len(sigmas),
                          len(grid_points),
                          num_band), dtype='double')
    ave_pp = np.zeros((len(grid_points), num_band), dtype='double')

    is_gamma_N_U_in = False
    is_ave_pp_in = False
    read_succeeded = True

    for j, sigma in enumerate(sigmas):
        data = read_gamma_from_hdf5(
            mesh,
            mesh_divisors=mesh_divisors,
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
            verbose=verbose)
        if data:
            gamma[j] = data['gamma']
            if 'gamma_isotope' in data:
                gamma_iso[j] = data['gamma_isotope']
            if 'gamma_N' in data:
                is_gamma_N_U_in = True
                gamma_N[j] = data['gamma_N']
                gamma_U[j] = data['gamma_U']
            if 'ave_pp' in data:
                is_ave_pp_in = True
                ave_pp[:] = data['ave_pp']
        else:
            for i, gp in enumerate(grid_points):
                data_gp = read_gamma_from_hdf5(
                    mesh,
                    mesh_divisors=mesh_divisors,
                    grid_point=gp,
                    sigma=sigma,
                    sigma_cutoff=sigma_cutoff,
                    filename=filename,
                    verbose=verbose)
                if data_gp:
                    gamma[j, :, i] = data_gp['gamma']
                    if 'gamma_iso' in data_gp:
                        gamma_iso[j, i] = data_gp['gamma_iso']
                    if 'gamma_N' in data_gp:
                        is_gamma_N_U_in = True
                        gamma_N[j, :, i] = data_gp['gamma_N']
                        gamma_U[j, :, i] = data_gp['gamma_U']
                    if 'ave_pp' in data_gp:
                        is_ave_pp_in = True
                        ave_pp[i] = data_gp['ave_pp']
                else:
                    for bi in range(num_band):
                        data_band = read_gamma_from_hdf5(
                            mesh,
                            mesh_divisors=mesh_divisors,
                            grid_point=gp,
                            band_index=bi,
                            sigma=sigma,
                            sigma_cutoff=sigma_cutoff,
                            filename=filename,
                            verbose=verbose)
                        if data_band:
                            gamma[j, :, i, bi] = data_band['gamma']
                            if 'gamma_iso' in data_band:
                                gamma_iso[j, i, bi] = data_band['gamma_iso']
                            if 'gamma_N' in data_band:
                                is_gamma_N_U_in = True
                                gamma_N[j, :, i, bi] = data_band['gamma_N']
                                gamma_U[j, :, i, bi] = data_band['gamma_U']
                            if 'ave_pp' in data_band:
                                is_ave_pp_in = True
                                ave_pp[i, bi] = data_band['ave_pp']
                        else:
                            read_succeeded = False

    if read_succeeded:
        br.set_gamma(gamma)
        if is_ave_pp_in:
            br.set_averaged_pp_interaction(ave_pp)
        if is_gamma_N_U_in:
            br.set_gamma_N_U(gamma_N, gamma_U)
        return True
    else:
        return False


class Conductivity_RTA(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=None,
                 sigmas=None,
                 sigma_cutoff=None,
                 is_isotope=False,
                 mass_variances=None,
                 boundary_mfp=None,  # in micrometre
                 use_ave_pp=False,
                 gamma_unit_conversion=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 is_kappa_star=True,
                 gv_delta_q=None,
                 is_full_pp=False,
                 read_pp=False,
                 store_pp=False,
                 pp_filename=None,
                 is_N_U=False,
                 is_gamma_detail=False,
                 log_level=0):
        self._pp = None
        self._temperatures = None
        self._sigmas = None
        self._sigma_cutoff = None
        self._is_kappa_star = None
        self._gv_delta_q = None
        self._is_full_pp = None
        self._is_N_U = is_N_U
        self._is_gamma_detail = is_gamma_detail
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

        self._read_gamma = False
        self._read_gamma_iso = False

        self._frequencies = None
        self._cv = None
        self._gv = None
        self._gv_sum2 = None
        self._gamma = None
        self._gamma_iso = None
        self._gamma_N = None
        self._gamma_U = None
        self._gamma_detail_at_q = None
        self._gamma_unit_conversion = gamma_unit_conversion
        self._use_ave_pp = use_ave_pp
        self._use_const_ave_pp = None
        self._averaged_pp_interaction = None
        self._num_ignored_phonon_modes = None
        self._num_sampling_grid_points = None

        self._mesh = None
        self._conversion_factor = None

        self._is_isotope = None
        self._isotope = None
        self._mass_variances = None
        self._grid_point_count = None

        Conductivity.__init__(self,
                              interaction,
                              symmetry,
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              sigma_cutoff=sigma_cutoff,
                              is_isotope=is_isotope,
                              mass_variances=mass_variances,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              boundary_mfp=boundary_mfp,
                              is_kappa_star=is_kappa_star,
                              gv_delta_q=gv_delta_q,
                              is_full_pp=is_full_pp,
                              log_level=log_level)

        self._use_const_ave_pp = self._pp.get_constant_averaged_interaction()
        self._read_pp = read_pp
        self._store_pp = store_pp
        self._pp_filename = pp_filename

        if self._temperatures is not None:
            self._allocate_values()

    def set_kappa_at_sigmas(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        for i, grid_point in enumerate(self._grid_points):
            cv = self._cv[:, i, :]
            gp = self._grid_points[i]
            frequencies = self._frequencies[gp]

            # Kappa
            for j in range(len(self._sigmas)):
                for k in range(len(self._temperatures)):
                    g_sum = self._get_main_diagonal(i, j, k)
                    for l in range(num_band):
                        if frequencies[l] < self._cutoff_frequency:
                            self._num_ignored_phonon_modes[j, k] += 1
                            continue

                        old_settings = np.seterr(all='raise')
                        try:
                            self._mode_kappa[j, k, i, l] = (
                                self._gv_sum2[i, l] * cv[k, l] /
                                (g_sum[l] * 2) * self._conversion_factor)
                        except FloatingPointError:
                            # supposed that g is almost 0 and |gv|=0
                            pass
                        except:
                            print("=" * 26 + " Warning " + "=" * 26)
                            print(" Unexpected physical condition of ph-ph "
                                  "interaction calculation was found.")
                            print(" g=%f at gp=%d, band=%d, freq=%f" %
                                  (g_sum[l], gp, l + 1, frequencies[l]))
                            print("=" * 61)
                        np.seterr(**old_settings)

        N = self._num_sampling_grid_points
        self._kappa = self._mode_kappa.sum(axis=2).sum(axis=2) / N

    def get_gamma_N_U(self):
        return (self._gamma_N, self._gamma_U)

    def set_gamma_N_U(self, gamma_N, gamma_U):
        self._gamma_N = gamma_N
        self._gamma_U = gamma_U

    def get_gamma_detail_at_q(self):
        return self._gamma_detail_at_q

    def get_number_of_ignored_phonon_modes(self):
        return self._num_ignored_phonon_modes

    def get_number_of_sampling_grid_points(self):
        return self._num_sampling_grid_points

    def set_averaged_pp_interaction(self, ave_pp):
        self._averaged_pp_interaction = ave_pp

    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        grid_point = self._grid_points[i]
        self._set_harmonic_properties(i, i)

        if self._read_gamma:
            if self._use_ave_pp:
                self._collision.set_grid_point(grid_point)
                self._set_gamma_at_sigmas(i)
        else:
            self._collision.set_grid_point(
                grid_point,
                stores_triplets_map=(self._is_full_pp or
                                     self._is_gamma_detail))
            num_triplets = len(self._pp.get_triplets_at_q()[0])
            if self._log_level:
                print("Number of triplets: %d" % num_triplets)

            if (self._is_full_pp or
                self._read_pp or
                self._store_pp or
                self._use_ave_pp or
                self._use_const_ave_pp or
                self._is_gamma_detail):
                self._set_gamma_at_sigmas(i)
            else:  # can save memory space
                self._set_gamma_at_sigmas_lowmem(i)

        if self._isotope is not None and not self._read_gamma_iso:
            gamma_iso = self._get_gamma_isotope_at_sigmas(i)
            self._gamma_iso[:, i, :] = gamma_iso[:, self._pp.get_band_indices()]

        if self._log_level:
            self._show_log(self._qpoints[i], i)

    def _allocate_values(self):
        num_band0 = len(self._pp.get_band_indices())
        num_grid_points = len(self._grid_points)
        num_temp = len(self._temperatures)
        self._kappa = np.zeros((len(self._sigmas), num_temp, 6),
                               order='C', dtype='double')
        self._mode_kappa = np.zeros((len(self._sigmas),
                                     num_temp,
                                     num_grid_points,
                                     num_band0,
                                     6),
                                    order='C', dtype='double')
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    num_temp,
                                    num_grid_points,
                                    num_band0),
                                   order='C', dtype='double')
            if self._is_gamma_detail or self._is_N_U:
                self._gamma_N = np.zeros_like(self._gamma)
                self._gamma_U = np.zeros_like(self._gamma)
        self._gv = np.zeros((num_grid_points, num_band0, 3),
                            order='C', dtype='double')
        self._gv_sum2 = np.zeros((num_grid_points, num_band0, 6),
                                 order='C', dtype='double')
        self._cv = np.zeros((num_temp, num_grid_points, num_band0),
                            order='C', dtype='double')
        if self._isotope is not None:
            self._gamma_iso = np.zeros(
                (len(self._sigmas), num_grid_points, num_band0),
                order='C', dtype='double')
        if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_grid_points, num_band0),
                order='C', dtype='double')
        self._num_ignored_phonon_modes = np.zeros(
            (len(self._sigmas), num_temp), order='C', dtype='intc')
        self._collision = ImagSelfEnergy(
            self._pp,
            with_detail=(self._is_gamma_detail or self._is_N_U),
            unit_conversion=self._gamma_unit_conversion)

    def _set_gamma_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
            self._collision.set_integration_weights()

            if self._log_level:
                text = "Collisions will be calculated with "
                if sigma is None:
                    text += "tetrahedron method."
                else:
                    text += "sigma=%s" % sigma
                    if self._sigma_cutoff is None:
                        text += "."
                    else:
                        text += "(%4.2f SD)." % self._sigma_cutoff
                print(text)

            if self._read_pp:
                pp, _g_zero = read_pp_from_hdf5(
                    self._mesh,
                    grid_point=self._grid_points[i],
                    sigma=sigma,
                    sigma_cutoff=self._sigma_cutoff,
                    filename=self._pp_filename,
                    verbose=(self._log_level > 0))
                _, g_zero = self._collision.get_integration_weights()
                if self._log_level:
                    if len(self._sigmas) > 1:
                        print("Multiple sigmas or mixing smearing and "
                              "tetrahedron method is not supported.")
                if _g_zero is not None and (_g_zero != g_zero).any():
                    raise ValueError("Inconsistency found in g_zero.")
                self._collision.set_interaction_strength(pp)
            elif self._use_ave_pp:
                self._collision.set_averaged_pp_interaction(
                    self._averaged_pp_interaction[i])
            elif self._use_const_ave_pp:
                if self._log_level:
                    print("Constant ph-ph interaction of %6.3e is used." %
                          self._pp.get_constant_averaged_interaction())
                self._collision.run_interaction()
                self._averaged_pp_interaction[i] = (
                    self._pp.get_averaged_interaction())
            elif j != 0 and (self._is_full_pp or self._sigma_cutoff is None):
                if self._log_level:
                    print("Existing ph-ph interaction is used.")
            else:
                if self._log_level:
                    print("Calculating ph-ph interaction...")
                self._collision.run_interaction(is_full_pp=self._is_full_pp)
                if self._is_full_pp:
                    self._averaged_pp_interaction[i] = (
                        self._pp.get_averaged_interaction())

            # Number of triplets depends on q-point.
            # So this is allocated each time.
            if self._is_gamma_detail:
                num_temp = len(self._temperatures)
                self._gamma_detail_at_q = np.empty(
                    ((num_temp,) + self._pp.get_interaction_strength().shape),
                    dtype='double', order='C')
                self._gamma_detail_at_q[:] = 0

            if self._log_level:
                print("Calculating collisions at temperatures...")
            for k, t in enumerate(self._temperatures):
                self._collision.set_temperature(t)
                self._collision.run()
                self._gamma[j, k, i] = self._collision.get_imag_self_energy()
                if self._is_N_U or self._is_gamma_detail:
                    g_N, g_U = self._collision.get_imag_self_energy_N_and_U()
                    self._gamma_N[j, k, i] = g_N
                    self._gamma_U[j, k, i] = g_U
                if self._is_gamma_detail:
                    self._gamma_detail_at_q[k] = (
                        self._collision.get_detailed_imag_self_energy())

    def _set_gamma_at_sigmas_lowmem(self, i):
        band_indices = self._pp.get_band_indices()
        (svecs,
         multiplicity,
         p2s,
         s2p,
         masses) = self._pp.get_primitive_and_supercell_correspondence()
        fc3 = self._pp.get_fc3()
        triplets_at_q, weights_at_q, _, _ = self._pp.get_triplets_at_q()
        bz_map = self._pp.get_bz_map()
        symmetrize_fc3_q = 0

        if None in self._sigmas:
            reclat = np.linalg.inv(self._pp.get_primitive().get_cell())
            thm = TetrahedronMethod(reclat, mesh=self._mesh)

        # It is assumed that self._sigmas = [None].
        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma)
            if self._is_N_U:
                collisions = np.zeros((2, len(self._temperatures),
                                       len(band_indices)),
                                      dtype='double', order='C')
            else:
                collisions = np.zeros((len(self._temperatures),
                                       len(band_indices)),
                                      dtype='double', order='C')
            import phono3py._phono3py as phono3c
            if sigma is None:
                phono3c.pp_collision(collisions,
                                     thm.get_tetrahedra(),
                                     self._frequencies,
                                     self._eigenvectors,
                                     triplets_at_q,
                                     weights_at_q,
                                     self._grid_address,
                                     bz_map,
                                     self._mesh,
                                     fc3,
                                     svecs,
                                     multiplicity,
                                     masses,
                                     p2s,
                                     s2p,
                                     band_indices,
                                     self._temperatures,
                                     self._is_N_U * 1,
                                     symmetrize_fc3_q,
                                     self._cutoff_frequency)
            else:
                if self._sigma_cutoff is None:
                    sigma_cutoff = -1
                else:
                    sigma_cutoff = float(self._sigma_cutoff)
                phono3c.pp_collision_with_sigma(collisions,
                                                sigma,
                                                sigma_cutoff,
                                                self._frequencies,
                                                self._eigenvectors,
                                                triplets_at_q,
                                                weights_at_q,
                                                self._grid_address,
                                                self._mesh,
                                                fc3,
                                                svecs,
                                                multiplicity,
                                                masses,
                                                p2s,
                                                s2p,
                                                band_indices,
                                                self._temperatures,
                                                self._is_N_U * 1,
                                                symmetrize_fc3_q,
                                                self._cutoff_frequency)
            col_unit_conv = self._collision.get_unit_conversion_factor()
            pp_unit_conv = self._pp.get_unit_conversion_factor()
            if self._is_N_U:
                col = collisions.sum(axis=0)
                col_N = collisions[0]
                col_U = collisions[1]
            else:
                col = collisions
            for k in range(len(self._temperatures)):
                self._gamma[j, k, i, :] = average_by_degeneracy(
                    col[k] * col_unit_conv * pp_unit_conv,
                    band_indices,
                    self._frequencies[self._grid_points[i]])
                if self._is_N_U:
                    self._gamma_N[j, k, i, :] = average_by_degeneracy(
                        col_N[k] * col_unit_conv * pp_unit_conv,
                        band_indices,
                        self._frequencies[self._grid_points[i]])
                    self._gamma_U[j, k, i, :] = average_by_degeneracy(
                        col_U[k] * col_unit_conv * pp_unit_conv,
                        band_indices,
                        self._frequencies[self._grid_points[i]])

    def _show_log(self, q, i):
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp][self._pp.get_band_indices()]
        gv = self._gv[i]
        if self._averaged_pp_interaction is not None:
            ave_pp = self._averaged_pp_interaction[i]
        else:
            ave_pp = None
        self._show_log_value_names()

        if self._log_level > 2:
            self._show_log_values_on_kstar(frequencies, gv, ave_pp, gp, q)
        else:
            self._show_log_values(frequencies, gv, ave_pp)

        sys.stdout.flush()

    def _show_log_values(self, frequencies, gv, ave_pp):
        if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
            for f, v, pp in zip(frequencies, gv, ave_pp):
                print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e" %
                      (f, v[0], v[1], v[2], np.linalg.norm(v), pp))
        else:
            for f, v in zip(frequencies, gv):
                print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f" %
                      (f, v[0], v[1], v[2], np.linalg.norm(v)))

    def _show_log_values_on_kstar(self, frequencies, gv, ave_pp, gp, q):
        rotation_map = get_grid_points_by_rotations(
            self._grid_address[gp],
            self._point_operations,
            self._mesh)
        for i, j in enumerate(np.unique(rotation_map)):
            for k, (rot, rot_c) in enumerate(zip(
                    self._point_operations, self._rotations_cartesian)):
                if rotation_map[k] != j:
                    continue

                print(" k*%-2d (%5.2f %5.2f %5.2f)" %
                      ((i + 1,) + tuple(np.dot(rot, q))))
                if (self._is_full_pp or
                    self._use_ave_pp or
                    self._use_const_ave_pp):
                    for f, v, pp in zip(frequencies,
                                        np.dot(rot_c, gv.T).T,
                                        ave_pp):
                        print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e" %
                              (f, v[0], v[1], v[2], np.linalg.norm(v), pp))
                else:
                    for f, v in zip(frequencies, np.dot(rot_c, gv.T).T):
                        print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f" %
                              (f, v[0], v[1], v[2], np.linalg.norm(v)))
        print('')

    def _show_log_value_names(self):
        if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"
        if self._gv_delta_q is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._gv_delta_q
        print(text)
