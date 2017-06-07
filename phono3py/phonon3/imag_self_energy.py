import sys
import numpy as np
from phonopy.units import VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from phonopy.phonon.degeneracy import degenerate_sets
from phono3py.phonon3.triplets import (get_triplets_integration_weights,
                                       gaussian, occupation)
from phono3py.file_IO import (write_gamma_detail_to_hdf5,
                              write_linewidth_at_grid_point,
                              write_imag_self_energy_at_grid_point)

def get_imag_self_energy(interaction,
                         grid_points,
                         sigmas,
                         frequency_step=None,
                         num_frequency_points=None,
                         temperatures=None,
                         scattering_event_class=None, # class 1 or 2
                         write_detail=False,
                         log_level=0):
    """Imaginary part of self energy at frequency points

    Band indices to be calculated at are kept in Interaction instance.

    Args:
        interaction: Ph-ph interaction
        grid_points: Grid-point indices to be caclculated on
        sigmas:
            A set of sigmas. simga=None means to use tetrahedron method,
            otherwise smearing method with real positive value of sigma.
        frequency_step: Pitch of frequency to be sampled.
        num_frequency_points: Number of sampling sampling points to be used
            instead of frequency_step.
        temperatures: Temperatures to be calculated at.
        scattering_event_class:
            Extract scattering event class 1 or 2.
        log_level: Log level. 0 or non 0 in this method.

    Returns:
        Tuple: (Imaginary part of self energy, sampling frequency points)

    """
    if temperatures is None:
        temperatures = [0.0, 300.0]

    if temperatures is None:
        print("Temperatures have to be set.")
        return False

    mesh = interaction.get_mesh_numbers()
    ise = ImagSelfEnergy(interaction, with_detail=write_detail)
    imag_self_energy = []
    frequency_points = []
    for i, gp in enumerate(grid_points):
        ise.set_grid_point(gp)
        if log_level:
            weights = interaction.get_triplets_at_q()[1]
            print("------------------- Imaginary part of self energy (%d/%d) "
                  "-------------------" % (i + 1, len(grid_points)))
            print("Grid point: %d" % gp)
            print("Number of ir-triplets: "
                  "%d / %d" % (len(weights), weights.sum()))
        ise.run_interaction()
        frequencies = interaction.get_phonons()[0]
        max_phonon_freq = np.amax(frequencies)

        if log_level:
            adrs = interaction.get_grid_address()[gp]
            q = adrs.astype('double') / mesh
            print("q-point: %s" % q)
            print("Phonon frequency:")
            text = "[ "
            for i, freq in enumerate(frequencies[gp]):
                if i % 6 == 0 and i != 0:
                    text += "\n"
                text += "%8.4f " % freq
            text += "]"
            print(text)
            sys.stdout.flush()

        gamma_sigmas = []
        fp_sigmas = []
        if write_detail:
            triplets, weights, _, _ = interaction.get_triplets_at_q()

        for j, sigma in enumerate(sigmas):
            if log_level:
                if sigma:
                    print("Sigma: %s" % sigma)
                else:
                    print("Tetrahedron method")

            ise.set_sigma(sigma)
            if sigma:
                fmax = max_phonon_freq * 2 + sigma * 4
            else:
                fmax = max_phonon_freq * 2
            fmax *= 1.005
            fmin = 0
            frequency_points_at_sigma = get_frequency_points(
                fmin,
                fmax,
                frequency_step=frequency_step,
                num_frequency_points=num_frequency_points)
            fp_sigmas.append(frequency_points_at_sigma)
            gamma = np.zeros(
                (len(temperatures), len(frequency_points_at_sigma),
                 len(interaction.get_band_indices())), dtype='double')

            if write_detail:
                num_band0 = len(interaction.get_band_indices())
                num_band = frequencies.shape[1]
                detailed_gamma = np.zeros(
                    (len(temperatures), len(frequency_points_at_sigma),
                     len(weights), num_band0, num_band, num_band),
                    dtype='double')

            for k, freq_point in enumerate(frequency_points_at_sigma):
                ise.set_frequency_points([freq_point])
                ise.set_integration_weights(
                    scattering_event_class=scattering_event_class)

                for l, t in enumerate(temperatures):
                    ise.set_temperature(t)
                    ise.run()
                    gamma[l, k] = ise.get_imag_self_energy()[0]
                    if write_detail:
                        detailed_gamma[l, k] = (
                            ise.get_detailed_imag_self_energy()[0])

            gamma_sigmas.append(gamma)

            if write_detail:
                filename = write_gamma_detail_to_hdf5(
                    detailed_gamma,
                    temperatures,
                    mesh,
                    gp,
                    sigma,
                    triplets,
                    weights,
                    frequency_points=frequency_points_at_sigma)

                if log_level:
                    print("Contribution of each triplet to imaginary part of "
                          "self energy is written in\n\"%s\"." % filename)

        imag_self_energy.append(gamma_sigmas)
        frequency_points.append(fp_sigmas)

    return imag_self_energy, frequency_points

def get_frequency_points(f_min,
                         f_max,
                         frequency_step=None,
                         num_frequency_points=None):
    if num_frequency_points is None:
        if frequency_step is not None:
            frequency_points = np.arange(
                f_min, f_max, frequency_step, dtype='double')
        else:
            frequency_points = np.array(np.linspace(
                f_min, f_max, 201), dtype='double')
    else:
        frequency_points = np.array(np.linspace(
            f_min, f_max, num_frequency_points), dtype='double')

    return frequency_points

def get_linewidth(interaction,
                  grid_points,
                  sigmas,
                  temperatures=np.arange(0, 1001, 10, dtype='double'),
                  write_detail=False,
                  log_level=0):
    ise = ImagSelfEnergy(interaction, with_detail=write_detail)
    band_indices = interaction.get_band_indices()
    mesh = interaction.get_mesh_numbers()
    gamma = np.zeros(
        (len(grid_points), len(sigmas), len(temperatures), len(band_indices)),
        dtype='double')

    print("--------------------------------"
          " Linewidth "
          "---------------------------------")
    for i, gp in enumerate(grid_points):
        ise.set_grid_point(gp)
        if log_level:
            print("======================= Grid point %d (%d/%d) "
                  "=======================" %
                  (gp, i + 1, len(grid_points)))
            weights = interaction.get_triplets_at_q()[1]
            print("Number of ir-triplets: "
                  "%d / %d" % (len(weights), weights.sum()))
            print("Calculating ph-ph interaction...")
        ise.run_interaction()
        frequencies = interaction.get_phonons()[0]
        if log_level:
            adrs = interaction.get_grid_address()[gp]
            q = adrs.astype('double') / mesh
            print("q-point: %s" % q)
            print("Phonon frequency:")
            print("%s" % frequencies[gp])
            sys.stdout.flush()

        if write_detail:
            triplets, weights, _, _ = interaction.get_triplets_at_q()

        for j, sigma in enumerate(sigmas):
            if log_level:
                text = "Calculating linewidths with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)
            ise.set_sigma(sigma)
            ise.set_integration_weights()

            if write_detail:
                num_band0 = len(interaction.get_band_indices())
                num_band = frequencies.shape[1]
                num_temp = len(temperatures)
                num_triplets = len(weights)
                detailed_gamma = np.zeros(
                    (num_temp, num_triplets, num_band0, num_band, num_band),
                    dtype='double')

            for k, t in enumerate(temperatures):
                ise.set_temperature(t)
                ise.run()
                gamma[i, j, k] = ise.get_imag_self_energy()
                if write_detail:
                    detailed_gamma[k] = ise.get_detailed_imag_self_energy()

            if write_detail:
                filename = write_gamma_detail_to_hdf5(
                    detailed_gamma,
                    temperatures,
                    mesh,
                    gp,
                    sigma,
                    triplets,
                    weights)

                if log_level:
                    print("Contribution of each triplet to imaginary part of "
                          "self energy is written in\n\"%s\"." % filename)
    return gamma

def write_linewidth(linewidth,
                    band_indices,
                    mesh,
                    grid_points,
                    sigmas,
                    temperatures,
                    filename=None,
                    is_mesh_symmetry=True):
    for i, gp in enumerate(grid_points):
        for j, sigma in enumerate(sigmas):
            for k, bi in enumerate(band_indices):
                pos = 0
                for l in range(k):
                    pos += len(band_indices[l])
                write_linewidth_at_grid_point(
                    gp,
                    bi,
                    temperatures,
                    linewidth[i, j, :, pos:(pos+len(bi))],
                    mesh,
                    sigma=sigma,
                    filename=filename,
                    is_mesh_symmetry=is_mesh_symmetry)

def write_imag_self_energy(imag_self_energy,
                           mesh,
                           grid_points,
                           band_indices,
                           frequency_points,
                           temperatures,
                           sigmas,
                           scattering_event_class=None,
                           filename=None,
                           is_mesh_symmetry=True):
    for gp, ise_sigmas, fp_sigmas in zip(grid_points,
                                         imag_self_energy,
                                         frequency_points):
        for sigma, ise_temps, fp in zip(sigmas, ise_sigmas, fp_sigmas):
            for t, ise in zip(temperatures, ise_temps):
                 for i, bi in enumerate(band_indices):
                     pos = 0
                     for j in range(i):
                         pos += len(band_indices[j])
                     write_imag_self_energy_at_grid_point(
                         gp,
                         bi,
                         mesh,
                         fp,
                         ise[:, pos:(pos + len(bi))].sum(axis=1) / len(bi),
                         sigma=sigma,
                         temperature=t,
                         scattering_event_class=scattering_event_class,
                         filename=filename,
                         is_mesh_symmetry=is_mesh_symmetry)

def average_by_degeneracy(imag_self_energy, band_indices, freqs_at_gp):
    deg_sets = degenerate_sets(freqs_at_gp)
    imag_se = np.zeros_like(imag_self_energy)
    for dset in deg_sets:
        bi_set = []
        for i, bi in enumerate(band_indices):
            if bi in dset:
                bi_set.append(i)
        for i in bi_set:
            if imag_self_energy.ndim == 1:
                imag_se[i] = (imag_self_energy[bi_set].sum() /
                              len(bi_set))
            else:
                imag_se[:, i] = (
                    imag_self_energy[:, bi_set].sum(axis=1) /
                    len(bi_set))
    return imag_se

class ImagSelfEnergy(object):
    def __init__(self,
                 interaction,
                 frequency_points=None,
                 temperature=None,
                 sigma=None,
                 with_detail=False,
                 unit_conversion=None,
                 lang='C'):
        self._pp = interaction
        self._sigma = None
        self.set_sigma(sigma)
        self._temperature = None
        self.set_temperature(temperature)
        self._frequency_points = None
        self.set_frequency_points(frequency_points)
        self._grid_point = None

        self._lang = lang
        self._imag_self_energy = None
        self._detailed_imag_self_energy = None
        self._pp_strength = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._with_detail = with_detail
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

        self._g = None # integration weights
        self._g_zero = None
        self._mesh = self._pp.get_mesh_numbers()
        self._is_collision_matrix = False

        # Unit to THz of Gamma
        if unit_conversion is None:
            self._unit_conversion = (18 * np.pi / (Hbar * EV) ** 2
                                     / (2 * np.pi * THz) ** 2
                                     * EV ** 2)
        else:
            self._unit_conversion = unit_conversion

    def run(self):
        if self._pp_strength is None:
            self.run_interaction()

        num_band0 = self._pp_strength.shape[1]
        if self._frequency_points is None:
            self._imag_self_energy = np.zeros(num_band0, dtype='double')
            if self._with_detail:
                self._detailed_imag_self_energy = np.empty_like(
                    self._pp_strength)
                self._detailed_imag_self_energy[:] = 0
                self._ise_N = np.zeros_like(self._imag_self_energy)
                self._ise_U = np.zeros_like(self._imag_self_energy)
            self._run_with_band_indices()
        else:
            self._imag_self_energy = np.zeros(
                (len(self._frequency_points), num_band0), dtype='double')
            if self._with_detail:
                self._detailed_imag_self_energy = np.zeros(
                    (len(self._frequency_points),) + self._pp_strength.shape,
                    dtype='double')
                self._ise_N = np.zeros_like(self._imag_self_energy)
                self._ise_U = np.zeros_like(self._imag_self_energy)
            self._run_with_frequency_points()

    def run_interaction(self, is_full_pp=True):
        if is_full_pp or self._frequency_points is not None:
            self._pp.run(lang=self._lang)
        else:
            self._pp.run(lang=self._lang, g_zero=self._g_zero)
        self._pp_strength = self._pp.get_interaction_strength()

    def set_integration_weights(self, scattering_event_class=None):
        if self._frequency_points is None:
            bi = self._pp.get_band_indices()
            f_points = self._frequencies[self._grid_point][bi]
        else:
            f_points = self._frequency_points

        self._g, _g_zero = get_triplets_integration_weights(
            self._pp,
            np.array(f_points, dtype='double'),
            self._sigma,
            is_collision_matrix=self._is_collision_matrix)
        if self._frequency_points is None:
            self._g_zero = _g_zero

        if scattering_event_class == 1 or scattering_event_class == 2:
            self._g[scattering_event_class - 1] = 0

    def get_imag_self_energy(self):
        if self._cutoff_frequency is None:
            return self._imag_self_energy
        else:
            return self._average_by_degeneracy(self._imag_self_energy)

    def get_imag_self_energy_N_and_U(self):
        if self._cutoff_frequency is None:
            return self._ise_N, self._ise_U
        else:
            return (self._average_by_degeneracy(self._ise_N),
                    self._average_by_degeneracy(self._ise_U))

    def get_detailed_imag_self_energy(self):
        return self._detailed_imag_self_energy

    def get_integration_weights(self):
        return self._g, self._g_zero

    def get_unit_conversion_factor(self):
        return self._unit_conversion

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            self._pp_strength = None
            (self._triplets_at_q,
             self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
            self._grid_point = grid_point
            self._frequencies, self._eigenvectors, _ = self._pp.get_phonons()

    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

        self.delete_integration_weights()

    def set_frequency_points(self, frequency_points):
        if frequency_points is None:
            self._frequency_points = None
        else:
            self._frequency_points = np.array(frequency_points, dtype='double')

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    def set_averaged_pp_interaction(self, ave_pp):
        # set_phonons is unnecessary now because all phonons are calculated in
        # self._pp.set_dynamical_matrix, though Gamma-point is an exception,
        # which is treatd at self._pp.set_grid_point.
        # self._pp.set_phonons(self._triplets_at_q.ravel())
        (self._frequencies,
         self._eigenvectors) = self._pp.get_phonons()[:2]

        num_triplets = len(self._triplets_at_q)
        num_band = self._pp.get_primitive().get_number_of_atoms() * 3
        num_grid = np.prod(self._mesh)
        bi = self._pp.get_band_indices()
        self._pp_strength = np.zeros(
            (num_triplets, len(bi), num_band, num_band), dtype='double')

        for i, v_ave in enumerate(ave_pp):
            self._pp_strength[:, i, :, :] = v_ave / num_grid

    def delete_integration_weights(self):
        self._g = None
        self._g_zero = None
        self._pp_strength = None

    def _run_with_band_indices(self):
        if self._g is not None:
            if self._lang == 'C':
                if self._with_detail:
                    # self._detailed_imag_self_energy.shape = 
                    #    (num_triplets, num_band0, num_band, num_band)
                    # self._imag_self_energy is also set.
                    self._run_c_detailed_with_band_indices_with_g()
                else:
                    # self._imag_self_energy.shape = (num_band0,)
                    self._run_c_with_band_indices_with_g()
            else:
                print("Running into _run_py_with_band_indices_with_g()")
                print("This routine is super slow and only for the test.")
                self._run_py_with_band_indices_with_g()
        else:
            print("get_triplets_integration_weights must be executed "
                  "before calling this method.")
            import sys
            sys.exit(1)

    def _run_with_frequency_points(self):
        if self._g is not None:
            if self._lang == 'C':
                if self._with_detail:
                    self._run_c_detailed_with_frequency_points_with_g()
                else:
                    self._run_c_with_frequency_points_with_g()
            else:
                print("Running into _run_py_with_frequency_points_with_g()")
                print("This routine is super slow and only for the test.")
                self._run_py_with_frequency_points_with_g()
        else:
            print("get_triplets_integration_weights must be executed "
                  "before calling this method.")
            import sys
            sys.exit(1)


    def _run_c_with_band_indices_with_g(self):
        import phono3py._phono3py as phono3c

        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape,
                               dtype='byte', order='C')
        else:
            _g_zero = self._g_zero

        phono3c.imag_self_energy_with_g(self._imag_self_energy,
                                        self._pp_strength,
                                        self._triplets_at_q,
                                        self._weights_at_q,
                                        self._frequencies,
                                        self._temperature,
                                        self._g,
                                        _g_zero,
                                        self._cutoff_frequency)
        self._imag_self_energy *= self._unit_conversion

    def _run_c_detailed_with_band_indices_with_g(self):
        import phono3py._phono3py as phono3c

        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape,
                               dtype='byte', order='C')
        else:
            _g_zero = self._g_zero

        phono3c.detailed_imag_self_energy_with_g(
            self._detailed_imag_self_energy,
            self._ise_N, # Normal
            self._ise_U, # Umklapp
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._pp.get_grid_address(),
            self._frequencies,
            self._temperature,
            self._g,
            _g_zero,
            self._cutoff_frequency)

        self._ise_N *= self._unit_conversion
        self._ise_U *= self._unit_conversion
        self._imag_self_energy = self._ise_N + self._ise_U

    def _run_c_with_frequency_points_with_g(self):
        import phono3py._phono3py as phono3c
        num_band0 = self._pp_strength.shape[1]
        g_shape = list(self._g.shape)
        g_shape[2] = num_band0
        g = np.zeros(tuple(g_shape), dtype='double', order='C')
        ise_at_f = np.zeros(num_band0, dtype='double')
        _g_zero = np.zeros(g_shape, dtype='byte', order='C')

        for i in range(len(self._frequency_points)):
            for j in range(num_band0):
                g[:, :, j, :, :] = self._g[:, :, i, :, :]
            phono3c.imag_self_energy_with_g(ise_at_f,
                                            self._pp_strength,
                                            self._triplets_at_q,
                                            self._weights_at_q,
                                            self._frequencies,
                                            self._temperature,
                                            g,
                                            _g_zero, # don't use g_zero
                                            self._cutoff_frequency)
            self._imag_self_energy[i] = ise_at_f
        self._imag_self_energy *= self._unit_conversion

    def _run_c_detailed_with_frequency_points_with_g(self):
        import phono3py._phono3py as phono3c
        num_band0 = self._pp_strength.shape[1]
        g_shape = list(self._g.shape)
        g_shape[2] = num_band0
        g = np.zeros((2,) + self._pp_strength.shape, dtype='double')
        detailed_ise_at_f = np.zeros(self._detailed_imag_self_energy.shape[1:5],
                                     dtype='double')
        ise_at_f_N = np.zeros(num_band0, dtype='double')
        ise_at_f_U = np.zeros(num_band0, dtype='double')
        _g_zero = np.zeros(g_shape, dtype='byte', order='C')

        for i in range(len(self._frequency_points)):
            for j in range(g.shape[2]):
                g[:, :, j, :, :] = self._g[:, :, i, :, :]
                phono3c.detailed_imag_self_energy_with_g(
                    detailed_ise_at_f,
                    ise_at_f_N,
                    ise_at_f_U,
                    self._pp_strength,
                    self._triplets_at_q,
                    self._weights_at_q,
                    self._pp.get_grid_address(),
                    self._frequencies,
                    self._temperature,
                    g,
                    _g_zero,
                    self._cutoff_frequency)
            self._detailed_imag_self_energy[i] = (detailed_ise_at_f *
                                                  self._unit_conversion)
            self._ise_N[i] = ise_at_f_N * self._unit_conversion
            self._ise_U[i] = ise_at_f_U * self._unit_conversion
            self._imag_self_energy[i] = self._ise_N[i] + self._ise_U[i]

    def _run_py_with_band_indices_with_g(self):
        if self._temperature > 0:
            self._ise_thm_with_band_indices()
        else:
            self._ise_thm_with_band_indices_0K()

    def _ise_thm_with_band_indices(self):
        freqs = self._frequencies[self._triplets_at_q[:, [1, 2]]]
        freqs = np.where(freqs > self._cutoff_frequency, freqs, 1)
        n = occupation(freqs, self._temperature)
        for i, (tp, w, interaction) in enumerate(zip(self._triplets_at_q,
                                                     self._weights_at_q,
                                                     self._pp_strength)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if (f1 > self._cutoff_frequency and
                    f2 > self._cutoff_frequency):
                    n2 = n[i, 0, j]
                    n3 = n[i, 1, k]
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k] # g2 - g3
                    self._imag_self_energy[:] += (
                        (n2 + n3 + 1) * g1 +
                        (n2 - n3) * (g2_g3)) * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_band_indices_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._pp_strength)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                self._imag_self_energy[:] += g1 * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _run_py_with_frequency_points_with_g(self):
        if self._temperature > 0:
            self._ise_thm_with_frequency_points()
        else:
            self._ise_thm_with_frequency_points_0K()

    def _ise_thm_with_frequency_points(self):
        for i, (tp, w, interaction) in enumerate(zip(self._triplets_at_q,
                                                     self._weights_at_q,
                                                     self._pp_strength)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if (f1 > self._cutoff_frequency and
                    f2 > self._cutoff_frequency):
                    n2 = occupation(f1, self._temperature)
                    n3 = occupation(f2, self._temperature)
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k] # g2 - g3
                    for l in range(len(interaction)):
                        self._imag_self_energy[:, l] += (
                            (n2 + n3 + 1) * g1 +
                            (n2 - n3) * (g2_g3)) * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_frequency_points_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._pp_strength)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                for l in range(len(interaction)):
                    self._imag_self_energy[:, l] += g1 * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _average_by_degeneracy(self, imag_self_energy):
        return average_by_degeneracy(imag_self_energy,
                                     self._pp.get_band_indices(),
                                     self._frequencies[self._grid_point])
