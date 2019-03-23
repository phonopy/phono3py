import numpy as np
from phonopy.cui.settings import Settings, ConfParser, fracval

class Phono3pySettings(Settings):
    def __init__(self):
        Settings.__init__(self)

        self._alm_options = None
        self._boundary_mfp = 1.0e6 # In micrometre. The default value is
                                   # just set to avoid divergence.
        self._coarse_mesh_shifts = None
        self._const_ave_pp = None
        self._create_displacements = False
        self._cutoff_fc3_distance = None
        self._cutoff_pair_distance = None
        self._gamma_conversion_factor = None
        self._grid_addresses = None
        self._grid_points = None
        self._ion_clamped = False
        self._is_bterta = False
        self._is_compact_fc = False
        self._is_frequency_shift = False
        self._is_full_pp = False
        self._is_gruneisen = False
        self._is_imag_self_energy = False
        self._is_isotope = False
        self._is_joint_dos = False
        self._is_kappa_star = True
        self._is_lbte = False
        self._is_N_U = False
        self._is_reducible_collision_matrix = False
        self._is_symmetrize_fc2 = False
        self._is_symmetrize_fc3_q = False
        self._is_symmetrize_fc3_r = False
        self._mass_variances = None
        self._max_freepath = None
        self._mesh_divisors = None
        self._read_collision = None
        self._read_fc2 = False
        self._read_fc3 = False
        self._read_gamma = False
        self._read_phonon = False
        self._read_pp = False
        self._phonon_supercell_matrix = None
        self._pinv_cutoff = 1.0e-8
        self._pinv_solver = 0
        self._pp_conversion_factor = None
        self._scattering_event_class = None  # scattering event class 1 or 2
        self._sigma_cutoff_width = None
        self._solve_collective_phonon = False
        self._temperatures = None
        self._use_alm_fc2 = False
        self._use_alm_fc3 = False
        self._use_ave_pp = False
        self._write_collision = False
        self._write_gamma_detail = False
        self._write_gamma = False
        self._write_phonon = False
        self._write_pp = False
        self._write_LBTE_solution = False

    def set_alm_options(self, alm_options):
        self._alm_options = alm_options

    def get_alm_options(self):
        return self._alm_options

    def set_boundary_mfp(self, boundary_mfp):
        self._boundary_mfp = boundary_mfp

    def get_boundary_mfp(self):
        return self._boundary_mfp

    def set_coarse_mesh_shifts(self, coarse_mesh_shifts):
        self._coarse_mesh_shifts = coarse_mesh_shifts

    def get_coarse_mesh_shifts(self):
        return self._coarse_mesh_shifts

    def set_create_displacements(self, create_displacements):
        self._create_displacements = create_displacements

    def get_create_displacements(self):
        return self._create_displacements

    def set_constant_averaged_pp_interaction(self, ave_pp):
        self._const_ave_pp = ave_pp

    def get_constant_averaged_pp_interaction(self):
        return self._const_ave_pp

    def set_cutoff_fc3_distance(self, cutoff_fc3_distance):
        self._cutoff_fc3_distance = cutoff_fc3_distance

    def get_cutoff_fc3_distance(self):
        return self._cutoff_fc3_distance

    def set_cutoff_pair_distance(self, cutoff_pair_distance):
        self._cutoff_pair_distance = cutoff_pair_distance

    def get_cutoff_pair_distance(self):
        return self._cutoff_pair_distance

    def set_gamma_conversion_factor(self, gamma_conversion_factor):
        self._gamma_conversion_factor = gamma_conversion_factor

    def get_gamma_conversion_factor(self):
        return self._gamma_conversion_factor

    def set_grid_addresses(self, grid_addresses):
        self._grid_addresses = grid_addresses

    def get_grid_addresses(self):
        return self._grid_addresses

    def set_grid_points(self, grid_points):
        self._grid_points = grid_points

    def get_grid_points(self):
        return self._grid_points

    def set_ion_clamped(self, ion_clamped):
        self._ion_clamped = ion_clamped

    def get_ion_clamped(self):
        return self._ion_clamped

    def set_is_bterta(self, is_bterta):
        self._is_bterta = is_bterta

    def get_is_bterta(self):
        return self._is_bterta

    def set_is_compact_fc(self, is_compact_fc):
        self._is_compact_fc = is_compact_fc

    def get_is_compact_fc(self):
        return self._is_compact_fc

    def set_is_frequency_shift(self, is_frequency_shift):
        self._is_frequency_shift = is_frequency_shift

    def get_is_frequency_shift(self):
        return self._is_frequency_shift

    def set_is_full_pp(self, is_full_pp):
        self._is_full_pp = is_full_pp

    def get_is_full_pp(self):
        return self._is_full_pp

    def set_is_gruneisen(self, is_gruneisen):
        self._is_gruneisen = is_gruneisen

    def get_is_gruneisen(self):
        return self._is_gruneisen

    def set_is_imag_self_energy(self, is_imag_self_energy):
        self._is_imag_self_energy = is_imag_self_energy

    def get_is_imag_self_energy(self):
        return self._is_imag_self_energy

    def set_is_isotope(self, is_isotope):
        self._is_isotope = is_isotope

    def get_is_isotope(self):
        return self._is_isotope

    def set_is_joint_dos(self, is_joint_dos):
        self._is_joint_dos = is_joint_dos

    def get_is_joint_dos(self):
        return self._is_joint_dos

    def set_is_kappa_star(self, is_kappa_star):
        self._is_kappa_star = is_kappa_star

    def get_is_kappa_star(self):
        return self._is_kappa_star

    def set_is_lbte(self, is_lbte):
        self._is_lbte = is_lbte

    def get_is_lbte(self):
        return self._is_lbte

    def set_is_N_U(self, is_N_U):
        self._is_N_U = is_N_U

    def get_is_N_U(self):
        return self._is_N_U

    def set_is_reducible_collision_matrix(self, is_reducible_collision_matrix):
        self._is_reducible_collision_matrix = is_reducible_collision_matrix

    def get_is_reducible_collision_matrix(self):
        return self._is_reducible_collision_matrix

    def set_is_symmetrize_fc2(self, is_symmetrize_fc2):
        self._is_symmetrize_fc2 = is_symmetrize_fc2

    def get_is_symmetrize_fc2(self):
        return self._is_symmetrize_fc2

    def set_is_symmetrize_fc3_q(self, is_symmetrize_fc3_q):
        self._is_symmetrize_fc3_q = is_symmetrize_fc3_q

    def get_is_symmetrize_fc3_q(self):
        return self._is_symmetrize_fc3_q

    def set_is_symmetrize_fc3_r(self, is_symmetrize_fc3_r):
        self._is_symmetrize_fc3_r = is_symmetrize_fc3_r

    def get_is_symmetrize_fc3_r(self):
        return self._is_symmetrize_fc3_r

    def set_mass_variances(self, mass_variances):
        self._mass_variances = mass_variances

    def get_mass_variances(self):
        return self._mass_variances

    def set_max_freepath(self, max_freepath):
        self._max_freepath = max_freepath

    def get_max_freepath(self):
        return self._max_freepath

    def set_mesh_divisors(self, mesh_divisors):
        self._mesh_divisors = mesh_divisors

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def set_phonon_supercell_matrix(self, matrix):
        self._phonon_supercell_matrix = matrix

    def get_phonon_supercell_matrix(self):
        return self._phonon_supercell_matrix

    def set_pinv_cutoff(self, pinv_cutoff):
        self._pinv_cutoff = pinv_cutoff

    def get_pinv_cutoff(self):
        return self._pinv_cutoff

    def set_pinv_solver(self, pinv_solver):
        self._pinv_solver = pinv_solver

    def get_pinv_solver(self):
        return self._pinv_solver

    def set_pp_conversion_factor(self, pp_conversion_factor):
        self._pp_conversion_factor = pp_conversion_factor

    def get_pp_conversion_factor(self):
        return self._pp_conversion_factor

    def set_read_collision(self, read_collision):
        self._read_collision = read_collision

    def get_read_collision(self):
        return self._read_collision

    def set_read_fc2(self, read_fc2):
        self._read_fc2 = read_fc2

    def get_read_fc2(self):
        return self._read_fc2

    def set_read_fc3(self, read_fc3):
        self._read_fc3 = read_fc3

    def get_read_fc3(self):
        return self._read_fc3

    def set_read_gamma(self, read_gamma):
        self._read_gamma = read_gamma

    def get_read_gamma(self):
        return self._read_gamma

    def set_read_phonon(self, read_phonon):
        self._read_phonon = read_phonon

    def get_read_phonon(self):
        return self._read_phonon

    def set_read_pp(self, read_pp):
        self._read_pp = read_pp

    def get_read_pp(self):
        return self._read_pp

    def set_scattering_event_class(self, scattering_event_class):
        self._scattering_event_class = scattering_event_class

    def get_scattering_event_class(self):
        return self._scattering_event_class

    def set_sigma_cutoff_width(self, sigma_cutoff_width):
        self._sigma_cutoff_width = sigma_cutoff_width

    def get_sigma_cutoff_width(self):
        return self._sigma_cutoff_width

    def set_solve_collective_phonon(self, solve_collective_phonon):
        self._solve_collective_phonon = solve_collective_phonon

    def get_solve_collective_phonon(self):
        return self._solve_collective_phonon

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures

    def set_use_alm_fc2(self, use_alm_fc2):
        self._use_alm_fc2 = use_alm_fc2

    def get_use_alm_fc2(self):
        return self._use_alm_fc2

    def set_use_alm_fc3(self, use_alm_fc3):
        self._use_alm_fc3 = use_alm_fc3

    def get_use_alm_fc3(self):
        return self._use_alm_fc3

    def set_use_ave_pp(self, use_ave_pp):
        self._use_ave_pp = use_ave_pp

    def get_use_ave_pp(self):
        return self._use_ave_pp

    def set_write_collision(self, write_collision):
        self._write_collision = write_collision

    def get_write_collision(self):
        return self._write_collision

    def set_write_gamma_detail(self, write_gamma_detail):
        self._write_gamma_detail = write_gamma_detail

    def get_write_gamma_detail(self):
        return self._write_gamma_detail

    def set_write_gamma(self, write_gamma):
        self._write_gamma = write_gamma

    def get_write_gamma(self):
        return self._write_gamma

    def set_write_phonon(self, write_phonon):
        self._write_phonon = write_phonon

    def get_write_phonon(self):
        return self._write_phonon

    def set_write_pp(self, write_pp):
        self._write_pp = write_pp

    def get_write_pp(self):
        return self._write_pp

    def set_write_LBTE_solution(self, write_LBTE_solution):
        self._write_LBTE_solution = write_LBTE_solution

    def get_write_LBTE_solution(self):
        return self._write_LBTE_solution


class Phono3pyConfParser(ConfParser):
    def __init__(self, filename=None, args=None):
        self._settings = Phono3pySettings()
        confs = {}
        if filename is not None:
            ConfParser.__init__(self, filename=filename)
            self.read_file()  # store .conf file setting in self._confs
            self._parse_conf()
            self._set_settings()
            confs.update(self._confs)
        if args is not None:
            ConfParser.__init__(self, args=args)
            self._read_options()
            self._parse_conf()
            self._set_settings()
            confs.update(self._confs)
        self._confs = confs

    def _read_options(self):
        self.read_options()  # store data in self._confs
        if 'phonon_supercell_dimension' in self._args:
            dim_fc2 = self._args.phonon_supercell_dimension
            if dim_fc2 is not None:
                self._confs['dim_fc2'] = " ".join(dim_fc2)

        if 'alm_options' in self._args:
            if self._args.alm_options is not None:
                self._confs['alm_options'] = self._args.alm_options

        if 'boundary_mfp' in self._args:
            if self._args.boundary_mfp is not None:
                self._confs['boundary_mfp'] = self._args.boundary_mfp

        if 'const_ave_pp' in self._args:
            const_ave_pp = self._args.const_ave_pp
            if const_ave_pp is not None:
                self._confs['const_ave_pp'] = const_ave_pp

        if 'cutoff_fc3_distance' in self._args:
            cutoff_fc3 = self._args.cutoff_fc3_distance
            if cutoff_fc3 is not None:
                self._confs['cutoff_fc3_distance'] = cutoff_fc3

        if 'cutoff_pair_distance' in self._args:
            cutoff_pair = self._args.cutoff_pair_distance
            if cutoff_pair is not None:
                self._confs['cutoff_pair_distance'] = cutoff_pair

        if 'gamma_conversion_factor' in self._args:
            g_conv_factor = self._args.gamma_conversion_factor
            if g_conv_factor is not None:
                self._confs['gamma_conversion_factor'] = g_conv_factor

        if 'grid_addresses' in self._args:
            grid_adrs = self._args.grid_addresses
            if grid_adrs is not None:
                self._confs['grid_addresses'] = " ".join(grid_adrs)

        if 'grid_points' in self._args:
            if self._args.grid_points is not None:
                self._confs['grid_points'] = " ".join(self._args.grid_points)

        if 'ion_clamped' in self._args:
            if self._args.ion_clamped:
                self._confs['ion_clamped'] = '.true.'

        if 'is_bterta' in self._args:
            if self._args.is_bterta:
                self._confs['bterta'] = '.true.'

        if 'is_compact_fc' in self._args:
            if self._args.is_compact_fc:
                self._confs['compact_fc'] = '.true.'

        if 'is_gruneisen' in self._args:
            if self._args.is_gruneisen:
                self._confs['gruneisen'] = '.true.'

        if 'is_displacement' in self._args:
            if self._args.is_displacement:
                self._confs['create_displacements'] = '.true.'

        if 'is_frequency_shift' in self._args:
            if self._args.is_frequency_shift:
                self._confs['frequency_shift'] = '.true.'

        if 'is_full_pp' in self._args:
            if self._args.is_full_pp:
                self._confs['full_pp'] = '.true.'

        if 'is_imag_self_energy' in self._args:
            if self._args.is_imag_self_energy:
                self._confs['imag_self_energy'] = '.true.'

        if 'is_isotope' in self._args:
            if self._args.is_isotope:
                self._confs['isotope'] = '.true.'

        if 'is_joint_dos' in self._args:
            if self._args.is_joint_dos:
                self._confs['joint_dos'] = '.true.'

        if 'no_kappa_stars' in self._args:
            if self._args.no_kappa_stars:
                self._confs['kappa_star'] = '.false.'

        if 'is_lbte' in self._args:
            if self._args.is_lbte:
                self._confs['lbte'] = '.true.'

        if 'is_N_U' in self._args:
            if self._args.is_N_U:
                self._confs['N_U'] = '.true.'

        if 'is_reducible_collision_matrix' in self._args:
            if self._args.is_reducible_collision_matrix:
                self._confs['reducible_collision_matrix'] = '.true.'

        if 'is_symmetrize_fc2' in self._args:
            if self._args.is_symmetrize_fc2:
                self._confs['symmetrize_fc2'] = '.true.'

        if 'is_symmetrize_fc3_q' in self._args:
            if self._args.is_symmetrize_fc3_q:
                self._confs['symmetrize_fc3_q'] = '.true.'

        if 'is_symmetrize_fc3_r' in self._args:
            if self._args.is_symmetrize_fc3_r:
                self._confs['symmetrize_fc3_r'] = '.true.'

        if 'mass_variances' in self._args:
            mass_variances = self._args.mass_variances
            if mass_variances is not None:
                self._confs['mass_variances'] = " ".join(mass_variances)

        if 'max_freepath' in self._args:
            if self._args.max_freepath is not None:
                self._confs['max_freepath'] = self._args.max_freepath

        if 'mesh_divisors' in self._args:
            mesh_divisors = self._args.mesh_divisors
            if mesh_divisors is not None:
                self._confs['mesh_divisors'] = " ".join(mesh_divisors)

        if 'pinv_cutoff' in self._args:
            if self._args.pinv_cutoff is not None:
                self._confs['pinv_cutoff'] = self._args.pinv_cutoff

        if 'pinv_solver' in self._args:
            if self._args.pinv_solver is not None:
                self._confs['pinv_solver'] = self._args.pinv_solver

        if 'pp_conversion_factor' in self._args:
            pp_conv_factor = self._args.pp_conversion_factor
            if pp_conv_factor is not None:
                self._confs['pp_conversion_factor'] = pp_conv_factor

        if 'read_fc2' in self._args:
            if self._args.read_fc2:
                self._confs['read_fc2'] = '.true.'

        if 'read_fc3' in self._args:
            if self._args.read_fc3:
                self._confs['read_fc3'] = '.true.'

        if 'read_gamma' in self._args:
            if self._args.read_gamma:
                self._confs['read_gamma'] = '.true.'

        if 'read_phonon' in self._args:
            if self._args.read_phonon:
                self._confs['read_phonon'] = '.true.'

        if 'read_pp' in self._args:
            if self._args.read_pp:
                self._confs['read_pp'] = '.true.'

        if 'read_collision' in self._args:
            if self._args.read_collision is not None:
                self._confs['read_collision'] = self._args.read_collision

        if 'scattering_event_class' in self._args:
            scatt_class = self._args.scattering_event_class
            if scatt_class is not None:
                self._confs['scattering_event_class'] = scatt_class

        if 'sigma_cutoff_width' in self._args:
            sigma_cutoff = self._args.sigma_cutoff_width
            if sigma_cutoff is not None:
                self._confs['sigma_cutoff_width'] = sigma_cutoff

        if 'solve_collective_phonon' in self._args:
            if self._args.solve_collective_phonon:
                self._confs['collective_phonon'] = '.true.'

        if 'temperatures' in self._args:
            if self._args.temperatures is not None:
                self._confs['temperatures'] = " ".join(self._args.temperatures)

        if 'use_alm_fc2' in self._args:
            if self._args.use_alm_fc2:
                self._confs['alm_fc2'] = '.true.'

        if 'use_alm_fc3' in self._args:
            if self._args.use_alm_fc3:
                self._confs['alm_fc3'] = '.true.'

        if 'use_ave_pp' in self._args:
            if self._args.use_ave_pp:
                self._confs['use_ave_pp'] = '.true.'

        if 'write_gamma_detail' in self._args:
            if self._args.write_gamma_detail:
                self._confs['write_gamma_detail'] = '.true.'

        if 'write_gamma' in self._args:
            if self._args.write_gamma:
                self._confs['write_gamma'] = '.true.'

        if 'write_collision' in self._args:
            if self._args.write_collision:
                self._confs['write_collision'] = '.true.'

        if 'write_phonon' in self._args:
            if self._args.write_phonon:
                self._confs['write_phonon'] = '.true.'

        if 'write_pp' in self._args:
            if self._args.write_pp:
                self._confs['write_pp'] = '.true.'

        if 'write_LBTE_solution' in self._args:
            if self._args.write_LBTE_solution:
                self._confs['write_LBTE_solution'] = '.true.'

    def _parse_conf(self):
        self.parse_conf()
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'create_displacements':
                if confs['create_displacements'].lower() == '.false.':
                    self.set_parameter('create_displacements', False)
                elif confs['create_displacements'].lower() == '.true.':
                    self.set_parameter('create_displacements', True)

            if conf_key == 'dim_fc2':
                matrix = [ int(x) for x in confs['dim_fc2'].split() ]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of dim2 has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            "Determinant of supercell matrix has " +
                            "to be positive.")
                    else:
                        self.set_parameter('dim_fc2', matrix)

            if conf_key == 'boundary_mfp':
                self.set_parameter('boundary_mfp',
                                   float(confs['boundary_mfp']))

            if conf_key in ('constant_averaged_pp_interaction'
                            'const_ave_pp'):
                self.set_parameter('const_ave_pp', float(confs['const_ave_pp']))

            if conf_key == 'cutoff_fc3_distance':
                self.set_parameter('cutoff_fc3_distance',
                                   float(confs['cutoff_fc3_distance']))

            if conf_key == 'cutoff_pair_distance':
                self.set_parameter('cutoff_pair_distance',
                                   float(confs['cutoff_pair_distance']))

            if conf_key == 'full_pp':
                if confs['full_pp'].lower() == '.false.':
                    self.set_parameter('is_full_pp', False)
                elif confs['full_pp'].lower() == '.true.':
                    self.set_parameter('is_full_pp', True)

            if conf_key == 'gamma_conversion_factor':
                self.set_parameter('gamma_conversion_factor',
                                   float(confs['gamma_conversion_factor']))

            if conf_key == 'grid_addresses':
                vals = [int(x) for x in
                        confs['grid_addresses'].replace(',', ' ').split()]
                if len(vals) % 3 == 0 and len(vals) > 0:
                    self.set_parameter('grid_addresses',
                                       np.reshape(vals, (-1, 3)))
                else:
                    self.setting_error("Grid addresses are incorrectly set.")

            if conf_key == 'grid_points':
                vals = [int(x) for x in
                        confs['grid_points'].replace(',', ' ').split()]
                self.set_parameter('grid_points', vals)

            if conf_key == 'ion_clamped':
                if confs['ion_clamped'].lower() == '.false.':
                    self.set_parameter('ion_clamped', False)
                elif confs['ion_clamped'].lower() == '.true.':
                    self.set_parameter('ion_clamped', True)

            if conf_key == 'bterta':
                if confs['bterta'].lower() == '.false.':
                    self.set_parameter('is_bterta', False)
                elif confs['bterta'].lower() == '.true.':
                    self.set_parameter('is_bterta', True)

            if conf_key == 'compact_fc':
                if confs['compact_fc'].lower() == '.false.':
                    self.set_parameter('is_compact_fc', False)
                elif confs['compact_fc'].lower() == '.true.':
                    self.set_parameter('is_compact_fc', True)

            if conf_key == 'frequency_shift':
                if confs['frequency_shift'].lower() == '.false.':
                    self.set_parameter('is_frequency_shift', False)
                elif confs['frequency_shift'].lower() == '.true.':
                    self.set_parameter('is_frequency_shift', True)

            if conf_key == 'gruneisen':
                if confs['gruneisen'].lower() == '.false.':
                    self.set_parameter('is_gruneisen', False)
                elif confs['gruneisen'].lower() == '.true.':
                    self.set_parameter('is_gruneisen', True)

            if conf_key == 'imag_self_energy':
                if confs['imag_self_energy'].lower() == '.false.':
                    self.set_parameter('is_imag_self_energy', False)
                elif confs['imag_self_energy'].lower() == '.true.':
                    self.set_parameter('is_imag_self_energy', True)

            if conf_key == 'isotope':
                if confs['isotope'].lower() == '.false.':
                    self.set_parameter('is_isotope', False)
                elif confs['isotope'].lower() == '.true.':
                    self.set_parameter('is_isotope', True)

            if conf_key == 'joint_dos':
                if confs['joint_dos'].lower() == '.false.':
                    self.set_parameter('is_joint_dos', False)
                elif confs['joint_dos'].lower() == '.true.':
                    self.set_parameter('is_joint_dos', True)

            if conf_key == 'lbte':
                if confs['lbte'].lower() == '.false.':
                    self.set_parameter('is_lbte', False)
                elif confs['lbte'].lower() == '.true.':
                    self.set_parameter('is_lbte', True)

            if conf_key == 'N_U':
                if confs['N_U'].lower() == '.false.':
                    self.set_parameter('is_N_U', False)
                elif confs['N_U'].lower() == '.true.':
                    self.set_parameter('is_N_U', True)

            if conf_key == 'reducible_collision_matrix':
                if confs['reducible_collision_matrix'].lower() == '.false.':
                    self.set_parameter('is_reducible_collision_matrix', False)
                elif confs['reducible_collision_matrix'].lower() == '.true.':
                    self.set_parameter('is_reducible_collision_matrix', True)

            if conf_key == 'symmetrize_fc2':
                if confs['symmetrize_fc2'].lower() == '.false.':
                    self.set_parameter('is_symmetrize_fc2', False)
                elif confs['symmetrize_fc2'].lower() == '.true.':
                    self.set_parameter('is_symmetrize_fc2', True)

            if conf_key == 'symmetrize_fc3_q':
                if confs['symmetrize_fc3_q'].lower() == '.false.':
                    self.set_parameter('is_symmetrize_fc3_q', False)
                elif confs['symmetrize_fc3_q'].lower() == '.true.':
                    self.set_parameter('is_symmetrize_fc3_q', True)

            if conf_key == 'symmetrize_fc3_r':
                if confs['symmetrize_fc3_r'].lower() == '.false.':
                    self.set_parameter('is_symmetrize_fc3_r', False)
                elif confs['symmetrize_fc3_r'].lower() == '.true.':
                    self.set_parameter('is_symmetrize_fc3_r', True)

            if conf_key == 'mass_variances':
                vals = [fracval(x) for x in confs['mass_variances'].split()]
                if len(vals) < 1:
                    self.setting_error("Mass variance parameters are incorrectly set.")
                else:
                    self.set_parameter('mass_variances', vals)

            if conf_key == 'max_freepath':
                self.set_parameter('max_freepath', float(confs['max_freepath']))

            if conf_key == 'mesh_divisors':
                vals = [x for x in confs['mesh_divisors'].split()]
                if len(vals) == 3:
                    self.set_parameter('mesh_divisors', [int(x) for x in vals])
                elif len(vals) == 6:
                    divs = [int(x) for x in vals[:3]]
                    is_shift = [x.lower() == 't' for x in vals[3:]]
                    for i in range(3):
                        if is_shift[i] and (divs[i] % 2 != 0):
                            is_shift[i] = False
                            self.setting_error("Coarse grid shift along the " +
                                               ["first", "second", "third"][i] +
                                               " axis is not allowed.")
                    self.set_parameter('mesh_divisors', divs + is_shift)
                else:
                    self.setting_error("Mesh divisors are incorrectly set.")

            if conf_key == 'kappa_star':
                if confs['kappa_star'].lower() == '.false.':
                    self.set_parameter('is_kappa_star', False)
                elif confs['kappa_star'].lower() == '.true.':
                    self.set_parameter('is_kappa_star', True)

            if conf_key == 'pinv_cutoff':
                self.set_parameter('pinv_cutoff', float(confs['pinv_cutoff']))

            if conf_key == 'pinv_solver':
                self.set_parameter('pinv_solver', int(confs['pinv_solver']))

            if conf_key == 'pp_conversion_factor':
                self.set_parameter('pp_conversion_factor',
                                   float(confs['pp_conversion_factor']))

            if conf_key == 'read_collision':
                if confs['read_collision'] == 'all':
                    self.set_parameter('read_collision', 'all')
                else:
                    vals = [int(x) for x in confs['read_collision'].split()]
                    self.set_parameter('read_collision', vals)

            if conf_key == 'read_fc2':
                if confs['read_fc2'].lower() == '.false.':
                    self.set_parameter('read_fc2', False)
                elif confs['read_fc2'].lower() == '.true.':
                    self.set_parameter('read_fc2', True)

            if conf_key == 'read_fc3':
                if confs['read_fc3'].lower() == '.false.':
                    self.set_parameter('read_fc3', False)
                elif confs['read_fc3'].lower() == '.true.':
                    self.set_parameter('read_fc3', True)

            if conf_key == 'read_gamma':
                if confs['read_gamma'].lower() == '.false.':
                    self.set_parameter('read_gamma', False)
                elif confs['read_gamma'].lower() == '.true.':
                    self.set_parameter('read_gamma', True)

            if conf_key == 'read_phonon':
                if confs['read_phonon'].lower() == '.false.':
                    self.set_parameter('read_phonon', False)
                elif confs['read_phonon'].lower() == '.true.':
                    self.set_parameter('read_phonon', True)

            if conf_key == 'read_pp':
                if confs['read_pp'].lower() == '.false.':
                    self.set_parameter('read_pp', False)
                elif confs['read_pp'].lower() == '.true.':
                    self.set_parameter('read_pp', True)

            if conf_key == 'scattering_event_class':
                self.set_parameter('scattering_event_class',
                                   confs['scattering_event_class'])

            if conf_key == 'sigma_cutoff_width':
                self.set_parameter('sigma_cutoff_width',
                                   float(confs['sigma_cutoff_width']))

            if conf_key == 'collective_phonon':
                if confs['collective_phonon'].lower() == '.false.':
                    self.set_parameter('collective_phonon', False)
                elif confs['collective_phonon'].lower() == '.true.':
                    self.set_parameter('collective_phonon', True)

            if conf_key == 'temperatures':
                vals = [fracval(x) for x in confs['temperatures'].split()]
                if len(vals) < 1:
                    self.setting_error("Temperatures are incorrectly set.")
                else:
                    self.set_parameter('temperatures', vals)

            if conf_key == 'alm_options':
                self.set_parameter('alm_options', confs['alm_options'])

            if conf_key == 'alm_fc2':
                if confs['alm_fc2'].lower() == '.false.':
                    self.set_parameter('alm_fc2', False)
                elif confs['alm_fc2'].lower() == '.true.':
                    self.set_parameter('alm_fc2', True)

            if conf_key == 'alm_fc3':
                if confs['alm_fc3'].lower() == '.false.':
                    self.set_parameter('alm_fc3', False)
                elif confs['alm_fc3'].lower() == '.true.':
                    self.set_parameter('alm_fc3', True)

            if conf_key == 'use_ave_pp':
                if confs['use_ave_pp'].lower() == '.false.':
                    self.set_parameter('use_ave_pp', False)
                elif confs['use_ave_pp'].lower() == '.true.':
                    self.set_parameter('use_ave_pp', True)

            if conf_key == 'write_gamma_detail':
                if confs['write_gamma_detail'].lower() == '.false.':
                    self.set_parameter('write_gamma_detail', False)
                elif confs['write_gamma_detail'].lower() == '.true.':
                    self.set_parameter('write_gamma_detail', True)

            if conf_key == 'write_gamma':
                if confs['write_gamma'].lower() == '.false.':
                    self.set_parameter('write_gamma', False)
                elif confs['write_gamma'].lower() == '.true.':
                    self.set_parameter('write_gamma', True)

            if conf_key == 'write_collision':
                if confs['write_collision'].lower() == '.false.':
                    self.set_parameter('write_collision', False)
                elif confs['write_collision'].lower() == '.true.':
                    self.set_parameter('write_collision', True)

            if conf_key == 'write_phonon':
                if confs['write_phonon'].lower() == '.false.':
                    self.set_parameter('write_phonon', False)
                elif confs['write_phonon'].lower() == '.true.':
                    self.set_parameter('write_phonon', True)

            if conf_key == 'write_pp':
                if confs['write_pp'].lower() == '.false.':
                    self.set_parameter('write_pp', False)
                elif confs['write_pp'].lower() == '.true.':
                    self.set_parameter('write_pp', True)

            if conf_key == 'write_LBTE_solution':
                if confs['write_LBTE_solution'].lower() == '.false.':
                    self.set_parameter('write_LBTE_solution', False)
                elif confs['write_LBTE_solution'].lower() == '.true.':
                    self.set_parameter('write_LBTE_solution', True)

    def _set_settings(self):
        self.set_settings()
        params = self._parameters

        # Is getting least displacements?
        if 'create_displacements' in params:
            if params['create_displacements']:
                self._settings.set_create_displacements('displacements')

        # Supercell dimension for fc2
        if 'dim_fc2' in params:
            self._settings.set_phonon_supercell_matrix(params['dim_fc2'])

        # Boundary mean free path for thermal conductivity calculation
        if 'boundary_mfp' in params:
            self._settings.set_boundary_mfp(params['boundary_mfp'])

        # Peierls type approximation for squared ph-ph interaction strength
        if 'const_ave_pp' in params:
            self._settings.set_constant_averaged_pp_interaction(
                params['const_ave_pp'])

        # Cutoff distance of third-order force constants. Elements where any
        # pair of atoms has larger distance than cut-off distance are set zero.
        if 'cutoff_fc3_distance' in params:
            self._settings.set_cutoff_fc3_distance(params['cutoff_fc3_distance'])

        # Cutoff distance between pairs of displaced atoms used for supercell
        # creation with displacements and making third-order force constants
        if 'cutoff_pair_distance' in params:
            self._settings.set_cutoff_pair_distance(
                params['cutoff_pair_distance'])

        # Gamma unit conversion factor
        if 'gamma_conversion_factor' in params:
            self._settings.set_gamma_conversion_factor(
                params['gamma_conversion_factor'])

        # Grid addresses (sets of three integer values)
        if 'grid_addresses' in params:
            self._settings.set_grid_addresses(params['grid_addresses'])

        # Grid points
        if 'grid_points' in params:
            self._settings.set_grid_points(params['grid_points'])

        # Atoms are clamped under applied strain in Gruneisen parameter calculation
        if 'ion_clamped' in params:
            self._settings.set_ion_clamped(params['ion_clamped'])

        # Calculate thermal conductivity in BTE-RTA
        if 'is_bterta' in params:
            self._settings.set_is_bterta(params['is_bterta'])

        # Compact force constants or full force constants
        if 'is_compact_fc' in params:
            self._settings.set_is_compact_fc(params['is_compact_fc'])

        # Calculate frequency_shifts
        if 'is_frequency_shift' in params:
            self._settings.set_is_frequency_shift(params['is_frequency_shift'])

        # Calculate full ph-ph interaction strength for RTA conductivity
        if 'is_full_pp' in params:
            self._settings.set_is_full_pp(params['is_full_pp'])

        # Calculate phonon-Gruneisen parameters
        if 'is_gruneisen' in params:
            self._settings.set_is_gruneisen(params['is_gruneisen'])

        # Calculate imaginary part of self energy
        if 'is_imag_self_energy' in params:
            self._settings.set_is_imag_self_energy(params['is_imag_self_energy'])

        # Calculate lifetime due to isotope scattering
        if 'is_isotope' in params:
            self._settings.set_is_isotope(params['is_isotope'])

        # Calculate joint-DOS
        if 'is_joint_dos' in params:
            self._settings.set_is_joint_dos(params['is_joint_dos'])

        # Calculate thermal conductivity in LBTE with Chaput's method
        if 'is_lbte' in params:
            self._settings.set_is_lbte(params['is_lbte'])

        # Calculate Normal and Umklapp processes
        if 'is_N_U' in params:
            self._settings.set_is_N_U(params['is_N_U'])

        # Solve reducible collision matrix but not reduced matrix
        if 'is_reducible_collision_matrix' in params:
            self._settings.set_is_reducible_collision_matrix(
                params['is_reducible_collision_matrix'])

        # Symmetrize fc2 by index exchange
        if 'is_symmetrize_fc2' in params:
            self._settings.set_is_symmetrize_fc2(params['is_symmetrize_fc2'])

        # Symmetrize phonon fc3 by index exchange
        if 'is_symmetrize_fc3_q' in params:
            self._settings.set_is_symmetrize_fc3_q(params['is_symmetrize_fc3_q'])

        # Symmetrize fc3 by index exchange
        if 'is_symmetrize_fc3_r' in params:
            self._settings.set_is_symmetrize_fc3_r(params['is_symmetrize_fc3_r'])

        # Mass variance parameters
        if 'mass_variances' in params:
            self._settings.set_mass_variances(params['mass_variances'])

        # Maximum mean free path
        if 'max_freepath' in params:
            self._settings.set_max_freepath(params['max_freepath'])

        # Divisors for mesh numbers
        if 'mesh_divisors' in params:
            self._settings.set_mesh_divisors(params['mesh_divisors'][:3])
            if len(params['mesh_divisors']) > 3:
                self._settings.set_coarse_mesh_shifts(
                    params['mesh_divisors'][3:])

        # Cutoff frequency for pseudo inversion of collision matrix
        if 'pinv_cutoff' in params:
            self._settings.set_pinv_cutoff(params['pinv_cutoff'])

        # Switch for pseudo-inverse solver
        if 'pinv_solver' in params:
            self._settings.set_pinv_solver(params['pinv_solver'])

        # Ph-ph interaction unit conversion factor
        if 'pp_conversion_factor' in params:
            self._settings.set_pp_conversion_factor(params['pp_conversion_factor'])

        # Read phonon-phonon interaction amplitudes from hdf5
        if 'read_amplitude' in params:
            self._settings.set_read_amplitude(params['read_amplitude'])

        # Read collision matrix and gammas from hdf5
        if 'read_collision' in params:
            self._settings.set_read_collision(params['read_collision'])

        # Read fc2 from hdf5
        if 'read_fc2' in params:
            self._settings.set_read_fc2(params['read_fc2'])

        # Read fc3 from hdf5
        if 'read_fc3' in params:
            self._settings.set_read_fc3(params['read_fc3'])

        # Read gammas from hdf5
        if 'read_gamma' in params:
            self._settings.set_read_gamma(params['read_gamma'])

        # Read phonons from hdf5
        if 'read_phonon' in params:
            self._settings.set_read_phonon(params['read_phonon'])

        # Read ph-ph interaction strength from hdf5
        if 'read_pp' in params:
            self._settings.set_read_pp(params['read_pp'])

        # Sum partial kappa at q-stars
        if 'is_kappa_star' in params:
            self._settings.set_is_kappa_star(params['is_kappa_star'])

        # Scattering event class 1 or 2
        if 'scattering_event_class' in params:
            self._settings.set_scattering_event_class(
                params['scattering_event_class'])

        # Cutoff width of smearing function (ratio to sigma value)
        if 'sigma_cutoff_width' in params:
            self._settings.set_sigma_cutoff_width(params['sigma_cutoff_width'])

        # Solve collective phonons
        if 'collective_phonon' in params:
            self._settings.set_solve_collective_phonon(
                params['collective_phonon'])

        # Temperatures
        if 'temperatures' in params:
            self._settings.set_temperatures(params['temperatures'])

        # List of ALM options as string separated by ','
        if 'alm_options' in params:
            self._settings.set_alm_options(params['alm_options'])

        # Use ALM for creating fc2
        if 'alm_fc2' in params:
            self._settings.set_use_alm_fc2(params['alm_fc2'])

        # Use ALM for creating fc3
        if 'alm_fc3' in params:
            self._settings.set_use_alm_fc3(params['alm_fc3'])

        # Use averaged ph-ph interaction
        if 'use_ave_pp' in params:
            self._settings.set_use_ave_pp(params['use_ave_pp'])

        # Write detailed imag-part of self energy to hdf5
        if 'write_gamma_detail' in params:
            self._settings.set_write_gamma_detail(
                params['write_gamma_detail'])

        # Write imag-part of self energy to hdf5
        if 'write_gamma' in params:
            self._settings.set_write_gamma(params['write_gamma'])

        # Write collision matrix and gammas to hdf5
        if 'write_collision' in params:
            self._settings.set_write_collision(params['write_collision'])

        # Write all phonons on grid points to hdf5
        if 'write_phonon' in params:
            self._settings.set_write_phonon(params['write_phonon'])

        # Write phonon-phonon interaction amplitudes to hdf5
        if 'write_pp' in params:
            self._settings.set_write_pp(params['write_pp'])

        # Write direct solution of LBTE to hdf5 files
        if 'write_LBTE_solution' in params:
            self._settings.set_write_LBTE_solution(
                params['write_LBTE_solution'])
