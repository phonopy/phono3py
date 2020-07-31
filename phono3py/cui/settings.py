# Copyright (C) 2015 Atsushi Togo
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
from phonopy.cui.settings import Settings, ConfParser, fracval


class Phono3pySettings(Settings):
    _default = {
        # In micrometre. The default value is just set to avoid divergence.
        'boundary_mfp': 1.0e6,
        'coarse_mesh_shifts': None,
        'constant_averaged_pp_interaction': None,
        'cutoff_fc3_distance': None,
        'cutoff_pair_distance': None,
        'gamma_conversion_factor': None,
        'grid_addresses': None,
        'grid_points': None,
        'ion_clamped': False,
        'is_bterta': False,
        'is_compact_fc': False,
        'is_frequency_shift': False,
        'is_full_pp': False,
        'is_gruneisen': False,
        'is_imag_self_energy': False,
        'is_isotope': False,
        'is_joint_dos': False,
        'is_kappa_star': True,
        'is_lbte': False,
        'is_N_U': False,
        'is_reducible_collision_matrix': False,
        'is_symmetrize_fc2': False,
        'is_symmetrize_fc3_q': False,
        'is_symmetrize_fc3_r': False,
        'lapack_zheev_uplo': 'L',
        'mass_variances': None,
        'max_freepath': None,
        'mesh_divisors': None,
        'read_collision': None,
        'read_fc2': False,
        'read_fc3': False,
        'read_gamma': False,
        'read_phonon': False,
        'read_pp': False,
        'phonon_supercell_matrix': None,
        'pinv_cutoff': 1.0e-8,
        'pinv_solver': 0,
        'pp_conversion_factor': None,
        'scattering_event_class': None,  # scattering event class 1 or 2
        'sigma_cutoff_width': None,
        'solve_collective_phonon': False,
        'use_ave_pp': False,
        'write_collision': False,
        'write_gamma_detail': False,
        'write_gamma': False,
        'write_phonon': False,
        'write_pp': False,
        'write_LBTE_solution': False
    }

    def __init__(self, default=None):
        Settings.__init__(self)
        self._v.update(Phono3pySettings._default.copy())
        if default is not None:
            self._v.update(default)

    def set_boundary_mfp(self, val):
        self._v['boundary_mfp'] = val

    def set_coarse_mesh_shifts(self, val):
        self._v['coarse_mesh_shifts'] = val

    def set_constant_averaged_pp_interaction(self, val):
        self._v['constant_averaged_pp_interaction'] = val

    def set_cutoff_fc3_distance(self, val):
        self._v['cutoff_fc3_distance'] = val

    def set_cutoff_pair_distance(self, val):
        self._v['cutoff_pair_distance'] = val

    def set_gamma_conversion_factor(self, val):
        self._v['gamma_conversion_factor'] = val

    def set_grid_addresses(self, val):
        self._v['grid_addresses'] = val

    def set_grid_points(self, val):
        self._v['grid_points'] = val

    def set_ion_clamped(self, val):
        self._v['ion_clamped'] = val

    def set_is_bterta(self, val):
        self._v['is_bterta'] = val

    def set_is_compact_fc(self, val):
        self._v['is_compact_fc'] = val

    def set_is_frequency_shift(self, val):
        self._v['is_frequency_shift'] = val

    def set_is_full_pp(self, val):
        self._v['is_full_pp'] = val

    def set_is_gruneisen(self, val):
        self._v['is_gruneisen'] = val

    def set_is_imag_self_energy(self, val):
        self._v['is_imag_self_energy'] = val

    def set_is_isotope(self, val):
        self._v['is_isotope'] = val

    def set_is_joint_dos(self, val):
        self._v['is_joint_dos'] = val

    def set_is_kappa_star(self, val):
        self._v['is_kappa_star'] = val

    def set_is_lbte(self, val):
        self._v['is_lbte'] = val

    def set_is_N_U(self, val):
        self._v['is_N_U'] = val

    def set_is_reducible_collision_matrix(self, val):
        self._v['is_reducible_collision_matrix'] = val

    def set_is_symmetrize_fc2(self, val):
        self._v['is_symmetrize_fc2'] = val

    def set_is_symmetrize_fc3_q(self, val):
        self._v['is_symmetrize_fc3_q'] = val

    def set_is_symmetrize_fc3_r(self, val):
        self._v['is_symmetrize_fc3_r'] = val

    def set_lapack_zheev_uplo(self, val):
        self._v['lapack_zheev_uplo'] = val

    def set_mass_variances(self, val):
        self._v['mass_variances'] = val

    def set_max_freepath(self, val):
        self._v['max_freepath'] = val

    def set_mesh_divisors(self, val):
        self._v['mesh_divisors'] = val

    def set_phonon_supercell_matrix(self, val):
        self._v['phonon_supercell_matrix'] = val

    def set_pinv_cutoff(self, val):
        self._v['pinv_cutoff'] = val

    def set_pinv_solver(self, val):
        self._v['pinv_solver'] = val

    def set_pp_conversion_factor(self, val):
        self._v['pp_conversion_factor'] = val

    def set_read_collision(self, val):
        self._v['read_collision'] = val

    def set_read_fc2(self, val):
        self._v['read_fc2'] = val

    def set_read_fc3(self, val):
        self._v['read_fc3'] = val

    def set_read_gamma(self, val):
        self._v['read_gamma'] = val

    def set_read_phonon(self, val):
        self._v['read_phonon'] = val

    def set_read_pp(self, val):
        self._v['read_pp'] = val

    def set_scattering_event_class(self, val):
        self._v['scattering_event_class'] = val

    def set_sigma_cutoff_width(self, val):
        self._v['sigma_cutoff_width'] = val

    def set_solve_collective_phonon(self, val):
        self._v['solve_collective_phonon'] = val

    def set_use_ave_pp(self, val):
        self._v['use_ave_pp'] = val

    def set_write_collision(self, val):
        self._v['write_collision'] = val

    def set_write_gamma_detail(self, val):
        self._v['write_gamma_detail'] = val

    def set_write_gamma(self, val):
        self._v['write_gamma'] = val

    def set_write_phonon(self, val):
        self._v['write_phonon'] = val

    def set_write_pp(self, val):
        self._v['write_pp'] = val

    def set_write_LBTE_solution(self, val):
        self._v['write_LBTE_solution'] = val


class Phono3pyConfParser(ConfParser):
    def __init__(self, filename=None, args=None, default_settings=None):
        self._settings = Phono3pySettings(default=default_settings)
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
        ConfParser.read_options(self)  # store data in self._confs
        if 'phonon_supercell_dimension' in self._args:
            dim_fc2 = self._args.phonon_supercell_dimension
            if dim_fc2 is not None:
                self._confs['dim_fc2'] = " ".join(dim_fc2)

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

        if 'lapack_zheev_uplo' in self._args:
            if self._args.lapack_zheev_uplo is not None:
                self._confs['lapack_zheev_uplo'] = self._args.lapack_zheev_uplo

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
        ConfParser.parse_conf(self)
        confs = self._confs

        for conf_key in confs.keys():
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
                if confs['full_pp'].lower() == '.true.':
                    self.set_parameter('is_full_pp', True)
                elif confs['full_pp'].lower() == '.false.':
                    self.set_parameter('is_full_pp', False)

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
                if confs['ion_clamped'].lower() == '.true.':
                    self.set_parameter('ion_clamped', True)
                elif confs['ion_clamped'].lower() == '.false.':
                    self.set_parameter('ion_clamped', False)

            if conf_key == 'bterta':
                if confs['bterta'].lower() == '.true.':
                    self.set_parameter('is_bterta', True)
                elif confs['bterta'].lower() == '.false.':
                    self.set_parameter('is_bterta', False)

            if conf_key == 'compact_fc':
                if confs['compact_fc'].lower() == '.true.':
                    self.set_parameter('is_compact_fc', True)
                elif confs['compact_fc'].lower() == '.false.':
                    self.set_parameter('is_compact_fc', False)

            if conf_key == 'frequency_shift':
                if confs['frequency_shift'].lower() == '.true.':
                    self.set_parameter('is_frequency_shift', True)
                elif confs['frequency_shift'].lower() == '.false.':
                    self.set_parameter('is_frequency_shift', False)

            if conf_key == 'gruneisen':
                if confs['gruneisen'].lower() == '.true.':
                    self.set_parameter('is_gruneisen', True)
                elif confs['gruneisen'].lower() == '.false.':
                    self.set_parameter('is_gruneisen', False)

            if conf_key == 'imag_self_energy':
                if confs['imag_self_energy'].lower() == '.true.':
                    self.set_parameter('is_imag_self_energy', True)
                elif confs['imag_self_energy'].lower() == '.false.':
                    self.set_parameter('is_imag_self_energy', False)

            if conf_key == 'isotope':
                if confs['isotope'].lower() == '.true.':
                    self.set_parameter('is_isotope', True)
                elif confs['isotope'].lower() == '.false.':
                    self.set_parameter('is_isotope', False)

            if conf_key == 'joint_dos':
                if confs['joint_dos'].lower() == '.true.':
                    self.set_parameter('is_joint_dos', True)
                elif confs['joint_dos'].lower() == '.false.':
                    self.set_parameter('is_joint_dos', False)

            if conf_key == 'lapack_zheev_uplo':
                self.set_parameter('lapack_zheev_uplo',
                                   confs['lapack_zheev_uplo'].upper())

            if conf_key == 'lbte':
                if confs['lbte'].lower() == '.true.':
                    self.set_parameter('is_lbte', True)
                elif confs['lbte'].lower() == '.false.':
                    self.set_parameter('is_lbte', False)

            if conf_key == 'N_U':
                if confs['N_U'].lower() == '.true.':
                    self.set_parameter('is_N_U', True)
                elif confs['N_U'].lower() == '.false.':
                    self.set_parameter('is_N_U', False)

            if conf_key == 'reducible_collision_matrix':
                if confs['reducible_collision_matrix'].lower() == '.true.':
                    self.set_parameter('is_reducible_collision_matrix', True)
                elif confs['reducible_collision_matrix'].lower() == '.false.':
                    self.set_parameter('is_reducible_collision_matrix', False)

            if conf_key == 'symmetrize_fc2':
                if confs['symmetrize_fc2'].lower() == '.true.':
                    self.set_parameter('is_symmetrize_fc2', True)
                elif confs['symmetrize_fc2'].lower() == '.false.':
                    self.set_parameter('is_symmetrize_fc2', False)

            if conf_key == 'symmetrize_fc3_q':
                if confs['symmetrize_fc3_q'].lower() == '.true.':
                    self.set_parameter('is_symmetrize_fc3_q', True)
                elif confs['symmetrize_fc3_q'].lower() == '.false.':
                    self.set_parameter('is_symmetrize_fc3_q', False)

            if conf_key == 'symmetrize_fc3_r':
                if confs['symmetrize_fc3_r'].lower() == '.true.':
                    self.set_parameter('is_symmetrize_fc3_r', True)
                elif confs['symmetrize_fc3_r'].lower() == '.false.':
                    self.set_parameter('is_symmetrize_fc3_r', False)

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
                if confs['kappa_star'].lower() == '.true.':
                    self.set_parameter('is_kappa_star', True)
                elif confs['kappa_star'].lower() == '.false.':
                    self.set_parameter('is_kappa_star', False)

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
                if confs['read_fc2'].lower() == '.true.':
                    self.set_parameter('read_fc2', True)
                elif confs['read_fc2'].lower() == '.false.':
                    self.set_parameter('read_fc2', False)

            if conf_key == 'read_fc3':
                if confs['read_fc3'].lower() == '.true.':
                    self.set_parameter('read_fc3', True)
                elif confs['read_fc3'].lower() == '.false.':
                    self.set_parameter('read_fc3', False)

            if conf_key == 'read_gamma':
                if confs['read_gamma'].lower() == '.true.':
                    self.set_parameter('read_gamma', True)
                elif confs['read_gamma'].lower() == '.false.':
                    self.set_parameter('read_gamma', False)

            if conf_key == 'read_phonon':
                if confs['read_phonon'].lower() == '.true.':
                    self.set_parameter('read_phonon', True)
                elif confs['read_phonon'].lower() == '.false.':
                    self.set_parameter('read_phonon', False)

            if conf_key == 'read_pp':
                if confs['read_pp'].lower() == '.true.':
                    self.set_parameter('read_pp', True)
                elif confs['read_pp'].lower() == '.false.':
                    self.set_parameter('read_pp', False)

            if conf_key == 'scattering_event_class':
                self.set_parameter('scattering_event_class',
                                   confs['scattering_event_class'])

            if conf_key == 'sigma_cutoff_width':
                self.set_parameter('sigma_cutoff_width',
                                   float(confs['sigma_cutoff_width']))

            if conf_key == 'collective_phonon':
                if confs['collective_phonon'].lower() == '.true.':
                    self.set_parameter('collective_phonon', True)
                elif confs['collective_phonon'].lower() == '.false.':
                    self.set_parameter('collective_phonon', False)

            if conf_key == 'use_ave_pp':
                if confs['use_ave_pp'].lower() == '.true.':
                    self.set_parameter('use_ave_pp', True)
                elif confs['use_ave_pp'].lower() == '.false.':
                    self.set_parameter('use_ave_pp', False)

            if conf_key == 'write_gamma_detail':
                if confs['write_gamma_detail'].lower() == '.true.':
                    self.set_parameter('write_gamma_detail', True)
                elif confs['write_gamma_detail'].lower() == '.false.':
                    self.set_parameter('write_gamma_detail', False)

            if conf_key == 'write_gamma':
                if confs['write_gamma'].lower() == '.true.':
                    self.set_parameter('write_gamma', True)
                elif confs['write_gamma'].lower() == '.false.':
                    self.set_parameter('write_gamma', False)

            if conf_key == 'write_collision':
                if confs['write_collision'].lower() == '.true.':
                    self.set_parameter('write_collision', True)
                elif confs['write_collision'].lower() == '.false.':
                    self.set_parameter('write_collision', False)

            if conf_key == 'write_phonon':
                if confs['write_phonon'].lower() == '.true.':
                    self.set_parameter('write_phonon', True)
                elif confs['write_phonon'].lower() == '.false.':
                    self.set_parameter('write_phonon', False)

            if conf_key == 'write_pp':
                if confs['write_pp'].lower() == '.true.':
                    self.set_parameter('write_pp', True)
                elif confs['write_pp'].lower() == '.false.':
                    self.set_parameter('write_pp', False)

            if conf_key == 'write_LBTE_solution':
                if confs['write_LBTE_solution'].lower() == '.true.':
                    self.set_parameter('write_LBTE_solution', True)
                elif confs['write_LBTE_solution'].lower() == '.false.':
                    self.set_parameter('write_LBTE_solution', False)

    def _set_settings(self):
        self.set_settings()
        params = self._parameters

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
            self._settings.set_cutoff_fc3_distance(
                params['cutoff_fc3_distance'])

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
            self._settings.set_is_imag_self_energy(
                params['is_imag_self_energy'])

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
            self._settings.set_is_symmetrize_fc3_q(
                params['is_symmetrize_fc3_q'])

        # Symmetrize fc3 by index exchange
        if 'is_symmetrize_fc3_r' in params:
            self._settings.set_is_symmetrize_fc3_r(
                params['is_symmetrize_fc3_r'])

        if 'lapack_zheev_uplo' in params:
            self._settings.set_lapack_zheev_uplo(params['lapack_zheev_uplo'])

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
            self._settings.set_pp_conversion_factor(
                params['pp_conversion_factor'])

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
