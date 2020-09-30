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

from phonopy.interface.phonopy_yaml import PhonopyYaml
import numpy as np


class Phono3pyYaml(PhonopyYaml):

    command_name = "phono3py"
    default_filenames = ("phono3py_disp.yaml", "phono3py.yaml")
    default_settings = {'force_sets': False,
                        'displacements': True,
                        'force_constants': False,
                        'born_effective_charge': True,
                        'dielectric_constant': True}

    def __init__(self,
                 configuration=None,
                 calculator=None,
                 physical_units=None,
                 settings=None):

        self.configuration = None
        self.calculator = None
        self.physical_units = None
        self.settings = None

        self.unitcell = None
        self.primitive = None
        self.supercell = None
        self.dataset = None
        self.supercell_matrix = None
        self.primitive_matrix = None
        self.nac_params = None
        self.force_constants = None

        self.symmetry = None  # symmetry of supercell
        self.s2p_map = None
        self.u2p_map = None
        self.frequency_unit_conversion_factor = None
        self.version = None

        #
        # phono3py only
        #
        # With DIM_FC2 given
        self.phonon_supercell_matrix = None
        self.phonon_dataset = None
        self.phonon_supercell = None
        self.phonon_primitive = None

        self._yaml = None

        super(Phono3pyYaml, self).__init__(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
            settings=settings)

    def set_phonon_info(self, phono3py):
        super(Phono3pyYaml, self).set_phonon_info(phono3py)
        self.phonon_supercell_matrix = phono3py.phonon_supercell_matrix
        self.phonon_dataset = phono3py.phonon_dataset
        self.phonon_primitive = phono3py.phonon_primitive
        self.phonon_supercell = phono3py.phonon_supercell

    def parse(self):
        super(Phono3pyYaml, self).parse()
        self._parse_fc3_dataset()

    def _parse_all_cells(self):
        """Parse all cells

        This method override PhonopyYaml._parse_all_cells.

        """

        super(Phono3pyYaml, self)._parse_all_cells()
        if 'phonon_primitive_cell' in self._yaml:
            self.phonon_primitive = self._parse_cell(
                self._yaml['phonon_primitive_cell'])
        if 'phonon_supercell' in self._yaml:
            self.phonon_supercell = self._parse_cell(
                self._yaml['phonon_supercell'])
        if 'phonon_supercell_matrix' in self._yaml:
            self.phonon_supercell_matrix = np.array(
                self._yaml['phonon_supercell_matrix'], dtype='intc', order='C')

    def _parse_dataset(self):
        """Parse phonon_dataset

        This method override PhonopyYaml._parse_dataset.

        """

        self.phonon_dataset = self._get_dataset(self.phonon_supercell)

    def _parse_fc3_dataset(self):
        """

        'duplicates' can be either dict (<v1.21) or list in phono3py.yaml.
        From v1.21, it was changed to list of list because
        dict with a key of int type is not allowed in JSON.

        """
        dataset = None
        if 'displacement_pairs' in self._yaml:
            disp = self._yaml['displacement_pairs'][0]
            if type(disp) is dict:  # type1
                dataset = self._parse_forces_fc3_type1(len(self.supercell))
            elif type(disp) is list:  # type2
                if 'displacement' in disp[0]:
                    dataset = self._parse_force_sets_type2()
        if 'displacement_pair_info' in self._yaml:
            info_yaml = self._yaml['displacement_pair_info']
            if 'cutoff_pair_distance' in info_yaml:
                dataset['cutoff_distance'] = info_yaml['cutoff_pair_distance']
            if 'duplicated_supercell_ids' in info_yaml:
                dataset['duplicates'] = info_yaml['duplicated_supercell_ids']
        self.dataset = dataset

    def _parse_forces_fc3_type1(self, natom):
        dataset = {'natom': natom, 'first_atoms': []}
        for d1 in self._yaml['displacement_pairs']:
            data1 = {
                'number': d1['atom'] - 1,
                'displacement': np.array(d1['displacement'], dtype='double'),
                'second_atoms': []}
            if 'forces' in d1:
                data1['forces'] = np.array(d1['forces'],
                                           dtype='double', order='C')
            d2_list = d1.get('paired_with')
            if d2_list is None:  # backward compatibility
                d2_list = d1.get('second_atoms')
            for d2 in d2_list:
                if 'forces' in d2:
                    data1['second_atoms'].append(
                        {'number': d2['atom'] - 1,
                         'displacement': np.array(d2['displacement'],
                                                  dtype='double'),
                         'forces': np.array(d2['forces'],
                                            dtype='double', order='C'),
                         'id': d2['displacement_id'],
                         'pair_distance': d2['pair_distance']})
                else:
                    disps = [{'number': d2['atom'] - 1,
                              'displacement': np.array(disp, dtype='double')}
                             for disp in d2['displacements']]
                    if 'pair_distance' in d2:
                        for d2_dict in disps:
                            d2_dict['pair_distance'] = d2['pair_distance']
                    if 'included' in d2:
                        for d2_dict in disps:
                            d2_dict['included'] = d2['included']
                    if 'displacement_ids' in d2:
                        for disp_id, d2_dict in zip(
                                d2['displacement_ids'], disps):
                            d2_dict['id'] = disp_id
                    data1['second_atoms'] += disps
            dataset['first_atoms'].append(data1)
        return dataset

    def _cell_info_yaml_lines(self):
        """Get YAML lines for information of cells

        This method override PhonopyYaml._cell_info_yaml_lines.

        """

        lines = super(Phono3pyYaml, self)._cell_info_yaml_lines()
        lines += self._supercell_matrix_yaml_lines(
            self.phonon_supercell_matrix, "phonon_supercell_matrix")
        lines += self._primitive_yaml_lines(self.phonon_primitive,
                                            "phonon_primitive_cell")
        lines += self._phonon_supercell_yaml_lines()
        return lines

    def _phonon_supercell_matrix_yaml_lines(self):
        lines = []
        if self.phonon_supercell_matrix is not None:
            lines.append("phonon_supercell_matrix:")
            for v in self.supercell_matrix:
                lines.append("- [ %3d, %3d, %3d ]" % tuple(v))
            lines.append("")
        return lines

    def _phonon_supercell_yaml_lines(self):
        lines = []
        if self.phonon_supercell is not None:
            s2p_map = getattr(self.phonon_primitive, 's2p_map', None)
            lines += self._cell_yaml_lines(
                self.phonon_supercell, "phonon_supercell", s2p_map)
            lines.append("")
        return lines

    def _nac_yaml_lines(self):
        """Get YAML lines for parameters of non-analytical term correction

        This method override PhonopyYaml._nac_yaml_lines.

        """

        if self.phonon_primitive is not None:
            return self._nac_yaml_lines_given_symbols(
                self.phonon_primitive.symbols)
        else:
            return self._nac_yaml_lines_given_symbols(
                self.primitive.symbols)

    def _displacements_yaml_lines(self, with_forces=False):
        """Get YAML lines for phonon_dataset and dataset.

        This method override PhonopyYaml._displacements_yaml_lines.
        PhonopyYaml._displacements_yaml_lines_2types is written
        to be also used by Phono3pyYaml.

        """

        lines = []
        if self.phonon_supercell_matrix is not None:
            lines += self._displacements_yaml_lines_2types(
                self.phonon_dataset, with_forces=with_forces)
        lines += self._displacements_yaml_lines_2types(
            self.dataset, with_forces=with_forces)
        return lines

    def _displacements_yaml_lines_type1(self, dataset, with_forces=False):
        """Get YAML lines for type1 phonon_dataset and dataset.

        This method override PhonopyYaml._displacements_yaml_lines_type1.
        PhonopyYaml._displacements_yaml_lines_2types calls
        Phono3pyYaml._displacements_yaml_lines_type1.

        """

        id_offset = len(dataset['first_atoms'])

        if 'second_atoms' in dataset['first_atoms'][0]:
            lines = ["displacement_pairs:"]
        else:
            lines = ["displacements:"]
        for i, d in enumerate(dataset['first_atoms']):
            lines.append("- atom: %4d" % (d['number'] + 1))
            lines.append("  displacement:")
            lines.append("    [ %19.16f, %19.16f, %19.16f ]"
                         % tuple(d['displacement']))
            id_num = i + 1
            if 'id' in d:
                assert id_num == d['id']
            lines.append("  displacement_id: %d" % id_num)
            if with_forces and 'forces' in d:
                lines.append("  forces:")
                for v in d['forces']:
                    lines.append(
                        "  - [ %19.16f, %19.16f, %19.16f ]" % tuple(v))
            if 'second_atoms' in d:
                ret_lines, id_offset = self._second_displacements_yaml_lines(
                    d['second_atoms'], id_offset, with_forces=with_forces)
                lines += ret_lines
        lines.append("")

        if 'second_atoms' in dataset['first_atoms'][0]:
            n_single = len(dataset['first_atoms'])
            n_pair = 0
            n_included = 0
            for d1 in dataset['first_atoms']:
                n_d2 = len(d1['second_atoms'])
                n_pair += n_d2
                for d2 in d1['second_atoms']:
                    if 'included' not in d2:
                        n_included += 1
                    elif d2['included']:
                        n_included += 1

            lines.append("displacement_pair_info:")
            if 'cutoff_distance' in dataset:
                lines.append("  cutoff_pair_distance: %11.8f"
                             % dataset['cutoff_distance'])
            lines.append("  number_of_singles: %d" % n_single)
            lines.append("  number_of_pairs: %d" % n_pair)
            if 'cutoff_distance' in dataset:
                lines.append("  number_of_pairs_in_cutoff: %d"
                             % n_included)

            # 'duplicates' is dict, but written as a list of list in yaml.
            # See the docstring of _parse_fc3_dataset for the reason.
            if 'duplicates' in dataset and dataset['duplicates']:
                lines.append("  duplicated_supercell_ids: "
                             "# 0 means perfect supercell")
                # Backward compatibility for dict type
                if type(dataset['duplicates']) is dict:
                    for i, j in dataset['duplicates'].items():
                        lines.append("  - [ %d, %d ]" % (int(i), j))
                else:
                    for (i, j) in dataset['duplicates']:
                        lines.append("  - [ %d, %d ]" % (i, j))
                lines.append("")

        return lines

    def _second_displacements_yaml_lines(self,
                                         dataset2,
                                         id_offset,
                                         with_forces=False):
        lines = []
        id_num = id_offset
        # lines.append("  second_atoms:")
        lines.append("  paired_with:")
        numbers = np.array([d['number'] for d in dataset2])
        unique_numbers = np.unique(numbers)
        for i in unique_numbers:
            indices_eq_i = np.sort(np.where(numbers == i)[0])
            if with_forces and 'forces' in dataset2[indices_eq_i[0]]:
                for j in indices_eq_i:
                    id_num += 1
                    lines.append("  - atom: %4d" % (i + 1))
                    lines.append("    pair_distance: %.8f"
                                 % dataset2[j]['pair_distance'])
                    lines.append("    displacement:")
                    lines.append("      [ %19.16f, %19.16f, %19.16f ]"
                                 % tuple(dataset2[j]['displacement']))

                    if 'id' in dataset2[j]:
                        assert dataset2[j]['id'] == id_num
                        lines.append("    displacement_id: %d" % id_num)

                    lines.append("    forces:")
                    for v in dataset2[j]['forces']:
                        lines.append(
                            "    - [ %19.16f, %19.16f, %19.16f ]" % tuple(v))
            else:
                lines.append("  - atom: %4d" % (i + 1))
                lines.append("    pair_distance: %.8f"
                             % dataset2[indices_eq_i[0]]['pair_distance'])
                if 'included' in dataset2[indices_eq_i[0]]:
                    included = dataset2[indices_eq_i[0]]['included']
                    lines.append("    included: %s"
                                 % ("true" if included else "false"))
                disp_ids = []
                lines.append("    displacements:")
                for j in indices_eq_i:
                    id_num += 1
                    d = tuple(dataset2[j]['displacement'])
                    lines.append("    - [ %19.16f, %19.16f, %19.16f ]" % d)
                    if 'id' in dataset2[j]:
                        assert dataset2[j]['id'] == id_num
                        disp_ids.append(dataset2[j]['id'])
                if disp_ids:
                    lines.append("    displacement_ids: [ %s ]"
                                 % ', '.join(["%d" % j for j in disp_ids]))

        return lines, id_num
