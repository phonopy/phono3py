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

from phonopy.interface.phonopy_yaml import PhonopyYaml
import numpy as np


class Phono3pyYaml(PhonopyYaml):

    command_name = "phono3py"
    default_filenames = ("phono3py.yaml", )

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
        #
        # same as self.supercell unless DIM_FC2 given
        #
        self.phonon_supercell = None
        self.phonon_primitive = None

        self._yaml = None

        super(Phono3pyYaml, self).__init__(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
            settings=settings)

    def __str__(self):
        lines = self.get_yaml_lines()
        if self.phonon_supercell_matrix is not None:
            i = lines.index("supercell_matrix:")
            i += 5
            lines.insert(i, "phonon_supercell_matrix:")
            for v in self.phonon_supercell_matrix:
                i += 1
                lines.insert(i, "- [ %3d, %3d, %3d ]" % tuple(v))
            i += 1
            lines.insert(i, "")
        return "\n".join(lines)

    def set_phonon_info(self, phono3py):
        super(Phono3pyYaml, self).set_phonon_info(phono3py)
        self.phonon_supercell_matrix = phono3py.phonon_supercell_matrix
        self.phonon_dataset = phono3py.phonon_dataset
        self.phonon_primitive = phono3py.phonon_primitive
        self.phonon_supercell = phono3py.phonon_supercell

    def _load(self, fp):
        super(Phono3pyYaml, self)._load(fp)
        if 'phonon_supercell_matrix' in self._yaml:
            self.phonon_supercell_matrix = np.array(
                self._yaml['phonon_supercell_matrix'],
                dtype='intc', order='C')

    def _displacements_yaml_lines_type1(self, with_forces=False):
        id_offset = len(self.dataset['first_atoms'])
        lines = []
        if 'second_atoms' in self.dataset['first_atoms'][0]:
            lines.append("displacement_pairs:")
        else:
            lines.append("displacements:")

        for i, d in enumerate(self.dataset['first_atoms']):
            lines.append("- atom: %4d" % (d['number'] + 1))
            lines.append("  displacement:")
            lines.append("    [ %19.16f, %19.16f, %19.16f ]"
                         % tuple(d['displacement']))
            if with_forces and 'forces' in d:
                lines.append("  forces:")
                for f in d['forces']:
                    lines.append("  - [ %19.16f, %19.16f, %19.16f ]" % tuple(f))
            if 'second_atoms' in d:
                lines += self._second_displacements_yaml_lines(
                    d['second_atoms'], id_offset, with_forces=with_forces)
        lines.append("")

        if 'second_atoms' in self.dataset['first_atoms'][0]:
            if 'duplicates' in self.dataset and self.dataset['duplicates']:
                lines.append("displacement_pair_duplicates:")
                for i in self.dataset['duplicates']:
                    # id-i and id-j give the same displacement pairs.
                    j = self.dataset['duplicates'][i]
                    lines.append("- %d : %d"
                                 % (i + id_offset + 1, j + id_offset + 1))

                lines.append("")
        return lines

    def _second_displacements_yaml_lines(self,
                                         dataset2,
                                         id_offset,
                                         with_forces=False):
        lines = []
        lines.append("  second_atoms:")
        numbers = np.array([d['number'] for d in dataset2])
        unique_numbers = np.unique(numbers)
        for i in unique_numbers:
            indices_eq_i = np.sort(np.where(numbers == i)[0])
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
                d = tuple(dataset2[j]['displacement'])
                lines.append("    - [ %19.16f, %19.16f, %19.16f ]" % d)
                disp_ids.append(dataset2[j]['id'] + id_offset + 1)
            lines.append("    displacement_ids: [ %s ]"
                         % ', '.join(["%d" % j for j in disp_ids]))

        return lines
