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
    def __init__(self,
                 configuration=None,
                 calculator=None,
                 physical_units=None):
        self._configuration = None
        self._calculator = None
        self._physical_units = None
        self._show_force_constants = None  # to be False
        self._show_displacements = None  # to be False
        self._settings = {}

        PhonopyYaml.__init__(self,
                             configuration=configuration,
                             calculator=calculator,
                             physical_units=physical_units)

        # Written in self.set_phonon_info
        self.unitcell = None
        self.primitive = None
        self.supercell = None
        self.yaml = None
        self.supercell_matrix = None
        self.phonon_supercell_matrix = None
        self.primitive_matrix = None
        self.nac_params = None
        self._supercell_matrix = None
        self._phonon_supercell_matrix = None
        self._primitive_matrix = None
        self._s2p_map = None
        self._u2p_map = None
        self._nac_params = None
        self._version = None

        # Overwrite this
        self._command_name = "phono3py"
        for key in self._settings:
            self._settings[key] = False
        self._settings['born_effective_charge'] = True
        self._settings['dielectric_constant'] = True

    def set_phonon_info(self, phono3py):
        super(Phono3pyYaml, self).set_phonon_info(phono3py)
        self._phonon_supercell_matrix = phono3py.phonon_supercell_matrix

    def _load(self, fp):
        super(Phono3pyYaml, self)._load(fp)
        if 'phonon_supercell_matrix' in self.yaml:
            self.phonon_supercell_matrix = np.array(
                self.yaml['phonon_supercell_matrix'],
                dtype='intc', order='C')

    def __str__(self):
        lines = self.get_yaml_lines()
        if self._phonon_supercell_matrix is not None:
            i = lines.index("supercell_matrix:")
            i += 5
            lines.insert(i, "phonon_supercell_matrix:")
            for v in self._phonon_supercell_matrix:
                i += 1
                lines.insert(i, "- [ %3d, %3d, %3d ]" % tuple(v))
            i += 1
            lines.insert(i, "")
        return "\n".join(lines)
