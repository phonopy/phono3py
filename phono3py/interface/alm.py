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

import numpy as np
from phonopy.interface.alm import run_alm


def get_fc3(supercell,
            primitive,
            displacements,
            forces,
            options=None,
            is_compact_fc=False,
            log_level=0):
    return run_alm(supercell,
                   primitive,
                   displacements,
                   forces,
                   2,
                   is_compact_fc=is_compact_fc,
                   options=options,
                   log_level=log_level)


def _extract_fc3_from_alm(alm,
                          natom,
                          is_compact_fc,
                          p2s_map=None,
                          p2p_map=None):
    p2s_map_alm = alm.getmap_primitive_to_supercell()[0]
    if (p2s_map is not None and
        len(p2s_map_alm) == len(p2s_map) and
        (p2s_map_alm == p2s_map).all()):
        fc3 = np.zeros((len(p2s_map), natom, natom, 3, 3, 3),
                       dtype='double', order='C')
        for (fc, indices) in zip(*alm.get_fc(2, mode='origin')):
            v1, v2, v3 = indices // 3
            c1, c2, c3 = indices % 3
            fc3[p2p_map[v1], v2, v3, c1, c2, c3] = fc
            fc3[p2p_map[v1], v3, v2, c1, c3, c2] = fc
    else:
        fc3 = np.zeros((natom, natom, natom, 3, 3, 3),
                       dtype='double', order='C')
        for (fc, indices) in zip(*alm.get_fc(2, mode='all')):
            v1, v2, v3 = indices // 3
            c1, c2, c3 = indices % 3
            fc3[v1, v2, v3, c1, c2, c3] = fc
            fc3[v1, v3, v2, c1, c3, c2] = fc

    return fc3
