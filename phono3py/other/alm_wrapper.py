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
from phono3py.phonon3.fc3 import distribute_fc3
from phonopy.harmonic.force_constants import distribute_force_constants

def get_fc2(supercell,
            forces_fc2,
            disp_dataset,
            symmetry):
    natom = supercell.get_number_of_atoms()
    force = np.array(forces_fc2, dtype='double', order='C')
    disp = np.zeros_like(force)
    lattice = supercell.get_cell()
    positions = supercell.get_scaled_positions()
    numbers = supercell.get_atomic_numbers()
    _set_disp_fc2(disp, disp_dataset)
    pure_trans = _collect_pure_translations(symmetry)
    rotations = np.array([np.eye(3, dtype='intc')] * len(pure_trans),
                         dtype='intc', order='C')

    print("------------------------------"
          " ALM FC2 start "
          "------------------------------")

    from alm import ALM
    with ALM(lattice, positions, numbers, 1) as alm:
        alm.set_displacement_and_force(disp, force)
        alm.run_fitting()
        fc2_alm = alm.get_fc(1)

    print("-------------------------------"
          " ALM FC2 end "
          "-------------------------------")

    fc2 = _expand_fc2(fc2_alm, supercell, pure_trans, rotations)

    return fc2

def get_fc3(supercell,
            forces_fc3,
            disp_dataset,
            symmetry):
    natom = supercell.get_number_of_atoms()
    force = np.array(forces_fc3, dtype='double', order='C')
    disp = np.zeros_like(force)
    lattice = supercell.get_cell()
    positions = supercell.get_scaled_positions()
    numbers = supercell.get_atomic_numbers()
    indices = _set_disp_fc3(disp, disp_dataset)
    pure_trans = _collect_pure_translations(symmetry)
    rotations = np.array([np.eye(3, dtype='intc')] * len(pure_trans),
                         dtype='intc', order='C')

    print("------------------------------"
          " ALM FC3 start "
          "------------------------------")

    from alm import ALM
    with ALM(lattice, positions, numbers, 2) as alm:
        alm.set_displacement_and_force(disp[indices], force[indices])
        if 'cutoff_distance' in disp_dataset:
            cut_d = disp_dataset['cutoff_distance']
            nkd = len(np.unique(numbers))
            rcs = np.ones((2, nkd, nkd), dtype='double')
            rcs[0] *= -1
            rcs[1] *= cut_d
            alm.set_cutoff_radii(rcs)
        alm.run_fitting()
        fc2_alm = alm.get_fc(1)
        fc3_alm = alm.get_fc(2)

    print("-------------------------------"
          " ALM FC3 end "
          "-------------------------------")

    fc2 = _expand_fc2(fc2_alm, supercell, pure_trans, rotations)
    fc3 = _expand_fc3(fc3_alm, supercell, pure_trans, rotations)

    return fc2, fc3

def _set_disp_fc2(disp, disp_dataset):
    count = 0
    for disp1 in disp_dataset['first_atoms']:
        disp[count, disp1['number']] = disp1['displacement']
        count += 1

def _set_disp_fc3(disp, disp_dataset):
    indices = []
    count = 0
    for disp1 in disp_dataset['first_atoms']:
        indices.append(count)
        disp[count, disp1['number']] = disp1['displacement']
        count += 1

    for disp1 in disp_dataset['first_atoms']:
        for disp2 in disp1['second_atoms']:
            if 'included' in disp2:
                if disp2['included']:
                    indices.append(count)
            else:
                indices.append(count)
            disp[count, disp1['number']] = disp1['displacement']
            disp[count, disp2['number']] = disp2['displacement']
            count += 1

    assert count == len(disp)

    return indices
                                       
def _expand_fc2(fc2_alm, supercell, pure_trans, rotations, symprec=1e-5):
    natom = supercell.get_number_of_atoms()
    fc2 = np.zeros((natom, natom, 3, 3), dtype='double', order='C')
    (fc_values, elem_indices) = fc2_alm
    first_atoms = np.unique(elem_indices[:, 0] // 3)

    for (fc, indices) in zip(fc_values, elem_indices):
        v1 = indices[0] // 3
        c1 = indices[0] % 3
        v2 = indices[1] // 3
        c2 = indices[1] % 3
        fc2[v1, v2, c1, c2] = fc

    lattice = np.array(supercell.get_cell().T, dtype='double', order='C')
    positions = supercell.get_scaled_positions()
    distribute_force_constants(fc2,
                               range(natom),
                               first_atoms,
                               lattice,
                               positions,
                               rotations,
                               pure_trans,
                               symprec)
    return fc2

def _expand_fc3(fc3_alm, supercell, pure_trans, rotations, symprec=1e-5):
    natom = supercell.get_number_of_atoms()
    fc3 = np.zeros((natom, natom, natom, 3, 3, 3), dtype='double', order='C')
    (fc_values, elem_indices) = fc3_alm
    first_atoms = np.unique(elem_indices[:, 0] // 3)

    for (fc, indices) in zip(fc_values, elem_indices):
        v1 = indices[0] // 3
        c1 = indices[0] % 3
        v2 = indices[1] // 3
        c2 = indices[1] % 3
        v3 = indices[2] // 3
        c3 = indices[2] % 3
        if v2 == v3 and c2 == c3:
            fc3[v1, v2, v3, c1, c2, c3] = fc
        else:
            fc3[v1, v2, v3, c1, c2, c3] = fc
            fc3[v1, v3, v2, c1, c3, c2] = fc

    lattice = np.array(supercell.get_cell().T, dtype='double', order='C')
    positions = supercell.get_scaled_positions()
    distribute_fc3(fc3,
                   first_atoms,
                   lattice,
                   positions,
                   rotations,
                   pure_trans,
                   symprec,
                   overwrite=True,
                   verbose=True)
    return fc3

def _collect_pure_translations(symmetry):
    pure_trans = []
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    for r, t in zip(rotations, translations):
        if (r == np.eye(3, dtype='intc')).all():
            pure_trans.append(t)
    return np.array(pure_trans, dtype='double', order='C')
