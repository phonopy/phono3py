# Copyright (C) 2019 Atsushi Togo
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


def get_fc3(supercell,
            primitive,
            displacements,
            forces,
            fc_calculator=None,
            fc_calculator_options=None,
            is_compact_fc=False,
            log_level=0):
    """Supercell 2nd order force constants (fc2) are calculated.

    The expected shape of supercell fc3 to be returned is
        (len(atom_list), num_atoms, num_atom, 3, 3, 3),
    where atom_list is either all atoms in primitive cell or supercell,
    which is chosen by is_compact_fc=True for primitive cell and False for
    supercell.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell
    primitive : Primitive
        Primitive cell
    displacements : array_like
        Displacements of atoms in supercell.
        shape=(num_snapshots, num_atoms, 3), dtype='double', order='C'
    forces : array_like
        Forces of atoms in supercell.
        shape=(num_snapshots, num_atoms, 3), dtype='double', order='C'
    fc_calculator : str, optional
        Currently only 'alm' is supported. Default is None, meaning invoking
        'alm'.
    fc_calculator_options : str, optional
        This is arbitrary string.
    log_level : integer or bool, optional
        Verbosity level. False or 0 means quiet. True or 1 means normal level
        of log to stdout. 2 gives verbose mode.

    Returns
    -------
    fc3 : ndarray
        3rd order force constants.
        shape=(len(atom_list), num_atoms, num_atoms, 3, 3, 3)
        dtype='double', order='C'.
        Here atom_list is either all atoms in primitive cell or supercell,
        which is chosen by is_compact_fc=True for primitive cell and False
        for supercell.

    """

    if fc_calculator == 'alm':
        from phono3py.other.alm import get_fc3
        return get_fc3(supercell,
                       primitive,
                       displacements,
                       forces,
                       options=fc_calculator_options,
                       is_compact_fc=is_compact_fc,
                       log_level=log_level)
    else:
        msg = ("Force constants calculator of %s was not found ."
               % fc_calculator)
        raise RuntimeError(msg)


def get_displacements_and_forces_fc3(disp_dataset):
    """Returns displacements and forces from disp_dataset

    Note
    ----
    Dipslacements of all atoms in supercells for all displacement
    configurations in phono3py are returned, i.e., most of
    displacements are zero. Only the configurations with 'included' ==
    True are included in the list of indices that is returned, too.

    Parameters
    ----------
    disp_dataset : dict
        Displacement dataset that may be obtained by
        file_IO.parse_disp_fc3_yaml.

    Returns
    -------
    disp : ndarray
        Displacements of atoms in supercells of all displacement
        configurations.
        shape=(ndisp, natom, 3)
        dtype='double'
    indices : list of int
        The indices of the displacement configurations with 'included' == True.

    """

    if 'first_atoms' in disp_dataset:
        natom = disp_dataset['natom']
        ndisp = len(disp_dataset['first_atoms'])
        for disp1 in disp_dataset['first_atoms']:
            ndisp += len(disp1['second_atoms'])
        displacements = np.zeros((ndisp, natom, 3), dtype='double', order='C')
        forces = np.zeros_like(displacements)
        indices = []
        count = 0
        for disp1 in disp_dataset['first_atoms']:
            indices.append(count)
            displacements[count, disp1['number']] = disp1['displacement']
            forces[count] = disp1['forces']
            count += 1

        for disp1 in disp_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                if 'included' in disp2:
                    if disp2['included']:
                        indices.append(count)
                else:
                    indices.append(count)
                displacements[count, disp1['number']] = disp1['displacement']
                displacements[count, disp2['number']] = disp2['displacement']
                forces[count] = disp2['forces']
                count += 1
        return (np.array(displacements[indices], dtype='double', order='C'),
                np.array(forces[indices], dtype='double', order='C'))
    elif 'forces' in disp_dataset and 'displacements' in disp_dataset:
        return disp_dataset['displacements'], disp_dataset['forces']
    else:
        raise RuntimeError("disp_dataset doesn't contain correct information.")
