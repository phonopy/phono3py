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

import sys
import numpy as np
from phonopy.interface.alm import extract_fc2_from_alm


def get_fc3(supercell,
            primitive,
            symmetry,
            displacements,
            forces,
            fc_calculator=None,
            fc_calculator_options=None,
            is_compact_fc=False,
            log_level=0):
    _alm_options = {}
    if fc_calculator_options is not None:
        _alm_options.update(fc_calculator_options)

    if is_compact_fc:
        p2s_map = primitive.p2s_map
        p2p_map = primitive.p2p_map
    else:
        p2s_map = None
        p2p_map = None

    if log_level:
        print("------------------------------"
              " ALM FC3 start "
              "------------------------------")
        print("ALM by T. Tadano, https://github.com/ttadano/ALM")
        if _alm_options:
            print("Settings:")
            for key, value in _alm_options.items():
                print("  %s: %s" % (key, value))
        if log_level == 1:
            print("Use -v option to watch detailed ALM log.")
        if log_level > 1:
            print("")
        sys.stdout.flush()

    fc2, fc3 = optimize(supercell.cell.T,
                        supercell.scaled_positions,
                        supercell.numbers,
                        displacements,
                        forces,
                        alm_options=_alm_options,
                        p2s_map=p2s_map,
                        p2p_map=p2p_map,
                        log_level=log_level)

    if log_level:
        print("-------------------------------"
              " ALM FC3 end "
              "-------------------------------")

    return fc2, fc3


def optimize(lattice,
             positions,
             numbers,
             displacements,
             forces,
             alm_options=None,
             p2s_map=None,
             p2p_map=None,
             log_level=0):
    """Calculate force constants

    lattice : array_like
        Basis vectors. a, b, c are given as column vectors.
        shape=(3, 3), dtype='double'
    positions : array_like
        Fractional coordinates of atomic points.
        shape=(num_atoms, 3), dtype='double'
    numbers : array_like
        Atomic numbers.
        shape=(num_atoms,), dtype='intc'
    displacements: ndarray
        Displacemetns of all atoms in supercells.
        shape=(supercells, atoms in supercell, 3)
        dtype='double', order='C',
    forces: ndarray
        Forces of all atoms in supercells.
        shape=(supercells, atoms in supercell, 3)}
        dtype='double', order='C'
    alm_options : dict, optional
        Default is None.
        List of keys
            cutoff_distance : float
            solver : str
                Either 'SimplicialLDLT' or 'dense'. Default is
                'SimplicialLDLT'.
            debiase_after_l1opt : int
            cross_validation : int
            l1_alpha : float,
            l1_alpha_min : float,
            l1_alpha_max : float,
            num_l1_alpha : int,
            l1_ratio': float,
            linear_model : int,
            ndata : int,
            output_filename_prefix : str

    """

    from alm import ALM
    with ALM(lattice, positions, numbers) as alm:
        options = {key.lower(): alm_options[key] for key in alm_options}
        nkd = len(np.unique(numbers))
        if 'cutoff_distance' not in options:
            rcs = -np.ones((2, nkd, nkd), dtype='double')
        elif type(options['cutoff_distance']) is float:
            rcs = np.ones((2, nkd, nkd), dtype='double')
            rcs[0] *= -1
            rcs[1] *= options['cutoff_distance']
            del options['cutoff_distance']

        if 'ndata' in options:
            ndata = options['ndata']
            del options['ndata']
        else:
            ndata = len(displacements)

        if 'solver' in options:
            solver = options['solver']
            del options['solver']
        else:
            solver = 'SimplicialLDLT'

        if 'output_filename_prefix' in options:
            output_filename_prefix = options['output_filename_prefix']
            del options['output_filename_prefix']
        else:
            output_filename_prefix = None

        if 'cross_validation' in options and options['cross_validation'] > 0:
            if 'linear_model' not in options:
                options['linear_model'] = 2
            elif 'linear_model' in options and options['linear_model'] != 2:
                options['linear_model'] = 2
                if log_level:
                    for key in alm_options:
                        if key.lower() == 'linear_model':
                            print("%s was set to 2 to run cross validation.")

        alm.set_verbosity((log_level > 1) * 1)
        if output_filename_prefix:
            alm.set_output_filename_prefix(output_filename_prefix)
        alm.define(2, rcs)
        alm.set_displacement_and_force(displacements[:ndata], forces[:ndata])

        if 'linear_model' in options and options['linear_model'] == 2:
            if 'cross_validation' in options:
                if options['cross_validation'] > 0:
                    alm.set_optimizer_control(options)
                    alm.optimize(solver=solver)
                    options['cross_validation'] = 0
                    options['l1_alpha'] = alm.get_cv_l1_alpha()
            alm.set_optimizer_control(options)
            alm.optimize(solver=solver)
        else:
            alm.optimize(solver=solver)

        natom = len(numbers)
        fc2 = extract_fc2_from_alm(alm,
                                   natom,
                                   atom_list=p2s_map,
                                   p2s_map=p2s_map,
                                   p2p_map=p2p_map)
        fc3 = _extract_fc3_from_alm(alm,
                                    natom,
                                    p2s_map=p2s_map,
                                    p2p_map=p2p_map)
        return fc2, fc3


def _extract_fc3_from_alm(alm,
                          natom,
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
        return disp_dataset['displacemens'], disp_dataset['forces']
    else:
        raise RuntimeError("disp_dataset doesn't contain correct information.")
