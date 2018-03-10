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

import sys
import numpy as np
from phonopy.harmonic.force_constants import (
    distribute_force_constants,
    show_drift_force_constants,
    symmetrize_force_constants,
    symmetrize_compact_force_constants)
from phono3py.phonon3.fc3 import show_drift_fc3
from phono3py.file_IO import (parse_disp_fc3_yaml,
                              parse_disp_fc2_yaml,
                              parse_FORCES_FC2,
                              parse_FORCES_FC3,
                              read_fc3_from_hdf5,
                              read_fc2_from_hdf5,
                              write_fc3_to_hdf5,
                              write_fc2_to_hdf5)
from phono3py.cui.show_log import (show_phono3py_force_constants_settings,
                                   print_error, file_exists)
from phono3py.phonon3.fc3 import (set_permutation_symmetry_fc3,
                                  set_translational_invariance_fc3)


def create_phono3py_force_constants(phono3py,
                                    phonon_supercell_matrix,
                                    settings,
                                    force_to_eVperA=None,
                                    distance_to_A=None,
                                    input_filename=None,
                                    output_filename=None,
                                    log_level=1):
    read_fc3 = settings.get_read_fc3()
    read_fc2 = settings.get_read_fc2()
    symmetrize_fc3r = (settings.get_is_symmetrize_fc3_r() or
                       settings.get_fc_symmetry())
    symmetrize_fc3q = settings.get_is_symmetrize_fc3_q()
    symmetrize_fc2 = (settings.get_is_symmetrize_fc2() or
                      settings.get_fc_symmetry())

    if log_level:
        show_phono3py_force_constants_settings(read_fc3,
                                               read_fc2,
                                               symmetrize_fc3r,
                                               symmetrize_fc3q,
                                               symmetrize_fc2,
                                               settings)

    # fc3
    if (settings.get_is_joint_dos() or
        (settings.get_is_isotope() and
         not (settings.get_is_bterta() or settings.get_is_lbte())) or
        settings.get_read_gamma() or
        settings.get_read_pp() or
        settings.get_write_phonon() or
        settings.get_constant_averaged_pp_interaction() is not None):
        pass
    else:
        if read_fc3: # Read fc3.hdf5
            if input_filename is None:
                filename = 'fc3.hdf5'
            else:
                filename = 'fc3.' + input_filename + '.hdf5'
            file_exists(filename, log_level)
            if log_level:
                print("Reading fc3 from %s" % filename)
            fc3 = read_fc3_from_hdf5(filename=filename)
            num_atom = phono3py.get_supercell().get_number_of_atoms()
            if fc3.shape[0] != num_atom:
                print("Matrix shape of fc3 doesn't agree with supercell size.")
                if log_level:
                    print_error()
                sys.exit(1)

            if symmetrize_fc3r:
                set_translational_invariance_fc3(fc3)
                set_permutation_symmetry_fc3(fc3)

            phono3py.set_fc3(fc3)
        else: # fc3 from FORCES_FC3
            if not _create_phono3py_fc3(phono3py,
                                        force_to_eVperA,
                                        distance_to_A,
                                        symmetrize_fc3r,
                                        symmetrize_fc2,
                                        settings.get_cutoff_fc3_distance(),
                                        input_filename,
                                        output_filename,
                                        settings.get_use_alm_fc3(),
                                        log_level):
                    print("fc3 was not created properly.")
                    if log_level:
                        print_error()
                    sys.exit(1)
        if log_level:
            show_drift_fc3(phono3py.get_fc3())

    # fc2
    phonon_primitive = phono3py.get_phonon_primitive()
    phonon_supercell = phono3py.get_phonon_supercell()
    phonon_symmetry = phono3py.get_phonon_supercell_symmetry()
    p2s_map = phonon_primitive.get_primitive_to_supercell_map()
    s2p_map = phonon_primitive.get_supercell_to_primitive_map()
    if read_fc2:
        if input_filename is None:
            filename = 'fc2.hdf5'
        else:
            filename = 'fc2.' + input_filename + '.hdf5'
        file_exists(filename, log_level)
        if log_level:
            print("Reading fc2 from %s" % filename)

        num_atom = phonon_supercell.get_number_of_atoms()
        phonon_fc2 = read_fc2_from_hdf5(filename=filename,
                                        p2s_map=p2s_map)
        if phonon_fc2.shape[1] != num_atom:
            print("Matrix shape of fc2 doesn't agree with supercell size.")
            if log_level:
                print_error()
            sys.exit(1)

        if symmetrize_fc2:
            if phonon_fc2.shape[0] == phonon_fc2.shape[1]:
                symmetrize_force_constants(phonon_fc2)
            else:
                symmetrize_compact_force_constants(phonon_fc2,
                                                   phonon_supercell,
                                                   phonon_symmetry,
                                                   s2p_map,
                                                   p2s_map)
        phono3py.set_fc2(phonon_fc2)
    else:
        if log_level:
            print("Solving fc2")

        if phonon_supercell_matrix is None:
            if not _create_phono3py_fc2(phono3py,
                                        force_to_eVperA,
                                        distance_to_A,
                                        symmetrize_fc2,
                                        input_filename,
                                        settings.get_use_alm_fc2(),
                                        log_level):
                print("fc2 was not created properly.")
                if log_level:
                    print_error()
                sys.exit(1)
        else:
            if not _create_phono3py_phonon_fc2(phono3py,
                                               force_to_eVperA,
                                               distance_to_A,
                                               symmetrize_fc2,
                                               input_filename,
                                               settings.get_use_alm_fc2(),
                                               log_level):
                    print("fc2 was not created properly.")
                    if log_level:
                        print_error()
                    sys.exit(1)
        if output_filename is None:
            filename = 'fc2.hdf5'
        else:
            filename = 'fc2.' + output_filename + '.hdf5'
        if log_level:
            print("Writing fc2 to %s" % filename)
        write_fc2_to_hdf5(phono3py.get_fc2(),
                          filename=filename,
                          p2s_map=p2s_map)

    if log_level:
        fc_orig = phono3py.get_fc2()
        if fc_orig.shape[0] == fc_orig.shape[1]:
            show_drift_force_constants(fc_orig, name='fc2')
        else:
            if fc_orig.shape[1] > 2000:
                show_drift_force_constants(fc_orig, name='fc2')
                print("  ** only the 2nd drift value is meaningful **")
            else:
                p2s_map = phonon_primitive.get_primitive_to_supercell_map()
                lattice = np.array(phonon_supercell.get_cell().T,
                                   dtype='double', order='C')
                positions = phonon_supercell.get_scaled_positions()
                rotations = phonon_symmetry.get_symmetry_operations()['rotations']
                trans = phonon_symmetry.get_symmetry_operations()['translations']
                symprec = phonon_symmetry.get_symmetry_tolerance()
                n_satom = phonon_supercell.get_number_of_atoms()
                fc = np.zeros((n_satom, n_satom, 3, 3), dtype='double', order='C')
                for i_p, i_s in enumerate(p2s_map):
                    fc[i_s] = fc_orig[i_p]
                distribute_force_constants(fc,
                                           np.arange(n_satom, dtype='intc'),
                                           p2s_map,
                                           lattice,
                                           positions,
                                           rotations,
                                           trans,
                                           symprec)
                show_drift_force_constants(fc, name='fc2')

def _create_phono3py_fc3(phono3py,
                         force_to_eVperA,
                         distance_to_A,
                         symmetrize_fc3r,
                         symmetrize_fc2,
                         cutoff_distance,
                         input_filename,
                         output_filename,
                         use_alm,
                         log_level):
    if input_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + input_filename + '.yaml'
    file_exists(filename, log_level)
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    disp_dataset = parse_disp_fc3_yaml(filename=filename)
    num_atom = phono3py.get_supercell().get_number_of_atoms()
    if disp_dataset['natom'] != num_atom:
        print("Number of atoms in supercell is not consistent with %s" %
              filename)
        if log_level:
            print_error()
        sys.exit(1)
    _convert_displacement_unit(disp_dataset, distance_to_A)

    file_exists("FORCES_FC3", log_level)
    if log_level:
        print("Sets of supercell forces are read from %s." % "FORCES_FC3")
    forces_fc3 = parse_FORCES_FC3(disp_dataset)
    if not forces_fc3:
        return False

    _convert_force_unit(forces_fc3, force_to_eVperA)

    phono3py.produce_fc3(
        forces_fc3,
        displacement_dataset=disp_dataset,
        cutoff_distance=cutoff_distance,
        symmetrize_fc3r=symmetrize_fc3r,
        use_alm=use_alm)
    if output_filename is None:
        filename = 'fc3.hdf5'
    else:
        filename = 'fc3.' + output_filename + '.hdf5'
    if log_level:
        print("Writing fc3 to %s" % filename)
    write_fc3_to_hdf5(phono3py.get_fc3(), filename=filename)

    return True

def _create_phono3py_fc2(phono3py,
                         force_to_eVperA,
                         distance_to_A,
                         symmetrize_fc2,
                         input_filename,
                         use_alm,
                         log_level):
    if input_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + input_filename + '.yaml'
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    file_exists(filename, log_level)
    disp_dataset = parse_disp_fc3_yaml(filename=filename)
    num_atom = phono3py.get_supercell().get_number_of_atoms()
    if disp_dataset['natom'] != num_atom:
        print("Number of atoms in supercell is not consistent with %s" %
              filename)
        if log_level:
            print_error()
        sys.exit(1)
    _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=True)

    if log_level:
        print("Sets of supercell forces are read from %s." % "FORCES_FC3")
    file_exists("FORCES_FC3", log_level)
    forces_fc2 = parse_FORCES_FC2(disp_dataset, filename="FORCES_FC3")
    if not forces_fc2:
        return False

    _convert_force_unit(forces_fc2, force_to_eVperA)

    phono3py.produce_fc2(
        forces_fc2,
        displacement_dataset=disp_dataset,
        symmetrize_fc2=symmetrize_fc2,
        use_alm=use_alm)

    return True

def _create_phono3py_phonon_fc2(phono3py,
                                force_to_eVperA,
                                distance_to_A,
                                symmetrize_fc2,
                                input_filename,
                                use_alm,
                                log_level):
    if input_filename is None:
        filename = 'disp_fc2.yaml'
    else:
        filename = 'disp_fc2.' + input_filename + '.yaml'
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    file_exists(filename, log_level)
    disp_dataset = parse_disp_fc2_yaml(filename=filename)
    num_atom = phono3py.get_phonon_supercell().get_number_of_atoms()
    if disp_dataset['natom'] != num_atom:
        print("Number of atoms in supercell is not consistent with %s" %
              filename)
        if log_level:
            print_error()
        sys.exit(1)
    _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=True)

    if log_level:
        print("Sets of supercell forces are read from %s." %
              "FORCES_FC2")
    file_exists("FORCES_FC2", log_level)
    forces_fc2 = parse_FORCES_FC2(disp_dataset)
    if not forces_fc2:
        return False

    _convert_force_unit(forces_fc2, force_to_eVperA)

    phono3py.produce_fc2(
        forces_fc2,
        displacement_dataset=disp_dataset,
        symmetrize_fc2=symmetrize_fc2,
        use_alm=use_alm)

    return True

def _convert_force_unit(force_sets, force_to_eVperA):
    if force_to_eVperA is not None:
        for forces in force_sets:
            if forces is not None:
                forces *= force_to_eVperA

def _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=False):
    if distance_to_A is not None:
        for first_atom in disp_dataset['first_atoms']:
            for i in range(3):
                first_atom['displacement'][i] *= distance_to_A
            if not is_fc2:
                for second_atom in first_atom['second_atoms']:
                    for i in range(3):
                        second_atom['displacement'][i] *= distance_to_A
