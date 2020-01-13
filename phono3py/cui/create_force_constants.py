# Copyright (C) 2020 Atsushi Togo
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

import sys
from phonopy.harmonic.force_constants import (
    show_drift_force_constants,
    symmetrize_force_constants,
    symmetrize_compact_force_constants)
from phonopy.file_IO import get_dataset_type2
from phono3py.phonon3.fc3 import show_drift_fc3
from phono3py.file_IO import (
    parse_disp_fc3_yaml, parse_disp_fc2_yaml, parse_FORCES_FC2,
    parse_FORCES_FC3, read_fc3_from_hdf5, read_fc2_from_hdf5,
    write_fc3_to_hdf5, write_fc2_to_hdf5, get_length_of_first_line)
from phono3py.cui.show_log import (
    show_phono3py_force_constants_settings, print_error, file_exists)
from phono3py.phonon3.fc3 import (
    set_permutation_symmetry_fc3, set_translational_invariance_fc3)


def create_phono3py_force_constants(phono3py,
                                    phonon_supercell_matrix,
                                    settings,
                                    force_to_eVperA=None,
                                    distance_to_A=None,
                                    compression="gzip",
                                    input_filename=None,
                                    output_filename=None,
                                    log_level=1):
    if settings.get_fc_calculator() is None:
        symmetrize_fc3r = (settings.get_is_symmetrize_fc3_r() or
                           settings.get_fc_symmetry())
        symmetrize_fc2 = (settings.get_is_symmetrize_fc2() or
                          settings.get_fc_symmetry())
    else:  # Rely on fc calculator the symmetrization of fc.
        symmetrize_fc2 = False
        symmetrize_fc3r = False

    if log_level:
        show_phono3py_force_constants_settings(settings)

    #######
    # fc3 #
    #######
    if (settings.get_is_joint_dos() or
        (settings.get_is_isotope() and
         not (settings.get_is_bterta() or settings.get_is_lbte())) or
        settings.get_read_gamma() or
        settings.get_read_pp() or
        (not settings.get_is_bterta() and settings.get_write_phonon()) or
        settings.get_constant_averaged_pp_interaction() is not None):
        pass
    else:
        if settings.get_read_fc3():  # Read fc3.hdf5
            _read_phono3py_fc3(phono3py,
                               symmetrize_fc3r,
                               input_filename,
                               log_level)
        else:  # fc3 from FORCES_FC3
            if not _create_phono3py_fc3(
                    phono3py,
                    force_to_eVperA,
                    distance_to_A,
                    symmetrize_fc3r,
                    symmetrize_fc2,
                    input_filename,
                    output_filename,
                    settings.get_is_compact_fc(),
                    settings.get_cutoff_pair_distance(),
                    settings.get_fc_calculator(),
                    settings.get_fc_calculator_options(),
                    compression,
                    log_level):

                    print("fc3 was not created properly.")
                    if log_level:
                        print_error()
                    sys.exit(1)

        cutoff_distance = settings.get_cutoff_fc3_distance()
        if cutoff_distance is not None and cutoff_distance > 0:
            if log_level:
                print("Cutting-off fc3 by zero (cut-off distance: %f)" %
                      cutoff_distance)
            phono3py.cutoff_fc3_by_zero(cutoff_distance)

        if log_level:
            show_drift_fc3(phono3py.fc3,
                           primitive=phono3py.primitive)

    #######
    # fc2 #
    #######
    phonon_primitive = phono3py.phonon_primitive
    p2s_map = phonon_primitive.p2s_map
    if settings.get_read_fc2():
        _read_phono3py_fc2(phono3py,
                           symmetrize_fc2,
                           input_filename,
                           log_level)
    else:
        if phonon_supercell_matrix is None:
            if settings.get_fc_calculator() is not None:
                pass
            elif not _create_phono3py_fc2(
                    phono3py,
                    force_to_eVperA,
                    distance_to_A,
                    symmetrize_fc2,
                    input_filename,
                    settings.get_is_compact_fc(),
                    settings.get_fc_calculator(),
                    settings.get_fc_calculator_options(),
                    log_level):
                print("fc2 was not created properly.")
                if log_level:
                    print_error()
                sys.exit(1)
        else:
            if not _create_phono3py_phonon_fc2(
                    phono3py,
                    force_to_eVperA,
                    distance_to_A,
                    symmetrize_fc2,
                    input_filename,
                    settings.get_is_compact_fc(),
                    settings.get_fc_calculator(),
                    settings.get_fc_calculator_options(),
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
            print("Writing fc2 to \"%s\"." % filename)
        write_fc2_to_hdf5(phono3py.fc2,
                          filename=filename,
                          p2s_map=p2s_map,
                          physical_unit='eV/Angstrom^2',
                          compression=compression)

    if log_level:
        show_drift_force_constants(phono3py.fc2,
                                   primitive=phonon_primitive,
                                   name='fc2')


def parse_forces(natom,
                 force_to_eVperA,
                 distance_to_A,
                 cutoff_pair_distance=None,
                 force_filename="FORCES_FC3",
                 disp_filename=None,
                 is_fc2=False,
                 log_level=0):
    disp_dataset = _get_type2_dataset(natom, filename=force_filename)
    if disp_dataset:  # type2
        if log_level:
            print("%d snapshots were found in %s."
                  % (len(disp_dataset['displacements']), "FORCES_FC3"))
        if force_to_eVperA is not None:
            disp_dataset['forces'] *= force_to_eVperA
        if distance_to_A is not None:
            disp_dataset['displacements'] *= distance_to_A
    else:  # type1
        if log_level:
            print("Displacement dataset for %s is read from \"%s\"."
                  % ("fc2" if is_fc2 else "fc3", disp_filename))
        if is_fc2:
            disp_dataset = parse_disp_fc2_yaml(filename=disp_filename)
        else:
            disp_dataset = parse_disp_fc3_yaml(filename=disp_filename)
        if cutoff_pair_distance:
            if ('cutoff_distance' not in disp_dataset or
                'cutoff_distance' in disp_dataset and
                cutoff_pair_distance < disp_dataset['cutoff_distance']):
                disp_dataset['cutoff_distance'] = cutoff_pair_distance
                if log_level:
                    print("Cutoff-pair-distance: %f" % cutoff_pair_distance)
        if disp_dataset['natom'] != natom:
            msg = ("Number of atoms in supercell is not consistent with "
                   "\"%s\"." % disp_filename)
            raise RuntimeError(msg)
        _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=is_fc2)
        if log_level:
            print("Sets of supercell forces are read from \"%s\"."
                  % force_filename)
            sys.stdout.flush()

        # forces are stored in disp_dataset.
        if is_fc2:
            parse_FORCES_FC2(disp_dataset,
                             filename=force_filename,
                             unit_conversion_factor=force_to_eVperA)
        else:
            parse_FORCES_FC3(disp_dataset,
                             filename=force_filename,
                             unit_conversion_factor=force_to_eVperA)

    return disp_dataset


def _read_phono3py_fc3(phono3py,
                       symmetrize_fc3r,
                       input_filename,
                       log_level):
    if input_filename is None:
        filename = 'fc3.hdf5'
    else:
        filename = 'fc3.' + input_filename + '.hdf5'
    file_exists(filename, log_level)
    if log_level:
        print("Reading fc3 from \"%s\"." % filename)

    p2s_map = phono3py.primitive.p2s_map
    try:
        fc3 = read_fc3_from_hdf5(filename=filename, p2s_map=p2s_map)
    except RuntimeError:
        import traceback
        traceback.print_exc()
        if log_level:
            print_error()
        sys.exit(1)
    num_atom = phono3py.supercell.get_number_of_atoms()
    if fc3.shape[1] != num_atom:
        print("Matrix shape of fc3 doesn't agree with supercell size.")
        if log_level:
            print_error()
        sys.exit(1)

    if symmetrize_fc3r:
        set_translational_invariance_fc3(fc3)
        set_permutation_symmetry_fc3(fc3)

    phono3py.fc3 = fc3


def _read_phono3py_fc2(phono3py,
                       symmetrize_fc2,
                       input_filename,
                       log_level):
    if input_filename is None:
        filename = 'fc2.hdf5'
    else:
        filename = 'fc2.' + input_filename + '.hdf5'
    file_exists(filename, log_level)
    if log_level:
        print("Reading fc2 from \"%s\"." % filename)

    num_atom = phono3py.phonon_supercell.get_number_of_atoms()
    p2s_map = phono3py.phonon_primitive.p2s_map
    try:
        phonon_fc2 = read_fc2_from_hdf5(filename=filename, p2s_map=p2s_map)
    except RuntimeError:
        import traceback
        traceback.print_exc()
        if log_level:
            print_error()
        sys.exit(1)

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
                                               phono3py.phonon_primitive)

    phono3py.fc2 = phonon_fc2


def _get_type2_dataset(natom, filename="FORCES_FC3"):
    with open(filename, 'r') as f:
        len_first_line = get_length_of_first_line(f)
        if len_first_line == 6:
            dataset = get_dataset_type2(f, natom)
        else:
            dataset = {}
    return dataset


def _create_phono3py_fc3(phono3py,
                         force_to_eVperA,
                         distance_to_A,
                         symmetrize_fc3r,
                         symmetrize_fc2,
                         input_filename,
                         output_filename,
                         is_compact_fc,
                         cutoff_pair_distance,
                         fc_calculator,
                         fc_calculator_options,
                         compression,
                         log_level):
    if input_filename is None:
        disp_filename = 'disp_fc3.yaml'
    else:
        disp_filename = 'disp_fc3.' + input_filename + '.yaml'
    natom = phono3py.supercell.get_number_of_atoms()
    try:
        disp_dataset = parse_forces(natom,
                                    force_to_eVperA,
                                    distance_to_A,
                                    cutoff_pair_distance=cutoff_pair_distance,
                                    force_filename="FORCES_FC3",
                                    disp_filename=disp_filename,
                                    log_level=log_level)
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.produce_fc3(displacement_dataset=disp_dataset,
                         symmetrize_fc3r=symmetrize_fc3r,
                         is_compact_fc=is_compact_fc,
                         fc_calculator=fc_calculator,
                         fc_calculator_options=fc_calculator_options)

    if output_filename is None:
        filename = 'fc3.hdf5'
    else:
        filename = 'fc3.' + output_filename + '.hdf5'
    if log_level:
        print("Writing fc3 to \"%s\"." % filename)

    write_fc3_to_hdf5(phono3py.fc3,
                      filename=filename,
                      p2s_map=phono3py.primitive.p2s_map,
                      compression=compression)

    return True


def _create_phono3py_fc2(phono3py,
                         force_to_eVperA,
                         distance_to_A,
                         symmetrize_fc2,
                         input_filename,
                         is_compact_fc,
                         fc_calculator,
                         fc_calculator_options,
                         log_level):
    if input_filename is None:
        disp_filename = 'disp_fc3.yaml'
    else:
        disp_filename = 'disp_fc3.' + input_filename + '.yaml'
    natom = phono3py.supercell.get_number_of_atoms()
    try:
        disp_dataset = parse_forces(natom,
                                    force_to_eVperA,
                                    distance_to_A,
                                    force_filename="FORCES_FC3",
                                    disp_filename=disp_filename,
                                    is_fc2=True,
                                    log_level=log_level)
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.produce_fc2(
        displacement_dataset=disp_dataset,
        symmetrize_fc2=symmetrize_fc2,
        is_compact_fc=is_compact_fc,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options)

    return True


def _create_phono3py_phonon_fc2(phono3py,
                                force_to_eVperA,
                                distance_to_A,
                                symmetrize_fc2,
                                input_filename,
                                is_compact_fc,
                                fc_calculator,
                                fc_calculator_options,
                                log_level):
    if input_filename is None:
        disp_filename = 'disp_fc2.yaml'
    else:
        disp_filename = 'disp_fc2.' + input_filename + '.yaml'
    natom = phono3py.phonon_supercell.get_number_of_atoms()
    try:
        disp_dataset = parse_forces(natom,
                                    force_to_eVperA,
                                    distance_to_A,
                                    force_filename="FORCES_FC2",
                                    disp_filename=disp_filename,
                                    is_fc2=True,
                                    log_level=log_level)
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.produce_fc2(
        displacement_dataset=disp_dataset,
        symmetrize_fc2=symmetrize_fc2,
        is_compact_fc=is_compact_fc,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options)

    return True


def _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=False):
    if distance_to_A is not None:
        for first_atom in disp_dataset['first_atoms']:
            for i in range(3):
                first_atom['displacement'][i] *= distance_to_A
            if not is_fc2:
                for second_atom in first_atom['second_atoms']:
                    for i in range(3):
                        second_atom['displacement'][i] *= distance_to_A
