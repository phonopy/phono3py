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

import os
import sys
from phonopy.harmonic.force_constants import (
    show_drift_force_constants,
    symmetrize_force_constants,
    symmetrize_compact_force_constants)
from phonopy.file_IO import get_dataset_type2
from phonopy.cui.phonopy_script import print_error, file_exists
from phonopy.interface.calculator import get_default_physical_units
from phono3py.phonon3.fc3 import show_drift_fc3
from phono3py.file_IO import (
    parse_disp_fc3_yaml, parse_disp_fc2_yaml, parse_FORCES_FC2,
    parse_FORCES_FC3, read_fc3_from_hdf5, read_fc2_from_hdf5,
    write_fc3_to_hdf5, write_fc2_to_hdf5, get_length_of_first_line)
from phono3py.cui.show_log import (
    show_phono3py_force_constants_settings)
from phono3py.phonon3.fc3 import (
    set_permutation_symmetry_fc3, set_translational_invariance_fc3)


def create_phono3py_force_constants(phono3py,
                                    settings,
                                    ph3py_yaml=None,
                                    input_filename=None,
                                    output_filename=None,
                                    phono3py_yaml_filename=None,
                                    log_level=1):
    if settings.fc_calculator is None:
        symmetrize_fc3r = (settings.is_symmetrize_fc3_r or
                           settings.fc_symmetry)
        symmetrize_fc2 = (settings.is_symmetrize_fc2 or
                          settings.fc_symmetry)
    else:  # Rely on fc calculator the symmetrization of fc.
        symmetrize_fc2 = False
        symmetrize_fc3r = False

    if log_level:
        show_phono3py_force_constants_settings(settings)

    #######
    # fc3 #
    #######
    if (settings.is_joint_dos or
        (settings.is_isotope and
         not (settings.is_bterta or settings.is_lbte)) or
        settings.read_gamma or
        settings.read_pp or
        (not settings.is_bterta and settings.write_phonon) or
        settings.constant_averaged_pp_interaction is not None):
        pass
    else:
        if settings.read_fc3:
            _read_phono3py_fc3(phono3py,
                               symmetrize_fc3r,
                               input_filename,
                               log_level)
        else:  # fc3 from FORCES_FC3
            _create_phono3py_fc3(phono3py,
                                 ph3py_yaml,
                                 symmetrize_fc3r,
                                 symmetrize_fc2,
                                 input_filename,
                                 output_filename,
                                 settings.is_compact_fc,
                                 settings.cutoff_pair_distance,
                                 settings.fc_calculator,
                                 settings.fc_calculator_options,
                                 log_level)
            if output_filename is None:
                filename = 'fc3.hdf5'
            else:
                filename = 'fc3.' + output_filename + '.hdf5'
            if log_level:
                print("Writing fc3 to \"%s\"." % filename)
            write_fc3_to_hdf5(phono3py.fc3,
                              filename=filename,
                              p2s_map=phono3py.primitive.p2s_map,
                              compression=settings.hdf5_compression)

        cutoff_distance = settings.cutoff_fc3_distance
        if cutoff_distance is not None and cutoff_distance > 0:
            if log_level:
                print("Cutting-off fc3 by zero (cut-off distance: %f)" %
                      cutoff_distance)
            phono3py.cutoff_fc3_by_zero(cutoff_distance)

        if log_level:
            show_drift_fc3(phono3py.fc3, primitive=phono3py.primitive)

    #######
    # fc2 #
    #######
    phonon_primitive = phono3py.phonon_primitive
    p2s_map = phonon_primitive.p2s_map
    if settings.read_fc2:
        _read_phono3py_fc2(phono3py,
                           symmetrize_fc2,
                           input_filename,
                           log_level)
    else:
        if phono3py.phonon_supercell_matrix is None:
            _create_phono3py_fc2(phono3py,
                                 ph3py_yaml,
                                 symmetrize_fc2,
                                 input_filename,
                                 settings.is_compact_fc,
                                 settings.fc_calculator,
                                 settings.fc_calculator_options,
                                 log_level)
        else:
            _create_phono3py_phonon_fc2(phono3py,
                                        ph3py_yaml,
                                        symmetrize_fc2,
                                        input_filename,
                                        settings.is_compact_fc,
                                        settings.fc_calculator,
                                        settings.fc_calculator_options,
                                        log_level)
        if output_filename is None:
            filename = 'fc2.hdf5'
        else:
            filename = 'fc2.' + output_filename + '.hdf5'
        if log_level:
            print("Writing fc2 to \"%s\"." % filename)
        write_fc2_to_hdf5(phono3py.fc2,
                          filename=filename,
                          p2s_map=p2s_map,
                          physical_unit='eV/angstrom^2',
                          compression=settings.hdf5_compression)

    if log_level:
        show_drift_force_constants(phono3py.fc2,
                                   primitive=phonon_primitive,
                                   name='fc2')


def parse_forces(phono3py,
                 ph3py_yaml=None,
                 cutoff_pair_distance=None,
                 force_filename="FORCES_FC3",
                 disp_filename=None,
                 fc_type=None,
                 log_level=0):
    if fc_type == 'phonon_fc2':
        natom = len(phono3py.phonon_supercell)
    else:
        natom = len(phono3py.supercell)
    dataset = _get_type2_dataset(natom, filename=force_filename)
    if dataset:  # type2
        physical_units = get_default_physical_units(phono3py.calculator)
        force_to_eVperA = physical_units['force_to_eVperA']
        distance_to_A = physical_units['distance_to_A']
        if log_level:
            print("%d snapshots were found in %s."
                  % (len(dataset['displacements']), "FORCES_FC3"))
        if force_to_eVperA is not None:
            dataset['forces'] *= force_to_eVperA
        if distance_to_A is not None:
            dataset['displacements'] *= distance_to_A
    else:  # type1
        dataset = _parse_forces_type1(phono3py,
                                      ph3py_yaml,
                                      cutoff_pair_distance,
                                      force_filename,
                                      disp_filename,
                                      fc_type,
                                      log_level)

    return dataset


def _parse_forces_type1(phono3py,
                        ph3py_yaml,
                        cutoff_pair_distance,
                        force_filename,
                        disp_filename,
                        fc_type,
                        log_level):
    dataset = None
    if fc_type == 'phonon_fc2':
        if ph3py_yaml and ph3py_yaml.phonon_dataset is not None:
            dataset = ph3py_yaml.phonon_dataset
        natom = len(phono3py.phonon_supercell)
    else:
        if ph3py_yaml and ph3py_yaml.dataset is not None:
            dataset = ph3py_yaml.dataset
        natom = len(phono3py.supercell)

    if disp_filename is None and dataset is None:
        msg = ("\"%s\" in type-1 format is given. "
               "But displacement dataset is not given properly."
               % force_filename)
        raise RuntimeError(msg)
    elif dataset is None:
        dataset = _read_disp_fc_yaml(disp_filename, fc_type, log_level)
    else:
        if log_level:
            print("Displacement dataset for %s was read from \"%s\"."
                  % (fc_type, ph3py_yaml.yaml_filename))

    physical_units = get_default_physical_units(phono3py.calculator)
    force_to_eVperA = physical_units['force_to_eVperA']
    distance_to_A = physical_units['distance_to_A']

    if cutoff_pair_distance:
        if ('cutoff_distance' not in dataset or
            'cutoff_distance' in dataset and
            cutoff_pair_distance < dataset['cutoff_distance']):
            dataset['cutoff_distance'] = cutoff_pair_distance
            if log_level:
                print("Cutoff-pair-distance: %f" % cutoff_pair_distance)

    if dataset['natom'] != natom:
        msg = ("Number of atoms in supercell is not consistent with "
               "\"%s\"." % disp_filename)
        raise RuntimeError(msg)

    if distance_to_A is not None:
        _convert_displacement_unit(dataset, distance_to_A)

    if fc_type == 'phonon_fc2':
        parse_FORCES_FC2(dataset,
                         filename=force_filename,
                         unit_conversion_factor=force_to_eVperA)
    else:
        parse_FORCES_FC3(dataset,
                         filename=force_filename,
                         unit_conversion_factor=force_to_eVperA)

    if log_level:
        print("Sets of supercell forces were read from \"%s\"."
              % force_filename)
        sys.stdout.flush()

    return dataset


def _read_disp_fc_yaml(disp_filename, fc_type, log_level):
    if fc_type == 'phonon_fc2':
        dataset = parse_disp_fc2_yaml(filename=disp_filename)
    else:
        dataset = parse_disp_fc3_yaml(filename=disp_filename)
    if log_level:
        print("Displacement dataset for %s was read from \"%s\"."
              % (fc_type, disp_filename))

    return dataset


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
                         ph3py_yaml,
                         symmetrize_fc3r,
                         symmetrize_fc2,
                         input_filename,
                         output_filename,
                         is_compact_fc,
                         cutoff_pair_distance,
                         fc_calculator,
                         fc_calculator_options,
                         log_level):
    """

    Note
    ----
    cutoff_pair_distance is the parameter to determine each displaced
    supercell is included to the computation of fc3. It is assumed that
    cutoff_pair_distance is stored in the step to create sets of
    displacements and the value is stored n the displacement dataset and
    also as the parameter 'included': True or False for each displacement.
    The parameter cutoff_pair_distance here can be used in the step to
    create fc3 by overwriting original cutoff_pair_distance value only
    when the former value is smaller than the later.

    """
    if input_filename is None:
        disp_filename = 'disp_fc3.yaml'
    else:
        disp_filename = 'disp_fc3.' + input_filename + '.yaml'

    # Try to use phono3py.dataset when the disp file not found
    if not os.path.isfile(disp_filename):
        disp_filename = None

    try:
        dataset = parse_forces(phono3py,
                               ph3py_yaml=ph3py_yaml,
                               cutoff_pair_distance=cutoff_pair_distance,
                               force_filename="FORCES_FC3",
                               disp_filename=disp_filename,
                               fc_type='fc3',
                               log_level=log_level)
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.produce_fc3(displacement_dataset=dataset,
                         symmetrize_fc3r=symmetrize_fc3r,
                         is_compact_fc=is_compact_fc,
                         fc_calculator=fc_calculator,
                         fc_calculator_options=fc_calculator_options)


def _create_phono3py_fc2(phono3py,
                         ph3py_yaml,
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

    # Try to use phono3py.dataset when the disp file not found
    if not os.path.isfile(disp_filename):
        disp_filename = None

    try:
        dataset = parse_forces(phono3py,
                               ph3py_yaml=ph3py_yaml,
                               force_filename="FORCES_FC3",
                               disp_filename=disp_filename,
                               fc_type='fc2',
                               log_level=log_level)
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.produce_fc2(
        displacement_dataset=dataset,
        symmetrize_fc2=symmetrize_fc2,
        is_compact_fc=is_compact_fc,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options)


def _create_phono3py_phonon_fc2(phono3py,
                                ph3py_yaml,
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

    # Try to use phono3py.phonon_dataset when the disp file not found
    if not os.path.isfile(disp_filename):
        disp_filename = None

    try:
        dataset = parse_forces(phono3py,
                               ph3py_yaml=ph3py_yaml,
                               force_filename="FORCES_FC2",
                               disp_filename=disp_filename,
                               fc_type='phonon_fc2',
                               log_level=log_level)
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.produce_fc2(
        displacement_dataset=dataset,
        symmetrize_fc2=symmetrize_fc2,
        is_compact_fc=is_compact_fc,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options)


def _convert_displacement_unit(dataset, distance_to_A):
    for d1 in dataset['first_atoms']:
        for i in range(3):
            d1['displacement'][i] *= distance_to_A
        if 'second_atoms' in d1:
            for d2 in d1['second_atoms']:
                for i in range(3):
                    d2['displacement'][i] *= distance_to_A
