# Copyright (C) 2019 Atsushi Togo
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
# * Neither the name of the phono3py project nor the names of its
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
import numpy as np
from phono3py import Phono3py
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.cui.create_force_constants import parse_forces
from phono3py.file_IO import read_fc3_from_hdf5, read_fc2_from_hdf5
from phono3py.phonon3.fc3 import show_drift_fc3
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_default_physical_units
import phonopy.cui.load_helper as load_helper


def load(phono3py_yaml=None,  # phono3py.yaml-like must be the first argument.
         supercell_matrix=None,
         primitive_matrix=None,
         phonon_supercell_matrix=None,
         mesh=None,
         is_nac=True,
         calculator=None,
         unitcell=None,
         supercell=None,
         nac_params=None,
         unitcell_filename=None,
         supercell_filename=None,
         born_filename=None,
         forces_fc3_filename=None,
         forces_fc2_filename=None,
         fc3_filename=None,
         fc2_filename=None,
         fc_calculator=None,
         factor=None,
         frequency_scale_factor=None,
         is_symmetry=True,
         symmetrize_fc=True,
         is_mesh_symmetry=True,
         is_compact_fc=False,
         symprec=1e-5,
         log_level=0):
    """Create Phono3py instance from parameters and/or input files.

    When unitcell and unitcell_filename are not given, file name that is
    default for the chosen calculator is looked for in the current directory
    as the default behaviour.

    if 'fc3.hdf5' is found, this is read.
    Unless 'fc3.hdf5' is found and if 'FORCES_FC3' and 'disp_fc3.yaml" are
    found, these are read and fc3 and fc2 are produced.

    if 'fc2.hdf5' is found, this is read.
    Unless 'fc2.hdf5' is found and if 'FORCES_FC2' and 'disp_fc2.yaml" are
    found, these are read and fc2 is produced.

    When force_sets_filename and force_constants_filename are not given,
    'FORCES_FC3' and 'FORCES_FC2' are looked for in the current directory
    as the default behaviour. When 'FORCES_FC3' ('FORCES_FC2') is given in
    the type-1 format, 'disp_fc3.yaml' ('disp_fc2.yaml') is also necessary
    and read.

    Parameters
    ----------
    phono3py_yaml : str, optional
        Filename of "phono3py.yaml"-like file. If this is given, the data
        such as unit cell structure,supercell matrix, and primitive_matrix,
        those for non-analytycal term correction are parsed. Default is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to input cell basis vectors.
        shape=(3, ) or (3, 3), where the former is considered a diagonal
        matrix. Default is the unit matrix.
        dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is
        the identity matrix. Default is None, which is equivalent to 'auto'.
        shape=(3, 3), dtype=float.
        When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix,
        the primitive matrix defined at
        https://spglib.github.io/spglib/definition.html
        is used.
    phonon_supercell_matrix : array_like, optional
        Supercell matrix used for fc2. In phono3py, supercell matrix for fc3
        and fc2 can be different to support longer range interaction of fc2
        than that of fc3. Unless setting this, supercell_matrix is used.
        This is only valide when unitcell or unitcell_filename is given.
        Default is None.
    mesh : array_like, optional
        Grid mesh numbers in reciprocal cell.
        shape=(3,), dtype='intc'
    is_nac : bool, optional
        If True, look for 'BORN' file. If False, NAS is turned off.
        The priority for NAC is nac_params > born_filename > is_nac ('BORN').
        Default is True.
    calculator : str, optional.
        Calculator used for computing forces. This is used to switch the set
        of physical units. Default is None, which is equivalent to "vasp".
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None. The priority for cell is
        unitcell_filename > supercell_filename > unitcell > supercell.
    supercell : PhonopyAtoms, optional
        Input supercell. Default value of primitive_matrix is set to
        'auto' (can be overwitten). supercell_matrix is ignored. Default is
        None. The priority for cell is
        unitcell_filename > supercell_filename > unitcell > supercell.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default is
        None. The priority for NAC is nac_params > born_filename > is_nac.
        {'born': Born effective charges
                 (array_like, shape=(primitive cell atoms, 3, 3), dtype=float),
         'dielectric': Dielectric constant matrix
                       (array_like, shape=(3, 3), dtype=float),
         'factor': unit conversion facotr (float)}
    unitcell_filename : str, optional
        Input unit cell filename. Default is None. The priority for cell is
        unitcell_filename > supercell_filename > unitcell > supercell.
    supercell_filename : str, optional
        Input supercell filename. When this is specified, supercell_matrix is
        ignored. Default is None. The priority for cell is
        1. unitcell_filename (with supercell_matrix)
        2. supercell_filename
        3. unitcell (with supercell_matrix)
        4. supercell.
    born_filename : str, optional
        Filename corresponding to 'BORN', a file contains non-analytical term
        correction parameters.
        The priority for NAC is nac_params > born_filename > is_nac ('BORN').
    forces_fc3_filename : sequence or str, optional
        A two-elemental sequence of filenames corresponding to
        ('FORCES_FC3', 'disp_fc3.yaml') in the type-1 format or a filename
        (str) corresponding to 'FORCES_FC3' in the type-2 format.
        Default is None. The priority to calculate or load force constants is
        fc3_filename > forces_fc3_filename > 'fc3.hdf5' > 'FORCES_FC3'.
    forces_fc2_filename : str or tuple, optional
        A two-elemental sequence of filenames corresponding to
        ('FORCES_FC2', 'disp_fc2.yaml') in the type-1 format or a filename
        (str) corresponding to 'FORCES_FC2' in the type-2 format.
        Default is None. The priority to calculate or load force constants is
        fc2_filename > forces_fc2_filename > 'fc2.hdf5' > 'FORCES_FC2'.
    fc3_filename : str, optional
        Filename of a file corresponding to 'fc3.hdf5', a file contains
        third-order force constants. Default is None.
        The priority to calculate or load force constants is
        fc3_filename > forces_fc3_filename > 'fc3.hdf5' > 'FORCES_FC3'.
    fc2_filename : str, optional
        Filename of a file corresponding to 'fc2.hdf5', a file contains
        second-order force constants. Default is None.
        The priority to calculate or load force constants is
        fc2_filename > forces_fc2_filename > 'fc2.hdf5' > 'FORCES_FC2'.
    fc_calculator : str, optional
        Force constants calculator. Currently only 'alm'. Default is None.
    factor : float, optional
        Phonon frequency unit conversion factor. Unless specified, default
        unit conversion factor for each calculator is used.
    frequency_scale_factor : float, optional
        Factor multiplied to calculated phonon frequency. Default is None,
        i.e., effectively 1.
    is_symmetry : bool, optional
        Setting False, crystal symmetry except for lattice translation is not
        considered. Default is True.
    symmetrize_fc : bool, optional
        Setting False, force constants are not symmetrized when creating
        force constants from displacements and forces. Default is True.
    is_mesh_symmetry : bool, optional
        Setting False, reciprocal mesh symmetry is not considered.
        Default is True.
    is_compact_fc : bool
        fc3 shape is
            False: (supercell, supercell, supecell, 3, 3, 3)
            True: (primitive, supercell, supecell, 3, 3, 3)
        where 'supercell' and 'primitive' indicate number of atoms in these
        cells. Default is False.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    log_level : int, optional
        Verbosity control. Default is 0.

    """

    if phono3py_yaml is None:
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=calculator,
            symprec=symprec)
        if phonon_supercell_matrix is not None:
            if unitcell is None and unitcell_filename is None:
                msg = ("phonon_supercell_matrix can be used only when "
                       "unitcell or unitcell_filename is given.")
                raise RuntimeError(msg)
            ph_smat = load_helper.get_supercell_matrix(phonon_supercell_matrix)
        else:
            ph_smat = None

        _nac_params = nac_params
    else:
        ph3py_yaml = Phono3pyYaml()
        ph3py_yaml.read(phono3py_yaml)
        cell = ph3py_yaml.unitcell
        smat = ph3py_yaml.supercell_matrix
        ph_smat = ph3py_yaml.phonon_supercell_matrix
        if smat is None:
            smat = np.eye(3, dtype='intc', order='C')
        if primitive_matrix == 'auto':
            pmat = 'auto'
        else:
            pmat = ph3py_yaml.primitive_matrix
        if is_nac:
            _nac_params = ph3py_yaml.nac_params
        else:
            _nac_params = None

    # units keywords: factor, nac_factor, distance_to_A
    units = get_default_physical_units(calculator)
    if factor is None:
        _factor = units['factor']
    else:
        _factor = factor

    ph3py = Phono3py(cell,
                     smat,
                     primitive_matrix=pmat,
                     phonon_supercell_matrix=ph_smat,
                     mesh=mesh,
                     frequency_factor_to_THz=_factor,
                     symprec=symprec,
                     is_symmetry=is_symmetry,
                     is_mesh_symmetry=is_mesh_symmetry,
                     calculator=calculator,
                     log_level=log_level)
    _nac_params = load_helper.get_nac_params(ph3py.primitive,
                                             _nac_params,
                                             born_filename,
                                             is_nac,
                                             units['nac_factor'])
    if _nac_params is not None:
        ph3py.nac_params = _nac_params

    _set_force_constants(ph3py,
                         units,
                         dataset=None,
                         fc3_filename=fc3_filename,
                         fc2_filename=fc2_filename,
                         forces_fc3_filename=forces_fc3_filename,
                         forces_fc2_filename=forces_fc2_filename,
                         fc_calculator=fc_calculator,
                         symmetrize_fc=symmetrize_fc,
                         is_compact_fc=is_compact_fc,
                         log_level=log_level)

    if mesh is not None:
        ph3py.init_phph_interaction(
            frequency_scale_factor=frequency_scale_factor)

    return ph3py


def _set_force_constants(
        ph3py,
        units,
        dataset=None,
        fc3_filename=None,
        fc2_filename=None,
        forces_fc3_filename=None,
        forces_fc2_filename=None,
        fc_calculator=None,
        fc_calculator_options=None,
        symmetrize_fc=True,
        is_compact_fc=False,
        log_level=0):
    p2s_map = ph3py.primitive.p2s_map
    if fc3_filename is not None:
        fc3 = read_fc3_from_hdf5(filename=fc3_filename, p2s_map=p2s_map)
        ph3py.fc3 = fc3
    elif forces_fc3_filename is not None:
        if type(forces_fc3_filename) is str:
            force_filename = forces_fc3_filename
            disp_filename = None
        else:
            force_filename, disp_filename = forces_fc3_filename
        _set_fc3(ph3py,
                 units,
                 force_filename,
                 disp_filename,
                 symmetrize_fc,
                 is_compact_fc,
                 fc_calculator,
                 fc_calculator_options,
                 log_level)
    elif os.path.isfile("fc3.hdf5"):
        ph3py.fc3 = read_fc3_from_hdf5(filename="fc3.hdf5", p2s_map=p2s_map)
        if log_level:
            print("fc3 was loaded from \"fc3.hdf5\".")
    elif os.path.isfile("FORCES_FC3") and os.path.isfile("disp_fc3.yaml"):
        _set_fc3(ph3py,
                 units,
                 "FORCES_FC3",
                 "disp_fc3.yaml",
                 symmetrize_fc,
                 is_compact_fc,
                 fc_calculator,
                 fc_calculator_options,
                 log_level)
    if log_level and ph3py.fc3 is not None:
        show_drift_fc3(ph3py.fc3, primitive=ph3py.primitive)

    if fc2_filename is not None:
        fc2 = read_fc2_from_hdf5(filename=fc2_filename, p2s_map=p2s_map)
        ph3py.fc2 = fc2
    elif forces_fc2_filename is not None:
        if type(forces_fc2_filename) == str:
            force_filename = forces_fc2_filename
            disp_filename = None
        else:
            force_filename, disp_filename = forces_fc2_filename
        _set_fc2(ph3py,
                 units,
                 force_filename,
                 disp_filename,
                 symmetrize_fc,
                 is_compact_fc,
                 fc_calculator,
                 fc_calculator_options,
                 log_level)
    elif os.path.isfile("fc2.hdf5"):
        ph3py.fc2 = read_fc2_from_hdf5(filename="fc2.hdf5", p2s_map=p2s_map)
        if log_level:
            print("fc2 was loaded from \"fc2.hdf5\".")
    elif os.path.isfile("FORCES_FC2") and os.path.isfile("disp_fc2.yaml"):
        _set_fc2(ph3py,
                 units,
                 "FORCES_FC2",
                 "disp_fc2.yaml",
                 symmetrize_fc,
                 is_compact_fc,
                 fc_calculator,
                 fc_calculator_options,
                 log_level)
    if log_level and ph3py.fc2 is not None:
        show_drift_force_constants(ph3py.fc2,
                                   primitive=ph3py.phonon_primitive,
                                   name='fc2')


def _set_fc3(ph3py,
             units,
             force_filename,
             disp_filename,
             symmetrize_fc,
             is_compact_fc,
             fc_calculator,
             fc_calculator_options,
             log_level):
    dataset = parse_forces(ph3py.supercell.get_number_of_atoms(),
                           units['force_to_eVperA'],
                           units['distance_to_A'],
                           force_filename=force_filename,
                           disp_filename=disp_filename,
                           log_level=log_level)
    ph3py.produce_fc3(displacement_dataset=dataset,
                      symmetrize_fc3r=symmetrize_fc,
                      is_compact_fc=is_compact_fc,
                      fc_calculator=fc_calculator,
                      fc_calculator_options=fc_calculator_options)
    if log_level and symmetrize_fc:
        print("Force constants were symmetrized.")


def _set_fc2(ph3py,
             units,
             force_filename,
             disp_filename,
             symmetrize_fc,
             is_compact_fc,
             fc_calculator,
             fc_calculator_options,
             log_level):
    dataset = parse_forces(ph3py.phonon_supercell.get_number_of_atoms(),
                           units['force_to_eVperA'],
                           units['distance_to_A'],
                           force_filename=force_filename,
                           disp_filename=disp_filename,
                           is_fc2=True,
                           log_level=log_level)
    ph3py.produce_fc2(displacement_dataset=dataset,
                      symmetrize_fc2=symmetrize_fc,
                      is_compact_fc=is_compact_fc,
                      fc_calculator=fc_calculator,
                      fc_calculator_options=fc_calculator_options)
    if log_level and symmetrize_fc:
        print("fc2 was symmetrized.")
