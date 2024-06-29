"""Phono3py loader."""

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
from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import phonopy.cui.load_helper as load_helper
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_default_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import determinant

from phono3py import Phono3py
from phono3py.cui.create_force_constants import (
    forces_in_dataset,
    parse_forces,
    run_pypolymlp_to_compute_forces,
)
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phono3py.interface.fc_calculator import extract_fc2_fc3_calculators
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon3.fc3 import show_drift_fc3


def load(
    phono3py_yaml: Optional[
        Union[str, bytes, os.PathLike]
    ] = None,  # phono3py.yaml-like must be the first argument.
    supercell_matrix: Optional[Union[Sequence, np.ndarray]] = None,
    primitive_matrix: Optional[Union[Sequence, np.ndarray]] = None,
    phonon_supercell_matrix: Optional[Union[Sequence, np.ndarray]] = None,
    is_nac: bool = True,
    calculator: Optional[str] = None,
    unitcell: Optional[PhonopyAtoms] = None,
    supercell: Optional[PhonopyAtoms] = None,
    nac_params: Optional[dict] = None,
    unitcell_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    supercell_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    born_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    forces_fc3_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    forces_fc2_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    fc3_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    fc2_filename: Optional[Union[str, bytes, os.PathLike]] = None,
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[str] = None,
    factor: Optional[float] = None,
    produce_fc: bool = True,
    is_symmetry: bool = True,
    symmetrize_fc: bool = True,
    is_mesh_symmetry: bool = True,
    is_compact_fc: bool = False,
    use_pypolymlp: bool = False,
    mlp_params: Optional[dict] = None,
    use_grg: bool = False,
    make_r0_average: bool = True,
    symprec: float = 1e-5,
    log_level: int = 0,
) -> Phono3py:
    """Create Phono3py instance from parameters and/or input files.

    "phono3py_yaml"-like file is parsed unless crystal structure information is
    given by unitcell_filename, supercell_filename, unitcell
    (PhonopyAtoms-like), or supercell (PhonopyAtoms-like). Even when
    "phono3py_yaml"-like file is parse, parameters except for crystal structure
    can be overwritten.

    'fc3.hdf5' is read if found in current directory. Unless 'fc3.hdf5' is found
    and if 'FORCES_FC3' and 'phono3py_disp.yaml" are found, these are read and
    fc3 and fc2 are produced.

    if 'fc2.hdf5' is found, this is read. Unless 'fc2.hdf5' is found and if
    'FORCES_FC2' and 'phono3py_disp.yaml" are found, these are read and fc2 is
    produced.

    When force_sets_filename and force_constants_filename are not given,
    'FORCES_FC3' and 'FORCES_FC2' are looked for in the current directory as the
    default behaviour. When 'FORCES_FC3' ('FORCES_FC2') is given in the type-1
    format, 'phono3py_disp.yaml' is also necessary and read.

    Crystal structure
    -----------------
    Means to provide crystal structure(s) and their priority:
        1. unitcell_filename (with supercell_matrix)
        2. supercell_filename
        3. unitcell (with supercell_matrix)
        4. supercell.
        5. phono3py_yaml-like

    Force sets or force constants
    -----------------------------
    Optional. Means to provide information to generate force constants and their
    priority:
        1. fc3_filename (fc2_filename)
        2. forces_fc3_filename (forces_fc2_filename). Do not forget that for
           type-1 format, phono3py_disp.yaml has to be given, too.
        3. 'fc3.hdf5' and 'fc2.hdf5' are searched in current directory.
        4. 'FORCES_FC3' and 'FORCES_FC2' are searched in current directory.
           'FORCES_FC2' is optional. For type-1 format, 'phono3py_disp.yaml' is
           also searched in current directory. When 'FORCES_FC2' is not found,
           'FORCES_FC3' is used to create fc2.

    Parameters for non-analytical term correctiion (NAC)
    ----------------------------------------------------
    Optional. Means to provide NAC parameters and their priority:
        1. born_filename
        2. nac_params
        3. phono3py_yaml_like.nac_params if existed and is_nac=True.
        4. 'BORN' is searched in current directory when is_nac=True.

    Parameters
    ----------
    phono3py_yaml : str, optional
        Filename of "phono3py.yaml"-like file. If this is given, the data in the
        file are parsed. Default is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to input cell basis vectors. shape=(3, ) or
        (3, 3), where the former is considered a diagonal matrix. Default is the
        unit matrix. dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is the
        identity matrix. When given as array_like, shape=(3, 3), dtype=float.
        When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix, the
        primitive matrix defined at
        https://spglib.github.io/spglib/definition.html is used. When 'auto' is
        given, the centring type ('F', 'I', 'A', 'C', 'R', or primitive 'P') is
        automatically chosen. Default is 'auto'.
    phonon_supercell_matrix : array_like, optional
        Supercell matrix used for fc2. In phono3py, supercell matrix for fc3 and
        fc2 can be different to support longer range interaction of fc2 than
        that of fc3. Unless setting this, supercell_matrix is used. This is only
        valide when unitcell or unitcell_filename is given. Default is None.
    is_nac : bool, optional
        If True, look for 'BORN' file. If False, NAS is turned off. Default is
        True.
    calculator : str, optional.
        Calculator used for computing forces. This is used to switch the set of
        physical units when parsing calculator input/output files. Default is
        None, which is equivalent to "vasp".
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None.
    supercell : PhonopyAtoms, optional
        Input supercell. With given, default value of primitive_matrix is set to
        'auto' (can be overwitten). supercell_matrix is ignored. Default is
        None.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default is None.
        {'born': Born effective charges
                 (array_like, shape=(primitive cell atoms, 3, 3), dtype=float),
         'dielectric': Dielectric constant matrix
                       (array_like, shape=(3, 3), dtype=float),
         'factor': unit conversion facotr (float)}
    unitcell_filename : os.PathLike, optional
        Input unit cell filename. Default is None.
    supercell_filename : os.PathLike, optional
        Input supercell filename. When this is specified, supercell_matrix is
        ignored. Default is None.
    born_filename : os.PathLike, optional
        Filename corresponding to 'BORN', a file contains non-analytical term
        correction parameters.
    forces_fc3_filename : sequence or os.PathLike, optional
        A two-elemental sequence of filenames corresponding to ('FORCES_FC3',
        'phono3py_disp.yaml') in the type-1 format or a filename (os.PathLike)
        corresponding to 'FORCES_FC3' in the type-2 format. Default is None.
    forces_fc2_filename : os.PathLike or sequence, optional
        A two-elemental sequence of filenames corresponding to ('FORCES_FC2',
        'phono3py_disp.yaml') in the type-1 format or a filename (os.PathLike)
        corresponding to 'FORCES_FC2' in the type-2 format. Default is None.
    fc3_filename : os.PathLike, optional
        Filename of a file corresponding to 'fc3.hdf5', a file contains
        third-order force constants. Default is None.
    fc2_filename : os.PathLike, optional
        Filename of a file corresponding to 'fc2.hdf5', a file contains
        second-order force constants. Default is None.
    fc_calculator : str, optional
        Force constants calculator. Currently only 'alm'. Default is None.
    fc_calculator_options : str, optional
        Optional parameters that are passed to the external fc-calculator. This
        is given as one text string. How to parse this depends on the
        fc-calculator. For alm, each parameter is splitted by comma ',', and
        each set of key and value pair is written in 'key = value'.
    factor : float, optional
        Phonon frequency unit conversion factor. Unless specified, default unit
        conversion factor for each calculator is used.
    produce_fc : bool, optional
        Setting False, force constants are not calculated from displacements and
        forces. Default is True.
    is_symmetry : bool, optional
        Setting False, crystal symmetry except for lattice translation is not
        considered. Default is True.
    symmetrize_fc : bool, optional
        Setting False, force constants are not symmetrized when creating force
        constants from displacements and forces. Default is True.
    is_mesh_symmetry : bool, optional
        Setting False, reciprocal mesh symmetry is not considered. Default is
        True.
    is_compact_fc : bool, optional
        fc3 are created in the array whose shape is
            True: (primitive, supercell, supecell, 3, 3, 3) False: (supercell,
            supercell, supecell, 3, 3, 3)
        and for fc2
            True: (primitive, supecell, 3, 3) False: (supercell, supecell, 3, 3)
        where 'supercell' and 'primitive' indicate number of atoms in these
        cells. Default is False.
    use_pypolymlp : bool, optional
        Use pypolymlp for generating force constants. Default is False.
    mlp_params : dict, optional
        A set of parameters used by machine learning potentials.
    use_grg : bool, optional
        Use generalized regular grid when True. Default is False.
    make_r0_average : bool, optional
        fc3 transformation from real to reciprocal space is done around three
        atoms and averaged when True. Default is False, i.e., only around the
        first atom. Setting False is for rough compatibility with v2.x. Default
        is True.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    log_level : int, optional
        Verbosity control. Default is 0.

    """
    if (
        supercell is not None
        or supercell_filename is not None
        or unitcell is not None
        or unitcell_filename is not None
    ):
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=calculator,
            symprec=symprec,
        )
        if phonon_supercell_matrix is not None:
            if unitcell is None and unitcell_filename is None:
                msg = (
                    "phonon_supercell_matrix can be used only when "
                    "unitcell or unitcell_filename is given."
                )
                raise RuntimeError(msg)
            ph_smat = phonon_supercell_matrix
        else:
            ph_smat = None
        _nac_params = nac_params
        ph3py_yaml = None
    elif phono3py_yaml is not None:
        ph3py_yaml = Phono3pyYaml()
        ph3py_yaml.read(phono3py_yaml)
        cell = ph3py_yaml.unitcell
        smat = ph3py_yaml.supercell_matrix
        ph_smat = ph3py_yaml.phonon_supercell_matrix
        if smat is None:
            smat = np.eye(3, dtype="intc", order="C")
        if primitive_matrix == "auto":
            pmat = "auto"
        else:
            pmat = ph3py_yaml.primitive_matrix

        if nac_params is not None:
            _nac_params = nac_params
        elif is_nac:
            _nac_params = ph3py_yaml.nac_params
        else:
            _nac_params = None

    # units keywords: factor, nac_factor, distance_to_A
    physical_units = get_default_physical_units(calculator)
    if factor is None:
        _factor = physical_units["factor"]
    else:
        _factor = factor
    ph3py = Phono3py(
        cell,
        smat,
        primitive_matrix=pmat,
        phonon_supercell_matrix=ph_smat,
        frequency_factor_to_THz=_factor,
        symprec=symprec,
        is_symmetry=is_symmetry,
        is_mesh_symmetry=is_mesh_symmetry,
        use_grg=use_grg,
        make_r0_average=make_r0_average,
        calculator=calculator,
        log_level=log_level,
    )

    # NAC params
    if born_filename is not None or _nac_params is not None or is_nac:
        ph3py.nac_params = load_helper.get_nac_params(
            ph3py.primitive,
            _nac_params,
            born_filename,
            is_nac,
            physical_units["nac_factor"],
            log_level=log_level,
        )

    read_fc = set_dataset_and_force_constants(
        ph3py,
        ph3py_yaml,
        fc3_filename=fc3_filename,
        fc2_filename=fc2_filename,
        forces_fc3_filename=forces_fc3_filename,
        forces_fc2_filename=forces_fc2_filename,
        phono3py_yaml_filename=phono3py_yaml,
        use_pypolymlp=use_pypolymlp,
        log_level=log_level,
    )

    if produce_fc:
        compute_force_constants_from_datasets(
            ph3py,
            read_fc,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            symmetrize_fc=symmetrize_fc,
            is_compact_fc=is_compact_fc,
            use_pypolymlp=use_pypolymlp,
            mlp_params=mlp_params,
            log_level=log_level,
        )

    if log_level and ph3py.fc3 is not None:
        show_drift_fc3(ph3py.fc3, primitive=ph3py.primitive)
    if log_level and ph3py.fc2 is not None:
        show_drift_force_constants(
            ph3py.fc2, primitive=ph3py.phonon_primitive, name="fc2"
        )

    return ph3py


def set_dataset_and_force_constants(
    ph3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml] = None,
    fc3_filename: Optional[os.PathLike] = None,
    fc2_filename: Optional[os.PathLike] = None,
    forces_fc3_filename: Optional[Union[os.PathLike, Sequence]] = None,
    forces_fc2_filename: Optional[Union[os.PathLike, Sequence]] = None,
    phono3py_yaml_filename: Optional[os.PathLike] = None,
    cutoff_pair_distance: Optional[float] = None,
    use_pypolymlp: bool = False,
    log_level: int = 0,
) -> dict:
    """Set displacements, forces, and create force constants.

    Most of properties are stored in ph3py.

    Returns
    -------
    dict
        This contains flags indicating whether fc2 and fc3 were read from
        file(s) or not. This information can be different from ph3py.fc3 is
        (not) None and ph3py.fc2 is (not) None. Items are as follows:
            fc3 : bool
            fc2 : bool

    """
    read_fc = {"fc2": False, "fc3": False}
    read_fc["fc3"], dataset = _get_dataset_or_fc3(
        ph3py,
        ph3py_yaml=ph3py_yaml,
        fc3_filename=fc3_filename,
        forces_fc3_filename=forces_fc3_filename,
        phono3py_yaml_filename=phono3py_yaml_filename,
        cutoff_pair_distance=cutoff_pair_distance,
        log_level=log_level,
    )
    if not read_fc["fc3"]:
        if use_pypolymlp:
            ph3py.mlp_dataset = dataset
        else:
            ph3py.dataset = dataset
    read_fc["fc2"], phonon_dataset = _get_dataset_phonon_dataset_or_fc2(
        ph3py,
        ph3py_yaml=ph3py_yaml,
        fc2_filename=fc2_filename,
        forces_fc2_filename=forces_fc2_filename,
        log_level=log_level,
    )
    if not read_fc["fc2"]:
        ph3py.phonon_dataset = phonon_dataset

    return read_fc


def compute_force_constants_from_datasets(
    ph3py: Phono3py,
    read_fc: dict,
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[Union[dict, str]] = None,
    symmetrize_fc: bool = True,
    is_compact_fc: bool = True,
    use_pypolymlp: bool = False,
    mlp_params: Optional[Union[dict, str]] = None,
    displacement_distance: Optional[float] = None,
    number_of_snapshots: Optional[int] = None,
    random_seed: Optional[int] = None,
    log_level: int = 0,
):
    """Compute force constants from datasets.

    Parameters
    ----------
    read_fc : dict
        This contains flags indicating whether fc2 and fc3 were read from
        file(s) or not. This information can be different from ph3py.fc3 is
        (not) None and ph3py.fc2 is (not) None. Items are as follows:
            fc3 : bool
            fc2 : bool

    """
    fc3_calculator = extract_fc2_fc3_calculators(fc_calculator, 3)
    fc2_calculator = extract_fc2_fc3_calculators(fc_calculator, 2)
    if not read_fc["fc3"] and (ph3py.dataset or ph3py.mlp_dataset):
        if ph3py.mlp_dataset and use_pypolymlp:
            run_pypolymlp_to_compute_forces(
                ph3py,
                mlp_params=mlp_params,
                displacement_distance=displacement_distance,
                number_of_snapshots=number_of_snapshots,
                random_seed=random_seed,
                log_level=log_level,
            )
        ph3py.produce_fc3(
            symmetrize_fc3r=symmetrize_fc,
            is_compact_fc=is_compact_fc,
            fc_calculator=fc3_calculator,
            fc_calculator_options=extract_fc2_fc3_calculators(fc_calculator_options, 3),
        )

        if log_level and symmetrize_fc and fc_calculator is None:
            print("fc3 was symmetrized.")

    if not read_fc["fc2"] and (ph3py.dataset or ph3py.phonon_dataset):
        ph3py.produce_fc2(
            symmetrize_fc2=symmetrize_fc,
            is_compact_fc=is_compact_fc,
            fc_calculator=fc2_calculator,
            fc_calculator_options=extract_fc2_fc3_calculators(fc_calculator_options, 2),
        )
        if log_level and symmetrize_fc and fc_calculator is None:
            print("fc2 was symmetrized.")


def _get_dataset_or_fc3(
    ph3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml] = None,
    fc3_filename: Optional[os.PathLike] = None,
    forces_fc3_filename: Optional[Union[os.PathLike, Sequence]] = None,
    phono3py_yaml_filename: Optional[os.PathLike] = None,
    cutoff_pair_distance: Optional[float] = None,
    log_level: int = 0,
) -> tuple[bool, dict]:
    p2s_map = ph3py.primitive.p2s_map
    read_fc3 = False
    dataset = None
    if fc3_filename is not None or pathlib.Path("fc3.hdf5").exists():
        if fc3_filename is None:
            _fc3_filename = "fc3.hdf5"
        else:
            _fc3_filename = fc3_filename
        fc3 = read_fc3_from_hdf5(filename=_fc3_filename, p2s_map=p2s_map)
        _check_fc3_shape(ph3py, fc3, filename=_fc3_filename)
        ph3py.fc3 = fc3
        read_fc3 = True
        if log_level:
            print(f'fc3 was read from "{_fc3_filename}".')
    elif (
        ph3py_yaml is not None
        and ph3py_yaml.dataset is not None
        and forces_in_dataset(ph3py_yaml.dataset)
    ):
        dataset = _get_dataset_for_fc3(
            ph3py,
            ph3py_yaml,
            None,
            phono3py_yaml_filename,
            cutoff_pair_distance,
            log_level,
        )
    elif forces_fc3_filename is not None or pathlib.Path("FORCES_FC3").exists():
        if forces_fc3_filename is None:
            force_filename = "FORCES_FC3"
        else:
            force_filename = forces_fc3_filename
        dataset = _get_dataset_for_fc3(
            ph3py,
            ph3py_yaml,
            force_filename,
            phono3py_yaml_filename,
            cutoff_pair_distance,
            log_level,
        )
        if not forces_in_dataset(dataset):
            dataset = None

    return read_fc3, dataset


def _get_dataset_phonon_dataset_or_fc2(
    ph3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml] = None,
    fc2_filename: Optional[os.PathLike] = None,
    forces_fc2_filename: Optional[Union[os.PathLike, Sequence]] = None,
    log_level: int = 0,
) -> tuple[bool, dict, dict]:
    phonon_p2s_map = ph3py.phonon_primitive.p2s_map
    read_fc2 = False
    phonon_dataset = None
    if fc2_filename is not None or pathlib.Path("fc2.hdf5").exists():
        if fc2_filename is None:
            _fc2_filename = "fc2.hdf5"
        else:
            _fc2_filename = fc2_filename
        fc2 = read_fc2_from_hdf5(filename=_fc2_filename, p2s_map=phonon_p2s_map)
        _check_fc2_shape(ph3py, fc2, filename=_fc2_filename)
        ph3py.fc2 = fc2
        read_fc2 = True
        if log_level:
            print(f'fc2 was read from "{fc2_filename}".')
    elif (
        ph3py_yaml is not None
        and ph3py_yaml.phonon_dataset is not None
        and forces_in_dataset(ph3py_yaml.phonon_dataset)
    ):
        phonon_dataset = _get_dataset_for_fc2(
            ph3py,
            ph3py_yaml,
            None,
            "phonon_fc2",
            log_level,
        )
    elif (
        forces_fc2_filename is not None or pathlib.Path("FORCES_FC2").exists()
    ) and ph3py.phonon_supercell_matrix:
        if forces_fc2_filename is None:
            force_filename = forces_fc2_filename
        else:
            force_filename = forces_fc2_filename
        phonon_dataset = _get_dataset_for_fc2(
            ph3py,
            ph3py_yaml,
            force_filename,
            "phonon_fc2",
            log_level,
        )
        if not forces_in_dataset(phonon_dataset):
            phonon_dataset = None

    return read_fc2, phonon_dataset


def _get_dataset_for_fc3(
    ph3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml],
    force_filename,
    phono3py_yaml_filename,
    cutoff_pair_distance,
    log_level,
) -> dict:
    dataset = parse_forces(
        ph3py,
        ph3py_yaml=ph3py_yaml,
        cutoff_pair_distance=cutoff_pair_distance,
        force_filename=force_filename,
        phono3py_yaml_filename=phono3py_yaml_filename,
        fc_type="fc3",
        log_level=log_level,
    )
    return dataset


def _get_dataset_for_fc2(
    ph3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml],
    force_filename,
    fc_type,
    log_level,
):
    dataset = parse_forces(
        ph3py,
        ph3py_yaml=ph3py_yaml,
        force_filename=force_filename,
        fc_type=fc_type,
        log_level=log_level,
    )
    return dataset


def _check_fc2_shape(ph3py: Phono3py, fc2, filename="fc2.hdf5"):
    if ph3py.phonon_supercell_matrix is None:
        smat = ph3py.supercell_matrix
    else:
        smat = ph3py.phonon_supercell_matrix
    _check_fc_shape(ph3py, fc2, smat, filename)


def _check_fc3_shape(ph3py: Phono3py, fc3, filename="fc3.hdf5"):
    smat = ph3py.supercell_matrix
    _check_fc_shape(ph3py, fc3, smat, filename)


def _check_fc_shape(ph3py: Phono3py, fc, smat, filename):
    if len(ph3py.unitcell) * determinant(smat) != fc.shape[1]:
        msg = (
            f'Supercell size mismatch between "{filename}" and supercell matrix {smat}.'
        )
        raise RuntimeError(msg)
