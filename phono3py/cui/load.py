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
import warnings
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import phonopy.cui.load_helper as load_helper
from numpy.typing import NDArray
from phonopy.file_IO import get_supported_file_extensions_for_compression
from phonopy.harmonic.displacement import DisplacementDataset
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_calculator_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import determinant

from phono3py import Phono3py
from phono3py._lang import resolve_lang
from phono3py.cui.create_force_constants import (
    develop_or_load_pypolymlp,
    parse_forces,
)
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phono3py.interface.fc_calculator import (
    extract_fc2_fc3_calculators,
    extract_fc2_fc3_calculators_options,
    update_cutoff_fc_calculator_options,
)
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon3.dataset import forces_in_dataset
from phono3py.phonon3.displacement_fc3 import Fc3DisplacementDataset
from phono3py.phonon3.fc3 import show_drift_fc3


def load(
    phono3py_yaml: str
    | os.PathLike
    | None = None,  # phono3py.yaml-like must be the first argument.
    supercell_matrix: Sequence[int]
    | Sequence[Sequence[int]]
    | NDArray[np.int64]
    | None = None,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None = "auto",
    phonon_supercell_matrix: Sequence[int]
    | Sequence[Sequence[int]]
    | NDArray[np.int64]
    | None = None,
    is_nac: bool = True,
    calculator: str | None = None,
    unitcell: PhonopyAtoms | None = None,
    supercell: PhonopyAtoms | None = None,
    nac_params: dict | None = None,
    unitcell_filename: str | os.PathLike | None = None,
    supercell_filename: str | os.PathLike | None = None,
    born_filename: str | os.PathLike | None = None,
    forces_fc3_filename: str | os.PathLike | None = None,
    forces_fc2_filename: str | os.PathLike | None = None,
    fc3_filename: str | os.PathLike | None = None,
    fc2_filename: str | os.PathLike | None = None,
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    factor: float | None = None,  # deprecated
    produce_fc: bool = True,
    is_symmetry: bool = True,
    symmetrize_fc: bool = True,
    is_mesh_symmetry: bool = True,
    is_compact_fc: bool = True,
    use_pypolymlp: bool = False,
    mlp_params: dict | None = None,
    use_grg: bool = False,
    make_r0_average: bool = True,
    symprec: float = 1e-5,
    log_level: int = 0,
    lang: Literal["C", "Rust"] = "Rust",
) -> Phono3py:
    """Create a Phono3py instance from parameters and/or input files.

    A ``"phono3py_yaml"``-like file is parsed unless crystal structure
    information is given through ``unitcell_filename``,
    ``supercell_filename``, ``unitcell`` (PhonopyAtoms-like), or
    ``supercell`` (PhonopyAtoms-like). Even when a ``"phono3py_yaml"``-
    like file is parsed, parameters other than the crystal structure
    can be overwritten by keyword arguments.

    ``"fc3.hdf5"`` is read if it is found in the current directory.
    Otherwise, if ``"FORCES_FC3"`` and ``"phono3py_disp.yaml"`` are
    found, they are read and fc3 (and fc2) are produced. Similarly,
    ``"fc2.hdf5"`` is read if found; otherwise ``"FORCES_FC2"`` plus
    ``"phono3py_disp.yaml"`` are used to produce fc2.

    When ``forces_fc3_filename`` / ``forces_fc2_filename`` are not
    given, ``"FORCES_FC3"`` and ``"FORCES_FC2"`` are searched in the
    current directory. With type-1 ``"FORCES_FC3"`` (or
    ``"FORCES_FC2"``), ``"phono3py_disp.yaml"`` is also required.

    **Crystal structure** -- in order of priority::

        1. unitcell_filename  (with supercell_matrix)
        2. supercell_filename
        3. unitcell           (with supercell_matrix)
        4. supercell
        5. phono3py_yaml-like

    **Force sets or force constants** (optional) -- in order of
    priority::

        1. fc3_filename       (fc2_filename)
        2. forces_fc3_filename (forces_fc2_filename); for type-1 format
           phono3py_disp.yaml must also be supplied.
        3. fc3.hdf5 and fc2.hdf5 in the current directory.
        4. FORCES_FC3 and FORCES_FC2 in the current directory.
           FORCES_FC2 is optional; for type-1 format
           phono3py_disp.yaml is also searched. When FORCES_FC2 is
           missing, FORCES_FC3 is used to create fc2.

    **Parameters for non-analytical term correction (NAC)**
    (optional) -- in order of priority::

        1. born_filename
        2. nac_params
        3. phono3py_yaml_like.nac_params when present and is_nac=True
        4. BORN in the current directory when is_nac=True

    Parameters
    ----------
    phono3py_yaml : str or os.PathLike, optional
        Path to a ``"phono3py.yaml"``-like file. When given, the
        contents are parsed. Default is ``None``.
    supercell_matrix : array_like, optional
        Transformation matrix to the supercell from the unit cell.
        ``shape=(3,)`` or ``(3, 3)``, ``dtype=int``. A 1D array is
        treated as the diagonal of a 3x3 matrix. Default is the
        identity matrix.
    primitive_matrix : str or array_like, optional
        Transformation matrix to the primitive cell from the unit
        cell. Default is ``"auto"``, which guesses the matrix from
        crystal symmetry (centring types ``"F"``, ``"I"``, ``"A"``,
        ``"C"``, ``"R"``, or primitive ``"P"``). To use the unit cell
        as the primitive cell (identity transformation), pass
        ``"P"``. ``None`` is treated the same as ``"auto"``. When a
        centring symbol is given, the primitive matrix defined at
        https://spglib.github.io/spglib/definition.html is used.
        When a ``"phono3py.yaml"``-like file is loaded and contains a
        ``primitive_matrix``, that value takes priority over the
        default ``"auto"``.
    phonon_supercell_matrix : array_like, optional
        Supercell matrix used for fc2 when a different dimension is
        desired from the one used for fc3. Same format as
        ``supercell_matrix``. Default is ``None``, which uses
        ``supercell_matrix`` for fc2. Only valid when ``unitcell`` or
        ``unitcell_filename`` is given.
    is_nac : bool, optional
        When True, look for ``"BORN"``. When False, NAC is turned
        off. Default is True.
    calculator : str, optional
        Calculator name (``"vasp"``, ``"qe"``, ...) used to switch
        the set of physical units when parsing calculator input/output
        files. Default is ``None``, which is equivalent to ``"vasp"``.
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is ``None``.
    supercell : PhonopyAtoms, optional
        Input supercell. When given, ``primitive_matrix`` defaults to
        ``"auto"`` (can be overwritten) and ``supercell_matrix`` is
        ignored. Default is ``None``.
    nac_params : dict, optional
        Parameters for non-analytical term correction::

            'born':       Born effective charges,
                          shape=(atoms in primitive, 3, 3),
                          dtype=float.
            'dielectric': dielectric constant matrix,
                          shape=(3, 3), dtype=float.
            'factor':     unit conversion factor (float).

        Default is ``None``.
    unitcell_filename : str or os.PathLike, optional
        Path to a unit-cell file. Default is ``None``.
    supercell_filename : str or os.PathLike, optional
        Path to a supercell file. When given, ``supercell_matrix`` is
        ignored. Default is ``None``.
    born_filename : str or os.PathLike, optional
        Path to a ``"BORN"`` file containing NAC parameters. Default
        is ``None``.
    forces_fc3_filename : str, os.PathLike, or sequence, optional
        Either a two-element sequence of paths corresponding to
        ``("FORCES_FC3", "phono3py_disp.yaml")`` for the type-1
        format, or a single path to ``"FORCES_FC3"`` for the type-2
        format. Default is ``None``.
    forces_fc2_filename : str, os.PathLike, or sequence, optional
        Same as ``forces_fc3_filename`` but for fc2. Default is
        ``None``.
    fc3_filename : str or os.PathLike, optional
        Path to a file storing fc3 (e.g. ``"fc3.hdf5"``). Default is
        ``None``.
    fc2_filename : str or os.PathLike, optional
        Path to a file storing fc2 (e.g. ``"fc2.hdf5"``). Default is
        ``None``.
    fc_calculator : str, optional
        Force-constants calculator. One of ``None``,
        ``"traditional"``, ``"symfc"``, or ``"alm"``. Default is
        ``None`` (equivalent to ``"traditional"``).
    fc_calculator_options : str, optional
        Options string forwarded to the chosen calculator. Use
        ``"<fc2_opts>|<fc3_opts>"`` to set separate options for fc2
        and fc3. For ``"alm"``, each parameter is split by ``","`` and
        each key/value pair is written as ``"key = value"``. Default
        is ``None``.
    factor : float, optional
        **Deprecated.** Default is ``None``.
    produce_fc : bool, optional
        Compute force constants from displacements and forces. When
        False, only the dataset is set up. Default is True.
    is_symmetry : bool, optional
        Use crystal symmetry (beyond lattice translation). Default
        is True.
    symmetrize_fc : bool, optional
        Symmetrize the force constants after producing them.
        Applied per fc order (fc2 / fc3): only effective when the
        corresponding calculator (after ``"|"`` splitting of
        ``fc_calculator``) is ``None`` or ``"traditional"``. fc2 or
        fc3 produced by ``"symfc"`` or ``"alm"`` is already
        symmetrized by the solver. Default is True.
    is_mesh_symmetry : bool, optional
        Use reciprocal-mesh symmetry. Default is True.
    is_compact_fc : bool, optional
        Use compact force-constant shape::

            fc3 True:  (primitive, supercell, supercell, 3, 3, 3)
            fc3 False: (supercell, supercell, supercell, 3, 3, 3)
            fc2 True:  (primitive, supercell, 3, 3)
            fc2 False: (supercell, supercell, 3, 3)

        Default is True.
    use_pypolymlp : bool, optional
        Use pypolymlp to generate force constants. Default is False.
    mlp_params : dict, optional
        Parameters for the machine-learning potential. Default is
        ``None``.
    use_grg : bool, optional
        Use a generalized regular grid (GRG). Default is False.
    make_r0_average : bool, optional
        Average the fc3 real-to-reciprocal-space transformation over
        the three atoms in each triplet when True (default). When
        False, only the first atom is used. ``False`` is provided for
        rough backward compatibility with v2.x results.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is ``1e-5``.
    log_level : int, optional
        Verbosity control. Default is ``0``.
    lang : Literal["C", "Rust"], optional
        Backend implementation for compute-heavy kernels. ``"C"``
        uses the existing C extension; ``"Rust"`` selects the
        experimental phonors backend. Default is ``"Rust"``.

    """
    lang = resolve_lang(lang)
    if primitive_matrix is None:
        primitive_matrix = "auto"
    if (
        supercell is not None
        or supercell_filename is not None
        or unitcell is not None
        or unitcell_filename is not None
    ):
        _calculator = calculator
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=_calculator,
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
        assert ph3py_yaml.unitcell is not None
        cell = ph3py_yaml.unitcell.copy()
        _calculator = ph3py_yaml.calculator
        smat = ph3py_yaml.supercell_matrix
        ph_smat = ph3py_yaml.phonon_supercell_matrix
        if smat is None:
            smat = np.eye(3, dtype="int64", order="C")
        # When the caller leaves primitive_matrix at the default "auto",
        # a value stored in the yaml takes priority (preserves the cell
        # transformation that was used originally).
        if primitive_matrix == "auto" and ph3py_yaml.primitive_matrix is not None:
            pmat = ph3py_yaml.primitive_matrix
        else:
            pmat = primitive_matrix

        if nac_params is not None:
            _nac_params = nac_params
        elif is_nac:
            _nac_params = ph3py_yaml.nac_params
        else:
            _nac_params = None

    # Convert distance unit of unit cell to Angstrom
    physical_units = get_calculator_physical_units(_calculator)
    factor_to_A = physical_units.distance_to_A
    assert cell is not None
    cell.cell = cell.cell * factor_to_A

    ph3py = Phono3py(
        cell,
        smat,
        primitive_matrix=pmat,
        phonon_supercell_matrix=ph_smat,
        symprec=symprec,
        is_symmetry=is_symmetry,
        is_mesh_symmetry=is_mesh_symmetry,
        use_grg=use_grg,
        make_r0_average=make_r0_average,
        calculator=_calculator,
        log_level=log_level,
        lang=lang,
    )
    if factor is not None:
        warnings.warn(
            "factor parameter is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        ph3py._frequency_factor_to_THz = factor

    # NAC params
    if born_filename is not None or _nac_params is not None or is_nac:
        ph3py.nac_params = load_helper.get_nac_params(
            ph3py.primitive,
            _nac_params,
            born_filename,
            is_nac,
            physical_units.nac_factor,
            log_level=log_level,
        )

    load_fc2_and_fc3(
        ph3py, fc3_filename=fc3_filename, fc2_filename=fc2_filename, log_level=log_level
    )

    ph3py.dataset = select_and_load_dataset(  # type: ignore[assignment]
        ph3py,
        ph3py_yaml=ph3py_yaml,
        forces_fc3_filename=forces_fc3_filename,
        phono3py_yaml_filename=phono3py_yaml,
        calculator=_calculator,
        log_level=log_level,
    )

    ph3py.phonon_dataset = select_and_load_phonon_dataset(  # type: ignore[assignment]
        ph3py,
        ph3py_yaml=ph3py_yaml,
        forces_fc2_filename=forces_fc2_filename,
        calculator=_calculator,
        log_level=log_level,
    )

    if use_pypolymlp and ph3py.fc3 is None and forces_in_dataset(ph3py.dataset):  # type: ignore[arg-type]
        assert ph3py.dataset is not None
        ph3py.mlp_dataset = ph3py.dataset  # type: ignore[assignment]
        ph3py.dataset = None

    if produce_fc:
        if ph3py.fc3 is None and use_pypolymlp:
            develop_or_load_pypolymlp(ph3py, mlp_params=mlp_params, log_level=log_level)

        compute_force_constants_from_datasets(
            ph3py,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            symmetrize_fc=symmetrize_fc,
            is_compact_fc=is_compact_fc,
            use_symfc_projector=True,
        )

    if log_level and ph3py.fc3 is not None:
        show_drift_fc3(ph3py.fc3, primitive=ph3py.primitive)
    if log_level and ph3py.fc2 is not None:
        show_drift_force_constants(
            ph3py.fc2, primitive=ph3py.phonon_primitive, name="fc2"
        )

    return ph3py


def load_fc2_and_fc3(
    ph3py: Phono3py,
    fc3_filename: str | os.PathLike | None = None,
    fc2_filename: str | os.PathLike | None = None,
    read_fc3: bool = True,
    read_fc2: bool = True,
    log_level: int = 0,
) -> None:
    """Set force constants."""
    if read_fc3 and (fc3_filename is not None or pathlib.Path("fc3.hdf5").exists()):
        _load_fc3(ph3py, fc3_filename=fc3_filename, log_level=log_level)

    if read_fc2 and (fc2_filename is not None or pathlib.Path("fc2.hdf5").exists()):
        _load_fc2(ph3py, fc2_filename=fc2_filename, log_level=log_level)


def compute_force_constants_from_datasets(
    ph3py: Phono3py,
    fc_calculator: Literal["traditional", "symfc", "alm"] | str | None = None,
    fc_calculator_options: str | None = None,
    cutoff_pair_distance: float | None = None,
    symmetrize_fc: bool = True,
    is_compact_fc: bool = True,
    use_symfc_projector: bool = False,
) -> None:
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
    fc3_calc_opts = extract_fc2_fc3_calculators_options(fc_calculator_options, 3)
    fc3_calc_opts = update_cutoff_fc_calculator_options(
        fc3_calc_opts, cutoff_pair_distance
    )
    fc2_calc_opts = extract_fc2_fc3_calculators_options(fc_calculator_options, 2)
    if ph3py.fc3 is None and forces_in_dataset(ph3py.dataset):  # type: ignore[arg-type]
        ph3py.produce_fc3(
            is_compact_fc=is_compact_fc,
            fc_calculator=fc3_calculator,
            fc_calculator_options=fc3_calc_opts,
        )
        if symmetrize_fc and fc3_calculator in (None, "traditional"):
            use_projector_fc3 = use_symfc_projector and fc3_calculator is None
            ph3py.symmetrize_fc3(
                use_symfc_projector=use_projector_fc3, options=fc3_calc_opts
            )
            # When phonon_supercell_matrix is None, produce_fc3 also
            # populates fc2; post-symmetrize it with the same scheme.
            if ph3py.phonon_supercell_matrix is None:
                ph3py.symmetrize_fc2(
                    use_symfc_projector=use_projector_fc3, options=fc2_calc_opts
                )

    if ph3py.fc2 is None or fc3_calculator != fc2_calculator:
        if (
            ph3py.phonon_supercell_matrix is None and forces_in_dataset(ph3py.dataset)  # type: ignore[arg-type]
        ) or (
            ph3py.phonon_supercell_matrix is not None
            and forces_in_dataset(ph3py.phonon_dataset)  # type: ignore[arg-type]
        ):
            ph3py.produce_fc2(
                is_compact_fc=is_compact_fc,
                fc_calculator=fc2_calculator,
                fc_calculator_options=fc2_calc_opts,
            )
            if symmetrize_fc and fc2_calculator in (None, "traditional"):
                use_projector_fc2 = use_symfc_projector and fc2_calculator is None
                ph3py.symmetrize_fc2(
                    use_symfc_projector=use_projector_fc2, options=fc2_calc_opts
                )


def _load_fc3(
    ph3py: Phono3py,
    fc3_filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> None:
    p2s_map = ph3py.primitive.p2s_map
    if fc3_filename is None:
        _fc3_filename = "fc3.hdf5"
    else:
        _fc3_filename = fc3_filename  # type: ignore[assignment]
    fc3 = read_fc3_from_hdf5(filename=_fc3_filename, p2s_map=p2s_map)
    if isinstance(fc3, dict):
        # fc3 is read from a file with type-1 format.
        assert "fc3" in fc3
        _check_fc3_shape(ph3py, fc3["fc3"], filename=_fc3_filename)
        ph3py.fc3 = fc3["fc3"]
        assert "fc3_nonzero_indices" in fc3
        ph3py.fc3_nonzero_indices = fc3["fc3_nonzero_indices"]
        if log_level:
            print(f'fc3 and fc3 nonzero indices were read from "{_fc3_filename}".')
    else:
        _check_fc3_shape(ph3py, fc3, filename=_fc3_filename)
        ph3py.fc3 = fc3
        if log_level:
            print(f'fc3 was read from "{_fc3_filename}".')


def select_and_load_dataset(
    ph3py: Phono3py,
    ph3py_yaml: Phono3pyYaml | None = None,
    forces_fc3_filename: str | os.PathLike | None = None,
    phono3py_yaml_filename: str | os.PathLike | None = None,
    cutoff_pair_distance: float | None = None,
    calculator: str | None = None,
    log_level: int = 0,
) -> Fc3DisplacementDataset | None:
    """Select and load dataset for fc3."""
    # displacements and forces are in phono3py-yaml-like file
    if ph3py_yaml is not None and forces_in_dataset(ph3py_yaml.dataset):  # type: ignore[arg-type]
        dataset = _get_dataset_for_fc3(
            ph3py,
            ph3py_yaml,
            None,
            phono3py_yaml_filename,
            cutoff_pair_distance,
            calculator,
            log_level,
        )
        return dataset

    # displacements and forces are in FORCES_FC3-like file
    force_filename = _get_filename_with_extension("FORCES_FC3")
    if forces_fc3_filename is not None:
        force_filename = forces_fc3_filename  # type: ignore[assignment]
    if force_filename is not None:
        dataset = _get_dataset_for_fc3(
            ph3py,
            ph3py_yaml,
            force_filename,
            phono3py_yaml_filename,
            cutoff_pair_distance,
            calculator,
            log_level,
        )
        return dataset

    # dataset is in phono3py-yaml-like file
    if ph3py_yaml is not None and ph3py_yaml.dataset is not None:
        # not forces_in_dataset(ph3py_yaml.dataset)
        # but want to read displacement dataset.
        dataset = _get_dataset_for_fc3(
            ph3py,
            ph3py_yaml,
            None,
            phono3py_yaml_filename,
            cutoff_pair_distance,
            calculator,
            log_level,
        )
        return dataset

    return None


def _get_filename_with_extension(filename: str | os.PathLike) -> os.PathLike | None:
    for ext in get_supported_file_extensions_for_compression():
        if pathlib.Path(f"{filename}{ext}").is_file():
            return pathlib.Path(f"{filename}{ext}")
    return None


def _load_fc2(
    ph3py: Phono3py, fc2_filename: str | os.PathLike | None = None, log_level: int = 0
) -> None:
    phonon_p2s_map = ph3py.phonon_primitive.p2s_map
    if fc2_filename is None:
        _fc2_filename = "fc2.hdf5"
    else:
        _fc2_filename = fc2_filename  # type: ignore[assignment]
    fc2 = read_fc2_from_hdf5(filename=_fc2_filename, p2s_map=phonon_p2s_map)
    _check_fc2_shape(ph3py, fc2, filename=_fc2_filename)
    if log_level:
        print(f'fc2 was read from "{_fc2_filename}".')
    ph3py.fc2 = fc2


def select_and_load_phonon_dataset(
    ph3py: Phono3py,
    ph3py_yaml: Phono3pyYaml | None = None,
    forces_fc2_filename: str | os.PathLike | None = None,
    calculator: str | None = None,
    log_level: int = 0,
) -> DisplacementDataset | None:
    """Select and load phonon dataset for fc2."""
    if ph3py.phonon_supercell_matrix is None:
        return None

    if ph3py_yaml is not None and forces_in_dataset(ph3py_yaml.phonon_dataset):
        phonon_dataset = _get_dataset_for_fc2(
            ph3py,
            ph3py_yaml,
            None,
            calculator,
            log_level,
        )
        return phonon_dataset

    force_filename = _get_filename_with_extension("FORCES_FC2")
    if forces_fc2_filename is not None:
        force_filename = forces_fc2_filename  # type: ignore[assignment]
    if force_filename is not None:
        phonon_dataset = _get_dataset_for_fc2(
            ph3py,
            ph3py_yaml,
            force_filename,
            calculator,
            log_level,
        )
        return phonon_dataset

    if ph3py_yaml is not None and ph3py_yaml.phonon_dataset is not None:
        # not forces_in_dataset(ph3py_yaml.dataset)
        # but want to read displacement dataset.
        phonon_dataset = _get_dataset_for_fc2(
            ph3py,
            ph3py_yaml,
            None,
            calculator,
            log_level,
        )
        return phonon_dataset

    return None


def _get_dataset_for_fc3(
    ph3py: Phono3py,
    ph3py_yaml: Phono3pyYaml | None,
    force_filename: str | os.PathLike | None,
    phono3py_yaml_filename: str | os.PathLike | None,
    cutoff_pair_distance: float | None,
    calculator: str | None,
    log_level: int,
) -> Fc3DisplacementDataset:
    dataset = parse_forces(
        ph3py,
        ph3py_yaml=ph3py_yaml,
        cutoff_pair_distance=cutoff_pair_distance,
        force_filename=force_filename,  # type: ignore[arg-type]
        phono3py_yaml_filename=phono3py_yaml_filename,
        fc_type="fc3",
        calculator=calculator,
        log_level=log_level,
    )
    return cast(Fc3DisplacementDataset, dataset)


def _get_dataset_for_fc2(
    ph3py: Phono3py,
    ph3py_yaml: Phono3pyYaml | None,
    force_filename: str | os.PathLike | None,
    calculator: str | None,
    log_level: int,
) -> DisplacementDataset:
    dataset = parse_forces(
        ph3py,
        ph3py_yaml=ph3py_yaml,
        force_filename=force_filename,  # type: ignore[arg-type]
        fc_type="phonon_fc2",
        calculator=calculator,
        log_level=log_level,
    )
    return cast(DisplacementDataset, dataset)


def _check_fc2_shape(
    ph3py: Phono3py,
    fc2: NDArray[np.double],
    filename: str | os.PathLike = "fc2.hdf5",
) -> None:
    if ph3py.phonon_supercell_matrix is None:
        smat = ph3py.supercell_matrix
    else:
        smat = ph3py.phonon_supercell_matrix
    _check_fc_shape(ph3py, fc2, smat, filename)


def _check_fc3_shape(
    ph3py: Phono3py,
    fc3: NDArray[np.double],
    filename: str | os.PathLike = "fc3.hdf5",
) -> None:
    smat = ph3py.supercell_matrix
    _check_fc_shape(ph3py, fc3, smat, filename)


def _check_fc_shape(
    ph3py: Phono3py,
    fc: NDArray[np.double],
    smat: NDArray[np.int64],
    filename: str | os.PathLike,
) -> None:
    if len(ph3py.unitcell) * determinant(smat) != fc.shape[1]:
        msg = (
            f'Supercell size mismatch between "{filename}" and supercell matrix {smat}.'
        )
        raise RuntimeError(msg)
