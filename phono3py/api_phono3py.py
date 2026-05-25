"""Phono3py main class."""

# Copyright (C) 2016 Atsushi Togo
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
from __future__ import annotations

import copy
import dataclasses
import os
import warnings
from collections.abc import Sequence
from typing import (  # List and Optional are for < python3.10
    Any,
    List,
    Literal,
    Optional,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from phonopy import Phonopy
from phonopy.api_phonopy import set_data_to_phonopy_yaml
from phonopy.harmonic.displacement import (
    DisplacementDataset,
    Type2DisplacementDataset,
    directions_to_displacement_dataset,
    get_least_displacements,
    get_random_displacements_dataset,
)
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.harmonic.force_constants import (
    set_permutation_symmetry,
    set_translational_invariance,
    symmetrize_compact_force_constants,
    symmetrize_force_constants,
)
from phonopy.interface.fc_calculator import get_fc_solver
from phonopy.interface.mlp import PhonopyMLP
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.pypolymlp import (
    PypolymlpParams,
)
from phonopy.interface.symfc import (
    SymfcFCSolver,
    parse_symfc_options,
    symmetrize_by_projector,
)
from phonopy.phonon.grid import BZGrid
from phonopy.physical_units import get_calculator_physical_units, get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    Primitive,
    Supercell,
    get_primitive,
    get_primitive_matrix_with_auto,
    get_supercell,
    shape_supercell_matrix,
    warn_if_primitive_matrix_auto_changed_cell,
)
from phonopy.structure.symmetry import Symmetry

from phono3py._lang import resolve_lang
from phono3py.conductivity.calculators import LBTECalculator, RTACalculator
from phono3py.conductivity.lbte_init import get_thermal_conductivity_LBTE
from phono3py.conductivity.rta_init import get_thermal_conductivity_RTA
from phono3py.interface.fc_calculator import (
    FC3Solver,
    extract_fc2_fc3_calculators_options,
)
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon3.dataset import forces_in_dataset
from phono3py.phonon3.displacement_fc3 import (
    Fc3DisplacementDataset,
    get_third_order_displacements,
)
from phono3py.phonon3.fc3 import (
    cutoff_fc3_by_zero,
    set_permutation_symmetry_compact_fc3,
    set_permutation_symmetry_fc3,
    set_translational_invariance_compact_fc3,
    set_translational_invariance_fc3,
)
from phono3py.phonon3.imag_self_energy import (
    get_imag_self_energy,
    write_imag_self_energy,
)
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.real_self_energy import (
    get_real_self_energy,
    write_real_self_energy,
)
from phono3py.phonon3.spectral_function import run_spectral_function


@dataclasses.dataclass
class ImagSelfEnergyValues:
    """Parameters for imaginary self-energy calculation."""

    frequency_points: NDArray[np.double] | None
    gammas: NDArray[np.double]
    scattering_event_class: Literal[1, 2] | None = None
    detailed_gammas: list[NDArray[np.double]] | None = None


class Phono3py:
    """Phono3py main API.

    A ``Phono3py`` instance is created from a unit cell, a supercell
    matrix, and a primitive matrix. It manages displacement generation
    for fc3 (and optionally fc2 when a separate
    ``phonon_supercell_matrix`` is given), construction of second- and
    third-order force constants from displacement-force datasets,
    phonon-phonon interaction calculation on a reciprocal-space grid,
    and derived quantities such as the imaginary and real parts of the
    phonon self-energy, the spectral function, the joint density of
    states, and the lattice thermal conductivity (RTA, direct solution
    of the linearized phonon Boltzmann equation, and the Wigner
    transport equation).

    Most attributes are exposed as ``@property`` accessors documented
    individually below. See :ref:`phono3py_api` for a tutorial-style
    overview of the typical workflow.

    Examples
    --------
    >>> import numpy as np
    >>> from phono3py import Phono3py
    >>> from phonopy.structure.atoms import PhonopyAtoms
    >>> a = 3.111
    >>> c = 4.978
    >>> x = 1.0 / 3
    >>> unitcell = PhonopyAtoms(
    ...     symbols=["Al", "Al", "N", "N"],
    ...     cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]],
    ...     scaled_positions=[
    ...         [x, 2 * x, 0],
    ...         [2 * x, x, 0.5],
    ...         [x, 2 * x, 0.1181],
    ...         [2 * x, x, 0.6181],
    ...     ],
    ... )
    >>> ph3 = Phono3py(
    ...     unitcell, supercell_matrix=[3, 3, 2], primitive_matrix="auto"
    ... )
    >>> ph3.generate_displacements()
    >>> # Obtain forces by running an external calculator on
    >>> # ph3.supercells_with_displacements, then:
    >>> # ph3.forces = sets_of_forces
    >>> # ph3.produce_fc3()
    >>> # ph3.mesh_numbers = 30
    >>> # ph3.init_phph_interaction()
    >>> # ph3.run_thermal_conductivity(temperatures=range(0, 1001, 10))

    """

    def __init__(
        self,
        unitcell: PhonopyAtoms,
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
        cutoff_frequency: float = 1e-4,
        frequency_factor_to_THz: float | None = None,
        is_symmetry: bool = True,
        is_mesh_symmetry: bool = True,
        use_grg: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
        make_r0_average: bool = True,
        symprec: float = 1e-5,
        calculator: str | None = None,
        log_level: int = 0,
        lang: Literal["C", "Rust"] = "Rust",
    ):
        """Init method.

        Parameters
        ----------
        unitcell : PhonopyAtoms
            Input unit cell.
        supercell_matrix : array_like, optional
            Transformation matrix to the supercell from the unit cell.
            ``shape=(3,)`` or ``(3, 3)``, ``dtype=int``. A 1D array is
            treated as the diagonal of a 3x3 matrix. Although the
            default ``None`` yields the identity matrix, it is strongly
            recommended to give ``supercell_matrix`` explicitly.
        primitive_matrix : str or array_like, optional
            Transformation matrix to the primitive cell from the unit
            cell. ``shape=(3, 3)``, ``dtype=float``. Default is
            ``"auto"``, which guesses the matrix from crystal symmetry
            (centring types ``"F"``, ``"I"``, ``"A"``, ``"C"``, ``"R"``,
            or primitive ``"P"``). To use the unit cell as the primitive
            cell (identity transformation), pass ``"P"``. ``None`` is
            treated the same as ``"auto"``. When a centring symbol is
            given, the primitive matrix defined at
            https://spglib.github.io/spglib/definition.html is used.
        phonon_supercell_matrix : array_like, optional
            Supercell matrix used for fc2 when a different supercell
            from the one used for fc3 is desired (typically larger, to
            capture longer-ranged harmonic interactions). Same format as
            ``supercell_matrix``. Default is ``None``, which uses
            ``supercell_matrix`` for fc2.
        cutoff_frequency : float, optional
            Phonon frequencies below this value (in THz) are treated as
            zero in scattering and self-energy calculations. Default is
            ``1e-4``.
        frequency_factor_to_THz : float, optional
            Deprecated. Passing a non-``None`` value emits a
            ``DeprecationWarning``. By default the conversion factor for
            the chosen calculator is used.
        is_symmetry : bool, optional
            Use crystal symmetry in most calculations when True. Default
            is True.
        is_mesh_symmetry : bool, optional
            Use crystal symmetry in reciprocal-space grid handling when
            True. Default is True.
        use_grg : bool, optional
            Use a generalized regular grid (GRG) when True. The grid is
            generated on the reciprocal basis vectors of the
            conventional unit cell of the primitive cell, which can
            reduce the required sampling density for primitive cells
            with high symmetry. Default is False.
        SNF_coordinates : Literal["reciprocal", "direct"], optional
            Space in which the grid-generating matrix is computed via
            Smith normal form. Default is ``"reciprocal"``.
        make_r0_average : bool, optional
            Average the fc3 real-to-reciprocal-space transformation over
            the three atoms in each triplet when True (default). When
            False, only the first atom is used. ``False`` is provided
            for rough backward compatibility with v2.x results.
        symprec : float, optional
            Tolerance used to find crystal symmetry. Default is ``1e-5``.
        calculator : str, optional
            Calculator name (``"vasp"``, ``"qe"``, ...) used to switch
            the set of physical units. Default is ``None``, which is
            equivalent to ``"vasp"``.
        log_level : int, optional
            Verbosity control: ``0``, ``1``, or ``2``. Default is ``0``.
        lang : Literal["C", "Rust"], optional
            Backend implementation for compute-heavy kernels. ``"C"``
            uses the existing C extension; ``"Rust"`` selects the
            experimental phonors backend. Default is ``"Rust"``.

        """
        self._symprec = symprec
        if frequency_factor_to_THz is None:
            self._frequency_factor_to_THz = get_physical_units().DefaultToTHz
        else:
            warnings.warn(
                "frequency_factor_to_THz parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_mesh_symmetry = is_mesh_symmetry
        self._use_grg = use_grg
        self._SNF_coordinates: Literal["reciprocal", "direct"] = SNF_coordinates

        self._make_r0_average = make_r0_average

        self._cutoff_frequency = cutoff_frequency
        self._calculator = calculator
        self._log_level = log_level
        self._lang: Literal["C", "Rust"] = resolve_lang(lang)

        # Create supercell and primitive cell
        self._unitcell = unitcell
        self._supercell_matrix = np.array(
            shape_supercell_matrix(supercell_matrix), dtype="int64", order="C"
        )
        self._primitive_matrix = get_primitive_matrix_with_auto(
            self._unitcell, primitive_matrix, symprec=self._symprec
        )
        warn_if_primitive_matrix_auto_changed_cell(
            primitive_matrix, self._primitive_matrix
        )
        self._nac_params: dict | None = None
        if phonon_supercell_matrix is not None:
            self._phonon_supercell_matrix = np.array(
                shape_supercell_matrix(phonon_supercell_matrix),
                dtype="int64",
                order="C",
            )
        else:
            self._phonon_supercell_matrix = None  # type: ignore[assignment]
        self._supercell: Supercell
        self._primitive: Primitive
        self._phonon_supercell: Supercell
        self._phonon_primitive: Primitive
        self._build_supercell()
        self._build_primitive_cell()
        self._build_phonon_supercell()
        self._build_phonon_primitive_cell()

        self._sigmas: list[float | None] = [None]
        self._sigma_cutoff: float | None = None

        # Grid
        self._bz_grid = None

        # Set supercell, primitive, and phonon supercell symmetries
        self._symmetry: Symmetry
        self._primitive_symmetry: Symmetry
        self._phonon_supercell_symmetry: Symmetry
        self._search_symmetry()
        self._search_primitive_symmetry()
        self._search_phonon_supercell_symmetry()

        # Displacements and supercells
        self._dataset: Fc3DisplacementDataset | None = None
        self._phonon_dataset: DisplacementDataset | None = None
        self._supercells_with_displacements: list[PhonopyAtoms | None] | None = None
        self._phonon_supercells_with_displacements: list[PhonopyAtoms] | None = None

        # Thermal conductivity
        # RTACalculator (RTA) or LBTECalculator (standard/Wigner LBTE).
        self._thermal_conductivity: RTACalculator | LBTECalculator | None = None

        # Imaginary part of self energy at frequency points
        self._ise_params: ImagSelfEnergyValues | None = None

        self._grid_points: NDArray[np.int64] | None = None
        self._temperatures: NDArray[np.double] | None = None

        # Force constants
        self._fc2: NDArray[np.double] | None = None
        self._fc2_cutoff: float | None = None  # available only symfc
        self._fc3: NDArray[np.double] | None = None
        # available only symfc
        self._fc3_nonzero_indices: NDArray[np.byte] | None = None
        self._fc3_cutoff: float | None = None  # available only symfc

        # MLP
        self._mlp: PhonopyMLP | None = None
        self._mlp_dataset: Type2DisplacementDataset | None = None
        self._phonon_mlp: PhonopyMLP | None = None
        self._phonon_mlp_dataset: Type2DisplacementDataset | None = None

        # Setup interaction
        self._interaction: Interaction | None = None
        self._band_indices: Sequence[NDArray[np.int64]]
        self._band_indices_flatten: NDArray[np.int64]
        self._set_band_indices()

    @property
    def version(self) -> str:
        """Return phono3py release version number."""
        from phono3py import __version__

        return __version__

    @property
    def calculator(self) -> str | None:
        """Return calculator name such as ``'vasp'``, ``'qe'``, etc."""
        return self._calculator

    @property
    def lang(self) -> Literal["C", "Rust"]:
        """Return the selected backend implementation.

        ``"C"`` uses the existing C extension; ``"Rust"`` selects the
        experimental phonors backend.

        """
        return self._lang

    @property
    def fc3(self) -> NDArray[np.double] | None:
        """Setter and getter of third-order force constants (fc3).

        ``shape=(s, s, s, 3, 3, 3)`` (full) or
        ``(p, s, s, 3, 3, 3)`` (compact), where ``s`` and ``p`` are the
        numbers of atoms in the supercell and the primitive cell.
        ``dtype='double'``, ``order='C'``.

        """
        return self._fc3

    @fc3.setter
    def fc3(self, fc3: NDArray[np.double] | None) -> None:
        self._fc3 = fc3

    @property
    def fc3_nonzero_indices(self) -> NDArray[np.byte] | None:
        """Setter and getter of the non-zero element mask of fc3.

        Boolean mask of atom-triplet indices whose fc3 elements are kept
        non-zero. This is produced by symfc when a cutoff distance is
        specified; ``None`` otherwise.
        ``shape=(s, s, s)`` or ``(p, s, s)`` matching the shape of
        :attr:`fc3`. ``dtype='byte'``, ``order='C'``.

        """
        return self._fc3_nonzero_indices

    @fc3_nonzero_indices.setter
    def fc3_nonzero_indices(self, fc3_nonzero_indices: NDArray[np.byte] | None) -> None:
        self._fc3_nonzero_indices = fc3_nonzero_indices

    @property
    def fc3_cutoff(self) -> float | None:
        """Return the fc3 cutoff distance in Angstroms.

        Only available when fc3 was computed with symfc and a cutoff
        was specified; ``None`` otherwise.

        """
        return self._fc3_cutoff

    @property
    def fc2(self) -> NDArray[np.double] | None:
        """Setter and getter of second-order force constants (fc2).

        ``shape=(s, s, 3, 3)`` (full) or ``(p, s, 3, 3)`` (compact),
        where ``s`` and ``p`` are the numbers of atoms in the supercell
        and the primitive cell. ``dtype='double'``, ``order='C'``.

        """
        return self._fc2

    @fc2.setter
    def fc2(self, fc2: NDArray[np.double] | None) -> None:
        self._fc2 = fc2

    @property
    def fc2_cutoff(self) -> float | None:
        """Return the fc2 cutoff distance in Angstroms.

        Only available when fc2 was computed with symfc and a cutoff
        was specified; ``None`` otherwise.

        """
        return self._fc2_cutoff

    @property
    def force_constants(self) -> NDArray[np.double] | None:
        """Return fc2. Phonopy-compatible alias for :attr:`fc2`."""
        return self.fc2

    @property
    def sigmas(self) -> list[float | None]:
        """Setter and getter of smearing widths used for delta functions.

        Each element is either a Gaussian standard deviation (in the
        same unit as phonon frequencies, typically THz) or ``None``.
        A ``None`` entry switches that calculation to the linear
        tetrahedron method instead of Gaussian smearing.

        """
        return self._sigmas

    @sigmas.setter
    def sigmas(self, sigmas: Sequence[float | None] | float | int | None) -> None:
        if sigmas is None:
            self._sigmas = [
                None,
            ]
        elif isinstance(sigmas, float) or isinstance(sigmas, int):
            self._sigmas = [
                float(sigmas),
            ]
        else:
            self._sigmas = []
            for s in sigmas:
                if isinstance(s, float) or isinstance(s, int):
                    self._sigmas.append(float(s))
                elif s is None:
                    self._sigmas.append(None)

    @property
    def sigma_cutoff(self) -> float | None:
        """Setter and getter of the smearing cutoff width.

        Given as a multiple of the Gaussian standard deviation. For
        example, ``5`` truncates the tail at 5 sigma. ``None`` disables
        truncation.

        """
        return self._sigma_cutoff

    @sigma_cutoff.setter
    def sigma_cutoff(self, sigma_cutoff: float | None) -> None:
        self._sigma_cutoff = sigma_cutoff

    @property
    def nac_params(self) -> dict[str, Any] | None:
        """Setter and getter of parameters for non-analytical term correction.

        The dict has the following keys::

            'born':       ndarray, Born effective charges,
                          shape=(atoms in primitive, 3, 3),
                          dtype='double', order='C'.
            'factor':     float, unit conversion factor.
            'dielectric': ndarray, dielectric constant tensor,
                          shape=(3, 3), dtype='double', order='C'.

        Unlike the
        `BORN file <https://phonopy.github.io/phonopy/input-files.html#born-optional>`__,
        Born effective charges of all atoms in the primitive cell must
        be supplied (not just the symmetrically independent ones).

        """
        return self._nac_params

    @nac_params.setter
    def nac_params(self, nac_params: dict[str, Any] | None) -> None:
        self._nac_params = nac_params
        if self._interaction is not None:
            self._init_dynamical_matrix()

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return the ``DynamicalMatrix`` instance (not the matrix itself)."""
        if self._interaction is None:
            return None
        else:
            return self._interaction.dynamical_matrix

    @property
    def primitive(self) -> Primitive:
        """Return primitive cell."""
        return self._primitive

    @property
    def unitcell(self) -> PhonopyAtoms:
        """Return input unit cell."""
        return self._unitcell

    @property
    def supercell(self) -> Supercell:
        """Return supercell for fc3."""
        return self._supercell

    @property
    def phonon_supercell(self) -> Supercell:
        """Return supercell for fc2."""
        return self._phonon_supercell

    @property
    def phonon_primitive(self) -> Primitive:
        """Return primitive cell for fc2.

        This represents the same primitive cell as :attr:`primitive`,
        but is constructed from the fc2 supercell and may therefore not
        be numerically identical bit-for-bit.

        """
        return self._phonon_primitive

    @property
    def symmetry(self) -> Symmetry:
        """Return symmetry of the supercell."""
        return self._symmetry

    @property
    def primitive_symmetry(self) -> Symmetry:
        """Return symmetry of the primitive cell."""
        return self._primitive_symmetry

    @property
    def phonon_supercell_symmetry(self) -> Symmetry:
        """Return symmetry of the fc2 supercell (``phonon_supercell``)."""
        return self._phonon_supercell_symmetry

    @property
    def supercell_matrix(self) -> NDArray[np.int64]:
        """Return transformation matrix to the supercell from the unit cell.

        ``shape=(3, 3)``, ``dtype='int64'``, ``order='C'``.

        """
        return self._supercell_matrix

    @property
    def phonon_supercell_matrix(self) -> NDArray[np.int64] | None:
        """Return transformation matrix to the fc2 supercell from the unit cell.

        ``shape=(3, 3)``, ``dtype='int64'``, ``order='C'``. ``None`` when
        the same supercell as fc3 is used.

        """
        return self._phonon_supercell_matrix

    @property
    def primitive_matrix(self) -> NDArray[np.double] | None:
        """Return transformation matrix to the primitive cell from the unit cell.

        ``shape=(3, 3)``, ``dtype='double'``, ``order='C'``.

        """
        return self._primitive_matrix

    @property
    def unit_conversion_factor(self) -> float:
        """Return phonon frequency unit conversion factor.

        This factor converts
        ``sqrt(<force> / <distance> / <AMU>) / 2pi / 1e12`` to the
        chosen phonon frequency unit. The default value converts to THz
        for displacements in Angstroms and forces in eV/Angstrom (i.e.
        the VASP default).

        """
        return self._frequency_factor_to_THz

    @property
    def dataset(self) -> Fc3DisplacementDataset | None:
        """Setter and getter of the fc3 displacement-force dataset.

        Dataset containing displacements in supercells, and optionally
        forces and supercell energies. The format is one of two types.

        **Type 1. Pairs of atomic displacements per supercell (for fc3)**::

            {'natom': number of atoms in supercell,
             'first_atoms': [
               {'number': atom index of first displaced atom,
                'displacement': displacement in Cartesian coordinates,
                'forces': forces on atoms in supercell,
                'id': displacement id (1, 2, ..., n_first_atoms),
                'second_atoms': [
                  {'number': atom index of second displaced atom,
                   'displacement': displacement in Cartesian coordinates,
                   'forces': forces on atoms in supercell,
                   'supercell_energy': energy of supercell,
                   'pair_distance': distance between paired atoms,
                   'included': bool flag (with cutoff_pair_distance,
                       whether this pair is used to compute fc3),
                   'id': displacement id (n_first_atoms + 1, ...)},
                  ...
                ]},
               ...
             ]}

        **Type 2. All atomic displacements in each supercell**::

            {'displacements':      ndarray, dtype='double', order='C',
                                   shape=(supercells, atoms in supercell, 3),
             'forces':             ndarray, dtype='double', order='C',
                                   shape=(supercells, atoms in supercell, 3),
             'supercell_energies': ndarray, dtype='double',
                                   shape=(supercells,)}

        In type 2, ``displacements`` and ``forces`` may be given as any
        array-like that can be reshaped to
        ``(supercells, atoms in supercell, 3)``.

        For type 1, ``duplicates`` and ``cutoff_distance`` may also be
        present when pair displacements are generated; ``duplicates``
        gives duplicated supercell ids as pairs.

        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Fc3DisplacementDataset | None) -> None:
        if dataset is None:
            self._dataset = None
        elif "first_atoms" in dataset:
            self._dataset = copy.deepcopy(dataset)
        elif "displacements" in dataset:
            self._dataset = {}  # type: ignore[assignment]
            self.displacements = dataset["displacements"]
            if "forces" in dataset:
                self.forces = dataset["forces"]
            if "supercell_energies" in dataset:
                self.supercell_energies = dataset["supercell_energies"]
        else:
            raise RuntimeError("Data format of dataset is wrong.")

        self._supercells_with_displacements = None
        self._phonon_supercells_with_displacements = None

    @property
    def phonon_dataset(self) -> DisplacementDataset | None:
        """Setter and getter of the fc2 displacement-force dataset.

        Dataset containing displacements in the fc2 supercells, and
        optionally forces and supercell energies. The format is one of
        two types.

        **Type 1. One atomic displacement per supercell**::

            {'natom': number of atoms in supercell,
             'first_atoms': [
               {'number': atom index of displaced atom,
                'displacement': displacement in Cartesian coordinates,
                'forces': forces on atoms in supercell,
                'supercell_energy': energy of supercell},
               ...
             ]}

        **Type 2. All atomic displacements in each supercell**::

            {'displacements':      ndarray, dtype='double', order='C',
                                   shape=(supercells, atoms in supercell, 3),
             'forces':             ndarray, dtype='double', order='C',
                                   shape=(supercells, atoms in supercell, 3),
             'supercell_energies': ndarray, dtype='double',
                                   shape=(supercells,)}

        In type 2, ``displacements`` and ``forces`` may be given as any
        array-like that can be reshaped to
        ``(supercells, atoms in supercell, 3)``.

        """
        return self._phonon_dataset

    @phonon_dataset.setter
    def phonon_dataset(self, dataset: DisplacementDataset | None) -> None:
        if dataset is None:
            self._phonon_dataset = None
        elif "first_atoms" in dataset:
            self._phonon_dataset = copy.deepcopy(dataset)
        elif "displacements" in dataset:
            self._phonon_dataset = {}  # type: ignore[assignment]
            self.phonon_displacements = dataset["displacements"]
            if "forces" in dataset:
                self.phonon_forces = dataset["forces"]
            if "supercell_energies" in dataset:
                self.phonon_supercell_energies = dataset["supercell_energies"]
        else:
            raise RuntimeError("Data format of dataset is wrong.")

        self._phonon_supercells_with_displacements = None

    @property
    def mlp_dataset(self) -> Type2DisplacementDataset | None:
        """Setter and getter of the displacement-force dataset used to train an MLP.

        Uses the same supercell as the fc3 displacement-force dataset.
        Only the type-2 format is supported; the dict must contain the
        keys ``"displacements"``, ``"forces"``, and ``"supercell_energies"``.

        """
        return self._mlp_dataset

    @mlp_dataset.setter
    def mlp_dataset(self, mlp_dataset: Type2DisplacementDataset) -> None:
        self._check_mlp_dataset(mlp_dataset)
        self._mlp_dataset = mlp_dataset

    @property
    def phonon_mlp_dataset(self) -> Type2DisplacementDataset | None:
        """Setter and getter of the phonon MLP displacement-force dataset.

        Uses the same supercell as the fc2 displacement-force dataset.
        Only the type-2 format is supported; the dict must contain the
        keys ``"displacements"``, ``"forces"``, and ``"supercell_energies"``.

        """
        return self._phonon_mlp_dataset

    @phonon_mlp_dataset.setter
    def phonon_mlp_dataset(self, mlp_dataset: Type2DisplacementDataset) -> None:
        self._check_mlp_dataset(mlp_dataset)
        self._phonon_mlp_dataset = mlp_dataset

    @property
    def mlp(self) -> PhonopyMLP | None:
        """Setter and getter of the ``PhonopyMLP`` machine-learning potential."""
        return self._mlp

    @mlp.setter
    def mlp(self, mlp: PhonopyMLP) -> None:
        self._mlp = mlp

    @property
    def phonon_mlp(self) -> PhonopyMLP | None:
        """Return the ``PhonopyMLP`` instance used to predict fc2 forces."""
        return self._phonon_mlp

    @property
    def band_indices(self) -> Sequence[NDArray[np.int64]]:
        """Setter and getter of band indices.

        List of integer-index arrays selecting the bands at which
        ph-ph-interaction-derived properties are computed. Each entry
        is an ``NDArray[np.int64]``.

        """
        return self._band_indices

    @band_indices.setter
    def band_indices(
        self, band_indices: Sequence[Sequence[int]] | NDArray[np.int64] | None
    ) -> None:
        self._set_band_indices(band_indices=band_indices)

    def _set_band_indices(
        self, band_indices: Sequence[Sequence[int]] | NDArray[np.int64] | None = None
    ) -> None:
        if band_indices is None:
            num_band = len(self._primitive) * 3
            self._band_indices = [np.arange(num_band, dtype="int64")]
        else:
            self._band_indices = [np.array(bi, dtype="int64") for bi in band_indices]
        self._band_indices_flatten = np.hstack(self._band_indices, dtype="int64")

    @property
    def masses(self) -> NDArray[np.double]:
        """Setter and getter of atomic masses of the primitive cell.

        ``shape=(atoms in primitive,)``, ``dtype='double'``. Setting
        propagates the new masses to the unit cell, supercell, phonon
        primitive cell, and phonon supercell so that the cells stay
        consistent.

        """
        return self._primitive.masses

    @masses.setter
    def masses(self, masses: Sequence[float] | NDArray[np.double] | None) -> None:
        if masses is None:
            return
        p_masses = np.array(masses)
        self._primitive.masses = p_masses
        p2p_map = self._primitive.p2p_map
        s_masses = p_masses[[p2p_map[x] for x in self._primitive.s2p_map]]
        self._supercell.masses = s_masses
        u2s_map = self._supercell.u2s_map
        u_masses = s_masses[u2s_map]
        self._unitcell.masses = u_masses
        self._phonon_primitive.masses = p_masses
        p2p_map = self._phonon_primitive.p2p_map
        s_masses = p_masses[[p2p_map[x] for x in self._phonon_primitive.s2p_map]]
        self._phonon_supercell.masses = s_masses

    @property
    def supercells_with_displacements(self) -> list[PhonopyAtoms | None]:
        """Return the fc3 supercells with displacements applied.

        Built from the displacement dataset generated by
        :meth:`generate_displacements`. Pair displacements that fall
        outside the cutoff distance appear as ``None``.

        """
        if self._dataset is None:
            raise RuntimeError("Displacement dataset is not set.")
        if self._supercells_with_displacements is None:
            self._build_supercells_with_displacements()
        assert self._supercells_with_displacements is not None
        return self._supercells_with_displacements

    @property
    def phonon_supercells_with_displacements(self) -> list[PhonopyAtoms]:
        """Return the fc2 supercells with displacements applied.

        Built from the phonon displacement dataset generated by
        :meth:`generate_displacements` or :meth:`generate_fc2_displacements`.

        """
        if self._phonon_supercell_matrix is None:
            raise RuntimeError(
                "Phono3py instance is not created with phonon supercell matrix."
            )
        if self._phonon_supercells_with_displacements is None:
            if self._phonon_dataset is None:
                raise RuntimeError("Phonon displacement dataset is not set.")
            self._phonon_supercells_with_displacements = (
                self._build_phonon_supercells_with_displacements(
                    self._phonon_supercell, self._phonon_dataset
                )
            )
        return self._phonon_supercells_with_displacements

    @property
    def mesh_numbers(self) -> NDArray[np.int64] | None:
        """Setter and getter of the reciprocal-space sampling mesh.

        The getter returns the diagonal of the grid-generating matrix
        (``BZGrid.D_diag``), ``shape=(3,)``, ``dtype='int64'``, or
        ``None`` if no grid has been set. The setter accepts a scalar
        distance-like value, a length-3 sequence of integers, or a
        ``(3, 3)`` integer matrix; see :ref:`phono3py_api` for details.

        """
        if self._bz_grid is None:
            return None
        else:
            return self._bz_grid.D_diag

    @mesh_numbers.setter
    def mesh_numbers(
        self,
        mesh_numbers: float
        | NDArray[np.int64]
        | Sequence[int]
        | Sequence[Sequence[int]],
    ) -> None:
        self._set_mesh_numbers(mesh_numbers)

    @property
    def thermal_conductivity(
        self,
    ) -> RTACalculator | LBTECalculator | None:
        """Return the thermal-conductivity calculator instance.

        Populated by :meth:`run_thermal_conductivity`. ``RTACalculator``
        for the relaxation-time approximation (``is_LBTE=False``),
        ``LBTECalculator`` for the direct solution of the linearized
        Boltzmann equation or the Wigner transport equation
        (``is_LBTE=True``), or ``None`` before the calculation has run.

        """
        return self._thermal_conductivity

    @property
    def displacements(self) -> NDArray[np.double]:
        """Setter and getter of displacements in the fc3 supercells.

        See the docstring of :attr:`dataset` for the type-1 and type-2
        formats. The getter always returns a type-2-style ndarray (it
        synthesizes one when the underlying dataset is type-1) with
        ``shape=(supercells, atoms in supercell, 3)``,
        ``dtype='double'``, ``order='C'``.

        The setter accepts an array-like of the same shape and stores
        it as a type-2 dataset. Setting raises ``RuntimeError`` when
        the existing dataset is type-1.

        """
        dataset = self._dataset

        if dataset is None:
            raise RuntimeError("displacement dataset is not set.")

        if "first_atoms" in dataset:
            num_scells = len(dataset["first_atoms"])  # type: ignore[typeddict-item]
            for disp1 in dataset["first_atoms"]:  # type: ignore[typeddict-item]
                num_scells += len(disp1["second_atoms"])
            displacements = np.zeros(
                (num_scells, len(self._supercell), 3),
                dtype="double",
                order="C",
            )
            i = 0
            for disp1 in dataset["first_atoms"]:  # type: ignore[typeddict-item]
                displacements[i, disp1["number"]] = disp1["displacement"]
                i += 1
            for disp1 in dataset["first_atoms"]:  # type: ignore[typeddict-item]
                for disp2 in disp1["second_atoms"]:
                    displacements[i, disp2["number"]] = disp2["displacement"]
                    i += 1
        elif "displacements" in dataset:
            displacements = dataset["displacements"]
        else:
            raise RuntimeError("displacement dataset has wrong format.")

        return displacements

    @displacements.setter
    def displacements(
        self,
        displacements: Sequence[Sequence[Sequence[float]]]
        | Sequence[NDArray[np.double]]
        | NDArray[np.double],
    ) -> None:
        disps = np.array(displacements, dtype="double", order="C")
        natom = len(self._supercell)
        if disps.ndim != 3 or disps.shape[1:] != (natom, 3):
            raise RuntimeError("Array shape of displacements is incorrect.")
        if self._dataset is None:
            self._dataset = {}  # type: ignore[assignment]
        elif "first_atoms" in self._dataset:
            raise RuntimeError("Displacements are incompatible with dataset.")
        self._dataset["displacements"] = disps  # type: ignore[typeddict-unknown-key, index]
        self._supercells_with_displacements = None

    @property
    def forces(self) -> NDArray[np.double] | None:
        """Setter and getter of supercell forces in the fc3 dataset.

        ``shape=(supercells with displacements, atoms in supercell, 3)``,
        ``dtype='double'``, ``order='C'``.

        The order of supercells must match the order in
        :attr:`supercells_with_displacements`. The setter accepts any
        array-like with the same shape.

        """
        return self._get_forces_energies(target="forces")

    @forces.setter
    def forces(
        self,
        values: NDArray[np.double]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]],
    ) -> None:
        self._set_forces_energies(values, target="forces")

    @property
    def supercell_energies(self) -> NDArray[np.double] | None:
        """Setter and getter of supercell energies in the fc3 dataset.

        ``shape=(supercells with displacements,)``, ``dtype='double'``.

        The order of supercells must match the order in
        :attr:`supercells_with_displacements`. The setter accepts any
        array-like with the same shape.

        """
        return self._get_forces_energies(target="supercell_energies")

    @supercell_energies.setter
    def supercell_energies(self, values: Sequence[float] | NDArray[np.double]) -> None:
        self._set_forces_energies(values, target="supercell_energies")

    @property
    def phonon_displacements(self) -> NDArray[np.double]:
        """Setter and getter of displacements in the fc2 supercells.

        See the docstring of :attr:`phonon_dataset` for the type-1 and
        type-2 formats. The getter always returns a type-2-style
        ndarray (it synthesizes one when the underlying dataset is
        type-1) with
        ``shape=(supercells, atoms in supercell, 3)``,
        ``dtype='double'``, ``order='C'``.

        The setter accepts an array-like of the same shape and stores
        it as a type-2 dataset. Setting raises ``RuntimeError`` when
        the existing dataset is type-1 or when
        :attr:`phonon_supercell_matrix` is not set.

        """
        if self._phonon_dataset is None:
            raise RuntimeError("phonon_dataset is not set.")
        if "first_atoms" in self._phonon_dataset:
            num_scells = len(self._phonon_dataset["first_atoms"])
            natom = len(self._phonon_supercell)
            displacements = np.zeros((num_scells, natom, 3), dtype="double", order="C")
            for i, disp1 in enumerate(self._phonon_dataset["first_atoms"]):
                displacements[i, disp1["number"]] = disp1["displacement"]
        elif (
            "forces" in self._phonon_dataset or "displacements" in self._phonon_dataset
        ):
            displacements = self._phonon_dataset["displacements"]
        else:
            raise RuntimeError("displacement dataset has wrong format.")

        return displacements

    @phonon_displacements.setter
    def phonon_displacements(
        self,
        displacements: Sequence[Sequence[Sequence[float]]]
        | Sequence[NDArray[np.double]]
        | NDArray[np.double],
    ) -> None:
        if self._phonon_supercell_matrix is None:
            raise RuntimeError("phonon_supercell_matrix is not set.")

        disps = np.asarray(displacements, dtype="double", order="C")
        natom = len(self._phonon_supercell)
        if disps.ndim != 3 or disps.shape[1:] != (natom, 3):
            raise RuntimeError("Array shape of displacements is incorrect.")
        if self._phonon_dataset is not None and "first_atoms" in self._phonon_dataset:
            raise RuntimeError("Displacements are incompatible with dataset.")

        self._phonon_dataset = {"displacements": disps}
        self._phonon_supercells_with_displacements = None

    @property
    def phonon_forces(self) -> NDArray[np.double] | None:
        """Setter and getter of supercell forces in the fc2 dataset.

        ``shape=(supercells with displacements, atoms in supercell, 3)``,
        ``dtype='double'``, ``order='C'``.

        The order of supercells must match the order in
        :attr:`phonon_supercells_with_displacements`. The setter accepts
        any array-like with the same shape.

        """
        return self._get_phonon_forces_energies(target="forces")

    @phonon_forces.setter
    def phonon_forces(
        self,
        values: NDArray[np.double]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]],
    ) -> None:
        self._set_phonon_forces_energies(values, target="forces")

    @property
    def phonon_supercell_energies(self) -> NDArray[np.double] | None:
        """Setter and getter of supercell energies in the fc2 dataset.

        ``shape=(supercells with displacements,)``, ``dtype='double'``.

        The order of supercells must match the order in
        :attr:`phonon_supercells_with_displacements`. The setter accepts
        any array-like with the same shape.

        """
        return self._get_phonon_forces_energies(target="supercell_energies")

    @phonon_supercell_energies.setter
    def phonon_supercell_energies(
        self, values: Sequence[float] | NDArray[np.double]
    ) -> None:
        self._set_phonon_forces_energies(values, target="supercell_energies")

    @property
    def phph_interaction(self) -> Interaction | None:
        """Return the ph-ph ``Interaction`` instance.

        Created by :meth:`init_phph_interaction`. ``None`` before
        initialization.

        """
        return self._interaction

    @property
    def detailed_gammas(self) -> list[NDArray[np.double]] | None:
        """Return scattering-event-resolved imaginary self-energies.

        Populated by
        :meth:`run_imag_self_energy(..., keep_gamma_detail=True)`.
        Returns ``None`` when keep_gamma_detail was not requested.
        Raises ``RuntimeError`` if :meth:`run_imag_self_energy` has not
        been called.

        """
        if self._ise_params is None:
            raise RuntimeError("Imaginary self energy parameters are not set.")
        return self._ise_params.detailed_gammas

    @property
    def grid(self) -> BZGrid | None:
        """Return the Brillouin-zone grid (``BZGrid``) used for the calculation.

        ``None`` before :attr:`mesh_numbers` has been set.

        """
        return self._bz_grid

    def init_phph_interaction(
        self,
        nac_q_direction: NDArray[np.double] | Sequence[float] | None = None,
        constant_averaged_interaction: float | None = None,
        frequency_scale_factor: float | None = None,
        symmetrize_fc3q: bool = False,
        lapack_zheev_uplo: Literal["L", "U"] = "L",
        openmp_per_triplets: bool | None = None,
    ) -> None:
        """Initialize ph-ph interaction calculation.

        This method creates an instance of Interaction class, which
        is necessary to run ph-ph interaction calculation.
        The input data such as grids, force constants, etc, are
        stored to be ready for the calculation.

        Note
        ----
        fc3 and fc2, and optionally nac_params have to be set before calling
        this method. fc3 and fc2 can be made either from sets of forces
        and displacements of supercells or be set simply via attributes.

        Parameters
        ----------
        nac_q_direction : array_like, optional
            Direction of q-vector watching from Gamma point used for
            non-analytical term correction. This is effective only at q=0
            (physically q->0). The direction is given in crystallographic
            (fractional) coordinates.
            shape=(3,), dtype='double'.
            Default value is None, which means this feature is not used.
        constant_averaged_interaction : float, optional
            Ph-ph interaction strength array is replaced by a scalar value.
            Default is None, which means this feature is not used.
        frequency_scale_factor : float, optional
            All phonon frequencies are scaled by this value. Default is None,
            which means phonon frequencies are not scaled.
        symmetrize_fc3q : bool, optional
            fc3 in phonon space is symmetrized by permutation symmetry.
            Default is False.
        lapack_zheev_uplo : str, optional
            'L' or 'U'. Default is 'L'. This is passed to LAPACK zheev
            used for phonon solver.
        openmp_per_triplets : bool or None, optional, default is None
            Normally this parameter should not be touched.
            When `True`, ph-ph interaction strength calculation runs with
            OpenMP distribution over triplets, and over bands when `False`.
            `None` will choose one of them automatically.

        """
        if self._bz_grid is None:
            msg = "Phono3py.mesh_numbers of instance has to be set."
            raise RuntimeError(msg)

        if self._fc2 is None:
            msg = "Phono3py.fc2 of instance is not found."
            raise RuntimeError(msg)

        self._interaction = Interaction(
            self._primitive,
            self._bz_grid,
            self._primitive_symmetry,
            fc3=self._fc3,
            fc3_nonzero_indices=self._fc3_nonzero_indices,
            band_indices=self._band_indices_flatten,
            constant_averaged_interaction=constant_averaged_interaction,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            frequency_scale_factor=frequency_scale_factor,
            cutoff_frequency=self._cutoff_frequency,
            is_mesh_symmetry=self._is_mesh_symmetry,
            symmetrize_fc3q=symmetrize_fc3q,
            make_r0_average=self._make_r0_average,
            lapack_zheev_uplo=lapack_zheev_uplo,
            openmp_per_triplets=openmp_per_triplets,
            lang=self._lang,
        )
        self._interaction.nac_q_direction = nac_q_direction
        self._init_dynamical_matrix()

    def set_phonon_data(
        self,
        frequencies: NDArray[np.double],
        eigenvectors: NDArray[np.cdouble],
        grid_address: NDArray[np.int64],
    ) -> None:
        """Set phonon frequencies and eigenvectors in Interaction instance.

        Harmonic phonon information is stored in Interaction instance. For
        example, this information store in a file is read and passed to
        Phono3py instance by using this method. The grid_address is used
        for the consistency check.

        Parameters
        ----------
        frequencies : array_like
            Phonon frequencies.
            shape=(num_grid_points, num_band), dtype='double', order='C'
        eigenvectors : array_like
            Phonon eigenvectors
            shape=(num_grid_points, num_band, num_band)
            dtype='cdouble', order='C'
        grid_address : array_like
            Grid point addresses by integers. The first dimension may not be
            prod(mesh) because it includes Brillouin zone boundary. The detail
            is found in the docstring of
            phono3py.phonon3.triplets.get_triplets_at_q.
            shape=(num_grid_points, 3), dtype=int

        """
        if self._interaction is not None:
            self._interaction.set_phonon_data(frequencies, eigenvectors, grid_address)

    def get_phonon_data(
        self,
    ) -> tuple[NDArray[np.double], NDArray[np.cdouble], NDArray[np.int64]]:
        """Return phonon frequencies, eigenvectors, and grid addresses.

        Grid addresses give the q-point location with respect to the
        reciprocal basis vectors by integers, with::

            q_points = grid_address / np.array(mesh, dtype='double')

        Returns
        -------
        frequencies : ndarray
            ``shape=(num_grid_points, num_band)``, ``dtype='double'``,
            ``order='C'``.
        eigenvectors : ndarray
            ``shape=(num_grid_points, num_band, num_band)``,
            ``dtype='cdouble'``, ``order='C'``.
        grid_address : ndarray
            Integer grid-point addresses (the first dimension may
            exceed ``prod(mesh)`` because BZ-boundary points are
            included).
            ``shape=(num_grid_points, 3)``, ``dtype=int``.

        Raises
        ------
        RuntimeError
            When :meth:`init_phph_interaction` has not been called.

        See Also
        --------
        set_phonon_data

        """
        if self._interaction is not None:
            freqs, eigvecs, _ = self._interaction.get_phonons()
            # In Phono3py, if self._interaction is not None, phonon data should be set.
            assert freqs is not None and eigvecs is not None
            return freqs, eigvecs, self._interaction.bz_grid.addresses
        else:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

    def run_phonon_solver(
        self, grid_points: Sequence[int] | NDArray[np.int64] | None = None
    ) -> None:
        """Run harmonic phonon calculation on grid points.

        Parameters
        ----------
        grid_points : array_like or None, optional
            A list of grid point indices of Phono3py.grid.addresses.
            Specifying None runs all phonons on the grid points unless
            those phonons were already calculated. Normally phonons at
            [0, 0, 0] point is already calculated before calling this method.
            Phonon calculations are performed automatically when needed
            internally for ph-ph calculation. Therefore calling this method
            is not necessary in most cases.
            The phonon results are obtained by Phono3py.get_phonon_data().

        """
        if self._interaction is not None:
            self._interaction.run_phonon_solver(grid_points=grid_points)
        else:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

    def generate_displacements(
        self,
        distance: float | None = None,
        cutoff_pair_distance: float | None = None,
        is_plusminus: bool | Literal["auto"] = "auto",
        is_diagonal: bool = True,
        number_of_snapshots: int | Literal["auto"] | None = None,
        random_seed: int | None = None,
        max_distance: float | None = None,
        number_estimation_factor: float | None = None,
    ) -> None:
        """Generate the fc3 displacement dataset in the supercell.

        Two modes are supported depending on whether
        ``number_of_snapshots`` is given.

        **Systematic mode** (``number_of_snapshots`` is ``None``):
        single and pair atomic displacements are generated using crystal
        symmetry. For fc3 two atoms are displaced per configuration.
        The first displacement is taken in the perfect supercell along
        its basis vectors -- this keeps more symmetry intact, which
        typically reduces the number of second displacements needed.
        The second displacement is then taken in the once-displaced
        supercell.

        **Random mode** (``number_of_snapshots`` is an int or
        ``"auto"``): the requested number of supercells with random
        atomic displacements is generated. When ``max_distance`` is
        given, displacements use random directions and random
        distances drawn uniformly from ``[distance, max_distance]``;
        otherwise all atoms are displaced by the same Euclidean
        distance equal to ``distance``.

        Note
        ----
        When ``phonon_supercell_matrix`` is not given, fc2 is computed
        from the same displacements as fc3. When it is given, fc2
        displacements in the phonon supercell are generated separately
        (unless they already exist). To control fc2 displacements
        independently, call :meth:`generate_fc2_displacements`.

        Parameters
        ----------
        distance : float, optional
            Constant displacement Euclidean distance in Angstroms.
            Default is ``None``, which means ``0.03``. In random-
            direction-random-distance mode this value also acts as the
            minimum distance: sampled distances smaller than this are
            clamped to it.
        cutoff_pair_distance : float, optional
            Cutoff Euclidean distance (in Angstroms) used to drop pair
            displacements that are too far apart from the fc3
            calculation. Default is ``None`` (no cutoff).
        is_plusminus : bool or "auto", optional
            With ``True``, atoms are displaced in both positive and
            negative directions. With ``False``, only one direction.
            With ``"auto"`` (default), both directions are used unless
            they are symmetrically equivalent, in which case only one is
            kept.
        is_diagonal : bool, optional
            With ``True`` (default), second displacements may be chosen
            off-axis if doing so reduces the displacement count. With
            ``False``, second displacements are taken strictly along
            the supercell basis vectors.
        number_of_snapshots : int, "auto", or None, optional
            Number of supercell snapshots with random displacements.
            With ``"auto"``, the minimum required number is estimated
            using symfc and then multiplied by
            ``number_estimation_factor``. Default is ``None``
            (systematic mode).
        random_seed : int or None, optional
            Random seed for random displacement generation. Default is
            ``None``.
        max_distance : float or None, optional
            When specified, displacements use random direction and
            random distance with ``distance`` as the lower bound and
            ``max_distance`` as the upper bound of a uniform
            distribution. Default is ``None``.
        number_estimation_factor : float, optional
            Multiplier applied to the symfc estimate when
            ``number_of_snapshots="auto"``. Default is ``None``, which
            uses ``8`` when ``max_distance`` is given and ``4``
            otherwise.

        """
        if distance is None:
            _distance = 0.03
        else:
            _distance = distance

        if number_of_snapshots is not None and (
            number_of_snapshots == "auto" or number_of_snapshots > 0
        ):
            if number_of_snapshots == "auto":
                if cutoff_pair_distance is None:
                    options = None
                else:
                    options = {"cutoff": {3: cutoff_pair_distance}}
                _number_of_snapshots = SymfcFCSolver(
                    self._supercell,
                    symmetry=self._symmetry,
                    options=options,
                ).estimate_numbers_of_supercells(orders=[3])[3]
                if number_estimation_factor is None:
                    if max_distance is None:
                        _number_of_snapshots *= 4
                    else:
                        _number_of_snapshots *= 8
                else:
                    _number_of_snapshots *= number_estimation_factor  # type: ignore[assignment]
                    _number_of_snapshots = int(_number_of_snapshots)
            else:
                _number_of_snapshots = number_of_snapshots
            self._dataset = self._generate_random_displacements(  # type: ignore[assignment]
                _number_of_snapshots,
                len(self._supercell),
                distance=_distance,
                is_plusminus=is_plusminus is True,
                random_seed=random_seed,
                max_distance=max_distance,
            )
            if cutoff_pair_distance is not None:
                self._dataset["cutoff_distance"] = cutoff_pair_distance  # type: ignore[typeddict-unknown-key, index]
        else:
            self._dataset = get_third_order_displacements(
                self._supercell,
                self._symmetry,
                _distance,
                is_plusminus=is_plusminus,
                is_diagonal=is_diagonal,
                cutoff_pair_distance=cutoff_pair_distance,
            )
        self._supercells_with_displacements = None

        if self._phonon_supercell_matrix is not None and self._phonon_dataset is None:
            self.generate_fc2_displacements(
                distance=_distance, is_plusminus=is_plusminus, is_diagonal=False
            )

    def generate_fc2_displacements(
        self,
        distance: float | None = None,
        is_plusminus: bool | Literal["auto"] = "auto",
        is_diagonal: bool = False,
        number_of_snapshots: int | Literal["auto"] | None = None,
        random_seed: int | None = None,
        max_distance: float | None = None,
    ) -> None:
        """Generate the fc2 displacement dataset in the phonon supercell.

        Two modes are supported. In **systematic mode** (default), single
        atomic displacements in the phonon supercell are generated using
        crystal symmetry. In **random mode** (when
        ``number_of_snapshots`` is given), the requested number of
        supercells with random displacements is generated. Calling this
        method clears any cached fc2 supercells with displacements.

        Note
        ----
        ``is_diagonal=False`` is the default so that the fc2
        first-displacement direction stays consistent with the
        first-displacement direction used by the fc3 pair generator.

        Parameters
        ----------
        distance : float, optional
            Constant displacement Euclidean distance in Angstroms.
            Default is ``None``, which means ``0.03``. In random-
            direction-random-distance mode this value also acts as the
            minimum distance: sampled distances smaller than this are
            clamped to it.
        is_plusminus : bool or "auto", optional
            With ``True``, atoms are displaced in both positive and
            negative directions. With ``False``, only one direction.
            With ``"auto"`` (default), both directions are used unless
            they are symmetrically equivalent, in which case only one
            is kept.
        is_diagonal : bool, optional
            With ``False`` (default), displacements are taken strictly
            along the supercell basis vectors. With ``True``, off-axis
            directions may be chosen if doing so reduces the
            displacement count.
        number_of_snapshots : int, "auto", or None, optional
            Number of supercell snapshots with random displacements.
            With ``"auto"``, the minimum required number is estimated
            using symfc and then doubled. Default is ``None``
            (systematic mode).
        random_seed : int or None, optional
            Random seed for random displacement generation. Default is
            ``None``.
        max_distance : float or None, optional
            When specified, displacements use random direction and
            random distance with ``distance`` as the lower bound and
            ``max_distance`` as the upper bound of a uniform
            distribution. Default is ``None``.

        """
        if distance is None:
            _distance = 0.03
        else:
            _distance = distance

        if number_of_snapshots is not None and (
            number_of_snapshots == "auto" or number_of_snapshots > 0
        ):
            if number_of_snapshots == "auto":
                _number_of_snapshots = (
                    SymfcFCSolver(
                        self._supercell, symmetry=self._symmetry
                    ).estimate_numbers_of_supercells(orders=[2])[2]
                    * 2
                )
            else:
                _number_of_snapshots = number_of_snapshots

            self._phonon_dataset = self._generate_random_displacements(
                _number_of_snapshots,
                len(self._phonon_supercell),
                distance=_distance,
                is_plusminus=is_plusminus is True,
                random_seed=random_seed,
                max_distance=max_distance,
            )
        else:
            phonon_displacement_directions = get_least_displacements(
                self._phonon_supercell_symmetry,
                is_plusminus=is_plusminus,
                is_diagonal=is_diagonal,
            )
            self._phonon_dataset = directions_to_displacement_dataset(  # type: ignore[assignment]
                phonon_displacement_directions, _distance, self._phonon_supercell
            )
        self._phonon_supercells_with_displacements = None

    def produce_fc3(
        self,
        symmetrize_fc3r: bool | None = None,
        is_compact_fc: bool = True,
        fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
        fc_calculator_options: str | None = None,
        use_symfc_projector: bool | None = None,
    ) -> None:
        """Calculate fc3 (and optionally fc2) from displacements and forces.

        The solver is chosen by ``fc_calculator``:

        - ``None`` (default) or ``"traditional"``: built-in
          finite-difference solver. The returned fc3 is **not**
          symmetrized; to enforce translational and permutation
          invariance, call :meth:`symmetrize_fc3` afterwards (and
          :meth:`symmetrize_fc2` when applicable).
        - ``"symfc"``: symfc solver, which returns symmetrized force
          constants in one shot. When a cutoff distance is configured
          via ``fc_calculator_options``, it is captured into
          :attr:`fc3_cutoff` (and :attr:`fc2_cutoff` when fc2 is
          produced here), and the boolean mask of non-zero atomic
          triplets is captured into :attr:`fc3_nonzero_indices`.
        - ``"alm"``: ALM solver, which handles symmetrization
          internally.

        When ``phonon_supercell_matrix`` is not set, fc2 is produced
        alongside fc3 and stored in :attr:`fc2`. When
        ``phonon_supercell_matrix`` is set, :attr:`fc2` is **not**
        populated here -- call :meth:`produce_fc2` to compute fc2 in
        the larger phonon supercell.

        Parameters
        ----------
        symmetrize_fc3r : bool, optional
            **Deprecated.** Passing any value emits a
            ``DeprecationWarning``. With the traditional solver,
            call :meth:`symmetrize_fc3` (and :meth:`symmetrize_fc2`
            when applicable) after :meth:`produce_fc3` instead.
            Default is ``None``.
        is_compact_fc : bool, optional
            fc3 shape::

                True:  (primitive, supercell, supercell, 3, 3, 3)
                False: (supercell, supercell, supercell, 3, 3, 3)

            Default is ``True``.
        fc_calculator : str, optional
            Force-constants calculator. One of ``None``,
            ``"traditional"``, ``"symfc"``, or ``"alm"``. Default is
            ``None`` (equivalent to ``"traditional"``).
        fc_calculator_options : str, optional
            Options string forwarded to the chosen calculator. Use
            ``"<fc2_opts>|<fc3_opts>"`` to set separate options for
            fc2 and fc3; without ``"|"`` the same options apply to
            both. For example, ``"cutoff=4|cutoff=3"`` sets cutoff 4
            for fc2 and 3 for fc3.
        use_symfc_projector : bool, optional
            **Deprecated.** Passing any value emits a
            ``DeprecationWarning``. Call
            ``symmetrize_fc3(use_symfc_projector=True)`` after
            :meth:`produce_fc3` instead. Default is ``None``.

        """
        if symmetrize_fc3r is not None:
            warnings.warn(
                "The symmetrize_fc3r parameter of Phono3py.produce_fc3 is "
                "deprecated. Call Phono3py.symmetrize_fc3 (and "
                "Phono3py.symmetrize_fc2 when applicable) after "
                "Phono3py.produce_fc3 instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if use_symfc_projector is not None:
            warnings.warn(
                "The use_symfc_projector parameter of Phono3py.produce_fc3 is "
                "deprecated. Call "
                "Phono3py.symmetrize_fc3(use_symfc_projector=True) after "
                "Phono3py.produce_fc3 instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        fc_solver_name = fc_calculator if fc_calculator is not None else "traditional"
        fc_solver = FC3Solver(
            fc_solver_name,
            self._supercell,
            symmetry=self._symmetry,
            dataset=self._dataset,  # type: ignore[arg-type]
            is_compact_fc=is_compact_fc,
            primitive=self._primitive,
            orders=[2, 3],
            options=fc_calculator_options,
            log_level=self._log_level,
            lang=self._lang,
        )
        fc2 = fc_solver.force_constants[2]
        fc3 = fc_solver.force_constants[3]

        self._fc3 = fc3  # type: ignore[assignment]
        self._fc3_nonzero_indices = None

        if fc_calculator == "traditional" or fc_calculator is None:
            if symmetrize_fc3r:
                fc3_calc_opts = extract_fc2_fc3_calculators_options(
                    fc_calculator_options, 3
                )
                if use_symfc_projector and fc_calculator is None:
                    self.symmetrize_fc3(use_symfc_projector=True, options=fc3_calc_opts)
                else:
                    self.symmetrize_fc3(options=fc3_calc_opts)
        elif fc_calculator == "symfc":
            symfc_solver = cast(SymfcFCSolver, fc_solver.fc_solver)
            fc3_nonzero_elems = symfc_solver.get_nonzero_atomic_indices_fc3()
            options = symfc_solver.options
            if options is not None and "cutoff" in options:
                self._fc3_cutoff = options["cutoff"].get(3, None)
                self._fc2_cutoff = options["cutoff"].get(2, None)
            if fc3_nonzero_elems is not None:
                if is_compact_fc:
                    self._fc3_nonzero_indices = np.array(  # type: ignore[assignment]
                        fc3_nonzero_elems[self._primitive.p2s_map],
                        dtype="byte",
                        order="C",
                    )
                else:
                    self._fc3_nonzero_indices = np.array(  # type: ignore[assignment]
                        fc3_nonzero_elems, dtype="byte", order="C"
                    )

        if self._phonon_supercell_matrix is None:
            self._fc2 = fc2
            if fc_calculator == "traditional" or fc_calculator is None:
                if symmetrize_fc3r:
                    fc2_calc_opts = extract_fc2_fc3_calculators_options(
                        fc_calculator_options, 2
                    )
                    if use_symfc_projector and fc_calculator is None:
                        self.symmetrize_fc2(
                            use_symfc_projector=True, options=fc2_calc_opts
                        )
                    else:
                        self.symmetrize_fc2(options=fc2_calc_opts)

    def symmetrize_fc3(
        self,
        use_symfc_projector: bool = False,
        options: str | None = None,
    ) -> None:
        """Symmetrize fc3 by symfc projector or the traditional approach.

        Parameters
        ----------
        use_symfc_projector : bool, optional
            If ``True``, symmetrize force constants with the symfc
            projector instead of the traditional approach.
        options : str or None, optional
            Options string. Accepted values depend on the backend::

                symfc projector:
                    "use_mkl=true"  call sparse_dot_mkl
                                    (requires sparse_dot_mkl to be
                                    installed).
                traditional:
                    "level=N"       apply translational and permutation
                                    symmetries alternately N times.
                                    Default level is 3.

        """
        if self._fc3 is None:
            raise RuntimeError("fc3 is not set. Call produce_fc3 first.")

        if use_symfc_projector:
            if self._log_level:
                print("Symmetrizing fc3 by symfc projector.", flush=True)
            if options is None:
                _options = None
            else:
                _options = parse_symfc_options(options, 3)
            self._fc3 = symmetrize_by_projector(
                self._supercell,
                self._fc3,
                3,
                primitive=self._primitive,
                options=_options,
                log_level=self._log_level,
                show_credit=True,
            )
        else:
            level = 3
            if options is not None:
                for option in options.split(","):
                    if "level" in option.lower():
                        try:
                            level = int(option.split("=")[1].split()[0])
                        except ValueError:
                            pass
                        break
            if self._log_level:
                print(
                    f"Symmetrizing fc3 by traditional approach (N={level}).", flush=True
                )
            for _ in range(level):
                if self._fc3.shape[0] == self._fc3.shape[1]:
                    set_translational_invariance_fc3(self._fc3)
                    set_permutation_symmetry_fc3(self._fc3, lang=self._lang)
                else:
                    set_translational_invariance_compact_fc3(
                        self._fc3, self._primitive, lang=self._lang
                    )
                    set_permutation_symmetry_compact_fc3(
                        self._fc3, self._primitive, lang=self._lang
                    )

    def produce_fc2(
        self,
        symmetrize_fc2: bool | None = None,
        is_compact_fc: bool = True,
        fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
        fc_calculator_options: str | None = None,
        use_symfc_projector: bool | None = None,
    ) -> None:
        """Calculate fc2 from displacements and forces.

        Uses the phonon supercell (equal to the fc3 supercell when
        ``phonon_supercell_matrix`` is not set). Forces are taken from
        :attr:`phonon_dataset` when present, otherwise from
        :attr:`dataset`. The solver is chosen by ``fc_calculator``:

        - ``None`` (default) or ``"traditional"``: built-in
          finite-difference solver. The returned fc2 is **not**
          symmetrized; to enforce translational and permutation
          invariance, call :meth:`symmetrize_fc2` afterwards.
        - ``"symfc"``: symfc solver, which returns symmetrized force
          constants in one shot. When a cutoff distance is configured
          via ``fc_calculator_options``, it is captured into
          :attr:`fc2_cutoff`.
        - ``"alm"``: ALM solver, which handles symmetrization
          internally.

        Parameters
        ----------
        symmetrize_fc2 : bool, optional
            **Deprecated.** Passing any value emits a
            ``DeprecationWarning``. With the traditional solver, call
            :meth:`symmetrize_fc2` after :meth:`produce_fc2` instead.
            Default is ``None``.
        is_compact_fc : bool, optional
            fc2 shape::

                True:  (primitive, supercell, 3, 3)
                False: (supercell, supercell, 3, 3)

            Default is ``True``.
        fc_calculator : str or None, optional
            Force-constants calculator. One of ``None``,
            ``"traditional"``, ``"symfc"``, or ``"alm"``. Default is
            ``None`` (equivalent to ``"traditional"``).
        fc_calculator_options : str or None, optional
            Options string forwarded to the chosen calculator.
        use_symfc_projector : bool, optional
            **Deprecated.** Passing any value emits a
            ``DeprecationWarning``. Call
            ``symmetrize_fc2(use_symfc_projector=True)`` after
            :meth:`produce_fc2` instead. Default is ``None``.

        """
        if symmetrize_fc2 is not None:
            warnings.warn(
                "The symmetrize_fc2 parameter of Phono3py.produce_fc2 is "
                "deprecated. Call Phono3py.symmetrize_fc2 after "
                "Phono3py.produce_fc2 instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if use_symfc_projector is not None:
            warnings.warn(
                "The use_symfc_projector parameter of Phono3py.produce_fc2 is "
                "deprecated. Call "
                "Phono3py.symmetrize_fc2(use_symfc_projector=True) after "
                "Phono3py.produce_fc2 instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self._phonon_dataset is None:
            disp_dataset = self._dataset
        else:
            disp_dataset = self._phonon_dataset  # type: ignore[assignment]

        if disp_dataset is None:
            raise RuntimeError("Displacement dataset is not set.")
        if not forces_in_dataset(disp_dataset):  # type: ignore[arg-type]
            raise RuntimeError("Forces are not set in the dataset.")

        if self._log_level:
            print("Computing phonon fc2.", flush=True)

        fc_solver = get_fc_solver(
            self._phonon_supercell,
            disp_dataset,  # type: ignore[arg-type]
            primitive=self._phonon_primitive,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            orders=[2],
            is_compact_fc=is_compact_fc,
            symmetry=self._phonon_supercell_symmetry,
            log_level=self._log_level,
            lang=self._lang,
        )
        self._fc2 = fc_solver.force_constants[2]  # type: ignore[assignment]

        if symmetrize_fc2 and (fc_calculator is None or fc_calculator == "traditional"):
            self.symmetrize_fc2(
                use_symfc_projector=use_symfc_projector and fc_calculator is None,
                options=fc_calculator_options,
            )

        if fc_calculator == "symfc":
            symfc_solver = cast(SymfcFCSolver, fc_solver.fc_solver)
            options = symfc_solver.options
            if options is not None and "cutoff" in options:
                self._fc2_cutoff = options["cutoff"].get(2, None)

    def symmetrize_fc2(
        self,
        use_symfc_projector: bool = False,
        options: str | None = None,
    ) -> None:
        """Symmetrize fc2 by symfc projector or the traditional approach.

        Parameters
        ----------
        use_symfc_projector : bool, optional
            If ``True``, symmetrize force constants with the symfc
            projector instead of the traditional approach.
        options : str or None, optional
            Options string. Accepted values depend on the backend::

                symfc projector:
                    "use_mkl=true"  call sparse_dot_mkl
                                    (requires sparse_dot_mkl to be
                                    installed).
                traditional:
                    "level=N"       apply translational and permutation
                                    symmetries alternately N times.
                                    Default level is 3.

        """
        if self._fc2 is None:
            raise RuntimeError(
                "fc2 is not set. Call produce_fc3 or produce_fc2 (phonon_fc2) first."
            )

        if self._phonon_supercell_matrix is None:
            supercell = self._supercell
            primitive = self._primitive
        else:
            supercell = self._phonon_supercell
            primitive = self._phonon_primitive
        assert self._fc2.shape[1] == len(supercell)

        if use_symfc_projector:
            if self._log_level:
                print("Symmetrizing fc2 by symfc projector.", flush=True)
            if options is None:
                _options = None
            else:
                _options = parse_symfc_options(options, 2)
            self._fc2 = symmetrize_by_projector(
                supercell,
                self._fc2,
                2,
                primitive=primitive,
                options=_options,
                log_level=self._log_level,
                show_credit=True,
            )
        else:
            level = 3
            if options is not None:
                for option in options.split(","):
                    if "level" in option.lower():
                        try:
                            level = int(option.split("=")[1].split()[0])
                        except ValueError:
                            pass
                        break
            if self._log_level:
                print(
                    f"Symmetrizing fc2 by traditional approach (N={level}).", flush=True
                )
            for _ in range(level):
                if self._fc2.shape[0] == self._fc2.shape[1]:
                    symmetrize_force_constants(self._fc2, lang=self._lang)
                else:
                    symmetrize_compact_force_constants(
                        self._fc2, primitive, lang=self._lang
                    )

    def cutoff_fc3_by_zero(
        self,
        cutoff_distance: float,
        fc3: NDArray[np.double] | None = None,
    ) -> None:
        """Set zero to fc3 elements out of cutoff distance.

        Note
        ----
        fc3 is overwritten.

        Parameters
        ----------
        cutoff_distance : float
            After creating force constants, fc elements where any pair
            distance in atom triplets larger than cutoff_distance are set zero.

        """
        if fc3 is None:
            _fc3 = self._fc3
        else:
            _fc3 = fc3
        if _fc3 is None:
            raise RuntimeError("fc3 is not set.")
        cutoff_fc3_by_zero(
            _fc3,
            self._supercell,
            cutoff_distance,
            p2s_map=self._primitive.p2s_map,
            symprec=self._symprec,
        )

    def set_permutation_symmetry(self) -> None:
        """Enforce permutation symmetry to fc2 and fc3."""
        if self._fc2 is not None:
            set_permutation_symmetry(self._fc2)
        if self._fc3 is not None:
            set_permutation_symmetry_fc3(self._fc3, lang=self._lang)

    def set_translational_invariance(self) -> None:
        """Enforce translation invariance.

        This subtracts drift divided by number of elements in each row and
        column.

        """
        if self._fc2 is not None:
            set_translational_invariance(self._fc2)
        if self._fc3 is not None:
            set_translational_invariance_fc3(self._fc3)

    def run_imag_self_energy(
        self,
        grid_points: Sequence[int] | NDArray[np.int64],
        temperatures: NDArray[np.double] | Sequence[float],
        frequency_points: NDArray[np.double] | Sequence[float] | None = None,
        frequency_step: float | None = None,
        num_frequency_points: int | None = None,
        num_points_in_batch: int | None = None,
        frequency_points_at_bands: bool = False,
        scattering_event_class: Literal[1, 2] | None = None,
        write_txt: bool = False,
        write_gamma_detail: bool = False,
        keep_gamma_detail: bool = False,
        output_filename: str | None = None,
    ) -> ImagSelfEnergyValues:
        """Calculate the imaginary part of the bubble self-energy (Gamma).

        The phonon self-energy is decomposed as
        ``Pi = Delta - i Gamma``. Gamma is computed at the given grid
        points and temperatures, as a function of frequency. Results
        are returned as :class:`ImagSelfEnergyValues` and also kept on
        the instance.

        Parameters
        ----------
        grid_points : array_like
            Grid-point indices at which Gamma is computed.
            ``shape=(grid_points,)``, ``dtype=int``.
        temperatures : array_like
            Temperatures at which Gamma is computed.
            ``shape=(temperatures,)``, ``dtype=float``.
        frequency_points : array_like, optional
            Frequency sampling points. ``shape=(frequency_points,)``,
            ``dtype=float``. Default is ``None``; when
            ``frequency_points_at_bands=False`` and this is ``None``,
            uniform sampling is generated from
            ``num_frequency_points`` or ``frequency_step``.
        frequency_step : float, optional
            Uniform pitch of frequency sampling points. Default is
            ``None``, which falls back to ``num_frequency_points``.
        num_frequency_points : int, optional
            Number of sampling points (including end points) used when
            ``frequency_step`` is not set. Default is ``None``, which
            gives 201.
        num_points_in_batch : int, optional
            Number of sampling points per batch. Larger batches allow
            more efficient multi-core utilization at the cost of
            memory. Default is ``None``, which gives 10.
        frequency_points_at_bands : bool, optional
            When ``True``, use the phonon band frequencies as the
            frequency points. Default is ``False``.
        scattering_event_class : int, optional
            Restrict Gamma to a specific scattering event class
            (``1`` or ``2``). When set, only the chosen class is
            accumulated into ``gammas`` (the usual full Gamma is not
            stored). Default is ``None``.
        write_txt : bool, optional
            Write frequency points and Gamma to text files. Default
            is ``False``.
        write_gamma_detail : bool, optional
            Write per-scattering-event Gamma to an HDF5 file. Default
            is ``False``.
        keep_gamma_detail : bool, optional
            Keep per-scattering-event Gamma on the instance
            (accessible via :attr:`detailed_gammas`). Default is
            ``False``.
        output_filename : str, optional
            Inserted into output filenames.

        Returns
        -------
        ImagSelfEnergyValues
            Container with ``frequency_points``, ``gammas``,
            ``scattering_event_class``, and (when
            ``keep_gamma_detail=True``) ``detailed_gammas``.

        Raises
        ------
        RuntimeError
            When :meth:`init_phph_interaction` has not been called.

        """
        if self._interaction is None:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

        if temperatures is None:
            self._temperatures = np.array([300.0], dtype="double")
        else:
            self._temperatures = np.asarray(temperatures, dtype="double")
        self._grid_points = np.asarray(grid_points, dtype="int64")
        vals = get_imag_self_energy(
            self._interaction,
            grid_points,
            temperatures,
            sigmas=self._sigmas,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            frequency_points_at_bands=frequency_points_at_bands,
            num_frequency_points=num_frequency_points,
            num_points_in_batch=num_points_in_batch,
            scattering_event_class=scattering_event_class,
            write_gamma_detail=write_gamma_detail,
            return_gamma_detail=keep_gamma_detail,
            output_filename=output_filename,
            log_level=self._log_level,
            lang=self._lang,
        )
        if keep_gamma_detail:
            self._ise_params = ImagSelfEnergyValues(
                frequency_points=vals[0],
                gammas=vals[1],
                scattering_event_class=scattering_event_class,
                detailed_gammas=vals[2],
            )
        else:
            self._ise_params = ImagSelfEnergyValues(
                frequency_points=vals[0], gammas=vals[1]
            )

        if write_txt:
            self._write_imag_self_energy(output_filename=output_filename)

        return self._ise_params

    def _write_imag_self_energy(self, output_filename: str | None = None) -> None:
        assert self._temperatures is not None
        if self._ise_params is None:
            raise RuntimeError("Imaginary self-energy is not calculated.")
        # if self._ise_params is not None
        assert self.mesh_numbers is not None
        assert self._grid_points is not None
        assert self._band_indices is not None
        write_imag_self_energy(
            self._ise_params.gammas,
            self.mesh_numbers,
            self._grid_points,
            self._band_indices,
            self._ise_params.frequency_points,
            self._temperatures,
            self._sigmas,
            scattering_event_class=self._ise_params.scattering_event_class,
            output_filename=output_filename,
            is_mesh_symmetry=self._is_mesh_symmetry,
            log_level=self._log_level,
        )

    def run_real_self_energy(
        self,
        grid_points: Sequence[int] | NDArray[np.int64],
        temperatures: NDArray[np.double] | Sequence[float],
        frequency_points_at_bands: bool = False,
        frequency_points: NDArray[np.double] | Sequence[float] | None = None,
        frequency_step: float | None = None,
        num_frequency_points: int | None = None,
        epsilons: Sequence[float | None] | None = None,
        write_txt: bool = False,
        write_hdf5: bool = False,
        output_filename: str | None = None,
    ) -> tuple[NDArray[np.double] | None, NDArray[np.double]]:
        """Calculate the real part of the bubble self-energy (Delta).

        The phonon self-energy is decomposed as
        ``Pi = Delta - i Gamma``. Delta is computed at the given grid
        points and temperatures, as a function of frequency.

        Parameters
        ----------
        grid_points : array_like
            Grid-point indices at which Delta is computed.
            ``shape=(grid_points,)``, ``dtype=int``.
        temperatures : array_like
            Temperatures at which Delta is computed.
            ``shape=(temperatures,)``, ``dtype=float``.
        frequency_points_at_bands : bool, optional
            With ``False`` (default), frequency shifts are calculated
            at sampling points. With ``True``, they are calculated at
            the phonon band frequencies.
        frequency_points : array_like, optional
            Frequency sampling points. ``shape=(frequency_points,)``,
            ``dtype=float``. Default is ``None``; when ``None``,
            uniform sampling is generated from
            ``num_frequency_points`` or ``frequency_step``.
        frequency_step : float, optional
            Uniform pitch of frequency sampling points. Default is
            ``None``, which falls back to ``num_frequency_points``.
        num_frequency_points : int, optional
            Number of sampling points (including end points) used when
            ``frequency_step`` is not set. Default is ``None``, which
            gives 201.
        epsilons : array_like, optional
            Smearing widths used to compute the principal part. When
            multiple values are given, frequency shifts for each are
            returned. ``shape=(epsilons,)``, ``dtype=float``. Default
            is ``None`` (use :attr:`sigmas`).
        write_txt : bool, optional
            Write frequency points and Delta to text files. Default
            is ``False``.
        write_hdf5 : bool, optional
            Write results to HDF5 files, one per (grid point, epsilon,
            temperature). Default is ``False``.
        output_filename : str, optional
            Inserted into output filenames.

        Returns
        -------
        frequency_points : ndarray or None
            Frequency sampling points. ``None`` when
            ``frequency_points_at_bands=True``.
            ``dtype='double'``.
        deltas : ndarray
            Real-part frequency shifts, indexed by
            ``(epsilon, grid_point, temperature, band)`` with a
            trailing frequency-points axis (collapsed to the band
            frequency when ``frequency_points_at_bands=True``).
            ``dtype='double'``.

        Raises
        ------
        RuntimeError
            When :meth:`init_phph_interaction` has not been called.

        """
        if self._interaction is None:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

        if epsilons is not None:
            _epsilons = epsilons
        else:
            _epsilons = self._sigmas

        # (epsilon, grid_point, temperature, band)
        _frequency_points, deltas = get_real_self_energy(
            self._interaction,
            grid_points,
            temperatures,
            epsilons=_epsilons,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
            frequency_points_at_bands=frequency_points_at_bands,
            write_hdf5=write_hdf5,
            output_filename=output_filename,
            log_level=self._log_level,
            lang=self._lang,
        )

        if write_txt:
            assert self.mesh_numbers is not None
            write_real_self_energy(
                deltas,
                self.mesh_numbers,
                grid_points,
                self._band_indices,
                _frequency_points,
                temperatures,
                _epsilons,
                output_filename=output_filename,
                is_mesh_symmetry=self._is_mesh_symmetry,
                log_level=self._log_level,
            )

        return _frequency_points, deltas

    def run_spectral_function(
        self,
        grid_points: Sequence[int] | NDArray[np.int64],
        temperatures: NDArray[np.double] | Sequence[float],
        frequency_points: NDArray[np.double] | Sequence[float] | None = None,
        frequency_step: float | None = None,
        num_frequency_points: int | None = None,
        num_points_in_batch: int | None = None,
        write_txt: bool = False,
        write_hdf5: bool = False,
        output_filename: str | None = None,
    ) -> None:
        """Calculate the phonon spectral function from the bubble self-energy.

        The spectral function
        ``A(omega) ~ Gamma / ((omega - Omega - Delta)^2 + Gamma^2)``
        is computed at the given grid points and temperatures, as a
        function of frequency. The result is stored on the instance.

        Parameters
        ----------
        grid_points : array_like
            Grid-point indices at which the spectral function is
            computed. ``shape=(grid_points,)``, ``dtype=int``.
        temperatures : array_like
            Temperatures at which the spectral function is computed.
            ``shape=(temperatures,)``, ``dtype=float``.
        frequency_points : array_like, optional
            Frequency sampling points. ``shape=(frequency_points,)``,
            ``dtype=float``. Default is ``None``; when ``None``,
            uniform sampling is generated from
            ``num_frequency_points`` or ``frequency_step``.
        frequency_step : float, optional
            Uniform pitch of frequency sampling points. Default is
            ``None``, which falls back to ``num_frequency_points``.
        num_frequency_points : int, optional
            Number of sampling points (including end points) used when
            ``frequency_step`` is not set. Default is ``None``, which
            gives 201.
        num_points_in_batch : int, optional
            Number of sampling points per batch. Larger batches allow
            more efficient multi-core utilization at the cost of
            memory. Default is ``None``, which gives 10.
        write_txt : bool, optional
            Write frequency points and spectral functions to text
            files. Default is ``False``.
        write_hdf5 : bool, optional
            Write results to HDF5 files, one per grid point. Default
            is ``False``.
        output_filename : str, optional
            Inserted into output filenames.

        Raises
        ------
        RuntimeError
            When :meth:`init_phph_interaction` has not been called.

        """
        if self._interaction is None:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

        self._spectral_function = run_spectral_function(
            self._interaction,
            grid_points,
            temperatures=temperatures,
            sigmas=self._sigmas,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
            num_points_in_batch=num_points_in_batch,
            band_indices=self._band_indices,
            write_txt=write_txt,
            write_hdf5=write_hdf5,
            output_filename=output_filename,
            log_level=self._log_level,
        )

    def run_thermal_conductivity(
        self,
        is_LBTE: bool = False,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        grid_points: Sequence[int] | NDArray[np.int64] | None = None,
        boundary_mfp: float | None = None,  # in micrometer
        solve_collective_phonon: bool = False,
        use_ave_pp: bool = False,
        is_reducible_collision_matrix: bool = False,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,  # for group velocity
        is_full_pp: bool = False,
        pinv_cutoff: float | None = None,  # for pseudo-inversion of collision matrix
        pinv_method: int = 0,  # for pseudo-inversion of collision matrix
        pinv_solver: int = 0,  # solver of pseudo-inversion of collision matrix
        write_gamma: bool = False,
        read_gamma: bool = False,
        is_N_U: bool = False,
        transport_type: str | None = None,
        write_kappa: bool = False,
        write_gamma_detail: bool = False,
        write_collision: bool = False,
        read_collision: str | Sequence[int] | None = None,
        write_pp: bool = False,
        read_pp: bool = False,
        read_elph: int | None = None,
        write_LBTE_solution: bool = False,
        compression: Literal["gzip", "lzf"] | int | None = "gzip",
        input_filename: str | None = None,
        output_filename: str | None = None,
        log_level: int | None = None,
    ) -> None:
        """Run a lattice thermal conductivity calculation.

        Result is stored in :attr:`thermal_conductivity` (an
        ``RTACalculator`` when ``is_LBTE=False``, an ``LBTECalculator``
        otherwise).

        Parameters
        ----------
        is_LBTE : bool, optional
            ``False`` for the relaxation-time approximation (RTA),
            ``True`` for the direct solution of the linearized Boltzmann
            equation (LBTE) and the Wigner transport equation. Default
            is ``False``.
        temperatures : array_like, optional
            Temperatures at which thermal conductivity is computed.
            ``shape=(temperature_points,)``, ``dtype='double'``. With
            ``None`` (default), the defaults depend on ``is_LBTE``::

                is_LBTE=False:  [0, 10, ..., 1000]
                is_LBTE=True:   [300]

        is_isotope : bool, optional
            Include isotope scattering. Default is ``False``.
        mass_variances : array_like, optional
            Mass variances for isotope scattering. With ``None``
            (default) and ``is_isotope=True``, the values stored in the
            phono3py instance are used.
            ``shape=(atoms_in_primitive,)``, ``dtype='double'``.
        grid_points : array_like, optional
            Grid-point indices at which mode thermal conductivities are
            computed. With ``None`` (default), all required grid points
            are chosen internally. ``shape=(num_grid_points,)``,
            ``dtype='int64'``.
        boundary_mfp : float, optional
            Mean free path in micrometers used to model a simple
            boundary-scattering contribution. ``None`` (default)
            disables this contribution.
        solve_collective_phonon : bool, optional
            Option for an under-development feature. Default is
            ``False``.
        use_ave_pp : bool, optional
            RTA only (``is_LBTE=False``). Use an averaged ph-ph
            interaction strength to compute phonon lifetimes. This does
            not reduce computational cost; it is mainly a modelling
            tool for analyzing the result. Default is ``False``.
        is_reducible_collision_matrix : bool, optional
            Direct-solution only (``is_LBTE=True``). Experimental: with
            ``True``, the full collision matrix is constructed and
            solved. Default is ``False``.
        is_kappa_star : bool, optional
            With ``True`` (default), use crystal symmetry to reduce the
            grid points at which mode thermal conductivities are
            sampled.
        gv_delta_q : float, optional
            Q-distance (in 1/Angstrom) used by the central
            finite-difference scheme for group velocity when
            non-analytical correction is in effect. Default is ``None``
            (effectively 1e-5).
        is_full_pp : bool, optional
            With ``True``, compute all elements of the ph-ph interaction
            strength. With ``False`` (default) and the tetrahedron
            method, elements known to be zero are skipped, giving a
            substantial speed-up. With the smearing method, all
            elements are computed regardless of this flag unless
            :attr:`sigma_cutoff` is set.
        pinv_cutoff : float, optional
            Direct-solution only (``is_LBTE=True``). Threshold to decide
            whether an eigenvalue is treated as zero in the
            pseudo-inverse of the collision matrix. See also
            ``pinv_method``. Default is ``None`` (typically ``1.0e-8``).
        pinv_method : int, optional
            Direct-solution only (``is_LBTE=True``). Pseudo-inverse
            zero-eigenvalue criterion::

                0:  abs(eigenvalue) < pinv_cutoff
                1:  eigenvalue       < pinv_cutoff

            Default is ``0``.
        pinv_solver : int, optional
            Direct-solution only (``is_LBTE=True``). Choice of solver
            for the pseudo-inverse of the collision matrix. ``0``
            selects the default automatically. Choices other than ``1``
            and ``4`` are dangerous and not recommended::

                0:  default (1 with MKL LAPACKE or scipy unavailable;
                    4 otherwise).
                1:  LAPACKE dsyev  -- smaller memory than dsyevd, but
                    slower. Default when MKL LAPACKE is integrated or
                    scipy is not installed.
                2:  LAPACKE dsyevd -- larger memory than dsyev, but
                    faster. Not recommended (occasional wrong result).
                3:  numpy dsyevd (linalg.eigh). Not recommended
                    (occasional wrong result).
                4:  scipy dsyev. Default when scipy is installed and
                    MKL LAPACKE is not integrated.
                5:  scipy dsyevd. Not recommended (occasional wrong
                    result).

            Default is ``0``.
        write_gamma : bool, optional
            RTA only (``is_LBTE=False``). Write mode thermal
            conductivity properties to files, one per grid point. When
            :attr:`band_indices` or multiple :attr:`sigmas` are
            specified, a file is written per band-index group and per
            sigma. Default is ``False``.
        read_gamma : bool, optional
            RTA only (``is_LBTE=False``). Read files written by
            ``write_gamma=True`` instead of recomputing ph-ph
            interaction strengths and imaginary self-energies. Default
            is ``False``.
        is_N_U : bool, optional
            RTA only (``is_LBTE=False``). Separate the imaginary
            self-energy into normal and Umklapp contributions. Default
            is ``False``.
        transport_type : str, optional
            ``"SMM19"``, ``"NJC23"``, or ``None``. Default is ``None``.
        write_kappa : bool, optional
            Write thermal conductivity and related properties to a
            file. With multiple :attr:`sigmas`, one file is written per
            sigma. Default is ``False``.
        write_gamma_detail : bool, optional
            RTA only (``is_LBTE=False``). Write detailed imaginary
            self-energy information alongside the files produced by
            ``write_gamma``. Default is ``False``.
        write_collision : bool, optional
            Direct-solution only (``is_LBTE=True``). Write the
            collision matrix to a file. With multiple :attr:`sigmas`,
            one file is written per sigma. The file can be very large.
            Default is ``False``.
        read_collision : str or Sequence[int], optional
            Direct-solution only (``is_LBTE=True``). Read the collision
            matrix from a file. Default is ``None``.
        write_pp : bool, optional
            Write ph-ph interaction strength to files, one per grid
            point. Assumes a single value in :attr:`sigmas`. Default
            is ``False``.
        read_pp : bool, optional
            Read ph-ph interaction strength from files. Default is
            ``False``.
        read_elph : int, optional
            Index used to read electron-phonon gammas from a file.
            Default is ``None``.
        write_LBTE_solution : bool, optional
            Direct-solution only (``is_LBTE=True``). Write the
            collision-matrix eigenvectors as row vectors (column
            vectors when ``pinv_solver=3``). With multiple
            :attr:`sigmas`, one file is written per sigma. The file
            can be very large. Default is ``False``.
        compression : str, optional
            HDF5 compression for large datasets. See the h5py
            documentation. Default is ``"gzip"``.
        input_filename : str, optional
            **Deprecated.** When set, the string is inserted before
            the filename extension in read paths. Default is ``None``.
        output_filename : str, optional
            **Deprecated.** When set, the string is inserted before
            the filename extension in write paths. Default is
            ``None``.
        log_level : int, optional
            Override the instance-level :attr:`log_level` for this
            call. Default is ``None`` (use the instance value).

        Raises
        ------
        RuntimeError
            When :meth:`init_phph_interaction` has not been called.

        """
        if input_filename is not None:
            warnings.warn(
                "input_filename parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        if output_filename is not None:
            warnings.warn(
                "output_filename parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self._interaction is None:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

        if log_level is None:
            _log_level = self._log_level
        else:
            _log_level = log_level

        if is_LBTE:
            self._thermal_conductivity = get_thermal_conductivity_LBTE(
                self._interaction,
                temperatures=temperatures,
                sigmas=self._sigmas,
                sigma_cutoff=self._sigma_cutoff,
                is_isotope=is_isotope,
                mass_variances=mass_variances,
                grid_points=grid_points,
                boundary_mfp=boundary_mfp,
                solve_collective_phonon=solve_collective_phonon,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                is_kappa_star=is_kappa_star,
                gv_delta_q=gv_delta_q,
                is_full_pp=is_full_pp,
                transport_type=transport_type,
                pinv_cutoff=pinv_cutoff,
                pinv_solver=pinv_solver,
                pinv_method=pinv_method,
                write_collision=write_collision,
                read_collision=read_collision,
                write_kappa=write_kappa,
                write_pp=write_pp,
                read_pp=read_pp,
                write_LBTE_solution=write_LBTE_solution,
                compression=compression,
                input_filename=input_filename,
                output_filename=output_filename,
                log_level=_log_level,
                lang=self._lang,
            )
        else:
            self._thermal_conductivity = get_thermal_conductivity_RTA(
                self._interaction,
                temperatures=temperatures,
                sigmas=self._sigmas,
                sigma_cutoff=self._sigma_cutoff,
                is_isotope=is_isotope,
                mass_variances=mass_variances,
                grid_points=grid_points,
                boundary_mfp=boundary_mfp,
                use_ave_pp=use_ave_pp,
                is_kappa_star=is_kappa_star,
                gv_delta_q=gv_delta_q,
                is_full_pp=is_full_pp,
                is_N_U=is_N_U,
                transport_type=transport_type,
                write_gamma=write_gamma,
                read_gamma=read_gamma,
                write_kappa=write_kappa,
                write_pp=write_pp,
                read_pp=read_pp,
                read_elph=read_elph,
                write_gamma_detail=write_gamma_detail,
                compression=compression,
                input_filename=input_filename,
                output_filename=output_filename,
                log_level=_log_level,
                lang=self._lang,
            )

    def save(
        self,
        filename: str | os.PathLike = "phono3py_params.yaml",
        settings: dict | None = None,
    ) -> None:
        """Save the Phono3py instance state to a YAML file.

        Parameters
        ----------
        filename : str or os.PathLike, optional
            Output file path. Default is ``"phono3py_params.yaml"``.
        settings : dict, optional
            Selects which sections to write. Only keys whose values
            differ from the defaults need to be supplied. The defaults
            are::

                {'force_sets':            True,
                 'displacements':         True,
                 'force_constants':       False,
                 'born_effective_charge': True,
                 'dielectric_constant':   True}

        """
        ph3py_yaml = self.to_phono3py_yaml(settings=settings)
        with open(filename, "w") as w:
            w.write(str(ph3py_yaml))

    def develop_mlp(
        self,
        params: PypolymlpParams | dict | str | None = None,
        test_size: float = 0.1,
    ) -> None:
        """Develop machine learning potential.

        Parameters
        ----------
        params : PypolymlpParams or dict, optional
            Parameters for developing MLP. Default is None. When dict is given,
            PypolymlpParams instance is created from the dict.
        test_size : float, optional
            Training and test data are split by this ratio. test_size=0.1
            means the first 90% of the data is used for training and the rest
            is used for test. Default is 0.1.

        """
        if self._mlp_dataset is None:
            raise RuntimeError("MLP dataset is not set.")

        self._mlp = PhonopyMLP(log_level=self._log_level)
        self._mlp.develop(
            self._mlp_dataset,  # type: ignore[arg-type]
            self._supercell,
            params=params,
            test_size=test_size,
        )

    def save_mlp(self, filename: str | os.PathLike | None = None) -> None:
        """Save machine learning potential."""
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        self._mlp.save(filename=filename)

    def load_mlp(self, filename: str | os.PathLike | None = None) -> None:
        """Load machine learning potential."""
        self._mlp = PhonopyMLP(log_level=self._log_level)
        self._mlp.load(filename=filename)

    def evaluate_mlp(self) -> None:
        """Evaluate machine learning potential.

        This method calculates the supercell energies and forces from the MLP
        for the displacements in self._dataset of type 2. The results are stored
        in self._dataset.

        The displacements may be generated by the produce_force_constants method
        with number_of_snapshots > 0. With MLP, a small distance parameter, such
        as 0.01, can be numerically stable for the computation of force
        constants.

        """
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        if self.supercells_with_displacements is None:
            raise RuntimeError("Displacements are not set. Run generate_displacements.")

        energies, forces, _ = self._mlp.evaluate(self.supercells_with_displacements)
        self.supercell_energies = energies
        self.forces = forces

    def develop_phonon_mlp(
        self,
        params: PypolymlpParams | dict | str | None = None,
        test_size: float = 0.1,
    ) -> None:
        """Develop MLP for fc2.

        Parameters
        ----------
        params : PypolymlpParams or dict, optional
            Parameters for developing MLP. Default is None. When dict is given,
            PypolymlpParams instance is created from the dict.
        test_size : float, optional
            Training and test data are split by this ratio. test_size=0.1
            means the first 90% of the data is used for training and the rest
            is used for test. Default is 0.1.

        """
        if self._phonon_mlp_dataset is None:
            raise RuntimeError("MLP dataset is not set.")

        self._phonon_mlp = PhonopyMLP(log_level=self._log_level)
        self._phonon_mlp.develop(
            self._phonon_mlp_dataset,  # type: ignore[arg-type]
            self._phonon_supercell,
            params=params,
            test_size=test_size,
        )

    def save_phonon_mlp(self, filename: str | os.PathLike | None = None) -> None:
        """Save machine learning potential."""
        if self._phonon_mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        self._phonon_mlp.save(filename=filename)

    def load_phonon_mlp(self, filename: str | os.PathLike | None = None) -> None:
        """Load machine learning potential."""
        self._phonon_mlp = PhonopyMLP(log_level=self._log_level)
        self._phonon_mlp.load(filename=filename)

    def evaluate_phonon_mlp(self) -> None:
        """Evaluate the machine learning potential.

        This method calculates the supercell energies and forces from the MLP
        for the displacements in self._dataset of type 2. The results are stored
        in self._dataset.

        The displacements may be generated by the produce_force_constants method
        with number_of_snapshots > 0. With MLP, a small distance parameter, such
        as 0.01, can be numerically stable for the computation of force
        constants.

        """
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")
        if self._phonon_mlp is None:
            raise RuntimeError("Phonon MLP is not developed yet.")

        if self.phonon_supercells_with_displacements is None:
            raise RuntimeError(
                "Displacements are not set. Run generate_fc2_displacements."
            )

        if self._phonon_mlp is None:
            mlp = self._mlp
        else:
            mlp = self._phonon_mlp
        energies, forces, _ = mlp.evaluate(self.phonon_supercells_with_displacements)
        self.phonon_supercell_energies = energies
        self.phonon_forces = forces

    def to_phono3py_yaml(
        self, configuration: dict | None = None, settings: dict | None = None
    ) -> Phono3pyYaml:
        """Return Phono3pyYaml class instance with this data."""
        units = get_calculator_physical_units(self.calculator)
        ph3py_yaml = Phono3pyYaml(
            configuration=configuration, physical_units=units, settings=settings
        )
        set_data_to_phonopy_yaml(cast(PhonopyYaml, ph3py_yaml), cast(Phonopy, self))
        if self.phonon_supercell_matrix is not None:
            ph3py_yaml.phonon_supercell_matrix = self.phonon_supercell_matrix
            if self.phonon_dataset is not None:
                ph3py_yaml.phonon_dataset = self.phonon_dataset
        ph3py_yaml.phonon_primitive = self.phonon_primitive
        ph3py_yaml.phonon_supercell = self.phonon_supercell
        return ph3py_yaml

    ###################
    # private methods #
    ###################
    def _search_symmetry(self) -> None:
        self._symmetry = Symmetry(
            self._supercell,
            symprec=self._symprec,
            is_symmetry=self._is_symmetry,
            lang=self._lang,
        )

    def _search_primitive_symmetry(self) -> None:
        self._primitive_symmetry = Symmetry(
            self._primitive,
            self._symprec,
            self._is_symmetry,
            lang=self._lang,
        )
        if len(self._symmetry.pointgroup_operations) != len(
            self._primitive_symmetry.pointgroup_operations
        ):
            print(
                "Warning: point group symmetries of supercell and primitive"
                "cell are different."
            )

    def _search_phonon_supercell_symmetry(self) -> None:
        if self._phonon_supercell_matrix is None:
            self._phonon_supercell_symmetry = self._symmetry
        else:
            self._phonon_supercell_symmetry = Symmetry(
                self._phonon_supercell,
                symprec=self._symprec,
                is_symmetry=self._is_symmetry,
                lang=self._lang,
            )

    def _build_supercell(self) -> None:
        self._supercell = get_supercell(
            self._unitcell, self._supercell_matrix, symprec=self._symprec
        )

    def _build_primitive_cell(self) -> None:
        """Create primitive cell.

        primitive_matrix:
          Relative axes of primitive cell to the input unit cell.
          Relative axes to the supercell is calculated by:
             supercell_matrix^-1 * primitive_matrix
          Therefore primitive cell lattice is finally calculated by:
             (supercell_lattice * (supercell_matrix)^-1 * primitive_matrix)^T

        """
        self._primitive = self._get_primitive_cell(
            self._supercell, self._supercell_matrix, self._primitive_matrix
        )

    def _build_phonon_supercell(self) -> None:
        """Create phonon supercell for fc2.

        phonon_supercell:
          This supercell is used for harmonic phonons (frequencies,
          eigenvectors, group velocities, ...)
        phonon_supercell_matrix:
          Different supercell size can be specified.

        """
        if self._phonon_supercell_matrix is None:
            self._phonon_supercell = self._supercell
        else:
            self._phonon_supercell = get_supercell(
                self._unitcell, self._phonon_supercell_matrix, symprec=self._symprec
            )

    def _build_phonon_primitive_cell(self) -> None:
        if self._phonon_supercell_matrix is None:
            self._phonon_primitive = self._primitive
        else:
            self._phonon_primitive = self._get_primitive_cell(
                self._phonon_supercell,
                self._phonon_supercell_matrix,
                self._primitive_matrix,
            )
            if (
                self._primitive is not None
                and (self._primitive.numbers != self._phonon_primitive.numbers).any()
            ):
                print(" Primitive cells for fc2 and fc3 can be different.")
                raise RuntimeError

    def _build_phonon_supercells_with_displacements(
        self, supercell: PhonopyAtoms, dataset: DisplacementDataset
    ) -> list[PhonopyAtoms]:
        supercells = []
        positions = supercell.positions
        magmoms = supercell.magnetic_moments
        masses = supercell.masses
        numbers = supercell.numbers
        lattice = supercell.cell

        if "displacements" in dataset:
            for disp in dataset["displacements"]:
                supercells.append(
                    PhonopyAtoms(
                        numbers=numbers,
                        masses=masses,
                        magnetic_moments=magmoms,
                        positions=positions + disp,
                        cell=lattice,
                    )
                )
        else:
            for disp1 in dataset["first_atoms"]:
                disp_cart1 = disp1["displacement"]
                positions = supercell.positions
                positions[disp1["number"]] += disp_cart1
                supercells.append(
                    PhonopyAtoms(
                        numbers=numbers,
                        masses=masses,
                        magnetic_moments=magmoms,
                        positions=positions,
                        cell=lattice,
                    )
                )

        return supercells

    def _build_supercells_with_displacements(self) -> None:
        assert self._dataset is not None

        magmoms = self._supercell.magnetic_moments
        masses = self._supercell.masses
        numbers = self._supercell.numbers
        lattice = self._supercell.cell

        # One displacement supercells
        supercells = cast(
            List[Optional[PhonopyAtoms]],  # For < python3.10
            self._build_phonon_supercells_with_displacements(
                self._supercell,
                self._dataset,  # type: ignore[arg-type]
            ),
        )

        # Two displacement supercells
        if "first_atoms" in self._dataset:  # type: ignore[typeddict-item]
            for disp1 in self._dataset["first_atoms"]:  # type: ignore[typeddict-item]
                disp_cart1 = disp1["displacement"]
                for disp2 in disp1["second_atoms"]:
                    if "included" in disp2:
                        included = disp2["included"]
                    else:
                        included = True
                    if included:
                        positions = self._supercell.positions
                        positions[disp1["number"]] += disp_cart1
                        positions[disp2["number"]] += disp2["displacement"]
                        supercells.append(
                            PhonopyAtoms(
                                numbers=numbers,
                                masses=masses,
                                magnetic_moments=magmoms,
                                positions=positions,
                                cell=lattice,
                            )
                        )
                    else:
                        supercells.append(None)

        self._supercells_with_displacements = supercells

    def _get_primitive_cell(
        self,
        supercell: Supercell,
        supercell_matrix: NDArray[np.int64],
        primitive_matrix: NDArray[np.double] | None,
    ) -> Primitive:
        inv_supercell_matrix = np.linalg.inv(supercell_matrix)
        if primitive_matrix is None:
            t_mat = inv_supercell_matrix
        else:
            t_mat = np.dot(inv_supercell_matrix, primitive_matrix)

        return get_primitive(
            supercell, t_mat, self._symprec, store_dense_svecs=True, lang=self._lang
        )

    def _set_mesh_numbers(
        self,
        mesh: float | NDArray[np.int64] | Sequence[int] | Sequence[Sequence[int]],
    ) -> None:
        self._interaction = None

        try:
            self._bz_grid = BZGrid(
                mesh,
                lattice=self._primitive.cell,
                symmetry_dataset=self._primitive_symmetry.dataset,
                is_time_reversal=self._is_symmetry,
                use_grg=self._use_grg,
                force_SNF=False,
                SNF_coordinates=self._SNF_coordinates,
                store_dense_gp_map=True,
                lang=self._lang,
            )
        except RuntimeError as e:
            if "Grid symmetry is broken." in str(e) and isinstance(mesh, (float, int)):
                self._bz_grid = BZGrid(
                    mesh,
                    lattice=self._primitive.cell,
                    symmetry_dataset=self._primitive_symmetry.dataset,
                    is_time_reversal=self._is_symmetry,
                    use_grg=True,
                    force_SNF=False,
                    SNF_coordinates=self._SNF_coordinates,
                    store_dense_gp_map=True,
                    lang=self._lang,
                )
            else:
                msg = (
                    "Grid symmetry is broken. If grid symmetry is uncertain, "
                    "try automatic mesh generation using a scalar value."
                )
                raise RuntimeError(msg) from e

    def _init_dynamical_matrix(self) -> None:
        if self._interaction is None:
            msg = (
                "Phono3py.init_phph_interaction has to be called "
                "before running this method."
            )
            raise RuntimeError(msg)

        assert self._fc2 is not None
        self._interaction.init_dynamical_matrix(
            self._fc2,
            self._phonon_supercell,
            self._phonon_primitive,
            nac_params=self._nac_params,
        )
        freqs, _, _ = self.get_phonon_data()
        gp_Gamma = self._interaction.bz_grid.gp_Gamma
        assert freqs is not None
        if np.sum(freqs[gp_Gamma] < self._cutoff_frequency) < 3:
            for i, f in enumerate(freqs[gp_Gamma, :3]):
                if not (f < self._cutoff_frequency):
                    freqs[gp_Gamma, i] = 0
                    print("=" * 26 + " Warning " + "=" * 26)
                    print(
                        " Phonon frequency of band index %d at Gamma "
                        "is calculated to be %f." % (i + 1, f)
                    )
                    print(" But this frequency is forced to be zero.")
                    print("=" * 61)

    def _get_forces_energies(
        self, target: Literal["forces", "supercell_energies"]
    ) -> NDArray[np.double] | None:
        """Return fc3 forces and supercell energies.

        Return None if tagert data is not found rather than raising exception.

        """
        if self._dataset is None:
            return None
        if not forces_in_dataset(self._dataset):  # type: ignore[arg-type]
            return None

        if target in self._dataset:  # type: ignore[operator]  # type-2
            return self._dataset[target]  # type: ignore[literal-required, typeddict-item]
        elif "first_atoms" in self._dataset:  # type: ignore[typeddict-item]  # type-1
            num_scells = len(self._dataset["first_atoms"])  # type: ignore[typeddict-item]
            for disp1 in self._dataset["first_atoms"]:  # type: ignore[typeddict-item]
                num_scells += len(disp1["second_atoms"])
            if target == "forces":
                values = np.zeros(
                    (num_scells, len(self._supercell), 3),
                    dtype="double",
                    order="C",
                )
                type1_target = "forces"
            elif target == "supercell_energies":
                values = np.zeros(num_scells, dtype="double")  # type: ignore[assignment]
                type1_target = "supercell_energy"
            count = 0
            for disp1 in self._dataset["first_atoms"]:  # type: ignore[typeddict-item]
                values[count] = disp1[type1_target]  # type: ignore[literal-required]
                count += 1
            for disp1 in self._dataset["first_atoms"]:  # type: ignore[typeddict-item]
                for disp2 in disp1["second_atoms"]:
                    values[count] = disp2[type1_target]  # type: ignore[literal-required]
                    count += 1
            return values
        return None

    def _set_forces_energies(
        self,
        values: NDArray[np.double]
        | Sequence[float]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]],
        target: Literal["forces", "supercell_energies"],
    ) -> None:
        if self._dataset is None:
            raise RuntimeError("Dataset is not available.")

        if "first_atoms" in self._dataset:  # type: ignore[typeddict-item]  # type-1
            count = 0
            for disp1 in self._dataset["first_atoms"]:  # type: ignore[typeddict-item]
                if target == "forces":
                    disp1[target] = np.array(values[count], dtype="double", order="C")
                elif target == "supercell_energies":
                    v = values[count]
                    assert isinstance(v, (float, np.floating))
                    disp1["supercell_energy"] = float(v)
                count += 1
            for disp1 in self._dataset["first_atoms"]:  # type: ignore[typeddict-item]
                for disp2 in disp1["second_atoms"]:
                    if target == "forces":
                        disp2[target] = np.array(
                            values[count], dtype="double", order="C"
                        )
                    elif target == "supercell_energies":
                        v = values[count]
                        assert isinstance(v, (float, np.floating))
                        disp2["supercell_energy"] = float(v)
                    count += 1
        elif "displacements" in self._dataset or "forces" in self._dataset:  # type-2
            self._dataset[target] = np.array(values, dtype="double", order="C")  # type: ignore[literal-required]
        else:
            raise RuntimeError("Set of FC3 displacements is not available.")

    def _get_phonon_forces_energies(
        self, target: Literal["forces", "supercell_energies"]
    ) -> NDArray[np.double] | None:
        """Return fc2 forces and supercell energies.

        Return None if tagert data is not found rather than raising exception.

        """
        if self._phonon_dataset is None:
            raise RuntimeError("Dataset for fc2 does not exist.")

        if "first_atoms" in self._phonon_dataset:  # type-1
            values = []
            for disp in self._phonon_dataset["first_atoms"]:
                if target == "forces":
                    if target in disp:
                        values.append(disp[target])
                elif target == "supercell_energies":
                    if "supercell_energy" in disp:
                        values.append(disp["supercell_energy"])
            if values:
                return np.array(values, dtype="double", order="C")
        else:
            if target in self._phonon_dataset:  # type-2
                return self._phonon_dataset[target]  # type: ignore

        return None

    def _set_phonon_forces_energies(
        self,
        values: NDArray[np.double]
        | Sequence[float]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]],
        target: Literal["forces", "supercell_energies"],
    ) -> None:
        if self._phonon_dataset is None:
            raise RuntimeError("Dataset for fc2 does not exist.")

        if "first_atoms" in self._phonon_dataset:
            for disp, v in zip(
                self._phonon_dataset["first_atoms"], values, strict=True
            ):
                if target == "forces":
                    disp[target] = np.array(v, dtype="double", order="C")
                elif target == "supercell_energies":
                    assert isinstance(v, (float, np.floating))
                    disp["supercell_energy"] = float(v)
        elif "displacements" in self._phonon_dataset:
            _values = np.array(values, dtype="double", order="C")
            natom = len(self._phonon_supercell)
            ndisps = len(self._phonon_dataset["displacements"])
            if target == "forces" and (
                _values.ndim != 3 or _values.shape != (ndisps, natom, 3)
            ):
                raise RuntimeError(f"Array shape of input {target} is incorrect.")
            elif target == "supercell_energies":
                if _values.ndim != 1 or _values.shape != (ndisps,):
                    raise RuntimeError(f"Array shape of input {target} is incorrect.")
            self._phonon_dataset[target] = _values
        else:
            raise RuntimeError("Set of FC2 displacements is not available.")

    def _generate_random_displacements(
        self,
        number_of_snapshots: int,
        number_of_atoms: int,
        distance: float = 0.03,
        is_plusminus: bool = False,
        random_seed: int | None = None,
        max_distance: float | None = None,
    ) -> Type2DisplacementDataset:
        if random_seed is not None and random_seed >= 0 and random_seed < 2**32:
            _random_seed = random_seed
        else:
            _random_seed = None
        d = get_random_displacements_dataset(
            number_of_snapshots,
            number_of_atoms,
            distance,
            random_seed=_random_seed,
            is_plusminus=is_plusminus,
            max_distance=max_distance,
        )
        if _random_seed is None:
            dataset: Type2DisplacementDataset = {"displacements": d}
        else:
            dataset: Type2DisplacementDataset = {
                "random_seed": _random_seed,
                "displacements": d,
            }
        return dataset

    def _check_mlp_dataset(self, mlp_dataset: Type2DisplacementDataset) -> None:
        if not isinstance(mlp_dataset, dict):
            raise TypeError("mlp_dataset has to be a dictionary.")
        if "displacements" not in mlp_dataset:
            raise RuntimeError("Displacements have to be given.")
        if "forces" not in mlp_dataset:
            raise RuntimeError("Forces have to be given.")
        if "supercell_energies" not in mlp_dataset:
            raise RuntimeError("Supercell energies have to be given.")
        if len(mlp_dataset["displacements"]) != len(mlp_dataset["forces"]):
            raise RuntimeError("Length of displacements and forces are different.")
        if len(mlp_dataset["displacements"]) != len(mlp_dataset["supercell_energies"]):
            raise RuntimeError(
                "Length of displacements and supercell_energies are different."
            )
