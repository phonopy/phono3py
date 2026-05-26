"""Thin experimental driver for 4th-order force constants (fc4).

``Phono4py`` orchestrates the experimental finite-difference fc4 workflow:
generate the fc4 displacement dataset, collect the supercells whose forces are
needed, take the forces back, and solve for fc4 from an equilibrium fc3. While
fc4 is experimental it is kept self-contained in ``phono3py.phonon4`` rather
than added to the main :class:`phono3py.Phono3py` class.

The fc3 part of the workflow (displacements, forces, ``produce_fc3``) is done
with an ordinary :class:`phono3py.Phono3py` instance; the resulting supercell,
symmetry, primitive, and fc3 are handed to ``Phono4py`` (most easily via
:meth:`Phono4py.from_phono3py`).

Typical use::

    ph3 = Phono3py(cell, supercell_matrix, primitive_matrix)
    # ... fc3 workflow: generate_displacements, set forces, produce_fc3 ...
    ph4 = Phono4py.from_phono3py(ph3)
    ph4.generate_displacements(distance=0.03)
    forces = evaluate(ph4.supercells_with_displacements)  # DFT or MLP
    ph4.forces = forces
    ph4.produce_fc4()
    fc4 = ph4.fc4

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.api_phonopy import set_data_to_phonopy_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon3.fc3 import compact_fc3_to_full_fc3
from phono3py.phonon4.dataset import (
    forces_in_dataset_fc4,
    get_displacements_and_forces_fc4,
    get_supercells_with_displacements_fc4,
    set_forces_in_dataset_fc4,
)
from phono3py.phonon4.displacement_fc4 import (
    Fc4Type1DisplacementDataset,
    get_fourth_order_displacements,
)
from phono3py.phonon4.fc4 import get_fc4
from phono3py.phonon4.file_IO import (
    parse_FORCES_FC4,
    read_fc4_from_hdf5,
    write_fc4_to_hdf5,
    write_FORCES_FC4,
)
from phono3py.phonon4.phono4py_yaml import Phono4pyYaml

if TYPE_CHECKING:
    from phono3py import Phono3py


class Phono4py:
    """Experimental finite-difference fc4 driver (see module docstring)."""

    def __init__(
        self,
        supercell: PhonopyAtoms,
        symmetry: Symmetry,
        primitive: Primitive | None = None,
        fc3: NDArray[np.double] | None = None,
        log_level: int = 0,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Initialize with the supercell and its symmetry.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell in which displacements are generated and fc4 is solved.
        symmetry : Symmetry
            Symmetry of the supercell.
        primitive : Primitive, optional
            Primitive cell, required only to expand a compact equilibrium fc3
            to the full layout in :meth:`produce_fc4`. Default is ``None``.
        fc3 : ndarray, optional
            Equilibrium fc3 (compact or full). Default is ``None``; it can also
            be set later via the :attr:`fc3` setter.
        log_level : int, optional
            Verbosity. Default is ``0``.
        lang : {"C", "Rust"}, optional
            Backend for the fc3 sub-computations. Default is ``"Rust"``.

        """
        self._supercell = supercell
        self._symmetry = symmetry
        self._primitive = primitive
        self._fc3 = fc3
        self._log_level = log_level
        self._lang: Literal["C", "Rust"] = lang
        self._dataset: Fc4Type1DisplacementDataset | None = None
        self._fc4: NDArray[np.double] | None = None
        # Source Phono3py kept for save() (cell/matrices for phono4py_disp.yaml).
        self._phono3py: Phono3py | None = None

    @classmethod
    def from_phono3py(
        cls, phono3py: Phono3py, fc3: NDArray[np.double] | None = None
    ) -> Phono4py:
        """Create a ``Phono4py`` reusing a configured ``Phono3py`` instance.

        The supercell, symmetry, primitive, backend, and (unless overridden)
        the equilibrium fc3 are taken from ``phono3py``.
        """
        ph4 = cls(
            phono3py.supercell,
            phono3py.symmetry,
            primitive=phono3py.primitive,
            fc3=phono3py.fc3 if fc3 is None else fc3,
            log_level=getattr(phono3py, "_log_level", 0),
            lang=getattr(phono3py, "_lang", "Rust"),
        )
        ph4._phono3py = phono3py
        return ph4

    @property
    def supercell(self) -> PhonopyAtoms:
        """Return the supercell."""
        return self._supercell

    @property
    def symmetry(self) -> Symmetry:
        """Return the supercell symmetry."""
        return self._symmetry

    @property
    def primitive(self) -> Primitive | None:
        """Return the primitive cell, if set."""
        return self._primitive

    @property
    def dataset(self) -> Fc4Type1DisplacementDataset | None:
        """Setter and getter of the fc4 displacement dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Fc4Type1DisplacementDataset | None) -> None:
        self._dataset = dataset

    @property
    def fc3(self) -> NDArray[np.double] | None:
        """Setter and getter of the equilibrium fc3 (compact or full)."""
        return self._fc3

    @fc3.setter
    def fc3(self, fc3: NDArray[np.double] | None) -> None:
        self._fc3 = fc3

    @property
    def fc4(self) -> NDArray[np.double] | None:
        """Setter and getter of fc4, shape ``(N, N, N, N, 3, 3, 3, 3)``."""
        return self._fc4

    @fc4.setter
    def fc4(self, fc4: NDArray[np.double] | None) -> None:
        self._fc4 = fc4

    @property
    def supercells_with_displacements(self) -> list[PhonopyAtoms]:
        """Return the fc4 supercells with displacements applied (ordered by id).

        Forces evaluated for these supercells must be assigned back, in the
        same order, via the :attr:`forces` setter.
        """
        if self._dataset is None:
            raise RuntimeError(
                "fc4 displacement dataset is not set. Run generate_displacements."
            )
        return get_supercells_with_displacements_fc4(self._supercell, self._dataset)

    @property
    def displacements(self) -> NDArray[np.double]:
        """Return the cumulative supercell displacements (ordered by id)."""
        if self._dataset is None:
            raise RuntimeError(
                "fc4 displacement dataset is not set. Run generate_displacements."
            )
        return get_displacements_and_forces_fc4(self._dataset)[0]

    @property
    def forces(self) -> NDArray[np.double] | None:
        """Setter and getter of the fc4 supercell forces (ordered by id)."""
        if self._dataset is None:
            return None
        return get_displacements_and_forces_fc4(self._dataset)[1]

    @forces.setter
    def forces(self, forces: NDArray[np.double] | list[NDArray[np.double]]) -> None:
        if self._dataset is None:
            raise RuntimeError(
                "fc4 displacement dataset is not set. Run generate_displacements."
            )
        set_forces_in_dataset_fc4(self._dataset, forces)

    def generate_displacements(
        self,
        distance: float = 0.03,
        is_plusminus: bool | Literal["auto"] = "auto",
        is_diagonal: bool = False,
    ) -> None:
        """Generate the fc4 displacement dataset in the supercell.

        Parameters
        ----------
        distance : float, optional
            Displacement distance in Angstrom. Default is ``0.03``.
        is_plusminus : bool or "auto", optional
            Whether to add minus displacements of the first atoms. Default is
            ``"auto"``.
        is_diagonal : bool, optional
            Whether to allow diagonal displacements of the second and third
            atoms. Default is ``False``.

        """
        self._dataset = get_fourth_order_displacements(
            self._supercell,
            self._symmetry,
            distance,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal,
        )
        self._fc4 = None

    def produce_fc4(
        self,
        is_compact_fc: bool = True,
        is_translational_symmetry: bool = True,
        is_permutation_symmetry: bool = True,
    ) -> None:
        """Solve for fc4 from the dataset forces and the equilibrium fc3.

        The equilibrium fc3 (:attr:`fc3`) is expanded to the full layout when
        it is compact. The same symmetrization is applied to the equilibrium
        and constrained fc3 before differencing and to the final fc4.

        Parameters
        ----------
        is_compact_fc : bool, optional
            Store fc4 in the compact layout ``(n_patom, N, N, N, 3, 3, 3, 3)``
            instead of the full ``(N, N, N, N, 3, 3, 3, 3)``. The compact layout
            requires the ``phonors`` Rust extension and a primitive cell.
            Default is ``True``.
        is_translational_symmetry : bool, optional
            Enforce the acoustic sum rule. Default is ``True``.
        is_permutation_symmetry : bool, optional
            Enforce permutation symmetry. Default is ``True``.

        """
        if self._dataset is None:
            raise RuntimeError(
                "fc4 displacement dataset is not set. Run generate_displacements."
            )
        if self._fc3 is None:
            raise RuntimeError("Equilibrium fc3 is not set.")
        if is_compact_fc and self._primitive is None:
            raise RuntimeError("Primitive cell is required for compact fc4.")

        fc3 = self._fc3
        num_satom = len(self._supercell)
        if fc3.shape[0] != num_satom:
            if self._primitive is None:
                raise RuntimeError(
                    "Primitive cell is required to expand compact fc3 to full fc3."
                )
            fc3 = compact_fc3_to_full_fc3(
                self._primitive, fc3, log_level=self._log_level, lang=self._lang
            )

        self._fc4 = get_fc4(
            self._supercell,
            self._dataset,
            fc3,
            self._symmetry,
            primitive=self._primitive,
            is_compact_fc=is_compact_fc,
            is_translational_symmetry=is_translational_symmetry,
            is_permutation_symmetry=is_permutation_symmetry,
            verbose=self._log_level > 0,
            lang=self._lang,
        )

    def save(
        self,
        filename: str | os.PathLike = "phono4py_disp.yaml",
        with_forces: bool | None = None,
    ) -> None:
        """Write a phono4py_disp.yaml with the cell and fc4 displacement dataset.

        Requires the instance to have been created via
        :meth:`from_phono3py` (the source Phono3py provides the cell and
        supercell/primitive matrices).

        Parameters
        ----------
        filename : str, optional
            Output filename. Default is ``"phono4py_disp.yaml"``.
        with_forces : bool, optional
            Whether to also write the dataset forces. Default is None, which
            writes them only when the dataset already carries forces.

        """
        if self._dataset is None:
            raise RuntimeError(
                "fc4 displacement dataset is not set. Run generate_displacements."
            )
        if self._phono3py is None:
            raise RuntimeError(
                "save() requires the source Phono3py; create the instance via "
                "Phono4py.from_phono3py()."
            )
        if with_forces is None:
            with_forces = forces_in_dataset_fc4(self._dataset)
        phono4py_yaml = Phono4pyYaml(settings={"force_sets": with_forces})
        set_data_to_phonopy_yaml(phono4py_yaml, self._phono3py)
        phono4py_yaml.dataset_fc4 = self._dataset
        with open(filename, "w") as w:
            w.write(str(phono4py_yaml))

    def load(self, filename: str | os.PathLike = "phono4py_disp.yaml") -> None:
        """Load the fc4 displacement dataset (and forces, if present) from yaml."""
        phono4py_yaml = Phono4pyYaml()
        phono4py_yaml.read(filename)
        self._dataset = phono4py_yaml.dataset_fc4
        self._fc4 = None

    def save_fc4(self, filename: str | os.PathLike = "fc4.hdf5") -> None:
        """Write fc4 to an hdf5 file (with p2s_map for the compact layout)."""
        if self._fc4 is None:
            raise RuntimeError("fc4 is not computed. Run produce_fc4.")
        p2s_map = None
        if self._fc4.shape[0] != len(self._supercell) and self._primitive is not None:
            p2s_map = self._primitive.p2s_map
        write_fc4_to_hdf5(self._fc4, filename=filename, p2s_map=p2s_map)

    def load_fc4(self, filename: str | os.PathLike = "fc4.hdf5") -> None:
        """Read fc4 from an hdf5 file."""
        p2s_map = self._primitive.p2s_map if self._primitive is not None else None
        self._fc4 = read_fc4_from_hdf5(filename=filename, p2s_map=p2s_map)

    def save_forces(self, filename: str | os.PathLike = "FORCES_FC4") -> None:
        """Write the dataset supercell forces to a FORCES_FC4 text file."""
        if self._dataset is None:
            raise RuntimeError("fc4 displacement dataset is not set.")
        write_FORCES_FC4(self._dataset, filename=filename)

    def load_forces(self, filename: str | os.PathLike = "FORCES_FC4") -> None:
        """Read supercell forces from FORCES_FC4 into the dataset (in place)."""
        if self._dataset is None:
            raise RuntimeError(
                "fc4 displacement dataset is not set. Run generate_displacements "
                "or load() first."
            )
        parse_FORCES_FC4(self._dataset, filename=filename)
