"""phono3py_yaml reader and writer."""

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

import dataclasses
import os
import typing
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from phonopy.interface.phonopy_yaml import (
    PhonopyYaml,
    PhonopyYamlDumperBase,
    PhonopyYamlLoaderBase,
    load_yaml,
)

if TYPE_CHECKING:
    from phono3py import Phono3py

from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYamlData
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell


@dataclasses.dataclass
class Phono3pyYamlData(PhonopyYamlData):
    """Phono3pyYaml data structure."""

    command_name: str = "phono3py"
    phonon_supercell_matrix: NDArray | None = None
    phonon_dataset: dict | None = None
    phonon_supercell: Supercell | PhonopyAtoms | None = None
    phonon_primitive: Primitive | PhonopyAtoms | None = None


class Phono3pyYamlLoader(PhonopyYamlLoaderBase):
    """Phono3pyYaml loader."""

    def __init__(
        self,
        yaml_data: dict,
        configuration: dict | None = None,
        calculator: str | None = None,
        physical_units: dict | None = None,
    ):
        """Init method.

        Parameters
        ----------
        yaml_data : dict

        """
        self._yaml = yaml_data
        self._data = Phono3pyYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )

    @property
    def data(self) -> Phono3pyYamlData:
        """Return Phono3pyYamlData instance."""
        return self._data

    def parse(self) -> "Phono3pyYamlLoader":
        """Yaml dict is parsed. See docstring of this class."""
        super().parse()
        self._parse_fc3_dataset()
        return self

    def _parse_all_cells(self):
        """Parse all cells.

        This method override PhonopyYaml._parse_all_cells.

        """
        super()._parse_all_cells()
        if "phonon_supercell_matrix" in self._yaml:
            self._data.phonon_supercell_matrix = np.array(
                self._yaml["phonon_supercell_matrix"], dtype="intc", order="C"
            )
            if "phonon_primitive_cell" in self._yaml:
                self._data.phonon_primitive = self._parse_cell(
                    self._yaml["phonon_primitive_cell"]
                )
            if "phonon_supercell" in self._yaml:
                self._data.phonon_supercell = self._parse_cell(
                    self._yaml["phonon_supercell"]
                )

    def _parse_dataset(self):
        """Parse phonon_dataset.

        This method override PhonopyYaml._parse_dataset.

        Phonon_displacements in type1 in old format is represented in yaml by

        displacements:

        But displacements (for fc3) in type2 in old format is also represented
        by

        displacements:

        Therefore, these have to be distinguished by phonon_supercell_matrix.

        In new format of type1

        displacements: -> phonon_displacements:

        and displacements: of type2 is put under phonon_dataset: block.

        """
        if (
            self._data.phonon_supercell_matrix is not None
            and "phonon_displacements" not in self._yaml
            and "displacements" in self._yaml
        ):  # old type1
            self._data.phonon_dataset = self._get_dataset(
                cast(PhonopyAtoms, self._data.phonon_supercell)
            )
        else:
            self._data.phonon_dataset = self._get_dataset(
                cast(PhonopyAtoms, self._data.phonon_supercell), key_prefix="phonon_"
            )

    def _parse_fc3_dataset(self):
        """Parse force dataset for fc3.

        'duplicates' can be either dict (<v1.21) or list in phono3py.yaml.
        From v1.21, it was changed to list of list because
        dict with a key of int type is not allowed in JSON.

        "displacements" gives type-II fc3 dataset at v2.2 or later, although
        it gave phonon-dataset at versions older than v2.2.

        """
        dataset = None
        if "displacement_pairs" in self._yaml:
            disp = self._yaml["displacement_pairs"][0]
            if isinstance(disp, dict):  # type1
                assert self._data.supercell is not None
                dataset = self._parse_fc3_dataset_type1(len(self._data.supercell))
            elif isinstance(disp, list):  # type2
                if "displacement" in disp[0]:
                    dataset = self._parse_force_sets_type2()
        if "displacement_pair_info" in self._yaml and dataset is not None:
            info_yaml = self._yaml["displacement_pair_info"]
            if "cutoff_pair_distance" in info_yaml:
                dataset["cutoff_distance"] = info_yaml["cutoff_pair_distance"]
            if "duplicated_supercell_ids" in info_yaml:
                dataset["duplicates"] = info_yaml["duplicated_supercell_ids"]
        self._data.dataset = dataset

        # This case should work only for v2.2 or later.
        if self._data.dataset is None:
            self._data.dataset = self._get_dataset(self._data.supercell)

    def _parse_fc3_dataset_type1(self, natom: int) -> dict:
        """Parse fc3 type1-dataset."""
        dataset = {"natom": natom, "first_atoms": []}
        disp2_id = len(self._yaml["displacement_pairs"])
        for disp1_id, d1 in enumerate(self._yaml["displacement_pairs"]):
            data1 = {
                "number": d1["atom"] - 1,
                "displacement": np.array(d1["displacement"], dtype="double"),
                "second_atoms": [],
            }
            if "displacement_id" in d1:
                d1_id = d1["displacement_id"]
                if disp1_id + 1 != d1_id:
                    msg = f"{d1_id} != {disp1_id + 1}. Dataset may be broken."
                    raise RuntimeError(msg)
                data1["id"] = d1_id
            if "forces" in d1:
                data1["forces"] = np.array(d1["forces"], dtype="double", order="C")
            if "supercell_energy" in d1:
                data1["supercell_energy"] = d1["supercell_energy"]
            d2_list = d1.get("paired_with")
            if d2_list is None:  # backward compatibility
                d2_list = d1.get("second_atoms")
            assert d2_list is not None
            for d2 in d2_list:
                if "displacements" in d2:
                    disp2_id = self._parse_fc3_dataset_type1_without_forces(
                        data1, d2, disp2_id
                    )
                else:
                    disp2_id = self._parse_fc3_dataset_type1_with_disp_pairs(
                        data1, d2, disp2_id
                    )

            dataset["first_atoms"].append(data1)
        return dataset

    def _parse_fc3_dataset_type1_with_disp_pairs(
        self, data1: dict, d2: dict, disp2_id: int
    ) -> int:
        """Parse fc3 type1-dataset with lists of displacement pairs.

        One displacement can couple with one force-set.

        """
        second_atom_dict = {
            "number": d2["atom"] - 1,
            "displacement": np.array(d2["displacement"], dtype="double"),
        }
        if "forces" in d2:
            second_atom_dict.update(
                {"forces": np.array(d2["forces"], dtype="double", order="C")}
            )
        if "displacement_id" in d2:
            d2_id = d2["displacement_id"]
            if disp2_id + 1 != d2_id:
                msg = f"{d2_id} != {disp2_id + 1}. Dataset may be broken."
                raise RuntimeError(msg)
            second_atom_dict["id"] = d2_id
        if "pair_distance" in d2:
            second_atom_dict["pair_distance"] = d2["pair_distance"]
        if "supercell_energy" in d2:
            second_atom_dict["supercell_energy"] = d2["supercell_energy"]
        disp2_id += 1
        data1["second_atoms"].append(second_atom_dict)
        return disp2_id

    def _parse_fc3_dataset_type1_without_forces(
        self, data1: dict, d2: dict, disp2_id: int
    ) -> int:
        """Parse fc3 type1-dataset that doesn't have forces in it.

        Displacements are stored in `displacements` as a list.

        """
        disps = [
            {
                "number": d2["atom"] - 1,
                "displacement": np.array(disp, dtype="double"),
            }
            for disp in d2["displacements"]
        ]
        if "pair_distance" in d2:
            for d2_dict in disps:
                d2_dict["pair_distance"] = d2["pair_distance"]
        if "included" in d2:
            for d2_dict in disps:
                d2_dict["included"] = d2["included"]
        if "displacement_ids" in d2:
            for d2_id, d2_dict in zip(d2["displacement_ids"], disps, strict=True):
                if disp2_id + 1 != d2_id:
                    msg = f"{d2_id} != {disp2_id + 1}. Dataset may be broken."
                    raise RuntimeError(msg)
                d2_dict["id"] = d2_id
                disp2_id += 1
        else:
            disp2_id += len(disps)
        data1["second_atoms"] += disps
        return disp2_id


class Phono3pyYamlDumper(PhonopyYamlDumperBase):
    """Phono3pyYaml dumper."""

    _default_dumper_settings = {
        "force_sets": True,
        "displacements": True,
        "force_constants": False,
        "born_effective_charge": True,
        "dielectric_constant": True,
    }

    def __init__(self, data: Phono3pyYamlData, dumper_settings: dict | None = None):
        """Init method."""
        self._data = data
        self._init_dumper_settings(dumper_settings)

    def _cell_info_yaml_lines(self) -> list:
        """Get YAML lines for information of cells.

        This method override PhonopyYaml._cell_info_yaml_lines.

        """
        lines = super()._cell_info_yaml_lines()
        if self._data.phonon_supercell_matrix is not None:
            lines += self._supercell_matrix_yaml_lines(
                self._data.phonon_supercell_matrix, "phonon_supercell_matrix"
            )
            lines += self._primitive_yaml_lines(
                self._data.phonon_primitive, "phonon_primitive_cell"
            )
            lines += self._phonon_supercell_yaml_lines()
        return lines

    def _phonon_supercell_yaml_lines(self) -> list:
        lines = []
        if self._data.phonon_supercell is not None:
            s2p_map = getattr(self._data.phonon_primitive, "s2p_map", None)
            lines += self._cell_yaml_lines(
                self._data.phonon_supercell, "phonon_supercell", s2p_map
            )
            lines.append("")
        return lines

    def _nac_yaml_lines(self) -> list:
        """Get YAML lines for parameters of non-analytical term correction.

        This method override PhonopyYaml._nac_yaml_lines.

        """
        if self._data.phonon_primitive is not None:
            return self._nac_yaml_lines_given_symbols(
                self._data.phonon_primitive.symbols
            )
        else:
            assert self._data.primitive is not None
            return self._nac_yaml_lines_given_symbols(self._data.primitive.symbols)

    def _displacements_yaml_lines(self, with_forces: bool = False) -> list:
        """Get YAML lines for phonon_dataset and dataset.

        This method override PhonopyYaml._displacements_yaml_lines.
        PhonopyYaml._displacements_yaml_lines_2types is written
        to be also used by Phono3pyYaml.

        """
        lines = []
        if self._data.phonon_dataset is not None:
            lines += self._displacements_yaml_lines_2types(
                self._data.phonon_dataset,
                with_forces=with_forces,
                key_prefix="phonon_",
            )
            lines.append("")
        lines += self._displacements_yaml_lines_2types(
            self._data.dataset, with_forces=with_forces
        )
        return lines

    def _displacements_yaml_lines_type1(
        self, dataset: dict, with_forces: bool = False, key_prefix: str = ""
    ) -> list:
        """Get YAML lines for type1 phonon_dataset and dataset.

        This method override PhonopyYaml._displacements_yaml_lines_type1.
        PhonopyYaml._displacements_yaml_lines_2types calls
        Phono3pyYaml._displacements_yaml_lines_type1.

        """
        return displacements_yaml_lines_type1(
            dataset, with_forces=with_forces, key_prefix=key_prefix
        )


class Phono3pyYaml(PhonopyYaml):
    """phono3py.yaml reader and writer.

    Details are found in the docstring of PhonopyYaml.
    The common usages are as follows:

    1. Set phono3py instance.
        p3yml = Phono3pyYaml()
        p3yml.set_phonon_info(phono3py_instance)
    2. Read phono3py.yaml file.
        p3yml = Phono3pyYaml()
        p3yml.read(filename)
    3. Parse yaml dict of phono3py.yaml.
        with open("phono3py.yaml", 'r') as f:
            p3yml.yaml_data = yaml.load(f, Loader=yaml.CLoader)
            p3yml.parse()
    4. Save stored data in Phono3pyYaml instance into a text file in yaml.
        with open(filename, 'w') as w:
            w.write(str(ph3py_yaml))

    """

    default_filenames = ("phono3py_disp.yaml", "phono3py.yaml")
    command_name = "phono3py"

    def __init__(
        self,
        configuration: dict | None = None,
        calculator: str | None = None,
        physical_units: dict | None = None,
        settings: dict | None = None,
    ):
        """Init method."""
        self._data = Phono3pyYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )
        self._dumper_settings = settings

    @property
    def phonon_primitive(self) -> PhonopyAtoms | None:
        """Return phonon primitive cell of phonopy calculation."""
        return self._data.phonon_primitive

    @phonon_primitive.setter
    def phonon_primitive(self, value: PhonopyAtoms):
        """Set phonon primitive cell of phonopy calculation."""
        self._data.phonon_primitive = value

    @property
    def phonon_supercell(self) -> PhonopyAtoms | None:
        """Return phonon supercell of phonopy calculation."""
        return self._data.phonon_supercell

    @phonon_supercell.setter
    def phonon_supercell(self, value: PhonopyAtoms):
        """Set phonon supercell of phonopy calculation."""
        self._data.phonon_supercell = value

    @property
    def phonon_dataset(self) -> dict | None:
        """Return phonon dataset of phonopy calculation."""
        return self._data.phonon_dataset

    @phonon_dataset.setter
    def phonon_dataset(self, value: dict):
        """Set phonon dataset of phonopy calculation."""
        self._data.phonon_dataset = value

    @property
    def phonon_supercell_matrix(self) -> NDArray | None:
        """Return phonon supercell matrix of phonopy calculation."""
        return self._data.phonon_supercell_matrix

    @phonon_supercell_matrix.setter
    def phonon_supercell_matrix(self, value: ArrayLike):
        """Set supercell matrix of phonopy calculation."""
        self._data.phonon_supercell_matrix = np.array(value, dtype="intc", order="C")

    def __str__(self):
        """Return string text of yaml output."""
        ph3yml_dumper = Phono3pyYamlDumper(
            self._data, dumper_settings=self._dumper_settings
        )
        return "\n".join(ph3yml_dumper.get_yaml_lines())

    def read(self, filename: str | os.PathLike | typing.IO) -> Phono3pyYaml:
        """Read Phono3pyYaml file."""
        self._data = read_phono3py_yaml(
            filename,
            configuration=self._data.configuration,
            calculator=self._data.calculator,
            physical_units=self._data.physical_units,
        )
        return self

    def set_phonon_info(self, phono3py: "Phono3py"):
        """Store data in Phono3py instance in this instance."""
        super().set_phonon_info(cast(Phonopy, phono3py))
        self._data.phonon_supercell_matrix = phono3py.phonon_supercell_matrix
        self._data.phonon_dataset = phono3py.phonon_dataset
        self._data.phonon_primitive = phono3py.phonon_primitive
        self._data.phonon_supercell = phono3py.phonon_supercell


def displacements_yaml_lines_type1(
    dataset: dict, with_forces: bool = False, key_prefix: str = ""
) -> list:
    """Get YAML lines for type1 phonon_dataset and dataset.

    This is a function but not class method because used by other function.

    """
    id_offset = len(dataset["first_atoms"])

    if "second_atoms" in dataset["first_atoms"][0]:
        lines = ["displacement_pairs:"]
    else:
        lines = [f"{key_prefix}displacements:"]
    for disp1_id, d1 in enumerate(dataset["first_atoms"]):
        lines.append("- atom: %4d" % (d1["number"] + 1))
        lines.append("  displacement:")
        lines.append("    [ %19.16f, %19.16f, %19.16f ]" % tuple(d1["displacement"]))
        if "id" in d1:
            assert disp1_id + 1 == d1["id"]
        lines.append(f"  displacement_id: {disp1_id + 1}")
        if with_forces and "forces" in d1:
            lines.append("  forces:")
            for v in d1["forces"]:
                lines.append("  - [ %19.16f, %19.16f, %19.16f ]" % tuple(v))
            if "supercell_energy" in d1:
                lines.append(
                    "  supercell_energy: {energy:.8f}".format(
                        energy=d1["supercell_energy"]
                    )
                )
        if "second_atoms" in d1:
            ret_lines, id_offset = _second_displacements_yaml_lines(
                d1["second_atoms"], id_offset, with_forces=with_forces
            )
            lines += ret_lines
    lines.append("")

    if "second_atoms" in dataset["first_atoms"][0]:
        lines += _displacements_yaml_lines_type1_info(dataset)

    return lines


def _displacements_yaml_lines_type1_info(dataset: dict) -> list:
    """Return lines of displacement-pair summary."""
    n_single = len(dataset["first_atoms"])
    n_pair = 0
    n_included = 0
    for d1 in dataset["first_atoms"]:
        n_pair += len(d1["second_atoms"])
        for d2 in d1["second_atoms"]:
            if "included" not in d2:
                n_included += 1
            elif d2["included"]:
                n_included += 1

    lines = []
    lines.append("displacement_pair_info:")
    if "cutoff_distance" in dataset:
        lines.append("  cutoff_pair_distance: %11.8f" % dataset["cutoff_distance"])
    lines.append("  number_of_singles: %d" % n_single)
    lines.append("  number_of_pairs: %d" % n_pair)
    if "cutoff_distance" in dataset:
        lines.append("  number_of_pairs_in_cutoff: %d" % n_included)

    # 'duplicates' is dict, but written as a list of list in yaml.
    # See the docstring of _parse_fc3_dataset for the reason.
    if "duplicates" in dataset and dataset["duplicates"]:
        lines.append("  duplicated_supercell_ids: # 0 means perfect supercell")
        # Backward compatibility for dict type
        if isinstance(dataset["duplicates"], dict):
            for disp1_id, j in dataset["duplicates"].items():
                lines.append("  - [ %d, %d ]" % (int(disp1_id), j))
        else:
            for disp1_id, j in dataset["duplicates"]:
                lines.append("  - [ %d, %d ]" % (disp1_id, j))
        lines.append("")

    return lines


def _second_displacements_yaml_lines(
    dataset2: list, id_offset: int, with_forces: bool = False
) -> tuple[list, int]:
    lines = []
    disp2_id = id_offset
    # lines.append("  second_atoms:")
    lines.append("  paired_with:")
    numbers = np.array([d["number"] for d in dataset2])
    unique_numbers = np.unique(numbers)
    for i in unique_numbers:
        indices_eq_i = np.sort(np.where(numbers == i)[0])
        if with_forces and "forces" in dataset2[indices_eq_i[0]]:
            for j in indices_eq_i:
                disp2_id += 1
                lines.append("  - atom: %4d" % (i + 1))
                lines.append("    pair_distance: %.8f" % dataset2[j]["pair_distance"])
                lines.append("    displacement:")
                lines.append(
                    "      [ %19.16f, %19.16f, %19.16f ]"
                    % tuple(dataset2[j]["displacement"])
                )

                if "id" in dataset2[j]:
                    assert dataset2[j]["id"] == disp2_id
                lines.append("    displacement_id: %d" % disp2_id)

                lines.append("    forces:")
                for v in dataset2[j]["forces"]:
                    lines.append("    - [ %19.16f, %19.16f, %19.16f ]" % tuple(v))
                if "supercell_energy" in dataset2[j]:
                    lines.append(
                        "    supercell_energy: {energy:.8f}".format(
                            energy=dataset2[j]["supercell_energy"]
                        )
                    )
        else:
            lines.append("  - atom: %4d" % (i + 1))
            lines.append(
                "    pair_distance: %.8f" % dataset2[indices_eq_i[0]]["pair_distance"]
            )
            if "included" in dataset2[indices_eq_i[0]]:
                included = dataset2[indices_eq_i[0]]["included"]
                lines.append("    included: %s" % ("true" if included else "false"))
            disp_ids = []
            lines.append("    displacements:")
            for j in indices_eq_i:
                disp2_id += 1
                d = tuple(dataset2[j]["displacement"])
                lines.append("    - [ %19.16f, %19.16f, %19.16f ]" % d)
                if "id" in dataset2[j]:
                    assert dataset2[j]["id"] == disp2_id
                disp_ids.append(disp2_id)
            lines.append(
                "    displacement_ids: [ %s ]" % ", ".join(["%d" % j for j in disp_ids])
            )

    return lines, disp2_id


def read_phono3py_yaml(
    filename: str | os.PathLike | typing.IO,
    configuration: dict | None = None,
    calculator: str | None = None,
    physical_units: dict | None = None,
) -> Phono3pyYamlData:
    """Read phono3py.yaml like file."""
    yaml_data = load_yaml(filename)
    if isinstance(yaml_data, str):
        msg = f'Could not load "{filename}" properly.'
        raise TypeError(msg)
    return load_phono3py_yaml(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )


def load_phono3py_yaml(
    yaml_data: dict,
    configuration: dict | None = None,
    calculator: str | None = None,
    physical_units: dict | None = None,
) -> Phono3pyYamlData:
    """Return Phono3pyYamlData instance loading yaml data.

    Parameters
    ----------
    yaml_data : dict

    """
    ph3yml_loader = Phono3pyYamlLoader(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )
    ph3yml_loader.parse()
    return ph3yml_loader.data
