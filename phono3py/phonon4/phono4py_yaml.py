"""phono4py_disp.yaml / phono4py.yaml reader and writer (experimental).

``Phono4pyYaml`` subclasses :class:`phono3py.interface.phono3py_yaml.Phono3pyYaml`
and reuses all of its cell and base machinery, adding only the fc4 displacement
dataset as a ``displacement_quartets`` section. The fc4 dataset is always
type-1, nested three levels deep (first / second / third atoms), and carries a
running ``displacement_id`` at every level; forces are written when present.
"""

from __future__ import annotations

import dataclasses
import os
import typing

import numpy as np
from phonopy.interface.phonopy_yaml import load_yaml
from phonopy.physical_units import CalculatorPhysicalUnits

from phono3py.interface.phono3py_yaml import (
    Phono3pyYaml,
    Phono3pyYamlData,
    Phono3pyYamlDumper,
    Phono3pyYamlLoader,
)
from phono3py.phonon4.displacement_fc4 import Fc4Type1DisplacementDataset


@dataclasses.dataclass
class Phono4pyYamlData(Phono3pyYamlData):
    """Phono4pyYaml data structure (adds the fc4 dataset)."""

    command_name: str = "phono4py"
    dataset_fc4: Fc4Type1DisplacementDataset | None = None


def _quartet_node_yaml_lines(node: dict, level: int, with_forces: bool) -> list[str]:
    """Return YAML lines for one displacement node at the given nesting level."""
    ind = "  " * level
    field = ind + "  "
    disp = node["displacement"]
    lines = [
        f"{ind}- atom: {node['number'] + 1}",
        f"{field}displacement:",
        f"{field}  [ {disp[0]:19.16f}, {disp[1]:19.16f}, {disp[2]:19.16f} ]",
    ]
    if "id" in node:
        lines.append(f"{field}displacement_id: {node['id']}")
    if with_forces and "forces" in node:
        lines.append(f"{field}forces:")
        for v in node["forces"]:
            lines.append(f"{field}- [ {v[0]:19.16f}, {v[1]:19.16f}, {v[2]:19.16f} ]")
    return lines


def displacement_quartets_yaml_lines(
    dataset: Fc4Type1DisplacementDataset, with_forces: bool = False
) -> list[str]:
    """Return YAML lines for a type-1 fc4 displacement dataset.

    The three nesting levels (first / second / third atoms) are written with
    nested ``paired_with`` blocks, mirroring the ``displacement_pairs`` format
    of fc3 one level deeper.
    """
    lines = ["displacement_quartets:"]
    for first_atom in dataset["first_atoms"]:
        lines += _quartet_node_yaml_lines(first_atom, 0, with_forces)
        lines.append("  paired_with:")
        for second_atom in first_atom["second_atoms"]:
            lines += _quartet_node_yaml_lines(second_atom, 1, with_forces)
            lines.append("    paired_with:")
            for third_atom in second_atom["third_atoms"]:
                lines += _quartet_node_yaml_lines(third_atom, 2, with_forces)
    lines.append("")
    return lines


class Phono4pyYamlDumper(Phono3pyYamlDumper):
    """Phono4pyYaml dumper (appends the fc4 displacement_quartets section)."""

    def _displacements_yaml_lines(self, with_forces: bool = False) -> list[str]:
        lines = super()._displacements_yaml_lines(with_forces=with_forces)
        dataset_fc4 = getattr(self._data, "dataset_fc4", None)
        if dataset_fc4 is not None:
            lines += displacement_quartets_yaml_lines(
                dataset_fc4, with_forces=with_forces
            )
        return lines


class Phono4pyYamlLoader(Phono3pyYamlLoader):
    """Phono4pyYaml loader (parses the fc4 displacement_quartets section)."""

    def __init__(
        self,
        yaml_data: dict,
        configuration: dict | None = None,
        calculator: str | None = None,
        physical_units: CalculatorPhysicalUnits | None = None,
    ) -> None:
        """Init method."""
        self._yaml = yaml_data
        self._data: Phono4pyYamlData = Phono4pyYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )

    @property
    def data(self) -> Phono4pyYamlData:
        """Return Phono4pyYamlData instance."""
        return self._data

    def parse(self) -> Phono4pyYamlLoader:
        """Parse yaml dict, then the fc4 dataset."""
        super().parse()
        self._parse_fc4_dataset()
        return self

    def _parse_fc4_dataset(self) -> None:
        if "displacement_quartets" not in self._yaml:
            return
        assert self._data.supercell is not None
        natom = len(self._data.supercell)
        first_atoms = []
        for d1 in self._yaml["displacement_quartets"]:
            first_atom = _parse_quartet_node(d1)
            first_atom["second_atoms"] = []
            for d2 in d1.get("paired_with", []):
                second_atom = _parse_quartet_node(d2)
                second_atom["third_atoms"] = [
                    _parse_quartet_node(d3) for d3 in d2.get("paired_with", [])
                ]
                first_atom["second_atoms"].append(second_atom)
            first_atoms.append(first_atom)
        self._data.dataset_fc4 = {"natom": natom, "first_atoms": first_atoms}


def _parse_quartet_node(node: dict) -> dict:
    """Parse one displacement node from a yaml dict."""
    parsed: dict = {
        "number": node["atom"] - 1,
        "displacement": np.array(node["displacement"], dtype="double"),
    }
    if "displacement_id" in node:
        parsed["id"] = node["displacement_id"]
    if "forces" in node:
        parsed["forces"] = np.array(node["forces"], dtype="double", order="C")
    return parsed


class Phono4pyYaml(Phono3pyYaml):
    """phono4py.yaml reader and writer (experimental).

    Reuses :class:`phono3py.interface.phono3py_yaml.Phono3pyYaml` for cells and
    base machinery, adding the fc4 displacement dataset via :attr:`dataset_fc4`.
    """

    default_filenames = ("phono4py_disp.yaml", "phono4py.yaml")
    command_name = "phono4py"

    def __init__(
        self,
        configuration: dict | None = None,
        calculator: str | None = None,
        physical_units: CalculatorPhysicalUnits | None = None,
        settings: dict | None = None,
    ) -> None:
        """Init method."""
        self._data: Phono4pyYamlData = Phono4pyYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )
        self._dumper_settings = settings

    @property
    def dataset_fc4(self) -> Fc4Type1DisplacementDataset | None:
        """Setter and getter of the fc4 displacement dataset."""
        return self._data.dataset_fc4

    @dataset_fc4.setter
    def dataset_fc4(self, value: Fc4Type1DisplacementDataset | None) -> None:
        self._data.dataset_fc4 = value

    def __str__(self) -> str:
        """Return string text of yaml output."""
        dumper = Phono4pyYamlDumper(self._data, dumper_settings=self._dumper_settings)
        return "\n".join(dumper.get_yaml_lines())

    def read(self, filename: str | os.PathLike | typing.IO) -> Phono4pyYaml:
        """Read a phono4py.yaml-like file."""
        self._data = read_phono4py_yaml(
            filename,
            configuration=self._data.configuration,
            calculator=self._data.calculator,
            physical_units=self._data.physical_units,
        )
        return self


def load_phono4py_yaml(
    yaml_data: dict,
    configuration: dict | None = None,
    calculator: str | None = None,
    physical_units: CalculatorPhysicalUnits | None = None,
) -> Phono4pyYamlData:
    """Return Phono4pyYamlData by loading a parsed yaml dict."""
    loader = Phono4pyYamlLoader(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )
    loader.parse()
    return loader.data


def read_phono4py_yaml(
    filename: str | os.PathLike | typing.IO,
    configuration: dict | None = None,
    calculator: str | None = None,
    physical_units: CalculatorPhysicalUnits | None = None,
) -> Phono4pyYamlData:
    """Read a phono4py.yaml-like file."""
    yaml_data = load_yaml(filename)
    if isinstance(yaml_data, str):
        raise TypeError(f'Could not load "{filename}" properly.')
    return load_phono4py_yaml(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )
