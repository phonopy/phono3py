"""File I/O for 4th-order force constants (fc4).

Experimental; kept self-contained in ``phono3py.phonon4``. Mirrors the fc3
hdf5 I/O in ``phono3py.file_IO`` (``write_fc3_to_hdf5`` / ``read_fc3_from_hdf5``)
one rank higher, supporting both the full and compact layouts (the latter via
an optional ``p2s_map``).
"""

from __future__ import annotations

import os

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from phono3py._version import __version__
from phonopy.file_IO import check_force_constants_indices, get_io_module_to_decompress

from phono3py.phonon4.dataset import (
    count_supercells_fc4,
    get_displacements_and_forces_fc4,
    set_forces_in_dataset_fc4,
)
from phono3py.phonon4.displacement_fc4 import Fc4Type1DisplacementDataset


def write_fc4_to_hdf5(
    fc4: NDArray[np.double],
    filename: str | os.PathLike = "fc4.hdf5",
    p2s_map: NDArray[np.int64] | None = None,
    compression: str | int | None = "gzip",
) -> None:
    """Write fc4 to an hdf5 file.

    Parameters
    ----------
    fc4 : ndarray
        Force constants, shape ``(n_satom, n_satom, n_satom, n_satom, 3, 3, 3,
        3)`` (full) or ``(n_patom, n_satom, n_satom, n_satom, 3, 3, 3, 3)``
        (compact), dtype=double.
    filename : str, optional
        Filename. Default is ``"fc4.hdf5"``.
    p2s_map : ndarray, optional
        Primitive-atom indices in the supercell-index system, shape
        ``(n_patom,)``, dtype=int64. Stored alongside compact fc4 so it can be
        checked on read. Default is None.
    compression : str or int, optional
        h5py lossless compression filter (e.g. ``"gzip"``, ``"lzf"``) or None
        for no compression. Default is ``"gzip"``.

    """
    with h5py.File(filename, "w") as w:
        w.create_dataset("version", data=np.bytes_(__version__))
        w.create_dataset("fc4", data=fc4, compression=compression)
        if p2s_map is not None:
            w.create_dataset("p2s_map", data=p2s_map)


def read_fc4_from_hdf5(
    filename: str | os.PathLike = "fc4.hdf5",
    p2s_map: NDArray[np.int64] | None = None,
) -> NDArray[np.double]:
    """Read fc4 from an hdf5 file.

    The full and compact layouts are distinguished by the array shape. When the
    file stores ``p2s_map`` (compact fc4) and ``p2s_map`` is given, their
    consistency is checked.
    """
    with h5py.File(filename, "r") as f:
        if "fc4" not in f:
            raise KeyError(
                f"{filename} does not have 'fc4' dataset. "
                "This file is not a valid fc4.hdf5."
            )
        fc4: NDArray[np.double] = f["fc4"][:]
        if not (fc4.dtype == np.dtype("double") and fc4.flags.c_contiguous):
            raise TypeError(
                f"{filename} has to be read by h5py as numpy ndarray of "
                "dtype='double' and c_contiguous."
            )
        if "p2s_map" in f:
            p2s_map_in_file: NDArray[np.int64] = f["p2s_map"][:]
            check_force_constants_indices(
                fc4.shape[:2], p2s_map_in_file, p2s_map, filename
            )
        return fc4


def write_FORCES_FC4(
    disp_dataset: Fc4Type1DisplacementDataset,
    forces_fc4: NDArray[np.double] | None = None,
    filename: str | os.PathLike = "FORCES_FC4",
) -> None:
    """Write supercell forces of the fc4 dataset to a text file.

    Supercells are written in ``id`` order (level-grouped: all first atoms, then
    second, then third), matching
    :func:`phono3py.phonon4.dataset.get_supercells_with_displacements_fc4`. Each
    block has a ``# File: <id>`` header and comment lines for the displaced
    atoms, followed by ``natom`` force lines. Comment lines are ignored on read.

    Parameters
    ----------
    disp_dataset : Fc4Type1DisplacementDataset
        fc4 displacement dataset (used for the supercell order and, when
        ``forces_fc4`` is None, for the stored forces).
    forces_fc4 : ndarray, optional
        Forces in id order, shape ``(n_supercells, natom, 3)``. When None, the
        forces stored in ``disp_dataset`` are used. Default is None.
    filename : str, optional
        Output filename. Default is ``"FORCES_FC4"``.

    """
    displacements, dataset_forces = get_displacements_and_forces_fc4(disp_dataset)
    forces = dataset_forces if forces_fc4 is None else np.asarray(forces_fc4)
    if forces is None:
        raise RuntimeError("No forces found in the fc4 dataset or arguments.")

    with open(filename, "w") as w:
        for i, (disp, force_set) in enumerate(zip(displacements, forces, strict=True)):
            w.write("# File: %d\n" % (i + 1))
            for atom in np.where(np.linalg.norm(disp, axis=1) > 1e-10)[0]:
                w.write("# %-5d %20.16f %20.16f %20.16f\n" % (atom + 1, *disp[atom]))
            for force in force_set:
                w.write("%15.10f %15.10f %15.10f\n" % tuple(force))


def parse_FORCES_FC4(
    disp_dataset: Fc4Type1DisplacementDataset,
    filename: str | os.PathLike = "FORCES_FC4",
    unit_conversion_factor: float | None = None,
) -> None:
    """Parse FORCES_FC4 and store the forces in ``disp_dataset`` (in place).

    The file is read in ``id`` order (matching :func:`write_FORCES_FC4`);
    comment lines starting with ``#`` are ignored.
    """
    natom = disp_dataset["natom"]
    ncells = count_supercells_fc4(disp_dataset)
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rt") as f:
        forces = np.loadtxt(f, dtype="double").reshape((ncells, natom, 3))
    if not forces.flags["C_CONTIGUOUS"]:
        forces = np.array(forces, dtype="double", order="C")
    if unit_conversion_factor is not None:
        forces *= unit_conversion_factor
    set_forces_in_dataset_fc4(disp_dataset, forces)
