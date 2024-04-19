"""Parse displacement dataset."""

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

import numpy as np


def get_displacements_and_forces_fc3(disp_dataset):
    """Return displacements and forces from disp_dataset.

    Note
    ----
    Dipslacements and forces of all atoms in supercells are returned.

    Parameters
    ----------
    disp_dataset : dict
        Displacement dataset.

    Returns
    -------
    displacements : ndarray
        Displacements of all atoms in all supercells.
        shape=(snapshots, supercell atoms, 3), dtype='double', order='C'
    forces : ndarray or None
        Forces of all atoms in all supercells.
        shape=(snapshots, supercell atoms, 3), dtype='double', order='C'
        None is returned when forces don't exist.

    """
    if "first_atoms" in disp_dataset:
        natom = disp_dataset["natom"]
        ndisp = len(disp_dataset["first_atoms"])
        for disp1 in disp_dataset["first_atoms"]:
            ndisp += len(disp1["second_atoms"])
        displacements = np.zeros((ndisp, natom, 3), dtype="double", order="C")
        forces = np.zeros_like(displacements)
        indices = []
        count = 0
        for disp1 in disp_dataset["first_atoms"]:
            indices.append(count)
            displacements[count, disp1["number"]] = disp1["displacement"]
            forces[count] = disp1["forces"]
            count += 1

        for disp1 in disp_dataset["first_atoms"]:
            for disp2 in disp1["second_atoms"]:
                if "included" in disp2:
                    if disp2["included"]:
                        indices.append(count)
                else:
                    indices.append(count)
                displacements[count, disp1["number"]] = disp1["displacement"]
                displacements[count, disp2["number"]] = disp2["displacement"]
                forces[count] = disp2["forces"]
                count += 1
        return (
            np.array(displacements[indices], dtype="double", order="C"),
            np.array(forces[indices], dtype="double", order="C"),
        )
    elif "forces" in disp_dataset and "displacements" in disp_dataset:
        return disp_dataset["displacements"], disp_dataset["forces"]
    else:
        raise RuntimeError("disp_dataset doesn't contain correct information.")
