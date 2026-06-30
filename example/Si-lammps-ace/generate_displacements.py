"""A script to generate fc3 supercells with displacements for LAMMPS.

Generating the supercells from a yaml unit cell keeps the cell in its original
(unrotated) Cartesian orientation in ``phono3py_disp.yaml``, in contrast to
reading a LAMMPS structure file, which is given in the rotated triclinic
convention. The generated ``supercell-xxxxx`` files follow the LAMMPS structure
input format; the forces obtained from LAMMPS are rotated back automatically when
FORCES_FC3 is created.

"""

from phonopy.interface.calculator import write_supercells_with_displacements
from phonopy.interface.phonopy_yaml import read_cell_yaml

import phono3py

cell = read_cell_yaml("phono3py_unitcell.yaml")
ph3 = phono3py.load(
    unitcell=cell,
    supercell_matrix=[2, 2, 2],  # use a larger size (e.g. [3, 3, 3]) for convergence
    calculator="lammps",
    produce_fc=False,
    log_level=1,
)
ph3.generate_displacements()
ph3.save("phono3py_disp.yaml")

ids = []
disp_cells = []
for i, scell in enumerate(ph3.supercells_with_displacements):
    if scell is not None:
        ids.append(i + 1)
        disp_cells.append(scell)
write_supercells_with_displacements(
    "lammps",
    ph3.supercell,
    disp_cells,
    displacement_ids=ids,
    zfill_width=5,
)
