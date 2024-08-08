This is the example of the wurtzite-type AlN phono3py calculation. The VASP code
was used with 500 eV and LDA as XC functional. The experimental lattice
parameters were used and the internal positions of atoms were relaxed by
calculation. The 3x3x2 and 5x5x3 supercells were chosen for fc3 and fc2. The
6x6x4, 2x2x2, 1x1x2 k-point sampling meshes with Gamma-centre in the basal plane
and off-Gamma-centre along c-axis were employed for the unit cell, fc3
supercell, and fc2 supercell, respectively. For the DFPT calculation of Born
effective charges and dielectric constant, the 12x12x8 k-point sampling mesh
with the similar shift was used.

Then the forces were calculated with the above settings. `FORCES_FC3` and
`FORCES_FC2` were created with subtracting residual forces of perfect supercell
from all displaced supercell forces.

Perfect and displaced supercells were created by

```bash
% phono3py --dim 3 3 2 -c POSCAR-unitcell -d
```

In the example directory, `FORCES_FC3` is compressed to `FORCES_FC3.xz`. After
unzipping `FORCES_FC3.xz` (e.g., using `xz -d`), to obtain `fc3.hdf5` and
`fc2.hdf5` using symfc (the results without using symfc, i.e., finite difference
method, are shown at the bottom of this README)

```bash
% phono3py-load --symfc -v
```

Lattice thermal conductivity is calculated by

```bash
% phono3py-load --mesh 40 --br --ts 300
```

`kappa-m15158.hdf5` is written as the result. Parameters for non-analytical term
correction (NAC) is automatically read from those stored in `phono3py_disp.yaml` or
`BORN` file. The lattice thermal conductivity is calculated as k_xx=242.8 and
k_zz=226.5 W/m-K at 300 K. Without NAC, k_xx=233.6 and k_zz=222.2.

Use of larger supercell for fc2 may change the shape of phonon band structure.
To see it, first regenerate `phono3py_disp.yaml` with `--dim-fc2` option,

```bash
% phono3py --dim 3 3 2 --dim-fc2 5 5 3 -c POSCAR-unitcell -d
```

Then re-create force constants and calculate thermal conductivity,

```bash
% phono3py-load --symfc -v
% phono3py-load --br --mesh=40 --ts 300
```

If `phono3py_disp.yaml` is renamed to `phono3py_disp_dimfc2.yaml`, it can be
specified at the first argument of `phono3py-load` command:

```bash
% phono3py-load phono3py_disp_dimfc2.yaml --symfc -v
% phono3py-load phono3py_disp_dimfc2.yaml --br --mesh=40 --ts 300
```

k_xx=240.2 and k_zz=230.1 are obtained. In the case of this example, we can see
that the larger fc2 supercell contributes little, which means that the 3x3x2
supercell was good enough to obtain a good shape of phonon band structure.

Using the finite difference method implemented in phono3py, lattice thermal
conductivities are obtained as k_xx=251.2 and k_zz=233,4 without using the large
fc2 supercell and k_xx=249.4 k_zz=236.9 using the large fc2 supercell.
