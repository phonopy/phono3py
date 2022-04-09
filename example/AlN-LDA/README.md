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

```
% phono3py --dim 3 3 2 -c POSCAR-unitcell -d
```

In the example directory, `FORCES_FC3` is compressed to `FORCES_FC3.lzma`. After
unzipping `FORCES_FC3.lzma` (e.g., using `tar xvfz` or `tar xvfa`), to obtain
`fc3.hdf5` and normal `fc2.hdf5`,

```
% phono3py --sym-fc
```

Using 13x13x9 sampling mesh, lattice thermal conductivity is calculated by

```
% phono3py --mesh 13 13 9 --fc3 --fc2 --br
```

`kappa-m13139.hdf5` is written as the result. The lattice thermal conductivity
is calculated as k_xx=228.2 and k_zz=224.1 W/m-K at 300 K.

With `--nac` option, non-analytical term correction is applied reading the Born
effective charges and dielectric constant from `BORN` file:

```
% phono3py --mesh 13 13 9 --fc3 --fc2 --br --nac
```

This changes thermal conductivity at 300 K to k_xx=235.7 and k_zz=219.1. The
shape of phonon band structure is important to fullfil energy and momentum
conservations.

Use of larger supercell of fc2 may change the shape of phonon band structure. To
see it, first regenerate `phono3py_disp.yaml` with `--dim-fc2` option,

```
% phono3py --dim 3 3 2 --dim-fc2 5 5 3 -c POSCAR-unitcell -d
```

Then re-create force constants and calculate thermal conductivity,

```
% phono3py --sym-fc
% phono3py --mesh="13 13 9" --fc3 --fc2 --br --nac
```

k_xx=236.0 and k_zz=222.2 are obtained. In the case of this example, we can see
that the larger fc2 supercell contributes little, which means that the 3x3x2
supercell was good enough to obtain a good shape of phonon band structure.
