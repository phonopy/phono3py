This is the example of NaCl calculation. Since all atoms are displaced, to
obtain force constants, an external force constants calculator is necessary,
i.e., build-in force constants calculator has no ability to compute force
constants for such dataset. In this example, ALM is used. See
https://phonopy.github.io/phonopy/setting-tags.html#alm. The easiest way to
install ALM is to use conda.

The supercell is 2x2x2 of the conventional unit cell. The VASP calculation was
made for force calculations with 500 eV, 2x2x2 off-Gamma-centre k-point sampling
mesh for the supercell, and PBE-sol. For dielectric constant and Born effective
charges were calculated in a similar calculation settings with `LEPSILON =
.TRUE.` and 4x4x4 off-Gamma-centre k-point sampling mesh.

200 supercells were generated with random displacements of 0.03 A displacement
distance. The calculated forces, displacements, dielectric constant, and Born
effective charges are all stored in `phono3py_params_NaCl222.yaml.xz`.

Force constants are calculated by

```
% phono3py-load phono3py_params_NaCl222.yaml.xz -v --alm
```

This calculation may require a few minutes depending on computer hardware and
some amount of memory space. By this `fc2.hdf5` and `fc3.hdf5` are obtained. The
lattice thermal conductivity at 300 K is calculated by

```
% phono3py-load phono3py_params_NaCl222.yaml.xz --mesh 50 --ts 300 --br
```

The result is ~7.2 W/m-K at 300 K.
