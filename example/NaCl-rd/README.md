# Example of using random directional displacements

## How to use symfc

This example utilizes an NaCl calculation result from A. Togo and A. Seko, J.
Chem. Phys. 160, 211001 (2024). Supercells of 2x2x2 and 4x4x4 conventional unit
cells are chosen for the third-order force constants (fc3) and second-order
force constants (fc2), respectively. Displacement-force datasets consisting of
100 supercells for fc3 and 2 supercells for fc2 are extracted and stored in
`phono3py_params_NaCl.yaml.xz`. Random directional displacements of a constant
0.03 Angstrom are used.

To calculate force constants, an external force constants calculator is
necessary. Here, the symfc tool (available at https://github.com/symfc/symfc) is
used, which can be easily installed via pip or conda.

The `fc3.hdf5` and `fc2.hdf5` are computed using the command:

```
% phono3py-load phono3py_params_NaCl.yaml.xz --symfc -v
```

Lattice thermal conductivity (LTC) is calculated with the following command:

```
% phono3py-load phono3py_params_NaCl.yaml.xz --br --ts 300 --mesh 50
```

By this, LTC is obtained around 8.3 W/m-k.


## How to use pypolymlp

When supercell energies are included in `phono3py_params.yaml` like file, the
polynomial machine learning potential (poly-MLP) by pypolymlp can be used to
calculate fc3 by the following command:

```bash
% phono3py-load phono3py_params_NaCl.yaml.xz --pypolymlp --symfc --rd 400 -v
```

the procedure below is performed:

1. Poly-MLPs are computed from the displacement-force dataset for fc3. This is
   activated by `--pypolymlp` option.
2. 800=400+400 supercells for random directional displacements are generated,
   where 400+400 means 400 supercells with random displacements (u) and 400
   supercells with opposite displacement vectors (-u). This is activated by
   `--rd 400` option. The default displacement distance is 0.001 Angstrom in
   `--pypolymlp` mode. Since random displacements are generated `--symfc` has to
   be specified for fc3. In this example, random displacements are used for fc2,
   too, `--symfc` is applied to both of fc3 and fc2. Without `--rd` option,
   systematic displacements are generated, for which the option `--fc-calc "symfc|"`
   has to be specified instead of `--symfc` (equivalent to `--fc-calc "symfc|symfc")`).
3. Forces on atoms in these 800 supercells are calculated using poly-MLP.
4. Force constants are calculated.


The `fc3.hdf5` and `fc2.hdf5` are obtained. Using these force constants, LTC is
calculated by

```bash
% phono3py-load phono3py_params_NaCl.yaml.xz --br --ts 300 --mesh 50
```

and the LTC value of around 8.2 W/m-k is obtained.

## Generating phono3py_params.yaml from vasprun.xml's

`phono3py_params.yaml` is generated from

```bash
% phono3py phono3py_disp.yaml --cf3 NaCl-vasprun/vasprun-{00001..00100}.xml --cf2 NaCl-vasprun/vasprun-ph0000{1,2}.xml --sp
```

This command reads electronic energies of supercells from `vasprun.xml`s and
writes them into `phono3py_params.yaml`, too. Here, `phono3py_disp.yaml` is not
included in this example, but `phono3py_params_NaCl.yaml.xz` can be used to run
this example since corresponding information of displacements is included in
this file, too.
