(random-displacements)=
# Force constants calculation with randan displacements of atoms

Random displacements and corresponding forces in supercells can be employed as a
displacement-force dataset for computing force constants. This requires an
external force constants calculator, e.g., symfc or ALM. Here, examples are
presented with using symfc that can be installed via pip or conda easily.

## Related command options

- {ref}`random_displacements_option` (`--rd`, `--random-seed`)
- {ref}`fc_calculator_option` (`--fc-calc`)
- {ref}`fc_calculator_options_option` (`--fc-calc-opt`)

## Generation of random directional displacements

The option `--rd NUM` is used instead of `-d` in generating displacements as follows:

```bash
% phono3py --rd 100 --dim 2 2 2 --pa auto -c POSCAR-unitcell
```

`NUM` means the number of supercells with random directional displacements. This
must be specified, and the initial guess may be from around the number of
supecells generated for the systematic displacements by `-d`. In the case of the
`NaCl-rd` example, 146 supercells are generated with `-d`, so similar
number `--rd 100` was chosen here.

If random directional displacements for fc2 are expected, `--rd-fc2` and
`--dim-fc2` have to be specified:

```bash
% phono3py --rd 100 --dim 2 2 2 --rd-fc2 2 --dim-fc2 4 4 4 --pa auto -c POSCAR-unitcell
```

where `--dim` is necessary but `--rd` is not.

## Create `FORCES_FC3` and `FORCES_FC2`

`FORCES_FC3` and optionally `FORCES_FC2`, which contains forces corresponding to
displacements, can be created by

```bash
% phono3py --cf3 vasprun_xmls/vasprun-00{001..100}.xml
```

Here it is assumed that the forces were calculated by VASP, and the output files
(`vasprun.xml`) are stored in `vasprun_xmls` directory after renaming. When
running this command, `phono3py_disp.yaml` is automatically read. For the
different file name, e.g. `phono3py_disp_rd.yaml`, it is specified with `-c`
option:

```bash
% phono3py -c phono3py_disp_rd.yaml --cf3 vasprun_xmls/vasprun-00{001..100}.xml
```

`FORCES_FC2` is created similarly, e.g., from the VASP output stored as
`vasprun_xmls/vasprun-ph000{1,2}.xml`,

```bash
% phono3py --cf2 vasprun_xmls/vasprun-ph000{1,2}.xml
```

## Create `phono3py_params.yaml`

Instead of creating `FORCES_FC3` and `FORCES_FC2`, more convenient data file to
store displacement-force dataset is created by `--sp` option:

```bash
% phono3py --cf3 vasprun_xmls/vasprun-00{001..100}.xml --cf2 vasprun_xmls/vasprun-ph0000{1,2}.xml --sp
```

The advantage to employ `phono3py_params.yaml` is that this file can contain all
the information required to run phono3py such as crystal structure, supercell
information, displacements, forces, and parameters for non-analytical term
correction. This file is immediately usable for `phono3py-load` command ({ref}`phono3py_load_command`).

## Calculation of force constants

If `phono3py_disp.yaml` is located in current directory, force constants are
calculated from `FORCES_FC3` (and optionally `FORCES_FC2`) and
`phono3py_disp.yaml` by

```bash
% phono3py-load --symfc -v
```

or

```bash
% phono3py-load phono3py_params.yaml --symfc -v
```

Similarly, it is performed by also using `phono3py` command,

```bash
% phono3py --symfc -v
```

or with `phono3py_params.yaml`

```bash
% phono3py -c phono3py_params.yaml --symfc -v
```


## Cutoff pair-distance for fc3 calculation

The number of supercells required for calculating fc3 depends on crystal
structure, crystal symmetry, and supercell size. For larger supercells, the
number can be very large. In addition, the required computational time and
memory space can also become large. In such case, it may be good to consider
introducing cutoff distance for pairs of atoms. It is performed by
`--fc-calc-opt` option as

```bash
% phono3py-load --symfc -v --fc-calc-opt "cutoff=8"
```

The shortcut of `--fc-calc-opt "cutoff=8"` is `--cutoff-pair 8`.

The convergence of fc3 has to be checked. With the same input of
displacement-force dataset, calculated force constants gradually converge by
increasing cutoff pair-distance. The convergence may be checked by lattice
thermal conductivity, but it may depend on quantity that is expected to be
calculated.
