# AlN lattice thermal conductivity calculation from dataset for pypolymlp

## Computational setting of VASP calculations

For supercell forces and energies

- Supercell 4x4x2 of wurtzite unit cell
- Random directional displacements of 0.03 Angstrom
- PBE-sol
- 520 eV cutoff energy
- Gamma centered 2x2x2 kpoint mesh
- LREAL = .FALSE.
- ADDGRID = .TRUE.

For parameters of non-analytical term correction,

- PBE-sol
- 520 eV cutoff energy
- Gamma centered 7x7x4 kpoint mesh
- LEPSION = .TRUE.
- LREAL = .FALSE.

These data are stored in `phonopy_params_mp-661.yaml.xz`.

## Example of lattice thermal conductivity calculation

MLPs by pypolymlp are developed by

```bash
% phono3py-load phonopy_params_mp-661.yaml.xz --pypolymlp -v
```

Dataset with 180 supercells is used for training and 20 for the test. This
calculation will take 5-10 minutes depending on computer resource.
`pypolymlp.yaml` is made by this command.

Force constants are calculated by

```bash
% phono3py-load phonopy_params_mp-661.yaml.xz --pypolymlp --relax-atomic-positions -d
```

With the `--relax-atomic-positions` option, internal atomic positions in unit
cell are optimized by pypolymlp. The displacement-force dataset is stored in
`phono3py_mlp_eval_dataset.yaml`. Force constants are symmetried using symfc,
but the phono3py's traditional symmetrizer can be used with the option
`--fc-calculator traditional`. The symmetry constraints applied by this
traditional symmetrizer is weaker, but the calculation demands less memory
space.

Lattice thermal conductivity is calculated by

```bash
% phono3py-load phonopy_params_mp-661.yaml.xz --mesh 40 --br
```

Steps written above are performed in one-shot by

```bash
% phono3py-load phonopy_params_mp-661.yaml.xz --pypolymlp --relax-atomic-positions -d --mesh 40 --br
```

The lattice thermal conductivity calculated at 300 K will be around k_xx=252 and k_zz=232.
