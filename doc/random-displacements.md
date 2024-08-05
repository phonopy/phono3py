(random-displacements)=
# Randan displacements

Random displacements and corresponding forces in supercells can be employed as a
displacement-force dataset for computing force constants. This requires an
external force constants calculator, e.g., symfc or ALM. Here, examples are
presented with using symfc that can be installed via pip or conda easily.

## Related setting tags

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
