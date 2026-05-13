(phono3py_init_command)=

# phono3py-init command

The `phono3py-init` command handles the setup steps that happen before
phonon and thermal-conductivity calculation:

- generate supercells with displacements (`-d`, `--rd`, `--rd-fc2`),
- collect calculator results into `FORCES_FC3` (`--cf3`, `--cf3-file`) or
  `FORCES_FC2` (`--cf2`),
- subtract residual forces (`--cfz`, `--cfz-fc2`),
- convert between `FORCE_SETS` and `FORCES_FC2` (`--cfs`, `--fs2f2`).

After this step, run the phonon and thermal-conductivity calculation with
{ref}`phono3py_command`.

## Examples

VASP, finite displacements:

```bash
% phono3py-init -d --dim 2 2 2 --dim-fc2 2 2 2 -c POSCAR-unitcell
% phono3py-init --cf3 disp-{00001..00146}/vasprun.xml
% phono3py-init --cf2 disp_fc2-{00001..00002}/vasprun.xml
```

VASP, random displacements:

```bash
% phono3py-init --rd 50 --rd-fc2 4 --dim 2 2 2 --dim-fc2 2 2 2 -c POSCAR-unitcell
% phono3py-init --cf3 disp-{00001..00050}/vasprun.xml
% phono3py-init --cf2 disp_fc2-{00001..00004}/vasprun.xml
```

Convert `FORCE_SETS` (phonopy format) to `FORCES_FC2`:

```bash
% phono3py-init --fs2f2
```

Convert `FORCES_FC2` to `FORCE_SETS` (phonopy format):

```bash
% phono3py-init --cfs
```

## Relation to the legacy `phono3py` command

Earlier versions of phono3py used a single `phono3py` command to cover both
the setup and the phonon/thermal-conductivity steps.  The two responsibilities
have been split:

| Old invocation                          | New invocation                                |
| --------------------------------------- | --------------------------------------------- |
| `phono3py -d --dim ... -c POSCAR`       | `phono3py-init -d --dim ... -c POSCAR`        |
| `phono3py --cf3 vasprun.xml ...`        | `phono3py-init --cf3 vasprun.xml ...`         |
| `phono3py --cf2 vasprun.xml ...`        | `phono3py-init --cf2 vasprun.xml ...`         |
| `phono3py --cfs`                        | `phono3py-init --cfs`                         |
| `phono3py-load --br --ts ... --mesh ...` | `phono3py --br --ts ... --mesh ...`           |

The new `phono3py` command rejects setup flags (`-d`, `--rd`, `--rd-fc2`,
`--cf3`, `--cf3-file`, `--cf2`, `--cfs`, `--fs2f2`) and points the user to
`phono3py-init`.
