(phono3py_command)=

# phono3py command

The `phono3py` command runs the phonon and thermal-conductivity calculation
step of the workflow.  It reads a `phono3py.yaml`-like file, constructs the
phonon and ph-ph interaction, and computes the requested property
(thermal conductivity, imaginary self energy, joint DOS, ...).

To prepare displacements and to collect calculator results into
`FORCES_FC3` / `FORCES_FC2` / `FORCE_SETS` files use
{ref}`phono3py_init_command`.

## Example

In the NaCl example for the VASP calculator,

```bash
% phono3py-init -d --dim 2 2 2 --dim-fc2 2 2 2 -c POSCAR-unitcell
% phono3py-init --cf3 disp-{00001..00146}/vasprun.xml
% phono3py-init --cf2 disp_fc2-{00001..00002}/vasprun.xml
```

After this, run the thermal-conductivity calculation as the post-process:

```bash
% phono3py --br --ts 300 --mesh 19 19 19
```

`phono3py` reads `phono3py_disp.yaml` (or `phono3py.yaml`) automatically when
the file is present in the current directory.  It can also be given as the
first argument; the file can be compressed (`xz`, `lzma`, `gz`, `bz2`).

```bash
% phono3py phono3py_params.yaml.xz --br --ts 300 --mesh 19 19 19
```

(phono3py_command_behaviour)=
## Behaviour

- `phono3py_xxx.yaml`-like file is always required, provided in either of two
  ways:
  1. as the first argument of the command, or
  2. found in the current directory as `phono3py_disp.yaml` or
     `phono3py.yaml` (`phono3py_disp.yaml` takes precedence).

- The `-c` option (read crystal structure separately) does not exist — the
  crystal structure is read from the yaml file.

- A phono3py configuration file ({ref}`use_config_with_option`) can be read
  through `--config` option.

- If parameters for non-analytical term correction (NAC) are found, NAC is
  automatically enabled.  Use `--nonac` to disable.

- When force constants are calculated from displacements and forces dataset,
  force constants are automatically symmetrized.  Use `--no-sym-fc` to opt
  out.

- `-o` takes one argument: the output yaml filename that replaces the
  default `phono3py.yaml`.

## Relation to `phono3py-load`

`phono3py-load` is the historical name of this command and is kept as a
deprecated alias.  It emits a warning and otherwise behaves identically to
`phono3py`.  Use `phono3py` in new scripts.
