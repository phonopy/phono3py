(migration_v4)=

# Migrating from phono3py v3 to v4

Phono3py v4 introduces several behaviour changes that affect existing
command lines, scripts, and saved workflows. This page lists the changes
in roughly decreasing order of user impact and shows how to update
existing usage.

Phono3py builds on phonopy, so the phonopy v4 changes also apply. See the
[phonopy v4 migration guide](https://phonopy.github.io/phonopy/migration-v4.html)
for details on options shared with phonopy (`--pa`, `--mesh`, etc.).

```{contents}
:depth: 2
:local:
```

## Command line split: `phono3py` and `phono3py-init`

The `phono3py` command has been split into two commands:

- `phono3py-init` -- setup operations: generate supercells with
  displacements, create `FORCES_FC3` / `FORCES_FC2` / `FORCE_SETS` files
  from external calculator results.
- `phono3py` -- phonon and thermal-conductivity calculation from a
  `phono3py.yaml`-like file.

The deprecated `phono3py-load` command is kept as an alias of `phono3py`.

Setup-related flags (`-d`/`--dim`, `--rd`/`--dim`, `--rd-fc2`, `--cf3`,
`--cf3-file`, `--cf2`, `--cfs`, `--fs2f2`, `--cfz`, `--cfz-fc2`, `-c`,
`--dim-fc2`, `--amplitude`, `--pm`, `--pm-fc2`, `--nodiag`,
`--cutoff-pair`) moved from `phono3py` to `phono3py-init`. Running them
on `phono3py` reports a migration error pointing here.

`-d` and `--rd` themselves are still accepted by `phono3py` because
`--rd` is also used by the pypolymlp workflow.  v3-style setup
invocations are caught via the companion `--dim` flag.

### Update existing shell scripts

**v3:**

```bash
phono3py -d --dim 2 2 2 -c POSCAR-unitcell
phono3py --cf3 vasprun.xml-{00001..00111}
phono3py --cf2 vasprun.xml-{00001..00002}
phono3py --cfs
phono3py --fs2f2
```

**v4:**

```bash
phono3py-init -d --dim 2 2 2 -c POSCAR-unitcell
phono3py-init --cf3 vasprun.xml-{00001..00111}
phono3py-init --cf2 vasprun.xml-{00001..00002}
phono3py-init --cfs
phono3py-init --fs2f2
```

For thermal-conductivity calculation (`--br`, `--lbte`, `-t`, etc.)
nothing changes:

```bash
phono3py --br --mesh 11 11 11
```

## `primitive_matrix` default changed to `"auto"`

In v3 the default for `primitive_matrix` was the 3x3 identity matrix,
i.e. no transformation was applied and the input unit cell was used
as the primitive cell as-is. In v4 the default is `"auto"`: phono3py
detects the primitive cell from crystal symmetry via spglib and
transforms the input cell accordingly.

When the auto-detected matrix is not the identity, the q-point
convention and band layout differ from v3 even though the input file
and command line are unchanged. Phono3py emits a runtime warning
(`PrimitiveMatrixAutoDefaultWarning`) in that situation, showing the
resolved matrix and pointing to this page.

### Update existing command lines

The `--pa` option (alias of `PRIMITIVE_AXES`) now defaults to `auto`
instead of the identity matrix. Existing command lines that relied on
the v3 default and did not pass `--pa` will silently switch to
auto-detection.

**v3** (no flag; input cell used as the primitive cell):

```bash
phono3py --br --mesh 11 11 11
```

**v4** (to keep v3 behaviour explicitly):

```bash
phono3py --pa P --br --mesh 11 11 11
```

### Update existing API calls

**v3:**

```python
ph3 = Phono3py(unitcell, supercell_matrix)  # input cell used as the primitive cell
```

**v4 (recommended new default, automatic primitive detection):**

```python
ph3 = Phono3py(unitcell, supercell_matrix)
# or, equivalently
ph3 = Phono3py(unitcell, supercell_matrix, primitive_matrix="auto")
```

**v4 (to keep v3 behaviour explicitly):**

```python
ph3 = Phono3py(unitcell, supercell_matrix, primitive_matrix="P")
```

### When the default does not change behaviour

If the input unit cell is already the primitive cell, auto-detection
returns the identity matrix (i.e. the input cell is used unchanged)
and no warning is emitted.

If the calculation loads a `phono3py.yaml` that already records a
`primitive_matrix`, that stored value takes priority over the new
default. Workflows driven by saved YAML files therefore reproduce v3
results exactly.

## Compact force constants are the default; `--cfc` / `--compact-fc` removed

The compact storage format for force constants (where the first atom
index runs over the primitive cell rather than the full supercell) is
now the default. In v3 the full-supercell array was the default and
`--cfc` / `--compact-fc` opted into the compact format.

`--cfc` / `--compact-fc` was removed in v4. Pass `--full-fc` on the
command line, or `is_compact_fc=False` to the API, to recover the v3
full-array layout.

**v3** (full-array default; `--cfc` opted in to compact):

```bash
phono3py --br --mesh 11 11 11        # full fc3
phono3py --cfc --br --mesh 11 11 11  # compact fc3
```

**v4:**

```bash
phono3py --br --mesh 11 11 11           # compact fc3 (new default)
phono3py --full-fc --br --mesh 11 11 11 # full fc3
```

## `--nac` removed

`--nac` was removed because non-analytical term correction is now
enabled automatically whenever the necessary data is available: a
`BORN` file in the working directory or `nac_params` stored in a
`phono3py.yaml`-like file. Pass `--nonac` to disable NAC explicitly.

**v3:**

```bash
phono3py --nac --br --mesh 11 11 11
```

**v4:**

```bash
phono3py --br --mesh 11 11 11   # NAC auto-detected from BORN / phono3py.yaml
phono3py --nonac --br --mesh 11 11 11   # explicit opt-out
```

## `phono3py.load` symmetrizes the traditional fc-solver via symfc-projector

When `phono3py.load(...)` is called without an explicit `fc_calculator`,
the traditional finite-difference fc-solver is now post-processed by
the symfc projector to enforce symmetry on the resulting force
constants. This matches the existing default of the `phono3py` CLI.

To recover the v3 behaviour (no projector applied to the traditional
solver output), pass `fc_calculator="traditional"` explicitly:

```python
ph3 = phono3py.load("phono3py.yaml", fc_calculator="traditional")
```

## Renamed / relocated modules

The grid, tetrahedron-method, and kaccum modules moved from phono3py
to phonopy. The new import paths are:

| v3 | v4 |
|----|----|
| `from phono3py.phonon.grid import BZGrid` | `from phonopy.phonon.grid import BZGrid` |
| `from phono3py.other.tetrahedron_method import ...` | `from phonopy.phonon.tetrahedron_method import ...` |
| `from phono3py.other.kaccum import ...` | replaced by the `phonopy.phonon.spectrum` accumulator API (`TetrahedronDOSAccumulator`, `SmearingDOSAccumulator`) |

There are no deprecation shims; the old names are gone in v4. The
`kaccum` CLI itself is unchanged for end users -- only the underlying
Python API moved.

## Rust backend (`phonors`) is now the default

The Rust kernels in [phonors](https://github.com/phonopy/phonors) are
the default backend in v4. `phonors` is a required runtime dependency
and is installed automatically with phono3py. The C extension is still
built and kept as a legacy backend that users can opt back into.

What this means for existing code:

- `Phono3py(...)`, `phono3py.load(...)`, and the CLI all run on the Rust
  backend out of the box. Numerical results are bit-identical to the
  C path on every kernel that has parity tests.
- Performance is generally similar to or better than the C extension
  thanks to rayon-based parallelism. The startup banner now prints
  `Rust backend (phonors) using rayon (N threads).`
- The C extension is still built by default and remains selectable
  per call via `lang="C"` / `--legacy-backend`.

### `--rust` is deprecated (no-op)

`--rust` used to enable the experimental Rust backend. In v4 it is a
deprecated no-op: the Rust backend is already active. The flag still
parses (so v3 command lines do not error) but emits a
`DeprecationWarning` and will be removed in a future release.

### `--legacy-backend` opts back into the C extension

To keep using the C kernels (for example to compare against v3
numerical output, or to work around a hypothetical phonors regression),
pass `--legacy-backend` on the CLI or `lang="C"` in the Python API.
The conf-file equivalent is `LEGACY_BACKEND = .true.`.

**v3 (or v4 with explicit opt-in):**

```bash
phono3py --legacy-backend --br --mesh 11 11 11
```

```python
ph3 = Phono3py(unitcell, supercell_matrix, lang="C")
ph3 = phono3py.load("phono3py.yaml", lang="C")
```

### Optional C-extension builds

Setting `PHONO3PY_NO_C_EXT=1` at build time still skips the C extension
entirely; in that case `lang="C"` silently falls back to Rust with a
one-time notice. See {ref}`rust_backend`.
