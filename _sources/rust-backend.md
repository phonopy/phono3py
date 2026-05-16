(rust_backend)=

# Rust backend

The computationally heavy parts of phono3py are implemented twice: as a
C extension module and as a Rust extension module. The Rust path lives in
the separate [phonors](https://github.com/phonopy/phonors) package
(built with [maturin](https://www.maturin.rs/) and
[PyO3](https://pyo3.rs/)) and is the **default backend** in phono3py v4.
The C extension is still built and kept as a legacy backend that users
can opt back into per call.

```{contents}
:depth: 2
:local:
```

## Installation

`phonors` is a required runtime dependency of phono3py and is installed
automatically by `pip install phono3py` and by the
[conda-forge phono3py package](https://anaconda.org/conda-forge/phono3py).
Pre-built wheels and conda packages are published for Linux, macOS, and
Windows on the supported Python versions, so no Rust toolchain is needed
for typical installations.

### Building `phonors` from source (optional)

To work against an unreleased `phonors` revision, clone the repository
alongside phono3py and build the extension in editable mode with
`maturin develop`:

```bash
% git clone https://github.com/phonopy/phonors.git
% cd phonors
% maturin develop --release
```

Requirements for a source build:

- A Rust toolchain (stable, edition 2021, `rustc >= 1.75`). The
  easiest way to install it is via [rustup](https://rustup.rs/).
- [maturin](https://www.maturin.rs/) 1.7 or newer (available on PyPI
  and conda-forge).
- Python 3.10 or newer. The extension is built against the stable ABI
  (`abi3-py310`), so one build works for all Python 3.10+ interpreters.

### Optional: native CPU tuning

By default, `maturin develop --release` builds with the Rust baseline
target (x86-64 v1 on x86_64, Armv8.0 on aarch64), so the resulting
module runs on any CPU of that architecture. For a local build that
will only run on the current machine, enabling the host CPU's full
instruction set can recover a few percent of wall-clock:

```bash
% RUSTFLAGS='-C target-cpu=native' maturin develop --release
```

## Usage

The Rust backend is the default; no flag or keyword is required. To
opt back into the legacy C extension, pass `--legacy-backend` on the
command line or `lang="C"` on the Python API. See
{ref}`migration_v4` for the full v3 -> v4 migration notes.

### Command line

```bash
% phono3py --br --mesh 11 11 11                  # default: Rust
% phono3py --legacy-backend --br --mesh 11 11 11 # opt back into C
```

When the Rust backend is active, the run header prints

```
Rust backend (phonors) using rayon (N threads).
```

where `N` follows rayon's defaults.

### Python API

The `Phono3py` constructor and the `phono3py.load` loader both accept
a `lang` keyword:

```python
import phono3py

ph3 = phono3py.load("phono3py.yaml")             # default: Rust
ph3 = phono3py.load("phono3py.yaml", lang="C")   # opt back into C
```

The current value is exposed as the read-only `Phono3py.lang`
property. Valid values are `"C"` and `"Rust"` (default `"Rust"`).

`lang` is threaded internally to every lang-aware consumer, including
`BZGrid` / `GridMatrix`, `Interaction`, the phonon solver
(dynamical-matrix construction; the `eigh` diagonalization stays in
Python/SciPy), `ImagSelfEnergy`, `RealSelfEnergy`, the RTA / LBTE
conductivity calculators, `Isotope`, and `JointDos`. No per-call
plumbing from the user side is required.

### Thread pool

The Rust kernels parallelize with [rayon](https://docs.rs/rayon/),
which uses its own thread pool. The thread count is controlled by
`RAYON_NUM_THREADS` (not `OMP_NUM_THREADS`, which only affects the C
path):

```bash
% RAYON_NUM_THREADS=8 phono3py ...
```

NumPy/SciPy BLAS multithreading used by phonon diagonalization is
independent and is controlled by the BLAS library's own variables
(e.g. `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`).

### Dispatch tracing

To verify which backend a given code path is actually running on, set
`PHONO3PY_TRACE_LANG=1` before launching:

```bash
% PHONO3PY_TRACE_LANG=1 phono3py ...
```

Each lang-aware call site emits one line to stderr, for example:

```
[phono3py.lang] dispatch name=Interaction.__init__ lang=Rust
```

The trace covers the following call sites:

- Construction events, fired once per instance: `BZGrid.__init__`,
  `GridMatrix.__init__`, `Interaction.__init__`,
  `ImagSelfEnergy.__init__`, `RealSelfEnergy.__init__`,
  `Isotope.__init__`, `JointDos.__init__`, and
  `conductivity_calculator[rta|lbte]`.
- `run_phonon_solver_rust`, fired on every invocation of the Rust
  phonon solver (the C and Python phonon solvers are not traced).

The trace is silent by default. It is attached to a dedicated logger
(`phono3py.lang`) and is independent of the `--loglevel` /
`log_level` setting, so enabling it does not affect the rest of the
run log.

### Batched grid-point path (opt-in)

The RTA low-memory scattering path can optionally process several grid points
per Rust call. This is disabled by default. It is useful on many-core machines
where individual per-grid-point calls do not fully saturate the thread pool.

The batch size can be set either through
`conductivity_calculator(..., rust_gp_batch_size=<int>)` or through
the `PHONO3PY_RUST_GP_BATCH_SIZE` environment variable. Resolution
order:

1. The `rust_gp_batch_size` argument if it is not `None`.
2. Otherwise the `PHONO3PY_RUST_GP_BATCH_SIZE` env var (default `0`).
3. A value `<= 0` falls back to the per-grid-point loop.

Example:

```bash
% PHONO3PY_RUST_GP_BATCH_SIZE=8 phono3py ...
```

The batched path applies only to the low-memory RTA solver and is
skipped automatically when `read_gamma` is set.

(rust_backend_no_c_ext)=

## Building phono3py without the C extension

For Rust-only deployments (or to validate that every dispatch site has
a Rust path), phono3py can be installed with the C extension skipped:

```bash
% PHONO3PY_NO_C_EXT=1 pip install -e . -vvv
```

When the env var is set, `CMakeLists.txt` returns early and
`phono3py._phono3py` / `phono3py._phononcalc` are not built. At runtime,
`Phono3py()` and `phono3py.load()` detect the missing C extension, emit
a one-time `[phono3py] C extension ... is not available; falling back
to lang='Rust' ...` message, and silently flip `lang="C"` requests to
`lang="Rust"`. Since `phonors` is a required dependency, it is always
present; an informative `ImportError` is raised only if it has been
manually uninstalled.

To restore the C extension, simply rebuild without the env var:

```bash
% pip install -e . -vvv
```

```{note}
This option is intended for testing the Rust path and for packagers
who want a Rust-only wheel. For day-to-day use the regular install
(with the C extension) remains the recommended path.
```

## Scope

Rust kernels are wired through the lang dispatch for the following
groups:

- Grid construction (`BZGrid`, `GridMatrix`: SNF, grid addresses and
  maps, reciprocal / transform rotations).
- Phonon-phonon interaction (`Interaction`).
- Imaginary / real self energy (`ImagSelfEnergy`, `RealSelfEnergy`),
  including the full-gamma variants reached through `is_full_pp`,
  `is_gamma_detail`, `read_pp`, `store_pp`, `use_ave_pp`.
- Lattice thermal conductivity: standard RTA (low-memory path) and
  LBTE (both irreducible and reducible) run end-to-end on Rust.
- Isotope scattering and joint DOS.

### Known limitations

- **Force constants.** A Rust implementation of the fc3 kernel exists
  in `phonors`, but `Phono3py.produce_fc3` still runs on the C /
  Python path because it goes through phonopy's `FCSolver`, whose
  signature does not yet accept a `lang` hint. The `lang` parameter
  therefore does not yet reach this step.
- **Dynamical-matrix diagonalization.** LAPACK calls (phonon
  eigenproblem, collision-matrix diagonalization, pseudo-inverse in
  the direct solution) stay in Python/SciPy by design. They are not
  part of the Rust port.

If any of these code paths is reached with `lang="Rust"`, phono3py
transparently uses the C (or Python) implementation for that step.

The Rust backend is exercised by the regular phono3py CI matrix on
Linux, macOS, and Windows.
