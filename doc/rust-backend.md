(rust_backend)=

# Rust backend (experimental)

The computationally heavy parts of phono3py have been implemented as C extension
modules called from Python through the Python/C API. An alternative Rust
implementation is now available experimentally. The Rust path is distributed as
a separate Python extension module, `phono3py_rs`, built with
[maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/). It is
experimental: behaviour is validated against the C path by the regression tests
under `test/`, but the C extension remains the default and both paths are kept
in the source tree for cross-checking.

```{contents}
:depth: 2
:local:
```

## Installation

The Rust backend is not bundled with the standard phono3py wheel and conda
package. It has to be built from the source tree under `rust/`.

### Requirements

- A Rust toolchain (stable, edition 2021, `rustc >= 1.75`). The
  easiest way to install it is via [rustup](https://rustup.rs/).
- [maturin](https://www.maturin.rs/) 1.7 or newer (available on PyPI
  and conda-forge).
- Python 3.10 or newer. The extension is built against the stable ABI
  (`abi3-py310`), so one build works for all Python 3.10+ interpreters.
- A working phono3py source checkout and its usual build/runtime
  dependencies (see {ref}`install_from_source_code`).

### Build and install

Build the `phono3py_rs` extension in editable mode with `maturin
develop`:

```bash
% cd rust
% maturin develop --release
```

After a successful build, the module should import from any Python process

```python
import phono3py_rs
```

The phono3py Python layer imports `phono3py_rs` lazily and only when
the Rust backend is selected, so installations without the extension
continue to work on the C path.

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

Once `phono3py_rs` is installed, the Rust backend is selected through
the `--rust` flag on the command line or the `lang` keyword on the
Python API. The C backend remains the default.

### Command line

Pass `--rust` to the `phono3py` command:

```bash
% phono3py --rust ...
```

When the flag is set, the *General settings* block of the run log
prints

```
Experimental Rust backend enabled (lang=Rust)
```

as a reminder.

### Python API

The `Phono3py` constructor and the `phono3py.load` loader both accept
a `lang` keyword:

```python
import phono3py

ph3 = phono3py.load("phono3py.yaml", lang="Rust")
```

The current value is exposed as the read-only `Phono3py.lang`
property. Valid values are `"C"` (default) and `"Rust"`.

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
% RAYON_NUM_THREADS=8 phono3py --rust ...
```

NumPy/SciPy BLAS multithreading used by phonon diagonalization is
independent and is controlled by the BLAS library's own variables
(e.g. `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`).

### Dispatch tracing

To verify that a given code path is actually running on the Rust
backend, set `PHONO3PY_TRACE_LANG=1` before launching:

```bash
% PHONO3PY_TRACE_LANG=1 phono3py --rust ...
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
where individual per-grid-point calls do not fully saturate the thread pool; on
typical workloads the gain is small (around 5% on the
[NaMgF3](https://mdr.nims.go.jp/datasets/93fe6da8-ea25-4239-8cd9-c299b53c9854)
20-atom benchmark at 128 threads on AMD EPYC 9754 128-Core Processor).

The batch size can be set either through
`conductivity_calculator(..., rust_gp_batch_size=<int>)` or through
the `PHONO3PY_RUST_GP_BATCH_SIZE` environment variable. Resolution
order:

1. The `rust_gp_batch_size` argument if it is not `None`.
2. Otherwise the `PHONO3PY_RUST_GP_BATCH_SIZE` env var (default `0`).
3. A value `<= 0` falls back to the per-grid-point loop.

Example:

```bash
% PHONO3PY_RUST_GP_BATCH_SIZE=8 phono3py --rust ...
```

The batched path applies only to the low-memory RTA solver and is
skipped automatically when `read_gamma` is set.

## Scope

Most of the 3-phonon scattering runtime is available on the Rust
backend:

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
  in `phono3py_rs`, but `Phono3py.produce_fc3` still runs on the C /
  Python path because it goes through phonopy's `FCSolver`, whose
  signature does not yet accept a `lang` hint. The `lang` parameter
  therefore does not yet reach this step.
- **Dynamical-matrix diagonalization.** LAPACK calls (phonon
  eigenproblem, collision-matrix diagonalization, pseudo-inverse in
  the direct solution) stay in Python/SciPy by design. They are not
  part of the Rust port.

If any of these code paths is reached with `lang="Rust"`, phono3py
transparently uses the C (or Python) implementation for that step.

The Rust backend is exercised by dedicated pull-request CI on Linux,
macOS, and Windows (see
`.github/workflows/phono3py-pytest-conda-rust*.yml`).
