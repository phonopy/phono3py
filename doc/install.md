(install)=

# Installation

The detailed installation processes for different environments are described
below. The easiest installation with a good computation performance is achieved
by using the phono3py conda package. Installation of phonopy before the
installation of phono3py is required. See how to install phonopy at
https://phonopy.github.io/phonopy/install.html. Phono3py relies on phonopy, so
please use the latest release of phonopy when installing phono3py.

<!--
```{contents}
:depth: 3
:local:
```
-->
## Installation using conda

Using conda is the easiest way for installation of phono3py for linux and macOS.
These packages are made by Jan Janssen. The installation is simply done by:

```bash
% conda install -c conda-forge phono3py
```

All dependent packages should be installed.

(install_from_source_code)=
## Installation from source code

When installing phono3py from the source code, cmake is required before running
`pip install`. A C/C++ compiler is required only when the legacy C extension is
also built (i.e. without `PHONO3PY_NO_C_EXT=1`).

The package version is derived from git tags by `setuptools_scm` (written to
`phono3py/_version.py`). Building from source therefore requires a git checkout
that includes tags. Building from an archive without git metadata (e.g. a
downloaded zip) may fail to determine the version.

These may be installed by the package manager of OS (e.g. `apt`) or conda
environment. Automatic search of required libraries and flags that are already
on the system is performed by cmake.

## Installation instruction of latest development version of phono3py

When using conda, `PYTHONPATH` should not be set if possible because potentially
wrong python libraries can be imported.

1. Download miniforge

   Miniforge is downloaded at https://github.com/conda-forge/miniforge. The
   detailed installation instruction is found in the same page. If usual conda
   or miniconda is used, the following `~/.condarc` setting is recommended:

   ```
   channel_priority: strict
   channels:
   - conda-forge
   ```

2. Initialization of conda and setup of conda environment

   ```bash
   % conda init <your_shell>
   ```

   `<your_shell>` is often `bash` but may be something else. It is important
   that after running `conda init`, your shell is needed to be closed and
   restarted. See more information by `conda init --help`.

   Then conda allows to make conda installation isolated by using conda's
   virtual environment.

   ```bash
   % conda create -n phono3py python
   % conda activate phono3py
   ```

3. Install necessary conda packages for phono3py

   ```bash
   % conda install numpy scipy h5py pyyaml matplotlib-base cmake spglib phonors
   ```

   Here `phonors` is the Rust backend, installed as a pre-built binary from
   conda; usually this is all that is needed. Only when a development version of
   `phonors` is required is it necessary to build `phonors` from source; see
   {ref}`rust_backend`.

   The above packages are enough for the default build below, which skips the C
   extension. To also build the legacy C extension, additionally install C/C++
   compilers:

   ```bash
   % conda install c-compiler cxx-compiler
   ```

   The latest phonopy and phono3py are obtained from github, and they are
   compiled and installed by:

   ```bash
   % mkdir dev
   % cd dev
   % git clone https://github.com/phonopy/phonopy.git
   % git clone https://github.com/phonopy/phono3py.git
   % cd phonopy
   % PHONOPY_NO_C_EXT=1 pip install -e . -vvv
   % cd ../phono3py
   % PHONO3PY_NO_C_EXT=1 pip install -e . -vvv
   ```

   Both phonopy and phono3py default to the Rust backend provided by
   `phonors`, which is installed automatically as a runtime dependency, so the
   C extension is not required. Setting `PHONOPY_NO_C_EXT=1` /
   `PHONO3PY_NO_C_EXT=1` makes each `CMakeLists.txt` return early and skips
   building the C extensions (`phonopy._phonopy` / `phonopy._recgrid` and
   `phono3py._phono3py` / `phono3py._phononcalc`); no C/C++ compiler is needed
   in this case. To also build the legacy C extensions, run `pip install -e .
   -vvv` without these env vars. See {ref}`rust_backend_no_c_ext` for more
   details.

## Dependent libraries

(install_lapacke)=
### LAPACKE (deprecated, legacy C-extension backend only)

```{deprecated} v4
The LAPACKE-linked C build is only useful when running with the legacy
C-extension backend (`--legacy-backend` / `lang="C"`). The default Rust
backend uses scipy/numpy for diagonalization and does not call into
LAPACKE. The LAPACKE-specific `--pinv-solver=1` and `--pinv-solver=2`
solvers are also deprecated; the default `--pinv-solver=4`
(`scipy.linalg.lapack.dsyev`) works with both backends.
```

LAPACK library is used in a few parts of the code to diagonalize matrices.
LAPACK*E* is the C-wrapper of LAPACK and LAPACK relies on BLAS. Both
single-thread or multithread BLAS can be used in phono3py. In the following,
multiple different ways of installation of LAPACKE are explained.

(install_with_lapacke)=
#### Building with linking LAPACKE

Phono3py can operate without linking to LAPACKE, which is the default
compilation setting. However, it is also possible to compile Phono3py with
LAPACKE support. When compiled this way, the diagonalization of the dynamical
matrix is handled by LAPACK routines within the C code of Phono3py.
Additionally, LAPACK is used for the diagonalization of the collision matrix
in the direct solution. The LAPACKE-using code paths are only reachable via
the legacy C-extension backend (`--legacy-backend` / `lang="C"`).

To compile phono3py with linking LAPACKE in C, use the following command:

```
% BUILD_WITHOUT_LAPACKE=OFF pip install -e . -vvv
```

For this, BLAS and LAPACKE libraries are required.

(install_an_example)=


#### OpenBLAS provided by conda

The installation of LAPACKE is easy by conda. It is:

```bash
% conda install -c conda-forge openblas
```

#### Netlib LAPACKE provided by Ubuntu package manager (with single-thread BLAS)

LAPACKE (http://www.netlib.org/lapack/lapacke.html) can be installed from the
Ubuntu package manager (`liblapacke` and `liblapacke-dev`):

```bash
% sudo apt-get install liblapack-dev liblapacke-dev
```

## Using HDF5 files on NFS mounted file system

If you are using HDF5 files on an NFS-mounted file system, you might need to
disable file locking. This can be done by setting the following environment
variable:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

<!-- (install_openmp)=
## Multithreading and its controlling by C macro

Phono3py uses multithreading concurrency in two ways. One is that written in the
code with OpenMP `parallel for`. The other is achieved by using multithreaded
BLAS. The BLAS multithreading is depending on which BLAS library is chosen by
users and the number of threads to be used may be controlled by the library's
environment variables (e.g., `OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS`). In
the phono3py C code, these two are written in a nested way, but of course the
nested use of multiple multithreadings has to be avoided. The outer loop of the
nesting is done by the OpenMP `parallel for` code. The inner loop calls LAPACKE
functions and then the LAPACKE functions call the BLAS routines. If both of the
inner and outer multithreadings can be activated, the inner multithreading must
be deactivated at the compilation time. This is achieved by setting the C macro
`MULTITHREADED_BLAS`, which can be written in `CMakeLists.txt`. Deactivating the
multithreading of BLAS using the environment variables is not recommended
because it is also used in the non-nested parts of the code and these
multithreadings are unnecessary to be deactivated. -->

## Trouble shooting

1. Phonopy version should be the latest to use the latest phono3py.
2. There are other pitfalls, see
   https://phonopy.github.io/phonopy/install.html#trouble-shooting.
