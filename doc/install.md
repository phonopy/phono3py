(install)=

# Installation

The detailed installation processes for different environments are described
below. The easiest installation with a good computation performance is achieved
by using the phono3py conda package (see {ref}`install_an_example`).

```{contents}
:depth: 3
:local:
```

Installation of phonopy before the installation of phono3py is required. See how
to install phonopy at https://phonopy.github.io/phonopy/install.html. Phono3py
relies on phonopy, so please use the latest release of phonopy when installing
phono3py.

## Installation using conda

Using conda is the easiest way for installation of phono3py if you are using
x86-64 linux system or macOS. These packages are made and maintained by Jan
Janssen. The installation is simply done by:

```bash
% conda install -c conda-forge phono3py
```

All dependent packages should be installed.

## Installation from source code

When installing phono3py using `setup.py` from the source code, a few libraries
are required before running `setup.py` script.

For phono3py, OpenMP library is necessary for the multithreding support. In
additon, BLAS, LAPACK, and LAPACKE are also needed. These packages are probably
installed using the package manager for each OS or conda.

Custom installation is achieved by creating `site.cfg` file on the same
directory as `setup.up`. Users will customize flags and additional libraries by
adding a `site.cfg` that conforms to numpy stile distutils
(`numpy.distutils.site.cfg` file). In most cases, users want to enable
multithreading support with OpenMP. Its minimum setting with gcc as the compiler
is:

```
[phono3py]
extra_compile_args = -fopenmp
```

Additional configuration can be necessary. It is recommended first starting only
with `extra_compile_args = -fopenmp` and run `setup.py`, then check if phono3py
works properly or not without error or warning. When phono3py doesn't work with
the above setting, add another configuration one by one to test compilation.
With conda-forge packages, at 11th Apr. 2022, the following configuration
worked.

With MKL (for gcc and clang), `site.cfg`

```
[phono3py]
extra_compile_args = -fopenmp
```

With openblas and clang

```
[phono3py]
extra_compile_args = -fopenmp
```

With openblas and gcc

```
[phono3py]
extra_compile_args = -fopenmp
extra_link_args = -lgomp
```

Mind that these configurations can be outdated instantly by the change of the
conda-forge packages.

More detailed information about `site.cfg` customization is found at
https://github.com/numpy/numpy/blob/main/site.cfg.example.

(install_lapacke)=

### Installation of LAPACKE

LAPACK library is used in a few parts of the code to diagonalize matrices.
LAPACK*E* is the C-wrapper of LAPACK and LAPACK relies on BLAS. Both
single-thread or multithread BLAS can be used in phono3py. In the following,
multiple different ways of installation of LAPACKE are explained.

### OpenBLAS provided by conda

The installtion of LAPACKE is easy by conda. It is:

```bash
% conda install -c conda-forge openblas
```

### OpenMP library of gcc

With system provided gcc, `libgomp1` may be necessary to enable OpenMP
multithreading support. This library is probably installed already in your
system. If you don't have it and you use Ubuntu linux, it is installed by:

```bash
% sudo apt-get install libgomp1
```

### Netlib LAPACKE provided by Ubuntu package manager (with single-thread BLAS)

LAPACKE (http://www.netlib.org/lapack/lapacke.html) can be installed from the
Ubuntu package manager (`liblapacke` and `liblapacke-dev`):

```bash
% sudo apt-get install liblapack-dev liblapacke-dev
```

### Compiling Netlib LAPACKE

The compilation procedure is found at the LAPACKE web site. After creating the
LAPACKE library, `liblapacke.a` (or the dynamic link library), `setup.py` must
be properly modified to link it. As an example, the procedure of compiling
LAPACKE is shown below.

```bash
% tar xvfz lapack-3.6.0.tgz
% cd lapack-3.6.0
% cp make.inc.example make.inc
% make lapackelib
```

BLAS, LAPACK, and LAPACKE, these all may have to be compiled with `-fPIC` option
to use it with python.

## Building using setup.py

If package installation is not possible or you want to compile with special
compiler or special options, phono3py is built using setup.py. In this case,
manual modification of `setup.py` may be needed.

1. Download the latest source code at

   https://pypi.python.org/pypi/phono3py

2. and extract it:

   ```
   % tar xvfz phono3py-1.11.13.39.tar.gz
   % cd phono3py-1.11.13.39
   ```

   The other option is using git to clone the phonopy repository from github:

   ```
   % git clone https://github.com/phonopy/phono3py.git
   % cd phono3py
   % git checkout master
   ```

3. Set up C-libraries for python C-API and python codes. This can be done as
   follows:

   Run `setup.py` script via pip:

   ```
   % pip install -e .
   ```

4. Set :envvar:`$PATH` and :envvar:`$PYTHONPATH`

   `PATH` and `PYTHONPATH` are set in the same way as phonopy, see
   https://phonopy.github.io/phonopy/install.html#building-using-setup-py.

(install_an_example)=

## Installation instruction of latest development version of phono3py

When using conda, `PYTHONPATH` should not be set if possible because potentially
wrong python libraries can be imported.

1. Download miniforge

   Miniforge is downloaded at https://github.com/conda-forge/miniforge. The
   detailed installation instruction is found in the same page.

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
   % conda create -n phono3py python=3.9
   % conda activate phono3py
   ```

3. Install necessary conda packages for phono3py

   For x86-64 system:

   ```bash
   % conda install numpy scipy h5py pyyaml matplotlib-base compilers "libblas=*=*mkl" spglib mkl-include
   ```

   A libblas library can be chosen among `[openblas, mkl, blis, netlib]`. If
   specific one is expected, it is installed by (e.g. openblas)

   ```bash
   % conda install "libblas=*=*openblas"
   ```

   For macOS ARM64 system, currently only openblas can be chosen:

   ```bash
   % conda install numpy scipy h5py pyyaml matplotlib-base compilers spglib
   ```

   Note that using hdf5 files on NFS mouted file system, you may have to disable
   file locking by setting

   ```bash
   export HDF5_USE_FILE_LOCKING=FALSE
   ```

   Install the latest phonopy and phono3py from github sources:

   ```bash
   % mkdir dev
   % cd dev
   % git clone https://github.com/phonopy/phonopy.git
   % git clone https://github.com/phonopy/phono3py.git
   % cd phonopy
   % git checkout develop
   % python setup.py build
   % pip install -e .
   % cd ../phono3py
   % git checkout develop
   % echo "[phono3py]" > site.cfg
   % echo "extra_compile_args = -fopenmp" >> site.cfg
   % python setup.py build
   % pip install -e .
   ```

   The conda packages dependency can often change and this recipe may not work
   properly. So if you find this instruction doesn't work, it is very
   appreciated if letting us know it in the phonopy mailing list.

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
`MULTITHREADED_BLAS`, which can be written in `setup.py`. Deactivating the
multithreading of BLAS using the environment variables is not recommended
because it is used in the non-nested parts of the code and these multithreadings
are unnecessary to be deactivated.

## Trouble shooting

1. Phonopy version should be the latest to use the latest phono3py.
2. There are other pitfalls, see
   https://phonopy.github.io/phonopy/install.html#trouble-shooting.
