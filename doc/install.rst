.. _install:

Installation
=============

The detailed installation processes for different environments
are described below. The easiest installation with a good computation
performance is achieved by using the phono3py conda package (see
:ref:`install_an_example`).

.. contents::
   :depth: 3
   :local:

Installation of phonopy before the installation of phono3py is
required. See how to install phonopy at
https://phonopy.github.io/phonopy/install.html. Phono3py relies on
phonopy, so please use the latest release of phonopy when installing
phono3py.

Installation using conda
-----------------------------

Using conda is the easiest way for installation of phono3py if you are
using x86-64 linux system or macOS. These packages are made and
maintained by Jan Janssen. The installation is simply done by::

   % conda install -c conda-forge phono3py

All dependent packages should be installed.

Installation using pip is not recommended
-----------------------------------------

PyPI packages are prepared for phonopy and phono3py releases. When
installing with PyPI, ``setup.py`` is executed locally to compile the
part of the code written in C, so a few libraries such as
lapacke must exist in the system. Those necessary libraries are
explained in the next section.

Installation from source code
------------------------------

When installing phono3py using ``setup.py`` from the source code, a
few libraries are required before running ``setup.py`` script.

For phono3py, OpenMP library is necessary for the multithreding
support. In additon, BLAS, LAPACK, and LAPACKE are also needed. These
packages are probably installed using the package manager for each OS
or conda.

When using gcc to compile phono3py, ``libgomp1`` is necessary to
enable OpenMP multithreading support. This library is probably
installed already in your system. If you don't have it and you use
Ubuntu linux, it is installed by::

   % sudo apt-get install libgomp1

.. _install_lapacke:

Installation of LAPACKE
~~~~~~~~~~~~~~~~~~~~~~~~

LAPACK library is used in a few parts of the code to diagonalize
matrices. LAPACK*E* is the C-wrapper of LAPACK and LAPACK relies on
BLAS. Both single-thread or multithread BLAS can be
used in phono3py. In the following, multiple different ways of
installation of LAPACKE are explained.

.. _install_mkl_lapacke:

MKL LAPACKE (with multithread BLAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Phono3py can be compiled with MKL for using LAPACKE.  If ``setup.py``
finds the file named ``setup_mkl.py``, the contents of ``setup_mkl.py`` is read
and those are included in the compilation setting.  For example, the
following setting prepared as ``setup_mkl.py`` seems working on Ubuntu 16.04
system::

   intel_root = "/opt/intel/composer_xe_2015.7.235"
   mkl_root = "%s/mkl" % intel_root
   compiler_root = "%s/compiler" % intel_root

   mkl_extra_link_args_lapacke = ['-L%s/lib/intel64' % mkl_root,
                                  '-lmkl_rt']
   mkl_extra_link_args_lapacke += ['-L%s/lib/intel64' % compiler_root,
                                   '-lsvml',
                                   '-liomp5',
                                   '-limf',
                                   '-lpthread']
   mkl_include_dirs_lapacke = ["%s/include" % mkl_root]

This setting considers to use ``icc`` but it may be compiled with
``gcc``. With ``gcc``, the compiler related setting shown above (i.e.,
around ``compiler_root``) is unnecessary. To achieve this
installation, not only the MKL library but also the header file are
necessary. The libraries are linked dynamically, so in most of the
cases, ``LD_LIBRARY_PATH`` environment variable has to be correctly
specified to let phono3py find those libraries.

.. _install_openblas_lapacke:

OpenBLAS provided by conda (with multithread BLAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installtion of LAPACKE is easy by conda. It is::

   % conda install -c conda-forge openblas libgfortran

The recent change of openblas package provided from anaconda makes to
install nomkl, i.e., numpy and scipy with Intel MKL cannot be used
together with openblas. At this moment, this is avoided to install
openblas from conda-forge channel. If the python libraries are not yet
installed::

   % conda install -c conda-forge numpy scipy h5py pyyaml matplotlib

When using hdf5 files from NFS mouted location, the latest h5py may
not work. In this case, installation of an older version is
recommended::

   % conda install hdf5=1.8.18

This openblas package contains BLAS, LAPACK, and LAPACKE. When this
``libopenblas`` is linked and the ``else`` statement of the C macro
definition section in ``setup.py`` is executed, the following macro
are activated::

   if use_setuptools:
       extra_compile_args += ['-DMULTITHREADED_BLAS']
   else:
       define_macros += [('MULTITHREADED_BLAS', None)]

Libraries or headers are not found at the build by ``setup.py``, the
following setting may be of the help::

    extra_link_args_lapacke += ['-lopenblas', '-lgfortran']
    include_dirs_lapacke += [
        os.path.join(os.environ['CONDA_PREFIX'], 'include'), ]


Netlib LAPACKE provided by Ubuntu package manager (with single-thread BLAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the versions of Ubuntu-12.10 or later, LAPACKE
(http://www.netlib.org/lapack/lapacke.html) can be installed from the
package manager (``liblapacke`` and ``liblapacke-dev``)::

   % sudo apt-get install liblapack-dev liblapacke-dev

Compiling Netlib LAPACKE
^^^^^^^^^^^^^^^^^^^^^^^^^

The compilation procedure is found at the LAPACKE web site. After
creating the LAPACKE library, ``liblapacke.a`` (or the dynamic link
library), ``setup.py`` must be properly modified to link it. As an
example, the procedure of compiling LAPACKE is shown below.

::

   % tar xvfz lapack-3.6.0.tgz
   % cd lapack-3.6.0
   % cp make.inc.example make.inc
   % make lapackelib

BLAS, LAPACK, and LAPACKE, these all may have to be compiled
with ``-fPIC`` option to use it with python.

Building using setup.py
~~~~~~~~~~~~~~~~~~~~~~~~

If package installation is not possible or you want to compile with
special compiler or special options, phono3py is built using
setup.py. In this case, manual modification of ``setup.py`` may be
needed.

1. Download the latest source code at

   https://pypi.python.org/pypi/phono3py

2. and extract it::

     % tar xvfz phono3py-1.11.13.39.tar.gz
     % cd phono3py-1.11.13.39

   The other option is using git to clone the phonopy repository from github::

     % git clone https://github.com/phonopy/phono3py.git
     % cd phono3py

2. Set up C-libraries for python C-API and python codes. This can be
   done as follows:

   Run ``setup.py`` script via pip::

      % pip install -e .

3. Set :envvar:`$PATH` and :envvar:`$PYTHONPATH`

   ``PATH`` and ``PYTHONPATH`` are set in the same way as phonopy, see
   https://phonopy.github.io/phonopy/install.html#building-using-setup-py.

.. _install_an_example:

Installation instruction of latest development version of phono3py
------------------------------------------------------------------

When using conda, ``PYTHONPATH`` should not be set if possible because
potentially wrong python libraries can be imported.

This installation instruction supposes linux x86-64 environment.

1. Download miniconda package

   Miniconda is downloaded at https://conda.io/miniconda.html.

   For usual 64-bit Linux system::

     % wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

   For macOS::

     % wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh

   The installation is made by

   ::

      % bash ~/miniconda.sh -b -p $HOME/miniconda
      % export PATH="$HOME/miniconda/bin:$PATH"

   The detailed installation instruction is found at https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

2. Initialization of conda and setup of conda environment

   ::

      % conda init <your_shell>

   ``<your_shell>`` is often ``bash`` but may be something else. It is
   important that after running ``conda init``, your shell is needed
   to be closed and restarted. See more information by ``conda init
   --help``.

   Then conda allows to make conda installation isolated by using conda's
   virtual environment.

   ::

      % conda create -n phono3py -c conda-forge python=3.7
      % conda activate phono3py

   Use of this is strongly recommended, otherwise proper settings of
   ``CONDA_PREFIX``, ``C_INCLUDE_PATH``, and ``LD_LIBRARY_PATH`` will
   be necessary.

2. Installation of compiler from conda

   For usual 64-bit Linux system::

      % conda install -c conda-forge gcc_linux-64

   For macOS::

      % conda install clang_osx-64 llvm-openmp

3. Install necessary conda packages for phono3py

   ::

      % conda install -c conda-forge numpy scipy h5py pyyaml matplotlib openblas libgfortran

   When using hdf5 files from NFS mouted location, the latest h5py may
   not work. In this case, installation of an older version is
   recommended::

      % conda install -c conda-forge hdf5=1.8.18

   Install the latest phonopy and phono3py::

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
      % python setup.py build
      % pip install -e .

   The conda packages dependency can often change and this recipe may
   not work properly. So if you find this instruction doesn't work, it
   is very appreciated if letting us know it in the phonopy mailing
   list.

Multithreading and its controlling by C macro
----------------------------------------------

Phono3py uses multithreading concurrency in two ways. One is that
written in the code with OpenMP ``parallel for``. The other is
achieved by using multithreaded BLAS. The BLAS multithreading is
depending on which BLAS library is chosen by users and the number of
threads to be used may be controlled by the library's environment
variables (e.g., ``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS``). In
the phono3py C code, these two are written in a nested way, but of
course the nested use of multiple multithreadings has to be
avoided. The outer loop of the nesting is done by the OpenMP
``parallel for`` code. The inner loop calls LAPACKE functions and then
the LAPACKE functions call the BLAS routines. If both of the inner and
outer multithreadings can be activated, the inner multithreading must
be deactivated at the compilation time. This is achieved by setting
the C macro ``MULTITHREADED_BLAS``, which can be written in
``setup.py``. Deactivating the multithreading of BLAS using the
environment variables is not recommended because it is used in the
non-nested parts of the code and these multithreadings are
unnecessary to be deactivated.

Trouble shooting
-----------------

1. Phonopy version should be the latest to use the latest phono3py.
2. There are other pitfalls, see
   https://phonopy.github.io/phonopy/install.html#trouble-shooting.
