.. _install:

Installation
=============

.. contents::
   :depth: 3
   :local:

Installation of phonopy before the installation of phono3py is
required. See how to install phonopy at
https://atztogo.github.io/phonopy/install.html. Phono3py relies on
phonopy, so please use the latest release of phonopy when installing
phono3py.

Installation using conda
-----------------------------

Using conda is the easiest way for installation of phono3py if you are
using usual 64 bit linux system. The conda packages for 64bit linux
are also found at ``atztogo`` channel::

   % conda install -c atztogo phono3py

All dependent packages are installed simultaneously.

Installation using pip
---------------------------

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

Installation of LAPACKE
~~~~~~~~~~~~~~~~~~~~~~~~

LAPACK library is used in a few parts of the code to diagonalize
matrices. LAPACK*E* is the C-wrapper of LAPACK and LAPACK relies on
BLAS. Both single-thread or multithread BLAS can be
used in phono3py. In the following, multiple different ways of
installation of LAPACKE are explained.

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
with -fPIC option to use it with python.

OpenBLAS provided by MacPorts (with single-thread BLAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MacPorts, the ``OpenBLAS`` package contains not only BLAS but also
LAPACK and LAPACKE in ``libopenblas``.

MKL LAPACKE (with multithread BLAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MKL LAPACKE can be used. For this, ``mkl.py`` file has to be
created. In this file, the locations of necessary MKL libraries are
provided such as follows::

   extra_link_args_lapacke += ['-L/opt/intel/mkl/lib/intel64',
                               '-lmkl_intel_ilp64', '-lmkl_intel_thread',
                               '-lmkl_core']
   library_dirs_lapacke += []
   include_dirs_lapacke += ['/opt/intel/mkl/include']


OpenBLAS provided by conda (with multithread BLAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installtion of LAPACKE is easy by conda. It is::

   % conda install openblas

or if the python libraries are not yet installed::

   % conda install openblas numpy scipy h5py pyyaml matplotlib

This openblas package contains BLAS, LAPACK, and LAPACKE.

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

     % git clone https://github.com/atztogo/phono3py.git
     % cd phono3py

2. Set up C-libraries for python C-API and python codes. This can be
   done as follows:

   Run ``setup.py`` script::

      % python setup.py install --user

3. Set :envvar:`$PATH` and :envvar:`$PYTHONPATH`

   ``PATH`` and ``PYTHONPATH`` are set in the same way as phonopy, see
   https://atztogo.github.io/phonopy/install.html#building-using-setup-py.

Installation on MacOSX
-----------------------

MacOSX users may be able to install phonopy and phono3py on recent
MacOSX. But it requires a basic knowledge on UNIX and python. So if
you are afraid of that, please prepare a computer or a virtual machine
with a normal linux OS such as Ubuntu-linux-64bit 14.04 or 16.04.

If you think you are familiar with MacOSX, unix system, and python,
the recommended installation process is written at
https://atztogo.github.io/phonopy/MacOSX.html, which is more-or-less
the same as phonopy, but with openblas, too. An example of the
procedure is summarized in the next section.

An example of installation process
-----------------------------------

1. Download miniconda package

   Miniconda is downloaded at https://conda.io/miniconda.html. 

   For usual 64-bit Linux system::

     % wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

   For MacOSX, it is assumed that gcc compiler is installed on your system. The
   compiler such as default clang on MacOSX can't handle OpenMP, so it
   can't be used. The gcc compiler may be installed using MacPort, e.g.::

     % sudo port install gcc7 wget

   where wget is optional. Then download using wget::

     % wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

2. Install conda packages

   It is supposed to have the following environment variable::

     export PATH=~/.miniconda3/bin:$PATH

   Then install and update conda::

     % bash miniconda.sh -b -p $HOME/.miniconda3
     % conda update conda

   The necessary python libraries and openBLAS are installed by::

     % conda install numpy scipy h5py pyyaml matplotlib openblas

   Install the latest phonopy and phono3py::

     % git clone https://github.com/atztogo/phonopy.git
     % cd phonopy
     % python setup.py install --user
     % cd ..
     % git clone https://github.com/atztogo/phono3py.git
     % cd phono3py
     % python setup.py install --user
     % cd ..

   Environment variables ``PATH`` and ``PYTHONPATH`` must be set
   appropriately to use phono3py. See see
   https://atztogo.github.io/phonopy/install.html#building-using-setup-py
   and
   https://atztogo.github.io/phonopy/install.html#set-correct-environment-variables-path-and-pythonpath.

Trouble shooting
-----------------

1. Phonopy version should be the latest to use the latest phono3py.
2. There are other pitfalls, see
   https://atztogo.github.io/phonopy/install.html#trouble-shooting.
