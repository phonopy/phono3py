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

From source code
-----------------

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

Installation of necessary libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here two ways to install necessary libraries are introduced.

By Ubuntu package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the case of ubuntu linux, it would be like::

   % sudo apt-get install libgomp1 liblapack-dev liblapacke-dev

In the versions of Ubuntu-12.10 or later, LAPACKE
(http://www.netlib.org/lapack/lapacke.html) can be installed from the
package manager (``liblapacke`` and ``liblapacke-dev``). In the recent
MacPorts, the ``OpenBLAS`` package contains not only BLAS but also
LAPACK and LAPACKE in ``libopenblas``. But in the older versions of
Ubuntu or in the other environments, you may have to compile LAPACKE
by yourself. The compilation procedure is found at the LAPACKE web
site. After creating the LAPACKE library, ``liblapacke.a`` (or the
dynamic link library), ``setup.py`` must be properly modified to link
it. As an example, the procedure of compiling LAPACKE is shown below.

::

   % tar xvfz lapack-3.6.0.tgz
   % cd lapack-3.6.0
   % cp make.inc.example make.inc
   % make lapackelib

BLAS, LAPACK, and LAPACKE, these all may have to be compiled
with -fPIC option to use it with python.

By conda
^^^^^^^^^

The installtion of LAPACKE is easy by conda. It is::

   % conda install openblas

or if the python libraries are not yet installed::

   % conda install openblas numpy scipy h5py pyyaml matplotlib

This openblas package contains BLAS, LAPACK, and LAPACKE.

..
   Multithreading support
   ------------------------

   Phono3py supports OpenMP multithreading and most users will need it,
   otherwise the calculation may take long time. The library options used
   for GCC, ``-lgomp`` and ``-fopenmp``, are written in ``setup.py``,
   but for the other compilers, you may have to change them.  If you need
   to compile without the OpenMP support, you can remove these options in
   ``setup.py``.

Building using setup.py
~~~~~~~~~~~~~~~~~~~~~~~~

If package installation is not possible or you want to compile with
special compiler or special options, phono3py is built using
setup.py. In this case, manual modification of ``setup.py`` may be
needed.

Download the latest source packages at
https://pypi.python.org/pypi/phono3py and extract it somewhere. In the
directory, open ``setup.py`` and specify the library and its path of a
lapacke library. Then

::

   % python setup.py install --user

Or you can install phono3py on the current directory by

::

   % python setup.py install --home=.

In this way to setup, ``PYTHONPATH`` has to be set so that python can
find harmonic and anharmonic phonopy libraries. If you have been
already a user of phonopy, ``PYTHONPATH`` for the original phonopy
version has to be removed. The ``PYTHONPATH`` setting depends on
shells that you use. For example in bash or zsh::

   export PYTHONPATH=~/phono3py-1.11.11/lib/python

or::

   export PYTHONPATH=$PYTHONPATH:~/phono3py-1.11.11/lib/python

Phono3py command is installed under ``bin`` directory. The location of
``bin`` directory is depending on ``--user`` or ``--home`` scheme when
running ``setup.py``. In the former case, it depends on your
operation system, e.g., ``~/.local/bin`` for Ubuntu linux and
``~/Library/Python/2.7`` for Mac (& python2.7). In the latter case,
``bin`` directory is found on the current directory if it was
``--home=.``.

Install using pip/conda
------------------------

PyPI and conda packages are prepared at phonopy and phono3py
releases. Using these packages, the phonopy and phono3py installations
are expected to be easily done. For more detail, see
https://atztogo.github.io/phonopy/install.html .

