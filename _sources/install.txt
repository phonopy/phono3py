.. _install:

Installation
=============

System requirement
-------------------

**From version 1.11.3, phono3py depends on phonopy (>1.11.2).** So the
installation of phonopy before the installation of phono3py is
required. See how to install phonopy at
https://atztogo.github.io/phonopy/install.html .

For phono3py, OpenMP library is necessary for the multithreding
support. In the case of the GNU OpenMP library, the library name may
be ``libgomp1``. In additon, LAPACK and LAPACKE are also needed. These
packages are probably installed using the package manager for each
OS. In the case of ubuntu linux, it would be like::

   % sudo apt-get install libgomp1 liblapack-dev liblapacke-dev

In the versions of Ubuntu-12.10 or later, LAPACKE
(http://www.netlib.org/lapack/lapacke.html) can be installed from the
package manager (``liblapacke`` and ``liblapacke-dev``). In the recent
MacPorts, the ``lapack`` or ``OpenBLAS`` package probably contains
LAPACKE. But in the older versions of Ubuntu or in the other
environments, you may have to compile LAPACKE by yourself. The
compilation procedure is found at the LAPACKE web site. After creating
the LAPACKE library, ``liblapacke.a`` (or the dynamic link library),
``setup3.py`` must be properly modified to link it. As an example, the
procedure of compiling LAPACKE is shown below.

::

   % tar xvfz lapack-3.5.0.tgz
   % cd lapack-3.5.0
   % cp make.inc.example make.inc
   % make lapackelib

BLAS, LAPACK, and LAPACKE, these all may have to be compiled
with -fPIC option to use it with python.

Multithreading support
------------------------

Phono3py supports OpenMP multithreading and most users will need it,
otherwise the calculation may take long time. The library options used
for GCC, ``-lgomp`` and ``-fopenmp``, are written in ``setup3.py``,
but for the other compilers, you may have to change them.  If you need
to compile without the OpenMP support, you can remove these options in
``setup3.py``.

Install using pip/conda
------------------------

Occasionally PyPI and conda packages are prepared at phonopy and
phono3py releases. Using these packages, the phonopy and phono3py
installations are expected to be easily done. For more detail, see 
https://atztogo.github.io/phonopy/install.html .

Building using setup.py
------------------------

If package installation is not possible or you want to compile with
special compiler or special options, phono3py is built using
setup.py. In this case, manual modification of ``setup.py`` may be
needed.

Download the latest source packages at

https://pypi.python.org/pypi/phono3py

and extract it somewhere. The version number here is not related to
the version number of harmonic (usual) phonopy. The harmonic phonopy
included in this package is a development version and can be different
from that distributed at sourceforge.net.

In the directory, open ``setup3.py`` and set the location of
lapacke. If you installed lapacke from the package manager, you can
remove the line related to lapacke. If you compiled it by yourself,
set the location of it. Before running ``setup3.py``, if you install
phonopy (not phono3py), you need to uninstall phonopy since
``setup3.py`` installs phonopy, too. Then

::

   % python setup3.py install --user

Or you can install phono3py on the current directory by

::

   % python setup3.py install --home=.

In this way to setup, ``PYTHONPATH`` has to be set so that python can
find harmonic and anharmonic phonopy libraries. If you have been
already a user of phonopy, ``PYTHONPATH`` for the original phonopy
version has to be removed. The ``PYTHONPATH`` setting depends on
shells that you use. For example in bash or zsh::

   export PYTHONPATH=~/phonopy-1.11.3/lib/python

or::

   export PYTHONPATH=$PYTHONPATH:~/phonopy-1.11.3/lib/python

Phono3py command is installed under ``bin`` directory. The location of
``bin`` directory is depending on ``--user`` or ``--home`` scheme when
running ``setup3.py``. In the former case, it depends on your
operation system, e.g., ``~/.local/bin`` for Ubuntu linux and
``~/Library/Python/2.7`` for Mac (& python2.7). In the latter case,
``bin`` directory is found on the current directory if it was
``--home=.``.
