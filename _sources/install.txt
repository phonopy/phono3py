.. _install:

Installation
=============

System requirement
-------------------

The following python libraries are required.

::

   python-dev python-numpy python-yaml python-h5py python-matplotlib 

``python-matplotlib`` is optional, but it is strongly recommended to
install it.  The OpenMP library is necessary for multithreding
support. The GNU OpenMP library is ``libgomp1``.  In the case of
ubuntu linux, these are installed using the package manager::

   % sudo apt-get install python-dev python-numpy python-matplotlib \
     python-yaml python-h5py libgomp1 liblapacke-dev

In the versions of Ubuntu-12.10 or later, LAPACKE
(http://www.netlib.org/lapack/lapacke.html) can be installed from the
package manager (``liblapacke`` and ``liblapacke-dev``). In the recent
MacPorts, the ``lapack`` package may contains LAPACKE. But in the
older versions of Ubuntu or in the other environments, you
may have to compile LAPACKE by yourself. The compilation procedure is
found at the LAPACKE web site. After creating the LAPACKE library,
``liblapacke.a`` (or the dynamic link library), ``setup3.py`` must be
properly modified to link it. As an example, the procedure of
compiling LAPACKE is shown below.

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

Installation procedure
------------------------

Download the latest version from
http://sourceforge.net/projects/phonopy/files/phono3py/ and extract it
somewhere. The version number here is not related to the version
number of harmonic (usual) phonopy. The harmonic phonopy included in
this package is a development version and can be different from that
distributed at sourceforge.net.

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

   export PYTHONPATH=~/phonopy-1.11.1/lib/python

or::

   export PYTHONPATH=$PYTHONPATH:~/phonopy-1.11.1/lib/python

Phono3py command is installed under ``bin`` directory. The location of
``bin`` directory is depending on ``--user`` or ``--home`` scheme when
running ``setup3.py``. In the former case, it depends on your
operation system, e.g., ``~/.local/bin`` for Ubuntu linux and
``~/Library/Python/2.7`` for Mac (& python2.7). In the latter case,
``bin`` directory is found on the current directory if it was
``--home=.``.
