=======================================================
Welcome to phono3py
=======================================================

This software calculates phonon-phonon interaction and related
properties using the supercell approach. For example, the following
physical properties are obtained:

- Lattice thermal conductivity
- Phonon lifetime/linewidth
- Imaginary part of self energy
- Joint density of states (JDOS) and weighted-JDOS

The theoretical background is summarized in the paper found at
http://dx.doi.org/10.1103/PhysRevB.91.094306 or the draft in arxiv at
http://arxiv.org/abs/1501.00691 .

Examples are found in ``example-phono3py`` directory. Phono3py API
example ``Si.py`` is found in ``example-phono3py/Si`` directory, but
the API document has not yet written.

:ref:`Interfaces to calculators <calculator_interfaces>` for VASP and
pwscf are built-in.


Documentation
=============

.. toctree::
   :maxdepth: 1
      
   install
   interfaces
   vasp
   pwscf
   command-options
   output-files
   tips
   auxiliary-tools
   citation
   changelog

Mailing list
============

For questions, bug reports, and comments, please visit following
mailing list:

https://lists.sourceforge.net/lists/listinfo/phonopy-users

Message body including attached files has to be smaller than 300 KB.

License
=======

New BSD

Contact
=======

* Author: `Atsushi Togo <http://atztogo.github.io/>`_

