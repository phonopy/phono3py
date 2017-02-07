.. _calculator_interfaces:

Interfaces to calculators
==========================

Currently the built-in interfaces for VASP, Pwscf, and CRYSTAL are
prepared. VASP is the default interface and no special option is
necessary to invoke it, but for the other interfaces, each special
option has to be specified, e.g. ``--pwscf`` or ``--crystal``.

.. toctree::
   :maxdepth: 1

   vasp
   pwscf
   crystal

Calculator specific behaviors
------------------------------

Physical unit
^^^^^^^^^^^^^^

The interfaces for VASP, Pwscf, and CRYSTAL  are built in to the phono3py command.

For each calculator, each physical unit system is used. The physical
unit systems used for the calculators are summarized below.

::

           | unit-cell  FORCES_FC3   disp_fc3.yaml
   -----------------------------------------------
   VASP    | Angstrom   eV/Angstrom  Angstrom
   Pwscf   | au (bohr)  Ry/au        au
   CRYSTAL | Angstrom   eV/Angstrom  Angstrom

``FORCES_FC2`` and ``disp_fc2.yaml`` have the same physical units as
``FORCES_FC3`` and ``disp_fc3.yaml``, respectively.

.. _default_unit_cell_file_name_for_calculator:

Default unit cell file name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default unit cell file names are also changed according to the calculators::
    
   VASP    | POSCAR     
   Pwscf   | unitcell.in
   CRYSTAL | crystal.o


.. _default_displacement_distance_for_calculator:

Default displacement distance created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default displacement distances created by ``-d`` option without
``--amplitude`` option are respectively as follows::

   VASP    | 0.03 Angstrom
   Pwscf   | 0.06 au (bohr)
   CRYSTAL | 0.03 Angstrom


