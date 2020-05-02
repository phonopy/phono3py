.. _qe_interface:

Quantum ESPRESSO (pw) & phono3py calculation
============================================

Quantum espresso package itself has a set of the force constants
calculation environment based on DFPT. But the document here explains how
to calculate phonon-phonon interaction and related properties using
phono3py, i.e., using the finite displacement and supercell approach.

An example for QE (pw) is found in the ``example-phono3py/Si-QE`` directory.

To invoke the QE (pw) interface, ``--qe`` option has to be always
specified::

   % phono3py --qe [options] [arguments]

When the file name of the unit cell is different from the default one
(see :ref:`default_unit_cell_file_name_for_calculator`), ``-c`` option
is used to specify the file name. QE (pw) unit cell file parser used in
phono3py is the same as that in phonopy. It can read
only limited number of keywords that are shown in the phonopy web site
(http://phonopy.github.io/phonopy/qe.html#qe-interface).

.. _qe_workflow:

Workflow
---------

1. Create supercells with displacements

   ::

      % phono3py --qe -d --dim="2 2 2" -c Si.in

   In this example, probably 111 different supercells with
   displacements are created. Supercell files (``supercell-xxx.in``)
   are created but they contain only the crystal
   structures. Calculation setting has to be added before running the
   calculation.

2. Run QE (pw) for supercell force calculations

   Let's assume that the calculations have been made in ``disp-xxx``
   directories with the file names of ``Si-supercell.in``. Then after
   finishing those calculations, ``Si-supercell.out`` may be created
   in each directory.

3. Collect forces

   ``FORCES_FC3`` is obtained with ``--cf3`` options collecting the
   forces on atoms in QE (pw) calculation results::

      % phono3py --qe --cf3 disp-00001/Si-supercell.out disp-00002/Si-supercell.out ...

   or in recent bash or zsh::

      % phono3py --qe --cf3 disp-{00001..00111}/Si-supercell.out

   ``disp_fc3.yaml`` is used to create ``FORCES_FC3``, therefore it
   must exist in current directory.

4) Calculate 3rd and 2nd order force constants

   ``fc3.hdf5`` and ``fc2.hdf5`` files are created by::

      % phono3py --qe --dim="2 2 2" -c Si.in --sym-fc

5) Calculate lattice thermal conductivity, e.g., by::

      % phono3py --qe --dim="2 2 2" -c Si.in --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" \
         --mesh="11 11 11" --fc3 --fc2 --br
