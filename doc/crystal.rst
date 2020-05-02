.. _crystal_interface:

CRYSTAL & phono3py calculation
==============================

CRYSTAL program package has a robust built-in phonon calculation
workflow for harmonic phonon properties. However, combining CRYSTAL
with Phono3py enables convenient access to anharmonic phonon properties.

An example for CRYSTAL is found in the ``example/Si-CRYSTAL`` directory.

To invoke the CRYSTAL interface, ``--crystal`` option has to be always
specified::

   % phono3py --crystal [options] [arguments]

When the file name of the unit cell is different from the default one
(see :ref:`default_unit_cell_file_name_for_calculator`), ``-c`` option
is used to specify the file name. CRYSTAL unit cell file parser used in
phono3py is the same as that in phonopy. It reads a limited number of
keywords that are documented in the phonopy web site
(http://phonopy.github.io/phonopy/crystal.html#crystal-interface).

.. _crystal_workflow:

Workflow
---------

In this example (Si-CRYSTAL), the CRYSTAL output file is crystal.o.
This is the default file name for the CRYSTAL interface,
so the -c crystal.o parameter is not needed.

1) Create supercells with displacements
   (4x4x4 for 2nd order FC, 2x2x2 for 3rd order FC)

   ::

      % phono3py --crystal --dim="2 2 2" --dim-fc2="4 4 4" -d

   57 supercell files (``supercell-xxx.d12/.ext``) for the third order
   force constants are created. In addition, one supercell file
   (``supercell_fc2-00001.d12/.ext``) is created for the second order
   force constants.

2) To make valid CRYSTAL input files, there are two possible options:

   a) Manually: modify the generated supercell-xxx.d12 and supercell_fc2-xxxxx.d12
      files by replacing the line ``Insert basis sets and parameters here`` with the
      basis set and computational parameters.

   b) Recommended option: before generating the supercells, include files named
      ``TEMPLATE`` and ``TEMPLATE3`` in the current directory. These files should
      contain the basis sets and computational parameters for CRYSTAL (see the example).
      When phono3py finds these files it automatically generates complete
      CRYSTAL input files in the step 1.

   Note that supercells with displacements must not be relaxed in the
   force calculations, because atomic forces induced by a small atomic
   displacement are what we need for phonon calculation. To get accurate
   forces, TOLDEE parameter should be 10 or higher. Phono3py includes this
   parameter and the necessary GRADCAL keyword automatically in the inputs.

   Then, CRYSTAL supercell calculations are executed to obtain forces on
   atoms, e.g., as follows::

     % runcry17 supercell-001.d12

3) Collect forces in ``FORCES_FC3`` and ``FORCES_FC2``::

     % phono3py --crystal --cf3 supercell-*o

     % phono3py --crystal --cf2 supercell_fc2-*o

   ``disp_fc3.yaml`` and ``disp_fc2.yaml`` are used to create ``FORCES_FC3`` and
   ``FORCES_FC2``, therefore they must exist in current directory.

4) Calculate 3rd and 2nd order force constants in files ``fc3.hdf5`` and ``fc2.hdf5``::

      % phono3py --crystal --dim="2 2 2" --dim-fc2="4 4 4" --sym-fc

   ``--sym-fc`` is used to symmetrize second- and third-order force constants.

5) Thermal conductivity calculation::

     % phono3py --crystal --fc3 --fc2 --dim="2 2 2" --dim-fc2="4 4 4" --mesh="20 20 20" --br

   ``--br`` invokes the Relaxation Time Approximation.
   Add ``--isotope`` for isotope scattering.
   Check the effect of ``--nac`` for polar systems.
   Carefully test the convergence with respect to ``--mesh``!
