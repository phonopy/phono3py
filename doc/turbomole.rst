.. _turbomole_interface:

TURBOMOLE & phono3py calculation
================================

The riper module of TURBOMOLE can be used to study periodic structures.
An example for TURBOMOLE is found in the ``example/Si-TURBOMOLE`` directory.

To invoke the TURBOMOLE interface, ``--turbomole`` option has to be always
specified::

   % phono3py --turbomole [options] [arguments]

When the file name of the unit cell is different from the default one
(see :ref:`default_unit_cell_file_name_for_calculator`), ``-c`` option
is used to specify the file name. TURBOMOLE unit cell file parser used in
phono3py is the same as that in phonopy. It reads a limited number of
keywords that are documented in the phonopy web site
(http://phonopy.github.io/phonopy/turbomole.html#turbomole-interface).

.. _turbomole_workflow:

Workflow
---------

In the example Si-TURBOMOLE, the TURBOMOLE input file is ``control``.
This is the default file name for the TURBOMOLE interface,
so the ``-c control`` parameter is not needed.

1) Create supercells with displacements (2x2x2 conventional cell for
   3rd order FC and 3x3x3 conventional cell for 2nd order FC)

   ::

      % phono3py --turbomole --dim="2 2 2" --dim-fc2="3 3 3" -d

   111 supercell directories (``supercell-00xxx``) for the third order
   force constants are created. In addition, one supercell directory
   (``supercell_fc2-00001``) is created for the second order
   force constants.

2) Complete TURBOMOLE inputs need to be prepared manually in the subdirectories.

   Note that supercells with displacements must not be relaxed in the
   force calculations, because atomic forces induced by a small atomic
   displacement are what we need for the phonon calculation. To get accurate
   forces, $scfconv should be 10. Phono3py includes this data group automatically
   in the ``control`` file. You also need to choose a k-point mesh for the force
   calculations. TURBOMOLE data group $riper may need to be adjusted to improve
   SCF convergence (see example files in subdirectory supercell-00001 for
   further details)

   Then, TURBOMOLE supercell calculations are executed to obtain forces on
   atoms, e.g., as follows::

     % riper > supercell-00001.out

3) Collect forces in ``FORCES_FC3`` and ``FORCES_FC2``::

     % phono3py --turbomole --cf3 supercell-*

     % phono3py --turbomole --cf2 supercell_fc2-*

   ``disp_fc3.yaml`` and ``disp_fc2.yaml`` are used to create ``FORCES_FC3`` and
   ``FORCES_FC2``, therefore they must exist in current directory. The Si-TURBOMOLE
   example contains pre-calculated force files.

4) Calculate 3rd and 2nd order force constants in files ``fc3.hdf5`` and ``fc2.hdf5``::

      % phono3py --turbomole --dim="2 2 2" --dim-fc2="3 3 3" --sym-fc

   ``--sym-fc`` is used to symmetrize second- and third-order force constants.

5) Thermal conductivity calculation::

     % phono3py --turbomole --primitive-axis="0 1/2 1/2  1/2 0 1/2  1/2 1/2 0" --fc3 --fc2 --dim="2 2 2" --dim-fc2="3 3 3" --mesh="20 20 20" --br

   ``--primitive-axis`` is used to get the results for the primitive 2-atom cell
   ``--br`` invokes the Relaxation Time Approximation.
   Carefully test the convergence with respect to ``--mesh``!
