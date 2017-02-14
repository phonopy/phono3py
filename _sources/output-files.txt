.. _output_files:

Output files
============

.. contents::
   :depth: 3
   :local:

The calculation results are written into files. Mostly the data are
stored in HDF5 format, therefore how to read the data
from HDF5 files is also shown.

Intermediate text files
------------------------

``disp_fc3.yaml``
^^^^^^^^^^^^^^^^^^

This is created with ``-d`` option. See :ref:`create_displacements_option`.

``disp_fc2.yaml``
^^^^^^^^^^^^^^^^^^

This is created with ``-d`` option with ``--dim_fc2`` option. See
:ref:`dim_fc2_option`.


``FORCES_FC3``
^^^^^^^^^^^^^^^

This is created with ``--cf3`` option. See :ref:`cf3_option`.

``FORCES_FC2``
^^^^^^^^^^^^^^^

This is created with ``--cf2`` option. See :ref:`cf2_option` and
:ref:`dim_fc2_option`.


HDF5 files
-------------

``kappa-*.hdf5``
^^^^^^^^^^^^^^^^^

See the detail at :ref:`kappa_hdf5_file`.

``fc3.hdf5``
^^^^^^^^^^^^^

Third order force constants are stored in :math:`\mathrm{eV}/\mathrm{\AA}^3`.

``fc2.hdf5``
^^^^^^^^^^^^^

Second order force constants are stored in
:math:`\mathrm{eV}/\mathrm{\AA}^3`.

``gamma-*.hdf5``
^^^^^^^^^^^^^^^^^

Imaginary parts of self energies at harmonic phonon frequencies
(:math:`\Gamma_\lambda(\omega_\lambda)` = half linewidths) are stored in
THz. See :ref:`write_gamma_option`.

``gamma_detail-*.hdf5``
^^^^^^^^^^^^^^^^^^^^^^^^

Q-point triplet contributions to imaginary parts of self energies at
phonon frequencies (half linewidths) are stored in THz.  See
:ref:`write_detailed_gamma_option`.

Simple text file 
-----------------

``gammas-*.dat``
^^^^^^^^^^^^^^^^^

Imaginary parts of self energies with respect to frequency 
:math:`\Gamma_\lambda(\omega)` are stored in THz. See :ref:`ise_option`.

``jdos-*.dat``
^^^^^^^^^^^^^^^

Joint densities of states are stored in Thz. See :ref:`jdos_option`.

``linewidth-*.dat``
^^^^^^^^^^^^^^^^^^^^

Linewidths (FWHM) at temperatures are stored in THz. See :ref:`lw_option`.

