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

The following files are not compatible with phonopy. But phonopy's
``FORCE_SETS`` file can be created using phono3py command options from
the following files. See the detail at :ref:`file_format_compatibility`.

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

.. _fc3_hdf5_file:

``fc3.hdf5``
^^^^^^^^^^^^^

Third order force constants (in real space) are stored in
:math:`\mathrm{eV}/\text{Angstrom}^3`.

In phono3py, this is stored in the numpy array ``dtype='double'`` and
``order='C'`` in the shape of::

   (num_atom, num_atom, num_atom, 3, 3, 3)

against :math:`\Phi_{\alpha\beta\gamma}(l\kappa, l'\kappa',
l''\kappa'')`. The first three ``num_atom`` are the atom indices in supercell
corresponding to :math:`l\kappa`, :math:`l'\kappa'`,
:math:`l''\kappa''`, respectively. The last three elements are the Cartesian
coordinates corresponding to :math:`\alpha`, :math:`\beta`,
:math:`\gamma`, respectively.

If you want to import a supercell structure and its fc3, you may
suffer from matching its atom index between the supercell and an
expected unit cell. This may be easily dealt with by letting phono3py
see your supercell as the unit cell (e.g., ``POSCAR``,
``unitcell.in``, etc) and find the unit (primitive) cell using
:ref:`--pa option <pa_option>`. For example, let us assume your
supercell is the 2x2x2 multiples of your unit cell that has no
centring, then your ``--pa`` setting will be ``1/2 0 0 0 1/2 0 0 1/2
0``. If your unit cell is a conventional unit cell and has a centring,
e.g., the face centring,

.. math::

   (\mathbf{a}_\text{p}, \mathbf{b}_\text{p}, \mathbf{c}_\text{p}) =
   (\mathbf{a}_\text{s}, \mathbf{b}_\text{s}, \mathbf{c}_\text{s})
   \begin{pmatrix}
   \frac{{1}}{2} & 0 & 0 \\
   0 & \frac{{1}}{2} & 0 \\
   0 & 0 & \frac{{1}}{2}
   \end{pmatrix}
   \begin{pmatrix}
   0 & \frac{{1}}{2} & \frac{{1}}{2} \\
   \frac{{1}}{2} & 0 & \frac{{1}}{2} \\
   \frac{{1}}{2} & \frac{{1}}{2} & 0
   \end{pmatrix} =
   (\mathbf{a}_\text{s}, \mathbf{b}_\text{s}, \mathbf{c}_\text{s})
   \begin{pmatrix}
   0 & \frac{{1}}{4} & \frac{{1}}{4} \\
   \frac{{1}}{4} & 0 & \frac{{1}}{4} \\
   \frac{{1}}{4} & \frac{{1}}{4} & 0
   \end{pmatrix}.

So what you have to set is ``--pa="0 1/4 1/4  1/4 0 1/4  1/4 1/4 0"``.

.. _fc2_hdf5_file:

``fc2.hdf5``
^^^^^^^^^^^^^

Second order force constants are stored in
:math:`\mathrm{eV}/\text{Angstrom}^2`.

In phono3py, this is stored in the numpy array ``dtype='double'`` and
``order='C'`` in the shape of::

   (num_atom, num_atom, 3, 3)

against :math:`\Phi_{\alpha\beta}(l\kappa, l'\kappa')`. More detail is
similar to the case for :ref:`fc3_hdf5_file`.

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
