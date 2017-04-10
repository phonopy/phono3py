.. _command_options:

Command options and setting tags
=================================

Command-user-interface of phono3py is operated with a variety of
command options. Here those command options are explained.

Using the latest phono3py and phonopy, a configuration file with
setting tags can be used instead of and together with the command
options. The setting tags are mostly equivalent to the most command
options, but when both are set simultaneously, the command options are
preferred. An example of configuration (e.g., saved in a file
``setting.conf``) is as follow::

   DIM = 2 2 2
   DIM_FC2 = 4 4 4
   PRIMITIVE_AXIS = 0 1/2 1/2 1/2 0 1/2 1/2 1/2 0
   MESH = 11 11 11
   BTERTA = .TRUE.
   NAC = .TRUE.
   READ_FC2 = .TRUE.
   READ_FC3 = .TRUE.
   CELL_FILENAME = POSCAR-unitcell

where the setting tag names are case insensitive. This is run by

::

   % phono3py [comannd options] setting.conf

.. contents::
   :depth: 2
   :local:

Calculator interface
---------------------

``-c``: Unit cell filename
~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``CELL_FILENAME``)

::

   % phono3py -c POSCAR-unitcell ... (many options)

``--pwscf``: PWSCF (Quantum espresso) interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using this option, PWSCF interface is invoked.
See the detail at :ref:`pwscf_interface`.

``--crystal``: CRYSTAL interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using this option, CRYSTAL interface is invoked.
See the detail at :ref:`crystal_interface`.

Force constants
----------------

.. _create_displacements_option:

``-d``: Create displacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``CREATE_DISPLACEMENTS``)

Supercell with displacements are created. Using with ``--amplitude``
option, atomic displacement distances are controlled. With this
option, files for supercells with displacements and ``disp_fc3.yaml``
file are created.

``--amplitude``: Amplitude of displacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``DISPLACEMENT_DISTANCE``)

Displacement distance. The default value depends on calculator. See
:ref:`default_displacement_distance_for_calculator`.

``--dim``: Supercell dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``DIM``)

Supercell size is specified. See the
detail at http://atztogo.github.io/phonopy/setting-tags.html#dim .

.. _dim_fc2_option:

``--dim_fc2``: Supercell dimension for 2nd order force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``DIM_FC2``)

A larger and different supercell size for 2nd order force constants
than that for 3rd order force constants can be specified with this
option. Often interaction between a pair of atoms has longer range in
real space than interaction among three atoms. Therefore to reduce
computational demand, choosing larger supercell size only for 2nd
order force constants may be a good idea.

Using this option with ``-d`` option, the structure files
(e.g. ``POSCAR_FC2-xxxxx`` or equivalent files for the other
interfaces) and ``disp_fc2.yaml`` are created. These are used to
calculate 2nd order force constants for the larger supercell size and
these force calculations have to be done in addition to the usual
force calculations for 3rd order force constants. 

::

   phono3py -d --dim="2 2 2" --dim_fc2="4 4 4" -c POSCAR-unitcell

After the force calculations, ``--cf2`` option is used to create
``FORCES_FC2``.

::

   phono3py --cf2 disp-{001,002}/vasprun.xml

To calculate 2nd order force constants for the larger supercell size,
``FORCES_FC2`` and ``disp_fc2.yaml`` are necessary. Whenever running
phono3py for the larger 2nd order force constants, ``--dim_fc2``
option has to be specified. ``fc2.hdf5`` created as a result of
running phono3py contains the 2nd order force constants with
larger supercell size. The filename is the same as that created in the
usual phono3py run without ``--dim_fc2`` option.

::

   phono3py --dim="2 2 2" --dim_fc2="4 4 4" -c POSCAR-unitcell ... (many options)

.. _pa_option:

``--pa``, ``--primitive_axis``: Transformation matrix to primitive cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``PRIMITIVE_AXIS``)

Transformation matrix from a non-primitive cell to the primitive
cell. See phonopy ``PRIMITIVE_AXIS`` tag (``--pa`` option) at
http://atztogo.github.io/phonopy/setting-tags.html#primitive-axis

``--fc2``: Read 2nd order force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``READ_FC2``, ``.TRUE.`` or ``.FALSE.``)

Read 2nd order force constants from ``fc2.hdf5``.

``--fc3``: Read 3nd order force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``READ_FC3``, ``.TRUE.`` or ``.FALSE.``)

Read 3rd order force constants from ``fc3.hdf5``.

``--sym_fc2``, ``--sym_fc3r``, ``--tsym``: Symmetries force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tags: ``SYMMETRIZE_FC2``, ``.TRUE.`` or ``.FALSE.``)
(Setting tags: ``SYMMETRIZE_FC3``, ``.TRUE.`` or ``.FALSE.``)
(Setting tags: ``TRANSLATION``, ``.TRUE.`` or ``.FALSE.``)

These are used to symmetrize second- and third-order force
constants. ``--sym_fc2`` and ``--sym_fc3r`` symmetrize those in real
space by the index exchange, respectively, and ``--tsym`` symmetrizes
by the translational invariance, respectively.

..
   ``--sym_fc3q`` symmetrizes third-order force constants in normal
   coordinates by the index exchange.

When those force constants are not read from the hdf5 files,
symmetrized force constants in real space are written into those hdf5
files.

``--cf3``: Create ``FORCES_FC3``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is used to create ``FORCES_FC3``. ``disp_fc3.yaml`` has to be
located at the current directory.

::

   % phono3py --cf3 disp-{00001..00755}/vasprun.xml

.. _cf3_file_option:

``--cf3_file``: Create ``FORCES_FC3`` from file name list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is used to create ``FORCES_FC3`` from a file name
list. ``disp_fc3.yaml`` has to be located at the current directory.

::

   % phono3py --cf3 phono3py --cf3_file file_list.dat

where ``file_list.dat`` contains file names that can be recognized
from the current directory and is expected to be like::

  disp-00001/vasprun.xml
  disp-00002/vasprun.xml
  disp-00003/vasprun.xml
  disp-00004/vasprun.xml
  ...

The order of the file names is important. This option may be useful
to be used together with ``--cutoff_pair`` option.

.. _cf2_option:

``--cf2``: Create ``FORCES_FC2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is used to create ``FORCES_FC2``. ``disp_fc2.yaml`` has to be
located at the current directory. This is optional. ``FORCES_FC2`` is
necessary to run with ``--dim_fc2`` option.

::

   % phono3py --cf2 disp_fc2-{00001..00002}/vasprun.xml

``--cutoff_fc3`` or ``--cutoff_fc3_distance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``CUTOFF_FC3_DISTANCE``)

This option is **not** used to reduce number of supercells with
displacements, but this option is used to set zero in elements of
given third-order force constants. The zero elements are selected by
the condition that any pair-distance of atoms in each atom triplet is
larger than the specified cut-off distance.

If one wants to reduce number of supercells, the first choice is to
reduce the supercell size and the second choice is using
``--cutoff_pair`` option.

.. _cutoff_pair_option:

``--cutoff_pair`` or ``--cutoff_pair_distance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``CUTOFF_PAIR_DISTANCE``)

This option is only used together with ``-d`` option.

A cutoff pair-distance in a supercell is used to reduce the number of
necessary supercells with displacements to obtain third order force
constants. As the drawback, a certain number of
third-order-force-constants elements are abandoned or computed with
less numerical accuracy. More details are found in the following link:

.. toctree::
   :maxdepth: 1

   cutoff-pair

Reciprocal space sampling mesh and grid points
-----------------------------------------------

``--mesh``: Sampling mesh
~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``MESH`` or ``MESH_NUMBERS``)

Phonon triples are chosen on the grid points on the sampling mesh
specified by this option. This mesh is made along reciprocal
axes and is always Gamma-centered.

..
   ``--md``
   ~~~~~~~~~

   Divisors of mesh numbers. Another sampling mesh is used to calculate
   phonon lifetimes. :math:`8\times 8\times 8` mesh is used for the
   calculation of phonon lifetimes when it is specified, e.g.,
   ``--mesh="11 11 11" --md="2 2 2"``.

.. _gp_option:

``--gp``: Grid points by their ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``GRID_POINTS``)

Grid points where imaginary part of self energy is calculated are
specified. Indices of grid points are specified by space or comma
(``,``) separated numbers. The mapping table between grid points to its
indices is obtained by running with ``--loglevel=2`` option.

::

   % phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --fc3 --fc2 --br --write_gamma --gp="0 1 2 3 4 5"

where ``--gp="0 1 2 3 4 5"`` can be also written
``--gp="0,1,2,3,4,5"``. There is a similar option as this option,
:ref:`--ga option <ga_option>`.

``--ga`` option may be also useful when a workload of thermal
conductivity calculation is expected to be distributed into different
computer nodes.

.. toctree::
   :maxdepth: 1

   workload-distribution


.. _ga_option:

``--ga``: Grid points by address with three integer values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``GRID_ADDRESSES``)

This option is used to specify grid points like ``--gp`` option but in
the different way. For example with ``--mesh="16 16 16"``, a q-point
of (0.5, 0.5, 0.5) is given by ``--ga="8 8 8"``. The values have to be
integers. If you want to specify the point on a path, ``--ga="0 0 0 1
1 1 2 2 2 3 3 3 ..."``, where each three values are recogninzed as a
grid point. The grid points given by ``--ga`` option are translated to
grid point indices as given by ``--gp`` option, and the values given
by ``--ga`` option will not be shown in log files.

.. _wgp_option:

``--wgp``: Write grid point information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Irreducible grid point indices are written into
``ir_grid_points.yaml``. This information may be used when we want to
calculate imaginary part of self energy at each grid point in
conjunction with :ref:`--gp option <gp_option>`.
``grid_address-mxxx.hdf5`` is also written. This file contains all the
grid points and their grid addresses in integers. Q-points
corresponding to grid points are calculated divided these integers by
sampling mesh numbers for respective reciprocal axes.

``--stp``: Show number of triplets to be calculated for each grid point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numbers of q-point triplets to be calculated for irreducible grid
points for specified sampling mesh numbers are shown. This can be used
to estimate how large a calculation is. Only those for specific grid
points are shown by using with ``--gp`` or ``--ga`` option.

Brillouin zone integration
---------------------------

``--thm``: Tetrahedron method (default choice)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``TETRAHEDRON``, ``.TRUE.`` or ``.FALSE.``)

Tetrahedron method is used for calculation of imaginary part of self
energy. This is the default option. Therefore it is not necessary to
specify this unless both results by tetrahedron method and
smearing method in one time execution are expected.

``--sigma``: Smearing method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``SIGMA``)

:math:`\sigma` value of Gaussian function for smearing when
calculating imaginary part of self energy. See the detail at
:ref:`brillouinzone_sum`.

Multiple :math:`\sigma` values are also specified by space separated
numerical values. This is used when we want to test several
:math:`\sigma` values simultaneously.

Physical properties
--------------------

``--br``: Thermal conductivity with relaxation time approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``BTERTA``, ``.TRUE.`` or ``.FALSE.``)

Run calculation of lattice thermal conductivity tensor with the single
mode relaxation time approximation (RTA) and linearized phonon
Boltzmann equation. Without specifying ``--gp`` (or ``--ga``) option,
all necessary phonon lifetime calculations for grid points are
sequentially executed and then thermal conductivity is calculated
under RTA. The thermal conductivity and many related properties are
written into ``kappa-mxxx.hdf5``. 

With ``--gp`` (or ``--ga``) option,
phonon lifetimes on the specified grid points are calculated. To save
the results, ``--write_gamma`` option has to be specified and the
physical properties belonging to the grid
points are written into ``kappa-mxxx-gx(-sx).hdf5``.

``--isotope``: Phonon-isotope scattering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``ISOTOPE``, ``.TRUE.`` or ``.FALSE.``)

Phonon-isotope scattering is calculated based on the formula by
Shin-ichiro Tamura, Phys. Rev. B, **27**, 858 (1983). Mass variance
parameters are read from database of the natural abundance data for
elements, which refers Laeter *et al.*, Pure Appl. Chem., **75**, 683
(2003).

::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --isotope

``--mass_variances`` or ``--mv``: Parameter for phonon-isotope scattering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``MASS_VARIANCES``)

This option is used to include isotope effect by reading specified
mass variance parameters. For example of GaN, this may be set like
``--mv="1.97e-4 1.97e-4 0 0"``. The number of elements has to
correspond to the number of atoms in the primitive cell.

Isotope effect to thermal conductivity may be checked first running
without isotope calculation::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br

Then running with isotope calculation::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br \
     --read_gamma --mv="1.97e-4 1.97e-4 0 0"

In the result hdf5 file, currently isotope scattering strength is not
written out, i.e., ``gamma`` is still imaginary part of self energy of
ph-ph scattering.

``--boundary_mfp``, ``--bmfp``: Very simple phonon-boundary scattering model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``BOUNDARY_MFP``)

A most simple boundary scattering treatment is
implemented. :math:`v_g/L` is just used as the scattering rate, where
:math:`v_g` is the group velocity and :math:`L` is the boundary mean
free path. The value is given in micrometre. The default value, 1
metre, is just used to avoid divergence of phonon lifetime and the
contribution to the thermal conducitivity is considered negligible.

.. _cf3_option:

``--tmax``, ``--tmin``, ``--tstep``, ``--ts``: Temperatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``TMAX``, ``TMIN``, ``TSTEP``, ``TEMPERATURES``)


Temperatures at equal interval are specified by ``--tmax``,
``--tmin``, ``--tstep``. See phonopy ``TMAX``, ``TMIN``, ``TSTEP``
tags (``--tmax``, ``--tmin``, ``--tstep`` options) at
http://atztogo.github.io/phonopy/setting-tags.html#tprop-tmin-tmax-tstep .

::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
     -c POSCAR-unitcell --br --tmin=100 --tmax=1000 --tstep=50


Specific temperatures are given by ``--ts``.

::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
     -c POSCAR-unitcell --br --ts="200 300 400"

``--nac``: Non-analytical term correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``NAC``, ``.TRUE.`` or ``.FALSE.``)

Non-analytical term correction for harmonic phonons. Like as phonopy,
``BORN`` file has to be put on the same directory. Always the default
value of unit conversion factor is used even if it is written in the
first line of ``BORN`` file.

``--q_direction``: Direction for non-analytical term correction at :math:`\mathbf{q}\rightarrow \mathbf{0}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``Q_DIRECTION``)

This is used with ``--nac`` to specify the direction to polarize in
reciprocal space. See the detail at
http://atztogo.github.io/phonopy/setting-tags.html#q-direction .

.. _write_gamma_option:

``--write_gamma``
~~~~~~~~~~~~~~~~~

(Setting tag: ``WRITE_GAMMA``, ``.TRUE.`` or ``.FALSE.``)

Imaginary parts of self energy at harmonic phonon frequencies
:math:`\Gamma_\lambda(\omega_\lambda)` are written
into file in hdf5 format.  The result is written into
``kappa-mxxx-gx(-sx).hdf5`` or ``kappa-mxxx-gx-bx(-sx).hdf5`` with
``--bi`` option. With ``--sigma`` option, ``-sx`` is inserted in front
of ``.hdf5``.

.. _read_gamma_option:

``--read_gamma``
~~~~~~~~~~~~~~~~

(Setting tag: ``READ_GAMMA``, ``.TRUE.`` or ``.FALSE.``)

Imaginary parts of self energy at harmonic phonon frequencies
:math:`\Gamma_\lambda(\omega_\lambda)`
are read from ``kappa`` file in hdf5 format.  Initially the usual
result file of ``kappa-mxxx(-sx).hdf5`` is searched. Unless it is
found, it tries to read ``kappa`` file for each grid point,
``kappa-mxxx-gx(-sx).hdf5``. Then, similarly,
``kappa-mxxx-gx(-sx).hdf5`` not found,
``kappa-mxxx-gx-bx(-sx).hdf5`` files for band indices are searched.

.. _write_detailed_gamma_option:

``--write_gamma_detail``
~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``WRITE_GAMMA_DETAIL``, ``.TRUE.`` or ``.FALSE.``)

Each q-point triplet contribution to imaginary part of self energy is
written into ``gamma_detail-mxxx-gx(-sx).hdf5`` file. Be careful that
this is large data.

In the output file in hdf5, following keys are used to extract the
detailed information.

====================================== =============================================================================================================================================
gamma_detail for ``--ise``             (temperature, sampling frequency point, symmetry reduced set of triplets at a grid point, band1, band2, band3) in THz (without :math:`2\pi`)
gamma_detail for ``--lw`` and ``--br`` (temperature, symmetry reduced set of triplets at a grid point, band1, band2, band3) in THz (without :math:`2\pi`)
mesh                                   Numbers of sampling mesh along reciprocal axes.
frequency_point for ``--ise``          Sampling frequency points in THz (without :math:`2\pi`), i.e., :math:`\omega` in :math:`\Gamma_\lambda(\omega)`
temperature                            (temperature,), Temperatures in K
triplet                                (symmetry reduced set of triplets at a grid point, 3), Triplets are given by the grid point indices (see below).
weight                                 (symmetry reduced set of triplets at a grid point,), Weight of each triplet to imaginary part of self energy
====================================== =============================================================================================================================================

Q-points corresponding to grid point indices are calculated from
grid addresses and sampling mesh numbers given in
``grid_address-mxxx.hdf5`` that is obtained by ``--wgp`` option. A
python script to obtain q-point triplets is shown below.

.. code-block:: python

    import h5py
    import numpy as np
    
    f = h5py.File("gamma_detail-mxxx-gx.hdf5")
    g = h5py.File("grid_address-mxxx.hdf5")
    grid_address = f['grid_address'][:]
    triplets = g['triplet'][:]
    mesh = f['mesh'][:]
    q = grid_address[triplets] / np.array(mesh, dtype='double')

Imaginary part of self energy or linewidth/2 is recovered by the
following script:

.. code-block:: python

    import h5py
    import numpy as np
    
    f = h5py.File("gamma_detail-mxxx-gx.hdf5")
    temp = 30 # index of temperature
    gamma_tp = f['gamma_detail'][:].sum(axis=-1).sum(axis=-1)
    weight = f['weight'][:]
    gamma = np.dot(weight, gamma_tp[temp])

For example, for ``--lw`` or ``--br``, this ``gamma`` gives
:math:`\Gamma_\lambda(\omega_\lambda)` of the band indices at the grid
point indicated by :math:`\lambda` at the temperature of index 30. If
any bands are degenerated, those ``gamma`` in ``kappa--mxxx-gx(-sx).hdf5``
or ``gamma--mxxx-gx(-sx).hdf5`` type file are averaged, but the ``gamma``
obtained here in this way are not symmetrized. Apart from this
symmetrization, the values must be equivalent between them.

..
   ``--write_amplitude``
   ~~~~~~~~~~~~~~~~~~~~~~

   Interaction strengths of triplets are written into file in hdf5
   format. This file can be huge and usually it is not recommended to
   write it out.

.. _ise_option:

``--ise``: Imaginary part of self energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``IMAG_SELF_ENERGY``, ``.TRUE.`` or ``.FALSE.``)

Imaginary part of self energy :math:`\Gamma_\lambda(\omega)` is
calculated with respect to :math:`\omega`. The output is written to
``gammas-mxxxx-gx(-sx)-tx-bx.dat`` in THz (without :math:`2\pi`) with
respect to frequency in THz (without :math:`2\pi`).

::

   % phono3py --fc3 --fc2 --dim="2 2  2" --mesh="16 16 16" -c POSCAR-unitcell \
     --nac --q_direction="1 0 0" --gp=0 --ise --bi="4 5, 6"

.. _lw_option:

``--lw``: Line width
~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``LINEWIDTH``, ``.TRUE.`` or ``.FALSE.``)

Linewidth :math:`2\Gamma_\lambda(\omega_\lambda)` is calculated with
respect to temperature. The output is written to
``linewidth-mxxxx-gx(-sx)-bx.dat`` in THz (without :math:`2\pi`).

::

   % phono3py --fc3 --fc2 --dim="2 2  2" --mesh="16 16 16" -c POSCAR-unitcell \
     --nac --q_direction="1 0 0" --gp=0 --lw --bi="4 5, 6"


.. _jdos_option:

``--jdos``: Joint density of states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``JOINT_DOS``, ``.TRUE.`` or ``.FALSE.``)

Two classes of joint density of states (JDOS) are calculated. The
result is written into ``jdos-mxxxxxx-gx(-sx).dat`` in
:math:`\text{THz}^{-1}` (without :math:`(2\pi)^{-1}`) with
respect to frequency in THz (without :math:`2\pi`). The first
column is the frequency, and the second and third columns are the
values given as follows, respectively,

.. math::
   
   &D_2^{(1)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda_1,\lambda_2}
   \left[\delta(\omega+\omega_{\lambda_1}-\omega_{\lambda_2}) +
   \delta(\omega-\omega_{\lambda_1}+\omega_{\lambda_2}) \right], \\
   &D_2^{(2)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda_1,\lambda_2}\delta(\omega-\omega_{\lambda_1}
   -\omega_{\lambda_2}).

::

   % phono3py --fc2 --dim="2 2 2" -c POSCAR-unitcell --mesh="16 16 16" \
     --nac --jdos --ga="0 0 0  8 8 8"

When temperatures are specified, two classes of weighted JDOS are
calculated. The result is written into
``jdos-mxxxxxx-gx(-sx)-txxx.dat`` in :math:`\text{THz}^{-1}` (without
:math:`(2\pi)^{-1}`) with respect to frequency in THz
(without :math:`2\pi`). In the file name, ``txxx`` shows the
temperature. The first column is the frequency, and the second and
third columns are the values given as follows, respectively,

.. math::

   &N_2^{(1)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda'\lambda''} \Delta(-\mathbf{q}+\mathbf{q}'+\mathbf{q}'')
   (n_{\lambda'} - n_{\lambda''}) [ \delta( \omega + \omega_{\lambda'} -
   \omega_{\lambda''}) - \delta( \omega - \omega_{\lambda'} +
   \omega_{\lambda''})], \\
   &N_2^{(2)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda'\lambda''} \Delta(-\mathbf{q}+\mathbf{q}'+\mathbf{q}'')
   (n_{\lambda'}+ n_{\lambda''}+1) \delta( \omega - \omega_{\lambda'} -
   \omega_{\lambda''}).

::

   % phono3py --fc2 --dim="2 2 2" -c POSCAR-unitcell --mesh="16 16 16" \
     --nac --jdos --ga="0 0 0  8 8 8" --ts=300

``--num_freq_points``, ``--freq_pitch``: Sampling frequency for distribution functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``NUM_FREQUENCY_POINTS``)

For spectrum like calculations of imaginary part of self energy and
JDOS, number of sampling frequency points is controlled by
``--num_freq_points`` or ``--freq_pitch``.

.. _bi_option:

``--bi``: Specific band index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``BAND_INDICES``)

Specify band indices. The output file name will be, e.g.,
``gammas-mxxxxxx-gxx(-sx)-bx.dat`` where ``bxbx...`` shows the band indices
used to be averaged. The calculated values at indices separated by
space are averaged, and those separated by comma are separately
calculated.

::

   % phono3py --fc3 --fc2 --dim="2 2 2" --mesh="16 16 16" \
     -c POSCAR-unitcell --nac --gp="34" --bi="4 5, 6"

.. _full_pp_option:

``--full_pp``: Calculate all elements of phonon-phonon interaction strength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``FULL_PP``, ``.TRUE.`` or ``.FALSE.``)

After version 1.10.5, for RTA thermal conductivity calculation with
using the linear tetrahedron method, only necessary part of
phonon-phonon interaction strengh among phonons,
:math:`\bigl|\Phi_{-\lambda\lambda'\lambda''}\bigl|^2`, is
calculated due to delta functions in calculation of
:math:`\Gamma_\lambda(\omega)`,

.. math::

   \Gamma_\lambda(\omega) = \frac{18\pi}{\hbar^2}
    \sum_{\lambda' \lambda''}
    \bigl|\Phi_{-\lambda\lambda'\lambda''}\bigl|^2 
    \left\{(n_{\lambda'}+ n_{\lambda''}+1) 
     \delta(\omega-\omega_{\lambda'}-\omega_{\lambda''}) \right.
     + (n_{\lambda'}-n_{\lambda''})
    \left[\delta(\omega+\omega_{\lambda'}-\omega_{\lambda''})
   - \left. \delta(\omega-\omega_{\lambda'}+\omega_{\lambda''})
   \right]\right\}.

But specifying this option, full elements of phonon-phonon interaction
strengh among phonons are calculated and averaged phonon-phonon
interaction strength (:math:`P_{\mathbf{q}j}`) is also given.

``--ave_pp``: Use averaged phonon-phonon interaction strength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``USE_AVE_PP``, ``.TRUE.`` or ``.FALSE.``)

Averaged phonon-phonon interaction strength (:math:`P_{\mathbf{q}j}`)
is used to calculate imaginary part of self energy in thermal
conductivity calculation. This option works
only when ``--read_gamma`` and ``--br`` options are activated where
the averaged phonon-phonon interaction that is read from
``kappa-mxxxxx.hdf5`` file is used if it exists in the file. Therefore the
averaged phonon-phonon interaction has to be stored before using this
option (see :ref:`full_pp_option`). The calculation result
**overwrites** ``kappa-mxxxxx.hdf5`` file. Therefore to use this
option together with ``-o`` option is strongly recommended.

First, run full conductivity calculation,

::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br

Then

::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br \
     --read_gamma --ave_pp -o ave_pp

``--const_ave_pp``: Use constant phonon-phonon interaction strength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``CONSTANT_AVERAGED_PP_INTERACTION``, ``.TRUE.`` or ``.FALSE.``)

Averaged phonon-phonon interaction (:math:`P_{\mathbf{q}j}`) is
replaced by this constant value in thermal conductivity
calculation. This option works only when ``--br`` options are
activated. Therefore third-order force constants are not necessary to
input. The physical unit of the value is :math:`\text{eV}^2`.

::
   
   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br \
     --const_ave_pp=1e-10

``--gruneisen``: Mode-Gruneisen parameter from 3rd order force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Setting tag: ``GRUNEISEN``, ``.TRUE.`` or ``.FALSE.``)

Mode-Gruneisen-parameters are calculated from fc3.

Mesh sampling mode::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="16 16 16" 
     -c POSCAR-unitcell --nac --gruneisen

Band path mode::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v \
     -c POSCAR-unitcell --nac --gruneisen --band="0 0 0  0 0 1/2"

