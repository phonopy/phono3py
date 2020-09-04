.. _hdf5_howto:

How to read the results stored in hdf5 files
=============================================

.. contents::
   :depth: 3
   :local:

How to use HDF5 python library
-------------------------------

It is assumed that ``python-h5py`` is installed on the computer you
interactively use. In the following, how to see the contents of
``.hdf5`` files in the interactive mode of Python. The basic usage of
reading ``.hdf5`` files using ``h5py`` is found at `here
<http://docs.h5py.org/en/latest/high/dataset.html#reading-writing-data>`_.
Usually for running interactive python, ``ipython`` is recommended to
use but not the plain python. In the following example, an MgO result
of thermal conductivity calculation is loaded and thermal conductivity
tensor at 300 K is watched.

::


   In [1]: import h5py

   In [2]: f = h5py.File("kappa-m111111.hdf5")

   In [3]: list(f)
   Out[3]:
   ['frequency',
    'gamma',
    'group_velocity',
    'gv_by_gv',
    'heat_capacity',
    'kappa',
    'kappa_unit_conversion',
    'mesh',
    'mode_kappa',
    'qpoint',
    'temperature',
    'weight']

   In [4]: f['kappa'].shape
   Out[4]: (101, 6)

   In [5]: f['kappa'][:]
   Out[5]:
   array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
          [  2.11702476e+05,   2.11702476e+05,   2.11702476e+05,
             6.64531043e-13,   6.92618921e-13,  -1.34727352e-12],
          [  3.85304024e+04,   3.85304024e+04,   3.85304024e+04,
             3.52531412e-13,   3.72706406e-13,  -7.07290889e-13],
          ...,
          [  2.95769356e+01,   2.95769356e+01,   2.95769356e+01,
             3.01803322e-16,   3.21661793e-16,  -6.05271364e-16],
          [  2.92709650e+01,   2.92709650e+01,   2.92709650e+01,
             2.98674274e-16,   3.18330655e-16,  -5.98999091e-16],
          [  2.89713297e+01,   2.89713297e+01,   2.89713297e+01,
             2.95610215e-16,   3.15068595e-16,  -5.92857003e-16]])

   In [6]: f['temperature'][:]
   Out[6]:
   array([    0.,    10.,    20.,    30.,    40.,    50.,    60.,    70.,
             80.,    90.,   100.,   110.,   120.,   130.,   140.,   150.,
            160.,   170.,   180.,   190.,   200.,   210.,   220.,   230.,
            240.,   250.,   260.,   270.,   280.,   290.,   300.,   310.,
            320.,   330.,   340.,   350.,   360.,   370.,   380.,   390.,
            400.,   410.,   420.,   430.,   440.,   450.,   460.,   470.,
            480.,   490.,   500.,   510.,   520.,   530.,   540.,   550.,
            560.,   570.,   580.,   590.,   600.,   610.,   620.,   630.,
            640.,   650.,   660.,   670.,   680.,   690.,   700.,   710.,
            720.,   730.,   740.,   750.,   760.,   770.,   780.,   790.,
            800.,   810.,   820.,   830.,   840.,   850.,   860.,   870.,
            880.,   890.,   900.,   910.,   920.,   930.,   940.,   950.,
            960.,   970.,   980.,   990.,  1000.])

   In [7]: f['kappa'][30]
   Out[7]:
   array([  1.09089896e+02,   1.09089896e+02,   1.09089896e+02,
            1.12480528e-15,   1.19318349e-15,  -2.25126057e-15])

   In [8]: f['mode_kappa'][30, :, :, :].sum(axis=0).sum(axis=0) / weight.sum()
   Out[8]:
   array([  1.09089896e+02,   1.09089896e+02,   1.09089896e+02,
            1.12480528e-15,   1.19318349e-15,  -2.25126057e-15])

   In [9]: g = f['gamma'][30]

   In [10]: import numpy as np

   In [11]: g = np.where(g > 0, g, -1)

   In [12]: lifetime = np.where(g > 0, 1.0 / (2 * 2 * np.pi * g), 0)

.. _kappa_hdf5_file:

Details of ``kappa-*.hdf5`` file
---------------------------------

Files name, e.g. ``kappa-m323220.hdf5``, is determined by some
specific options. ``mxxx``, show the numbers of sampling
mesh. ``sxxx`` and ``gxxx`` appear optionally. ``sxxx`` gives the
smearing width in the smearing method for Brillouin zone integration
for phonon lifetime, and ``gxxx`` denotes the grid number. Using the
command option of ``-o``, the file name can be modified slightly. For
example ``-o nac`` gives ``kappa-m323220.nac.hdf5`` to
memorize the option ``--nac`` was used.

Currently ``kappa-*.hdf5`` file (not for the specific grid points)
contains the properties shown below.

mesh
^^^^^

(Versions 1.10.11 or later)

The numbers of mesh points for reciprocal space sampling along
reciprocal axes, :math:`a^*, b^*, c^*`

frequency
^^^^^^^^^^

Phonon frequencies. The physical unit is THz, where THz
is in the ordinal frequency not the angular frequency.

The array shape is (irreducible q-point, phonon band).

.. _kappa_hdf5_file_gamma:

gamma
^^^^^^

Imaginary part of self energy. The physical unit is THz, where THz
is in the ordinal frequency not the angular frequency.

The array shape for all grid-points (irreducible q-points) is
(temperature, irreducible q-point, phonon band).

The array shape for a specific grid-point is
(temperature, phonon band).

Phonon lifetime (:math:`\tau_\lambda=1/2\Gamma_\lambda(\omega_\lambda)`) may
be estimated from ``gamma``. :math:`2\pi` has to be multiplied with
``gamma`` values in the hdf5 file to convert the unit of ordinal
frequency to angular frequency. Zeros in ``gamma`` values mean that
those elements were not calculated such as for three acoustic modes at
:math:`\Gamma` point. The below is the copy-and-paste from the
previous section to show how to obtain phonon lifetime in pico
second::

   In [8]: g = f['gamma'][30]

   In [9]: import numpy as np

   In [10]: g = np.where(g > 0, g, -1)

   In [11]: lifetime = np.where(g > 0, 1.0 / (2 * 2 * np.pi * g), 0)


gamma_isotope
^^^^^^^^^^^^^^

Isotope scattering of :math:`1/2\tau^\mathrm{iso}_\lambda`.
The physical unit is same as that of gamma.

The array shape is same as that of frequency.

group_velocity
^^^^^^^^^^^^^^^

Phonon group velocity, :math:`\nabla_\mathbf{q}\omega_\lambda`. The
physical unit is :math:`\text{THz}\cdot\text{Angstrom}`, where THz
is in the ordinal frequency not the angular frequency.

The array shape is (irreducible q-point, phonon band, 3 = Cartesian coordinates).

heat_capacity
^^^^^^^^^^^^^^

Mode-heat-capacity defined by

.. math::

    C_\lambda = k_\mathrm{B}
     \left(\frac{\hbar\omega_\lambda}{k_\mathrm{B} T} \right)^2
     \frac{\exp(\hbar\omega_\lambda/k_\mathrm{B}
     T)}{[\exp(\hbar\omega_\lambda/k_\mathrm{B} T)-1]^2}.

The physical unit is eV/K.

The array shape is (temperature, irreducible q-point, phonon band).

.. _output_kappa:

kappa
^^^^^^

Thermal conductivity tensor. The physical unit is W/m-K.

The array shape is (temperature, 6 = (xx, yy, zz, yz, xz, xy)).

.. _output_mode_kappa:

mode-kappa
^^^^^^^^^^^

Thermal conductivity tensors at k-stars (:math:`{}^*\mathbf{k}`):

.. math::

   \sum_{\mathbf{q} \in {}^*\mathbf{k}} \kappa_{\mathbf{q}j}.

The sum of this over :math:`{}^*\mathbf{k}` corresponding to
irreducible q-points divided by number of grid points gives
:math:`\kappa` (:ref:`output_kappa`), e.g.,::

   kappa_xx_at_index_30 = mode_kappa[30, :, :, 0].sum()/ weight.sum()

Be careful that until version 1.12.7, mode-kappa values were divided
by number of grid points.

The physical unit is W/m-K. Each tensor element is the sum of tensor
elements on the members of :math:`{}^*\mathbf{k}`, i.e., symmetrically
equivalent q-points by crystallographic point group and time reversal
symmetry.

The array shape is (temperature, irreducible q-point, phonon band, 6 =
(xx, yy, zz, yz, xz, xy)).


gv_by_gv
^^^^^^^^^

Outer products of group velocities for k-stars
(:math:`{}^*\mathbf{k}`) for each irreducible q-point and phonon band
(:math:`j`):

.. math::

   \sum_{\mathbf{q} \in {}^*\mathbf{k}} \mathbf{v}_{\mathbf{q}j} \otimes
   \mathbf{v}_{\mathbf{q}j}.

The physical unit is
:math:`\text{THz}^2\cdot\text{Angstrom}^2`, where THz is in the
ordinal frequency not the angular frequency.

The array shape is (irreducible q-point, phonon band, 6 = (xx, yy, zz,
yz, xz, xy)).

q-point
^^^^^^^^

Irreducible q-points in reduced coordinates.

The array shape is (irreducible q-point, 3 = reduced
coordinates in reciprocal space).

temperature
^^^^^^^^^^^^

Temperatures where thermal conductivities are calculated. The physical
unit is K.

weight
^^^^^^^

Weights corresponding to irreducible q-points. Sum of weights equals to
the number of mesh grid points.

ave_pp
^^^^^^^

Averaged phonon-phonon interaction in :math:`\text{eV}^2`,
:math:`P_{\mathbf{q}j}`:

.. math::

   P_{\mathbf{q}j} = \frac{1}{(3n_\mathrm{a})^2} \sum_{\lambda'\lambda''}
   |\Phi_{\lambda\lambda'\lambda''}|^2.

This is not going to be calculated in the RTA thermal coductivity
calculation mode by default. To calculate this, ``--full_pp`` option
has to be specified (see :ref:`full_pp_option`).

kappa_unit_conversion
^^^^^^^^^^^^^^^^^^^^^^

This is used to convert the physical unit of lattice thermal
conductivity made of ``heat_capacity``, ``group_velocity``, and
``gamma``, to W/m-K. In the single mode relaxation time (SMRT) method,
a mode contribution to the lattice thermal conductivity is given by

.. math::

   \kappa_\lambda = \frac{1}{V_0} C_\lambda \mathbf{v}_\lambda \otimes
   \mathbf{v}_\lambda \tau_\lambda^{\mathrm{SMRT}}.

For example, :math:`\kappa_{\lambda,{xx}}` is calculated by::

   In [1]: import h5py

   In [2]: f = h5py.File("kappa-m111111.hdf5")

   In [3]: kappa_unit_conversion = f['kappa_unit_conversion'][()]

   In [4]: weight = f['weight'][:]

   In [5]: heat_capacity = f['heat_capacity'][:]

   In [6]: gv_by_gv = f['gv_by_gv'][:]

   In [7]: gamma = f['gamma'][:]

   In [8]: kappa_unit_conversion * heat_capacity[30, 2, 0] * gv_by_gv[2, 0] / (2 * gamma[30, 2, 0])

   Out[8]:
   array([  1.02050241e+03,   1.02050241e+03,   1.02050241e+03,
            4.40486382e-15,   0.00000000e+00,  -4.40486382e-15])

   In [9]: f['mode_kappa'][30, 2, 0]
   Out[9]:
   array([  1.02050201e+03,   1.02050201e+03,   1.02050201e+03,
            4.40486209e-15,   0.00000000e+00,  -4.40486209e-15])

How to know grid point number corresponding to grid address
------------------------------------------------------------

Runngin with ``--write-gamma``, hdf5 files are written out file names
with grid point numbers such as ``kappa-m202020-g4200.hdf5``. You may
want to know the grid point number with given grid address. This is
done using ``get_grid_point_from_address`` as follows::

   In [1]: from phono3py.phonon3.triplets import get_grid_point_from_address

   In [2]: get_grid_point_from_address([0, 10, 10], [20, 20, 20])
   Out[2]: 4200

Here the first argument of this method is the grid address and the
second argument is the mesh numbers.
