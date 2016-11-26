.. _auxiliary_tools:

Auxiliary tools
===============

``kaccum``
-----------

Cumulative physical properties with respect to frequency or mean free
path are calculated using this command.

For example, cumulative thermal conductivity is defined by

.. math::

   \kappa^\text{c}(\omega) = 
    \int^\omega_0 \sum_\lambda
   \kappa_\lambda \delta(\omega_\lambda - \omega) d\omega

where :math:`\kappa_\lambda` of phono3py for single-mode RTA is given as

.. math::

   \kappa_\lambda =
   C_\lambda \mathbf{v}_\lambda \otimes \mathbf{v}_\lambda
   \tau_\lambda.

(The notations are found in http://arxiv.org/abs/1501.00691.)

How to use ``kaccum``
~~~~~~~~~~~~~~~~~~~~~

Let's computer lattice thermal conductivity of Si using an example
found in the example directory.

::

   % phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --sym_fc3r --sym_fc2 --tsym --br

Then using the output file, ``kappa-m191919.hdf5``, run ``kaccum`` as follows::

   % kaccum --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell kappa-m191919.hdf5 |tee kaccum.dat

Here ``--pa`` is optional. The definition of ``--pa`` option is same as
:ref:`pa_option`. ``POSCAR-unitcell`` is the unit cell filename that
is specified with ``-c`` option.
``kappa-m191919.hdf5`` is the output file of thermal conductivity
calculation, which is passed to ``kaccum`` as the first argument.

The format of the output is as follows: The first column gives
frequency in THz, and the second to seventh columns give the
cumulative lattice thermal conductivity of 6 elements, xx, yy, zz, yz,
xz, xy. The eighth to 13th columns give the derivatives. There are
sets of frequencies, which are separated by blank lines. Each set is
for a temperature. There are the groups corresponding to the number of
temperatures calculated.

To plot the output by gnuplot at temperature index 30 that may
correspond to 300 K,

::

   % gnuplot
   ...
   gnuplot> p "kaccum.dat" i 30 u 1:2 w l, "kaccum.dat" i 30 u 1:8 w l

The plot like below is displayed.

.. |i0| image:: Si-kaccum.png
        :width: 50%

|i0|

General option
~~~~~~~~~~~~~~~

``--pa``
^^^^^^^^^

See :ref:`pa_option`.

``-c``
^^^^^^^

Unit cell filename is specified with this option, e.g., ``-c POSCAR-unitcell``.

``--temperature``
^^^^^^^^^^^^^^^^^^

Pick up one temperature point. For example, ``--temperature=300`` for
300 K, which works only if thermal conductivity is calculated at
temperatures including 300 K.

``--nsp``
^^^^^^^^^^

Number of points to be sampled in the x-axis.

``kaccum`` for tensor properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of columns of output data is 13 as explained above. With
``--average`` and ``--trace`` options, number of columns of output
data becomes 3.

``--mfp``
^^^^^^^^^^

Mean free path is used instead of frequency for the x-axis. The unit
of MFP is Angstrom.

The figure below shows the results of Si example with the
:math:`19\times 19\times 19` and :math:`11\times 11\times 11` sampling
meshes used for the lattice thermal conductivity calculation. They look
differently. Especially for the result of the :math:`11\times 11\times
11` sampling mesh, the MFP seems converging but we can see it's not
true to look at that of the :math:`19\times 19\times 19` sampling
mesh. To show this type of plot, be careful about the sampling mesh
convergence.

.. |i1| image:: Si-kaccum-MFP.png
        :width: 50%

|i1|


``--gv``
^^^^^^^^^

Outer product of group velocities :math:`\mathbf{v}_\lambda \otimes
\mathbf{v}_\lambda` (in THz^2 x Angstrom^2)

``--average``
^^^^^^^^^^^^^^

Output the traces of the tensors divided by 3 rather than the unique
elements.

``--trace``
^^^^^^^^^^^^

Output the traces of the tensors rather than the unique elements.

``kaccum`` for scalar properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the following properties, those intensities are normalized by
the number of grid points. Number of columns of output data is three,
frequency, cumulative property, and derivative of cumulative property
such like DOS.

``--gamma``
^^^^^^^^^^^^

:math:`\Gamma_\lambda(\omega_\lambda)` (in THz) 

``--tau``
^^^^^^^^^^^

Lifetime :math:`\tau_\lambda = \frac{1}{2\Gamma_\lambda(\omega_\lambda)}` (in ps) 

``--cv``
^^^^^^^^^

Modal heat capacity :math:`C_\lambda` (in eV/K)

``--gv_norm``
^^^^^^^^^^^^^^

Absolute value of group velocity :math:`|\mathbf{v}_\lambda|` (in
THz x Angstrom) 

``--pqj``
^^^^^^^^^^^^^^

Averaged phonon-phonon interaction :math:`P_{\mathbf{q}j}` (in eV^2) 
