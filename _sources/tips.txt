.. _tips:

Tips
=====

Convergence check in calculation
---------------------------------

.. _brillouinzone_sum:

Brillouin zone summation
~~~~~~~~~~~~~~~~~~~~~~~~~

Brillouin zone sums appear at different two points for phonon lifetime
calculation. First it is used for the Fourier transform of force
constans, and then to obtain imaginary part of phonon-self-energy.  In
the numerical calculation, uniform sampling meshes are employed for
these summations. To obtain more accurate result, it is always better
to use denser meshes. But the denser mesh requires more
computationally demanding.

The second Brillouin zone sum contains delta functions. In phono3py
calculation, a linear tetrahedron method (``--thm``, default option)
and a smearing method (``--sigma``) can be used for this Brillouin
zone integration. Smearing parameter is used to approximate delta
functions. Small ``sigma`` value is better to describe the detailed
structure of three-phonon-space, but it requires a denser mesh to
converge.

..
   The first and second meshes have to be same or the first
   mesh is integral multiple of the second mesh, i.e., the first and
   second meshes have to overlap and the first mesh is the same as or
   denser than the second mesh.

To check the convergence with respect to the ``sigma`` value, multiple
sigma values can be set. This can be computationally efficient, since
it is avoided to re-calculate phonon-phonon interaction strength for
different ``sigma`` values in this case.

Convergence with respect to the sampling mesh and smearing parameter
strongly depends on materials. A :math:`20\times 20\times 20` sampling
mesh (or 8000 reducible sampling points) and 0.1 THz smearing value
for reciprocal of the volume of an atom may be a good starting choice.

The tetrahedron method requires no parameter such as the smearing
width, therefore it is easier to use than the smearing method and
recommended to use. A drawback of using the tetrahedron method is that
it is slower and consumes more memory space.

Numerical quality of force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Third-order force constants are much weaker to numerical noise of a
force calculator than second-order force constants. Therefore
supercell force calculations have to be done by enough high numerical
accuracy.

The phono3py default displacement distance is 0.03
:math:`\text{\AA}`. In some cases, accurate result may not be obtained
due to the numerical noise of the force calculator. Usually increasing
the displacement distance by ``--amplitude`` option reduces
the numerical noise, but increases error from higher order anharmonicity.

It is not easy to check the numerical quality of force constants. It
is suggested firstly to check deviation from the translational
invariance condition by watching output where the output lines start
with ``max drift of ...``. The drift value smaller than 1 may be
acceptable but of course it is dependent on cases. The most practical
way may be to compare thermal conductivities calculated with and
without symmetrizing third-order force constants by ``--sym_fc3r``,
``--sym_fc2``, and ``--tsym`` options.

Mode-Gruneisen-parameters calculated from third-order force constants
look very sensitive to numerical noise near the Gamma point. Therefore
symmetrization is recommended.

