.. _tips:

Tips
=====

.. _brillouinzone_sum:

Brillouin zone summation
-------------------------

Brillouin zone sums appear at different two points for phonon lifetime
calculation. First it is used for the Fourier transform of force
constans, and then to obtain imaginary part of phonon-self-energy.  In
the numerical calculation, uniform sampling meshes are employed for
these summations. To obtain more accurate result, it is always better
to use denser meshes. But the denser mesh is more
computationally demanding.

The second Brillouin zone sum contains delta functions. In phono3py
calculation, a linear tetrahedron method (``--thm``, default option)
and a smearing method (``--sigma``) can be used for this Brillouin
zone integration. In most cases, the tetrahedron method is better,
therefore it is the default choice in phono3py. Especially in high
thermal conductivity materials, the smearing method results in
underestimation of thermal conductivity.

The figure below shows Si thermal conductivity convergence with
respect to number of mesh points along an axis from n=19 to 65. This
is calculated with RTA and the linear tetrahedron method. Within the
methods and phono3py implementation, it is converging at around n=55,
however this computational demanding is not trivial. Extrapolation to
:math:`1/n \rightarrow 0` seems not a good idea, since it is
converging. This plot shows that we have to decide how much value is
acceptable as thermal conductivity value. What is important is that
the obtained value has to be shown accompanied with the information of
the computational settings. The BZ integration method and sampling
mesh are definitely those of them.

.. |isiconv| image:: Si-convergence.png
        :width: 25%

|isiconv|



In case the smearing method is necessary to use, the convergence of
q-point mesh together with smearing width has to be checked
carefully. Smearing parameter is used to approximate delta
functions. Small ``sigma`` value is better to describe the detailed
structure of three-phonon-space, but it requires a denser mesh to
converge.

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
-------------------------------------

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

Overall, numerical quality of forces given by force calculators is
the most important factor for the numerical quality of the thermal
conductivity. We may be able to apply symmetry constraints to the
force constants during the calculation e.g. using statistical
approach, but the quality of force constants will be bad if that of
forces are bad. Just they suffice the symmetry and the intensity is
not reliable. Therefore what we can do best is to use the best
calculator as the first priority. If we use ab-initio code, the
knowledge about the ab-initio calculation from practical points like
usage to method and theory is mandatory for the good thermal
conductivity calculation.

To reduce computational demands
--------------------------------

Here it is assumed ab-initio code is used as the force
calculator. Then the most heavy part of thermal conductivity
calculation is a set of many supercell force calculations by ab-initio code.

The number of force calculation is reduced by employing crystal
symmetry. This is only valid if the crystal we focus on has high
symmetry. Therefore we need another strategy. Introducing cutoff
distance to consider interaction among atoms is an idea. For this
phono3py has a marginal option but it is not very recommended to use
since there is a better code to do this task, which is the `ALM code
<http://alamode.readthedocs.io/en/latest/input/inputalm.html>`_ in
alamode package. The ALM interface for phono3py is now preparing.
