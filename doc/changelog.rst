.. _changelog:

Change Log
==========

Mar-17-2021: Version 1.22.3
---------------------------
- Fix ``--read-gamma`` option to work.

Feb-21-2021: Version 1.22.2
---------------------------
- Fix PyPI source distribution package

Feb-21-2021: Version 1.22.1
---------------------------
- ``phono3py`` command didn't work. This was fixed.
- Fix behaviour when specifying ``--thm`` and ``--sigma`` simultaneously.

Jan-29-2021: Version 1.22.0
---------------------------
- Maintenance release to follow phonopy v2.9.0.

Sep-30-2020: Version 1.21.0
---------------------------

- Maintenance release to follow the change of phonopy at v2.8.1
- Improvements of phono3py loader (``phono3py.load``), ``phono3py-load``
  command, API, and ``phono3py_disp.yaml``.
- Harmonic phonon calculation on mesh was multithreaded. This is
  effective when using very dense mesh with non-analytical term
  correction (probably rare case).
- Real and imaginary parts of self energy and spectral function of
  bubble diagram at API level

Mar-3-2020: Version 1.20.0
--------------------------

- ``phono3py_disp.yaml`` is made when creating displacements in
  addition to ``disp_fc3.yaml`` and
  ``disp_fc2.yaml``. ``phono3py_disp.yaml`` will be used instead of
  ``disp_fc3.yaml`` and ``disp_fc2.yaml`` in the future major release
  (v2.0).

Mar-3-2020: Version 1.19.1
--------------------------

- Release for pypi and conda (atztogo channel) packagings

Mar-2-2020: Version 1.19.0
--------------------------

- Improvements of phono3py loader and API.
- Improvement of interfaces to calculators. Now it is expected to be
  much easier to implement calculator interface if it exists in
  phonopy.
- Fixed dependency to phonopy v2.6.0.

Dec-22-2019: Version 1.18.2
---------------------------

- Initial version of phono3py loader (``phono3py.load``) was
  implemented. See docstring of ``phono3py.load``.

Oct-17-2019: Version 1.18.1
---------------------------

- Fix of phono3py-kaccum to follow the latest phonopy.

Oct-9-2019: Version 1.18.0
---------------------------

- Maintenance release

Apr-18-2019: Version 1.17.0
---------------------------
- ``--cfz`` option was made to subtract residual forces. See
  :ref:`cfz_option`.
- ``--cutoff-pair`` was made to override the cutoff pair distance
  written in ``disp_fc3.yaml`` when using on calculating force
  constants. This is useful when checking cutoff distance
  dependency. So the use case of having fully computed ``FORCES_FC3``
  is assumed.
- TURBOMOLE interface is provided by Antti Karttunen
  (``--turbomole``).
- Compatibility of ``fc2.hdf5`` and ``force_constants.hdf5`` was
  improved for all calculators to store physical unit information in
  the hdf5 file. See :ref:`file_format_compatibility`.

Mar-24-2019: Version 1.16.0
---------------------------
- Bug fixes and catching up the updates of phonopy.
- Most of hdf5 output files are compressed by ``gzip`` as
  default. This compression can be set off, see
  :ref:`hdf5_compression_option`.
- (Experimental) ``phono3py`` command accepts ``phono3py.yaml`` type
  file as an input crystal structure by ``-c`` option. When ``DIM``
  and any structure file are not given, ``phono3py_disp.yaml``
  (primary) or ``phono3py.yaml`` (secondary) is searched in the current
  directory. Then ``phono3py.yaml`` type file is used as the input.
  By this, semi-automatic phono3py mode is invocked, which acts as

  (1) ``supercell_matrix`` corresponding to ``DIM`` in the
      ``phono3py.yaml`` type file is used if it exists.
  (2) ``phonon_supercell_matrix``  corresponding to ``DIM_FC2`` in the
      ``phono3py.yaml`` type file is used if it exists.
  (3) ``primitive_matrix`` in the ``phono3py.yaml`` type file
      is used if it exists. Otherwise, set ``PRIMITIVE_AXES = AUTO``
      when ``PRIMITIVE_AXES`` is not given.
  (4) NAC params are read (``NAC = .TRUE.``) if NAC params are
      contained (primary) in the ``phono3py.yaml`` type file or if
      ``BORN`` file exists in the current directory (secondary).

Nov-22-2018: version 1.14.3
----------------------------
- Update to work with phonopy v1.14.2.
- Ph-ph interaction can be read (``--read-pp``) and write
  (``--write-pp``) in RTA thermal conductivity calculation, too. Mind
  that the data stored are different with and without
  ``--full-pp``. Wihtout ``--full-pp`` the data are stored in
  complicated way to save data side, so it is not considered readable
  by usual users.

June-20-2018: version 1.13.3
----------------------------

- ``--lw`` (linewidth) option was removed. Use ``--br`` option and
  find 2*gamma values as linewidths in ``kappa-xxx.hdf5`` file.
- Documentation of ``--lbte`` option is available at
  :ref:`direct_solution`.
- This version is dependent on phonopy>=1.13.2.

May-17-2018: version 1.13.1
----------------------------

- Compact force constants are implemented (See :ref:`compact_fc_option`).

Mar-16-2018: version 1.12.9
----------------------------

- Definition of ``mode_kappa`` values in output hdf5 file is
  changed. Previously they were divided by number of grid points, but
  now not. Therefore users who compute ``kappa`` from ``mode_kappa``
  need to be careful about this change. This does not affect to
  ``phono3py-kaccum`` results.

Feb-1-2018: version 1.12.7
----------------------------

- ``--tsym`` option is removed. Now with ``--sym-fc3r`` and
  ``--sym-fc2`` options,
  translational invariance symmetry is also applied.
- ``--sym-fc`` option is added. This is just an alias to specify both
  ``--sym-fc3r`` and ``--sym-fc2`` together.
- Documentation on ``--write-phonon`` and ``--read-phonon`` options is
  written. These options are used to save harmonic phonon infromation
  on strage.

Nov-22-2017: version 1.12.5
-----------------------------

- Bug fix of RTA thermal conductivity. This bug exists from version
  1.10.11.18 (git e40cd059). This bug exhibits when all the following
  conditions are met:

  1. RTA thermal conductivity calculation.
  2. Tetrahedron method for Brillouin zone integration is used.
  3. Number of triplets is smaller than number of bands at each grid point.
  4. Without using ``--full-pp``.


  (3) happens when the primitive cell is relatively large. Number of
  triplets can be shown using ``--stp`` option. A race condition of
  OpenMP multithreading is the source of the bug. Therefore, if it
  occurs, the same calculation comes up with the different thermal
  conductivity value in every run time, for whcih it behaves like
  randomly.

- RTA thermal conductivity with smearing method (``--sigma``) is made
  to run with smaller memory consumption as similar as tetrahedron
  method (``--thm``).

Nov-17-2017: version 1.12.3
----------------------------

- Command option parser of the phonopy tools is replaced from
  ``optparse`` to ``argparse``.
- The filenames used with these options were the positional arguments
  previously. Now they are the command-line arguments, i.e., filenames
  have to be put just after the option name like ``-f vasprun.xml-001
  vasprun.xml-002 ...``.
- The names of auxiliary tools (``kdeplot`` and ``kaccum``) are
  changed, for which the prefix phono3py- is added to the old names to
  avoid accidental conflict with other script names already existing
  under bin directory.
- :ref:`sigma_cutoff_option` option was created.

Jun-18-2017: version 1.11.13
----------------------------

- :ref:`normal_umklapp_option` option was made.
- Many minor updates: fixing bugs, improving usabilities.
- Improve of :ref:`auxiliary_tools_kaccum` and :ref:`auxiliary_tools_kdeplot`.

Mar-31-2017: version 1.11.11
----------------------------

- Abinit code interface is implemented and now under the testing.
- Reduction of memory usage in RTA thermal conductivity
  calculation. This is especially effective for larger unit cell
  case. Currently combinations with --full_pp, --write_gamma_detail,
  and --simga(smearing method) are not supported for this. Performance
  tuning is under going. In some case, computation can be slower than
  the previous versions.

Feb-9-2017: version 1.11.9
---------------------------

- This version works coupled with phonopy-1.11.8 or later.
- CRYSTAL code interface is implemented by Antti J. Karttunen.

Dec-14-2016: version 1.11.7
------------------------------

- This is a maintenance release. This version must be used with
  phonopy-1.11.6 or later.

Nov-27-2016: version 1.11.5
------------------------------

- ``gaccum`` is merged to ``kaccum``. ``gaccum`` is removed. See
  :ref:`auxiliary_tools_kaccum`.
- ``kdeplot`` is added. See :ref:`auxiliary_tools_kdeplot`.

Apr-24-2016: version 1.10.9
------------------------------

- Failure of writing ``kappa-mxxx-gx.hdf5`` was fixed.

Apr-16-2016: version 1.10.7
------------------------------

- API example is prepared and it is found in ``Si`` example. No
  doucment yet.
- Si pwscf example was placed in ``example-phono3py`` directory.
- User interface bug fix.

Mar-15-2016: version 1.10.5
------------------------------

- Numbering way of phono3py version was just changed (No big updates
  were made against previous version.) The number is given based on
  the phonopy version. For example, the harmonic part of
  phono3py-1.10.5 is based on the code close to phonopy-1.10.4.
- Python3 support
- For the RTA thermal conductivity calculation mode with using the
  linear tetrahedron method, only necessary part of phonon-phonon
  interaction strengh among phonons. This improves lifetime
  calculation performance, but as the drawback, averaged ph-ph
  interaction strength can not be given. See :ref:`full_pp_option`.
- Pwscf interface (:ref:`calculator_interfaces`)

Oct-10-2015: version 0.9.14
------------------------------

- Computational performance tuning for phonon-phonon interaction
  strength calculation was made by Jonathan Skelton. Depending on
  systems, but 10-20% performance improvement may be possible.
- ``--stp`` option is created to show numbers of q-point triplets to
  be calculated. See :ref:`command_options`.
- ``--write_gamma`` and ``--read_gamma`` support using with ``--bi``
  option. Therefore a thermal conductivity calculation can be
  distributed over band index, too. This may be useful for the system
  whose unit cell is large.

Sep-26-2015: version 0.9.13
------------------------------

- Changed so that ``--wgp`` option writes ``grid_address-mxxx.hdf5``
  instead of ``grid_address-mxxx.dat``.
- ``--write_detailed_gamma`` is implemented. See :ref:`command_options`.
- When running without setting ``--thm`` and ``--sigma`` options,
  linear tetrahedron method corresponding to ``--thm`` is used as the
  default behavior.
- ``--ise`` options is created.

Aug-12-2015: version 0.9.12
------------------------------

- Spglib update to version 1.8.2.1.
- Improve computational performance of ``kaccum`` and ``gaccum``.

Jun-18-2015: version 0.9.10.1
------------------------------

- Bug fix of ``gcaccum``

Jun-17-2015: version 0.9.10
----------------------------

- Fix bug in ``kaccum``. When using with ``--pa`` option, irreducible
  q-points were incorrectly indexed.
- ``gaccum`` is implemented. ``gaccum`` is very similar to ``kaccum``,
  but for :math:`\Gamma_\lambda(\omega_\lambda)`.
- spglib update.

Changes in version 0.9.7
-------------------------

- The definition of MSPP is modified so as to be averaged ph-ph
  interaction defined as :math:`P_{\mathbf{q}j}` in the arXiv
  manuscript. The key in the kappa hdf5 file is changed from ``mspp``
  to ``ave_pp``. The physical unit of :math:`P_{\mathbf{q}j}` is set
  to :math:`\text{eV}^2`.

Changes in version 0.9.6
------------------------

- Silicon example is put in ``example-phono3py`` directory.
- Accumulated lattice thermal conductivity is calculated by ``kaccum``
  script.
- JDOS output format was changed.

Changes in version 0.9.5
-------------------------

- In ``kappa-xxx.hdf5`` file, ``heat_capacity`` format was changed
  from ``(irreducible q-point, temperature, phonon band)`` to
  ``(temperature, irreducible q-point, phonon band)``. For ``gamma``,
  previous document was wrong in the array shape. It is
  ``(temperature, irreducible q-point, phonon band)``


Changes in version 0.9.4
------------------------

- The option of ``--cutoff_mfp`` is renamed to ``--boundary_mfp`` and
  now it's on the document.
- Detailed contribution of ``kappa`` at each **q**-point and phonon
  mode is output to .hdf5 with the keyword ``mode_kappa``.

Changes in version 0.8.11
-------------------------

- A new option of ``--cutoff_mfp`` for including effective boundary
  mean free path.
- The option name ``--cutfc3`` is changed to ``--cutoff_fc3``.
- The option name ``--cutpair`` is changed to ``--cutoff_pair``.
- A new option ``--ga`` is created.
- Fix spectrum plot of joint dos and imaginary part of self energy

Changes in version 0.8.10
-------------------------

- Different supercell size of fc2 from fc3 can be specified using
  ``--dim_fc2`` option.
- ``--isotope`` option is implemented. This is used instead of
  ``--mass_variances`` option without specifying the values. Mass
  variance parameters are read from database.

Changes in version 0.8.2
------------------------

- Phono3py python interface is rewritten and a lot of changes are
  introduced.
- ``FORCES_SECOND`` and ``FORCES_THIRD`` are no more used. Instead just
  one file of ``FORCES_FC3`` is used. Now ``FORCES_FC3`` is generated
  by ``--cf3`` option and the backward compatibility is simple: ``cat
  FORCES_SECOND FORCES_THIRD > FORCES_FC3``.
- ``--multiple_sigmas`` is removed. The same behavior is achieved by
  ``--sigma``.

Changes in version 0.8.0
------------------------

- ``--q_direction`` didn't work. Fix it.
- Implementation of tetrahedron method whcih is activated by
  ``--thm``.
- Grid addresses are written out by ``--wgp`` option.

Changes in version 0.7.6
------------------------

- Cut-off distance for fc3 is implemented. This is activated by
  ``--cutfc3`` option. FC3 elements where any atomic pair has larger
  distance than cut-off distance are set zero.
- ``--cutpair`` works only when creating displacements. The cut-off
  pair distance is written into ``disp_fc3.yaml`` and FC3 is created
  from ``FORCES_THIRD`` with this information. Usually sets of pair
  displacements are more redundant than that needed for creating fc3
  if index permutation symmetry is considered. Therefore using index
  permutation symmetry, some elements of fc3 can be recovered even if
  some of supercell force calculations are missing. In paticular, all
  pair distances among triplet atoms are larger than cutoff pair
  distance, any fc3 elements are not recovered, i.e., the element will
  be zero.

Changes in version 0.7.2
------------------------

- Default displacement distance is changed to 0.03.
- Files names of displacement supercells now have 5 digits numbering,
  ``POSCAR-xxxxx``.
- Cutoff distance between pair displacements is implemented. This is
  triggered by ``--cutpair`` option. This option works only for
  calculating atomic forces in supercells with configurations of pairs
  of displacements.

Changes in version 0.7.1
------------------------

- It is changed to sampling q-points in Brillouin zone. Previously
  q-points are sampled in reciprocal primitive lattice. Usually this
  change affects very little to the result.
- q-points of phonon triplets are more carefully sampled when a
  q-point is on Brillouin zone boundary. Usually this
  change affects very little to the result.
- Isotope effect to thermal conductivity is included.

Changes in version 0.6.0
------------------------

- ``disp.yaml`` is renamed to ``disp_fc3.yaml``. Old calculations with
  ``disp.yaml`` can be used without any problem just by changing the
  file name.
- Group velocity is calculated from analytical derivative of dynamical
  matrix.
- Group velocities at degenerate phonon modes are better handled.
  This improves the accuracy of group velocity and thus for thermal
  conductivity.
- Re-implementation of third-order force constants calculation from
  supercell forces, which makes the calculation much faster
- When any phonon of triplets can be on the Brillouin zone boundary, i.e.,
  when a mesh number is an even number, it is more carefully treated.
