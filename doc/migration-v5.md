(migration_v5)=

# Migrating toward phono3py v5

Some options of phono3py are off by default in v4.x and will be on by default in
v5.0. Each section of this page describes one of these options, shows how to
get the v5.0 result now, and shows how to keep the v4.x result after v5.0.

Phono3py builds on phonopy, so the phonopy v5 changes also apply. See the
[phonopy v5 migration guide](https://phonopy.github.io/phonopy/migration-v5.html)
for the changes in phonopy.

```{note}
v5.0 has not been released yet. This page is provisional. The set of options
whose defaults change, and the timing of the change, may still be revised
before v5.0. Check the {ref}`changelog` of the version you upgrade to for the
final list.
```

```{contents}
:depth: 2
:local:
```

## Changed default: acoustic frequencies at Gamma set to zero

The frequencies of the three acoustic modes at the Gamma point are zero in
theory. In a calculation, they have small nonzero values from rounding, and
these values depend on the linear algebra library. The results that use these
modes can therefore differ between computers.

The option `exclude_gamma_acoustic` sets the three frequencies at the Gamma
point with the smallest absolute values to zero after the phonons are solved.
The option is described under
{ref}`--exclude-gamma-acoustic <exclude_gamma_acoustic_option>`. It is off by
default in v4.x and will be on by default in v5.0.

To get the v5.0 result now, add `--exclude-gamma-acoustic` to the command, or
`EXCLUDE_GAMMA_ACOUSTIC = .TRUE.` to the configuration file:

```bash
% phono3py --mesh 19 19 19 --br --exclude-gamma-acoustic
```

From Python, pass `exclude_gamma_acoustic=True` to
`Phono3py.init_phph_interaction`, `Phono3pyJointDos` or `Phono3pyIsotope`. To
keep the v4.x result after v5.0, use `--no-exclude-gamma-acoustic`,
`EXCLUDE_GAMMA_ACOUSTIC = .FALSE.` or `exclude_gamma_acoustic=False`.

## Changed default: triplet tetrahedron weights averaged over degenerate modes

At some q-points, two or three phonon modes have the same frequency. Any
orthonormal combination of their eigenvectors is also a set of eigenvectors,
and the combination returned by the eigenvalue solver depends on the linear
algebra library. The imaginary part of the self energy is a sum over the
triplets of q-points $(\mathbf{q}, \mathbf{q}', \mathbf{q}'')$. For each
triplet, the tetrahedron method gives degenerate modes at $\mathbf{q}'$ and
$\mathbf{q}''$ different integration weights. The imaginary part of the self
energy therefore depends on which combination was returned, and so can differ
between computers.

The option `average_degenerate_weights` averages the integration weights of
each triplet over each set of degenerate modes. The imaginary part of the self energy then does
not depend on the choice of eigenvectors. The option is described under
{ref}`--average-degenerate-weights <average_degenerate_weights_option>`. It is
off by default in v4.x and will be on by default in v5.0.

To get the v5.0 result now, add `--average-degenerate-weights` to the command,
or `AVERAGE_DEGENERATE_WEIGHTS = .TRUE.` to the configuration file:

```bash
% phono3py --mesh 19 19 19 --br --average-degenerate-weights
```

From Python, pass `average_degenerate_weights=True` to
`Phono3py.init_phph_interaction`. To keep the v4.x result after v5.0, use
`--no-average-degenerate-weights`, `AVERAGE_DEGENERATE_WEIGHTS = .FALSE.` or
`average_degenerate_weights=False`.

## Changed default: isotope tetrahedron weights from both sides of vertices

The isotope scattering rate of a phonon mode at $\mathbf{q}$ is calculated with
the integration weights of the tetrahedron method at the frequency of that mode.
The vertices of the tetrahedra include $\mathbf{q}$ itself and the grid points
that are symmetrically equivalent to $\mathbf{q}$. At these vertices, the
frequencies of the same band are equal to the frequency of the mode in theory.
In a calculation, they differ by rounding errors, and the integration weights of
some modes depend on these rounding errors. The isotope scattering rates can
therefore differ between computers.

The option `isotope_two_sided_weights` calculates the integration weights at the
frequency of the mode shifted by $-10^{-12}$ THz and by $+10^{-12}$ THz, and uses
their average. The integration weights then do not depend on the rounding
errors. The option is described under
{ref}`--isotope-two-sided-weights <isotope_two_sided_weights_option>`. It is off
by default in v4.x and will be on by default in v5.0.

To get the v5.0 result now, add `--isotope-two-sided-weights` to the command, or
`ISOTOPE_TWO_SIDED_WEIGHTS = .TRUE.` to the configuration file:

```bash
% phono3py --mesh 19 19 19 --br --isotope --isotope-two-sided-weights
```

From Python, pass `use_isotope_two_sided_weights=True` to
`Phono3py.run_thermal_conductivity`, or `use_two_sided_weights=True` to
`Phono3pyIsotope`. To keep the v4.x result after v5.0, use
`--no-isotope-two-sided-weights`, `ISOTOPE_TWO_SIDED_WEIGHTS = .FALSE.`,
`use_isotope_two_sided_weights=False` or `use_two_sided_weights=False`.

## Deprecated API: `get_phonons` and the old `set_phonons`

`Interaction.get_phonons()`, `JointDos.get_phonons()` and `Isotope.get_phonons()`
return the tuple of frequencies, eigenvectors and `phonon_done`. They emit a
`DeprecationWarning` and will be removed in v5.0. Instead, the `phonons`
property of `Interaction`, `JointDos` and `Isotope` returns a
`phono3py.phonon.solver.PhononData` instance whose attributes are
`frequencies`, `eigenvectors`, `phonon_done` and `degenerate_ids`. The arrays
are those used in the instance, not copies.

```python
# v4.x
frequencies, eigenvectors, phonon_done = interaction.get_phonons()

# v5.0
phonons = interaction.phonons
frequencies = phonons.frequencies
eigenvectors = phonons.eigenvectors
phonon_done = phonons.phonon_done
```

Passing frequencies, eigenvectors and `phonon_done` to
`Isotope.set_phonons(frequencies, eigenvectors, phonon_done, dm=None)` is
deprecated in the same way. Pass a `PhononData` instance,
`Isotope.set_phonons(phonons, dm=None)`. Pass `dm` by keyword.

## Getting all the v5.0 defaults now

To get the v5.0 result of a thermal conductivity calculation now, add `--v5` to
the command, or `V5 = .TRUE.` to the configuration file. They turn on all the
options whose defaults change, described on this page:

```bash
% phono3py --mesh 19 19 19 --br --isotope --v5
```

An option given explicitly takes priority over `--v5`. For example,
`--v5 --no-average-degenerate-weights` turns on the other options and turns off
the averaging over degenerate modes. See {ref}`--v5 <v5_option>`.

From Python, pass each option. Pass `exclude_gamma_acoustic=True` and
`average_degenerate_weights=True` to `Phono3py.init_phph_interaction`, and
`use_isotope_two_sided_weights=True` to `Phono3py.run_thermal_conductivity`.

Tests that compare results with reference values made by v4.x will need new
reference values after v5.0, or these options turned off.
