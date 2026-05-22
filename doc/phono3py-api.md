(phono3py_api)=

# Phono3py API

```{contents}
:depth: 2
:local:
```

This page describes how to use the phono3py API, with snippets that work on
the
[`AlN-LDA` example](https://github.com/phonopy/phono3py/tree/develop/example/AlN-LDA).

## Crystal structure

Crystal structures in phono3py are represented as `PhonopyAtoms` instances. To
create one from a file written in a force-calculator format, use the
`read_crystal_structure` function provided by phonopy. For example, in the
AlN-LDA example:

```python
In [1]: from phonopy.interface.calculator import read_crystal_structure

In [2]: unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode='vasp')

In [3]: print(unitcell)
lattice:
- [     3.110999999491908,     0.000000000000000,     0.000000000000000 ] # a
- [    -1.555499999745954,     2.694205030733368,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     4.978000000000000 ] # c
points:
- symbol: Al # 1
  coordinates: [  0.333333333333333,  0.666666666666667,  0.000948820000000 ]
  mass: 26.981539
- symbol: Al # 2
  coordinates: [  0.666666666666667,  0.333333333333333,  0.500948820000000 ]
  mass: 26.981539
- symbol: N  # 3
  coordinates: [  0.333333333333333,  0.666666666666667,  0.619051180000000 ]
  mass: 14.006700
- symbol: N  # 4
  coordinates: [  0.666666666666667,  0.333333333333333,  0.119051180000000 ]
  mass: 14.006700
```

Otherwise, construct a {class}`~phonopy.structure.atoms.PhonopyAtoms` instance
directly:

```python
In [1]: import numpy as np

In [2]: from phonopy.structure.atoms import PhonopyAtoms

In [3]: a = 3.111

In [4]: c = 4.978

In [5]: lattice = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]

In [6]: x = 1.0 / 3

In [7]: points = [[x, 2 * x, 0], [2 * x, x, 0.5], [x, 2 * x, 0.1181], [2 * x, x, 0.6181]]

In [8]: symbols = ["Al", "Al", "N", "N"]

In [9]: unitcell = PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)
```

## `Phono3py` class

To drive phono3py from Python, use the phono3py API. The main class,
`Phono3py`, is imported as:

```python
from phono3py import Phono3py
```

As described in {ref}`workflow`, the phono3py workflow is roughly divided into
three steps, and the `Phono3py` class is used at every step. The minimum set
of inputs to instantiate it are `unitcell` (1st argument), `supercell_matrix`
(2nd argument), and `primitive_matrix` (3rd argument):

```python
ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix="auto")
```

`unitcell` is a {class}`~phonopy.structure.atoms.PhonopyAtoms` instance.
`supercell_matrix` and `primitive_matrix` are the transformation matrices used
to generate the supercell and primitive cell from `unitcell` (see the
[definitions in the phonopy docs](https://phonopy.github.io/phonopy/phonopy-module.html#definitions-of-variables)).
This step is analogous to instantiating the {class}`~phonopy.Phonopy` class
(see the
[pre-processing section](https://phonopy.github.io/phonopy/phonopy-module.html#pre-process)
of the phonopy docs).

`Phono3py` accepts many other parameters; see {ref}`api_reference` for the
full list of arguments and attributes (or use `help(Phono3py)` in an
interactive shell).

## Displacement dataset generation

Step (1) in {ref}`workflow` generates sets of displacements in the supercell.
The displaced supercells are passed as input crystal-structure models to an
external force calculator (e.g. a first-principles code). The calculator
returns sets of forces for the displaced supercells; phono3py itself only
provides the crystal structures. We refer to the set of displacements as the
`displacement dataset`, to the set of supercell forces as the `force sets`,
and to the pair as simply the `dataset` used to compute the second- and
third-order force constants.

After instantiating `Phono3py` with `unitcell`, displacements are generated as
follows:

```python
In [4]: ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix="auto")

In [5]: ph3.generate_displacements()

In [6]: len(ph3.supercells_with_displacements)
Out[6]: 1254

In [7]: type(ph3.supercells_with_displacements[0])
Out[7]: phonopy.structure.atoms.PhonopyAtoms
```

In this example, 1254 supercells with displacements were generated.

The displacement dataset is needed later for the force-constants
calculation, so it is recommended to save it to a file when the Python
session holding the `Phono3py` instance is going to be terminated. The
displacement dataset and crystal-structure information are saved via the
`Phono3py.save()` method:

```python
In [8]: ph3.save("phono3py_disp.yaml")
```

## Supercell force calculation

Forces on the displaced supercells are computed by an external force
calculator such as a first-principles code.

The computed supercell forces are stored in the `Phono3py` instance through
the `Phono3py.forces` attribute by assigning an array-like with shape
`(num_supercells, num_atoms_in_supercell, 3)`. In the example above, the
shape is `(1254, 72, 3)`.

If the force sets are stored in a {ref}`iofile_FORCES_FC3` file, the numpy
array of `forces` can be obtained by:

```python
forces = np.loadtxt("FORCES_FC3").reshape(-1, num_atoms_in_supercell, 3)
assert len(forces) == num_supercells
```

## Force constants calculation

Computing the force constants requires both the displacement dataset and the
force sets. {ref}`api-phono3py` is a convenient function that loads these
data from files and sets them on the `Phono3py` instance. When only the
displacement dataset is needed, the low-level phono3py-yaml parser
`Phono3pyYaml` is useful.

```python
In [1]: from phono3py.interface.phono3py_yaml import Phono3pyYaml

In [2]: ph3yml = Phono3pyYaml()

In [3]: ph3yml.read("phono3py_disp.yaml")

In [4]: disp_dataset = ph3yml.dataset
```

The steps below show how to compute force constants using `ph3yml`, assuming
that `FORCES_FC3` is in the current directory. The displacement dataset and
force sets are assigned to the `Phono3py` instance as follows:

```python
In [5]: unitcell = ph3yml.unitcell

In [6]: from phono3py import Phono3py

In [7]: ph3 = Phono3py(unitcell, supercell_matrix=ph3yml.supercell_matrix, primitive_matrix=ph3yml.primitive_matrix)

In [8]: import numpy as np

In [9]: forces = np.loadtxt("FORCES_FC3").reshape(-1, len(ph3.supercell), 3)

In [10]: ph3.dataset = disp_dataset

In [11]: ph3.forces = forces
```

Now we are ready to compute force constants. With the default
(`fc_calculator=None`, i.e. the traditional finite-difference solver),
`produce_fc3` does **not** symmetrize the result; call `symmetrize_fc3` and
`symmetrize_fc2` explicitly afterwards to enforce translational and
permutation invariance:

```python
In [12]: ph3.produce_fc3()

In [13]: ph3.symmetrize_fc3()

In [14]: ph3.symmetrize_fc2()
```

When `phonon_supercell_matrix` is set, fc2 is **not** produced by
`produce_fc3`; call `ph3.produce_fc2()` (followed by `ph3.symmetrize_fc2()`)
instead. With `fc_calculator="symfc"` or `"alm"`, the chosen solver already
returns symmetrized force constants, so the `symmetrize_*` calls are not
needed. See {ref}`api_reference` for the available calculators and options.

## Non-analytical term correction parameters

Born effective charges and the dielectric constant tensor, obtained from
experiments or calculations, are passed as parameters for the non-analytical
term correction (NAC). These parameters are stored as a python dict:

```
'born': ndarray
    Born effective charges
    shape=(num_atoms_in_primitive_cell, 3, 3), dtype='double', order='C'
'factor': float
    Unit conversion factor
'dielectric': ndarray
    Dielectric constant tensor
    shape=(3, 3), dtype='double', order='C'
```

Note that the phono3py API requires Born effective charges for **all** atoms
in the primitive cell, whereas the
[`BORN` file](https://phonopy.github.io/phonopy/input-files.html#born-optional)
only stores those of the symmetrically independent atoms. See
[the phonopy documentation](https://phonopy.github.io/phonopy/phonopy-module.html#getting-parameters-for-non-analytical-term-correction)
for more information on NAC parameters.

These NAC parameters are assigned to the `Phono3py` instance via the
`Phono3py.nac_params` attribute. They can be read from a `BORN` file:

```python
In [13]: from phonopy.file_IO import parse_BORN

In [14]: nac_params = parse_BORN(ph3.primitive, filename="BORN")

In [15]: ph3.nac_params = nac_params
```

where `parse_BORN` requires the corresponding primitive cell as a
`PhonopyAtoms` instance.

## Regular grid for **q**-point sampling

Phonons are normally sampled on a $\Gamma$-centred regular grid in reciprocal
space. There are three ways to specify the grid:

1. Three integer values
2. One value
3. 3x3 integer matrix

One of these is assigned to the `mesh_numbers` attribute of the `Phono3py`
class. For example:

```python
ph3.mesh_numbers = [10, 10, 10]
ph3.mesh_numbers = 50
ph3.mesh_numbers = [[-10, 10, 10], [10, -10, 10], [10, 10, -10]]
```

### Three integer values

Three integer values ($n_1$, $n_2$, $n_3$) are specified. They define a
conventional regular grid in which the $\mathbf{q}$-points are given by a
linear combination of reciprocal basis vectors divided by the respective
integers:

```{math}
:label: three-integer-grid
\mathbf{q} = \left( \frac{\mathbf{b}^*_1}{n_1} \; \frac{\mathbf{b}^*_2}{n_2} \;
\frac{\mathbf{b}^*_3}{n_3} \right)
\begin{pmatrix} m_1 \\ m_2 \\ m_3 \end{pmatrix},
```

where $m_i \in \{ 0, 1, \ldots, n_i - 1 \}$.

### One value

A length-like value $l$ is specified, from which a regular grid is generated.
By default, the three integer values of the conventional regular grid are
computed as:

```{math}
:label: one-value-grid
n_i = \max[1, \mathrm{nint}(l|\mathbf{b}^*_i|)]
```

As an experimental feature, the generalized regular grid is also supported.
Pass `use_grg=True` when instantiating `Phono3py` to enable it. The procedure
is as follows: first, the conventional unit cell corresponding to the
primitive cell is found; second, Eq. {eq}`one-value-grid` is applied to the
reciprocal basis vectors of that conventional unit cell; and finally,
$\mathbf{q}$-points are sampled following Eq. {eq}`three-integer-grid` with
respect to those reciprocal basis vectors. The parallelepiped defined by
$\mathbf{b}^*_i$ of the conventional unit cell can be smaller than that of
the primitive cell; in such a case, additional $\mathbf{q}$-points are
sampled to fill the latter parallelepiped.

### 3x3 integer matrix (experimental)

This form defines the generalized regular grid explicitly. The generalized
regular grid is designed to be generated automatically from a single
length-like value while respecting symmetry; however, a 3x3 integer matrix
(array) is also accepted as long as it is compatible with the crystal
symmetry.

## Phonon-phonon interaction calculation

The three-phonon interaction strength is computed once the regular grid is
defined. When `phono3py_disp.yaml` and `FORCES_FC3` (and optionally `BORN`)
are available, it is convenient to obtain a configured `Phono3py` instance via
`phono3py.load`:

```python
In [1]: import phono3py

In [2]: ph3 = phono3py.load("phono3py_disp.yaml", log_level=1)
NAC params were read from "BORN".
Displacement dataset for fc3 was read from "phono3py_disp.yaml".
Sets of supercell forces were read from "FORCES_FC3".
Computing fc3[ 1, x, x ] using numpy.linalg.pinv with displacements:
    [ 0.0300  0.0000  0.0000]
    [ 0.0000  0.0000  0.0300]
    [ 0.0000  0.0000 -0.0300]
Computing fc3[ 37, x, x ] using numpy.linalg.pinv with displacements:
    [ 0.0300  0.0000  0.0000]
    [ 0.0000  0.0000  0.0300]
    [ 0.0000  0.0000 -0.0300]
Expanding fc3.
fc3 was symmetrized.
Max drift of fc3: 0.000000 (yxx) 0.000000 (xyx) 0.000000 (xxy)
Displacement dataset for fc2 was read from "phono3py_disp.yaml".
Sets of supercell forces were read from "FORCES_FC3".
fc2 was symmetrized.
Max drift of fc2: 0.000000 (yy) 0.000000 (yy)
```

The three-phonon interaction calculation is ready to run after the following
lines:

```python
In [3]: ph3.mesh_numbers = 30

In [4]: ph3.init_phph_interaction()
```

It is implemented in the `phono3py.phonon3.interaction.Interaction` class. By
design, $\mathbf{q}_1$ of the three phonons
$(\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3)$ is fixed, and the calculation
iterates over different $\mathbf{q}_2$; $\mathbf{q}_3$ is then uniquely
determined by $\mathbf{q}_1$ and $\mathbf{q}_2$. Crystal symmetry is used to
skip symmetrically redundant $\mathbf{q}_2$ at the fixed $\mathbf{q}_1$. A
single `Interaction` instance stores the data for only one fixed
$\mathbf{q}_1$.

The three-phonon interaction strength requires a large amount of memory.
Therefore, in lattice thermal conductivity calculations it is computed on
demand and discarded after use. This calculation is typically the most
computationally demanding step in practical applications, so detailed
techniques are used (when requested) to skip elements of the interaction
strength that are not needed.

The three-phonon interaction strength calculation is the engine for more
macroscopic calculations such as lattice thermal conductivity. The outer
function that drives it sets the computational configuration of this kernel
so that resources can be tuned to each use case.

## Lattice thermal conductivity calculation

Once the three-phonon interaction has been prepared by
`ph3.init_phph_interaction()`, the lattice thermal conductivity calculation is
ready to run. The available parameters are documented in {ref}`api_reference`,
but even with the defaults, the calculation under the relaxation time
approximation (RTA) is performed as follows:

```python
In [5]: ph3.run_thermal_conductivity()
```

Pass `is_LBTE=True` to run the direct solution of the linearized Boltzmann
equation (and the Wigner transport equation) instead.

## Use of different supercell dimensions for 2nd and 3rd order FCs

Phono3py supports different supercell dimensions for the second- and third-order
force constants (fc2 and fc3). By default the same dimension is used for both.
Although fc3 requires many more supercells than fc2 for the same supercell size,
fc3 in our experience has a shorter interaction range in direct space.
Therefore a smaller supercell can usually be used for fc3 than for fc2; the
`phonon_supercell_matrix` parameter specifies the fc2 supercell dimension
independently.

Using `POSCAR-unitcell` in the AlN-LDA example,

```python
In [1]: from phonopy.interface.calculator import read_crystal_structure

In [2]: unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode="vasp")

In [3]: from phono3py import Phono3py

In [4]: ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix="auto", phonon_supercell_matrix=[5, 5, 3])

In [5]: ph3.generate_displacements()

In [6]: ph3.save("phono3py_disp.yaml")
```

`Phono3py.generate_fc2_displacements()` generates the fc2 displacement
dataset. Calling it explicitly is normally unnecessary because
`Phono3py.generate_displacements()` already calls it. Use it when you need to
control the displacement pattern with non-default settings; see
{ref}`api_reference` for the details.

When `phonon_supercell_matrix` is set, the following attributes are
available:

```python
Phono3py.phonon_supercell_matrix
Phono3py.phonon_dataset
Phono3py.phonon_forces
Phono3py.phonon_supercells_with_displacements
```

Their meanings are documented in {ref}`api_reference`, but should be obvious
from the names.

(api-phono3py)=

## `phono3py.load`

`phono3py.load` creates a `Phono3py` instance with the basic parameters
loaded from a phono3py-yaml-type file, and additionally tries to set up the
force constants and non-analytical term correction automatically from
phono3py files in the current directory. See {ref}`api_reference` for the
full list of arguments.

In the AlN-LDA example, the unit cell, supercell matrix, and primitive matrix
are stored in `phono3py_disp_dimfc2.yaml`. The file is read in ipython (or a
Jupyter notebook) as follows:

```python
In [1]: import phono3py

In [2]: ph3 = phono3py.load("phono3py_disp_dimfc2.yaml", produce_fc=False)

In [3]: type(ph3)
Out[3]: phono3py.api_phono3py.Phono3py

In [4]: print(ph3.unitcell)
lattice:
- [     3.110999999491908,     0.000000000000000,     0.000000000000000 ] # a
- [    -1.555499999745954,     2.694205030733368,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     4.978000000000000 ] # c
points:
- symbol: Al # 1
  coordinates: [  0.333333333333333,  0.666666666666667,  0.000948820000000 ]
  mass: 26.981539
- symbol: Al # 2
  coordinates: [  0.666666666666667,  0.333333333333333,  0.500948820000000 ]
  mass: 26.981539
- symbol: N  # 3
  coordinates: [  0.333333333333333,  0.666666666666667,  0.619051180000000 ]
  mass: 14.006700
- symbol: N  # 4
  coordinates: [  0.666666666666667,  0.333333333333333,  0.119051180000000 ]
  mass: 14.006700

In [5]: ph3.supercell_matrix
Out[5]:
array([[3, 0, 0],
       [0, 3, 0],
       [0, 0, 2]])

In [6]: ph3.nac_params
Out[6]:
{'born': array([[[ 2.51264750e+00,  0.00000000e+00,  0.00000000e+00],
         [ 0.00000000e+00,  2.51264750e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00,  2.67353000e+00]],

        [[ 2.51264750e+00, -2.22044605e-16,  0.00000000e+00],
         [ 0.00000000e+00,  2.51264750e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00,  2.67353000e+00]],

        [[-2.51264750e+00,  0.00000000e+00,  0.00000000e+00],
         [ 0.00000000e+00, -2.51264750e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00, -2.67353000e+00]],

        [[-2.51264750e+00,  2.22044605e-16,  0.00000000e+00],
         [ 0.00000000e+00, -2.51264750e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00, -2.67353000e+00]]]),
 'factor': 14.39965172592227,
 'dielectric': array([[4.435009, 0.      , 0.      ],
        [0.      , 4.435009, 0.      ],
        [0.      , 0.      , 4.653269]])}
```
