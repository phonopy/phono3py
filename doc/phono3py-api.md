(phono3py_api)=

# Phono3py API

```{contents}
:depth: 2
:local:
```

How to use phono3py API is described below along with snippets that work with
[`AlN-LDA` example](https://github.com/phonopy/phono3py/tree/develop/example/AlN-LDA).

## Crystal structure

Crystal structures in phono3py are usually `PhonopyAtoms` class instances. When
we want to obtain the `PhonopyAtoms` class instances from a file written in a
force-calculator format, the `read_crystal_structure` function in phonopy may be
used, e.g., in the AlN-LDA example,

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

Otherwise, it is directly created from
[`PhonopyAtoms` class](https://phonopy.github.io/phonopy/phonopy-module.html#phonopyatoms-class).

```python
In [1]: from phonopy.structure.atoms import PhonopyAtoms

In [2]: a = 3.11

In [3]: a = 3.111

In [4]: c = 4.978

In [5]: lattice = [[a, 0, 0], [-a / 2,  a * np.sqrt(3) / 2, 0], [0, 0, c]]

In [6]: x = 1. / 3

In [7]: points = [[x, 2 * x, 0], [x * 2, x, 0.5], [x, 2 * x, 0.1181], [2 * x, x, 0.6181]]

In [8]: symbols = ['Al', 'Al', 'N', 'N']

In [9]: unitcell = PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)
```

## `Phono3py` class

To operate phono3py from python, there is phono3py API. The main class is
`Phono3py` that is imported by

```python
from phono3py import Phono3py
```

As written in {ref}`workflow`, phono3py workflow is roughly divided into three
steps. `Phono3py` class is used at each step. The minimum set of inputs to
instantiate `Phono3py` class are `unitcell` (1st argument), `supercell_matrix`
(2nd argument), and `primitive_matrix`, which are used as

```python
ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix='auto')
```

The `unitcell` is an instance of the
[`PhonopyAtoms` class](https://phonopy.github.io/phonopy/phonopy-module.html#phonopyatoms-class).
`supercell_matrix` and `primitive_matrix` are the transformation matrices to
generate supercell and primitive cell from `unitcell` (see
[definitions](https://phonopy.github.io/phonopy/phonopy-module.html#definitions-of-variables)).
This step is similar to the
[instantiation of `Phonopy` class](https://phonopy.github.io/phonopy/phonopy-module.html#pre-process).

There are many parameters that can be given to `Phono3py` class. The details are
written in the docstring, which is shown by

```python
help(Phono3py)
```

## Displacement dataset generation

The step (1) in {ref}`workflow` generates sets of displacements in supercell.
Supercells with the displacements are used as input crystal structure models of
force calculator that is, e.g., first-principles calculation code. Using the
force calculator, sets of forces of the supercells with the displacements are
obtained out of phono3py environment, i.e., phono3py only provides crystal
structures. Here we call the sets of the displacements as
`displacement dataset`, the sets of the supercell forces as `force sets`, and
the pair of `displacement dataset` and `force sets` as simply `dataset` for
computing second and third force constants.

After instantiating `Phono3py` with `unitcell`, displacements are generated as
follows:

```python
In [4]: ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix='auto')

In [5]: ph3.generate_displacements()

In [6]: len(ph3.supercells_with_displacements)
Out[6]: 1254

In [7]: type(ph3.supercells_with_displacements[0])
Out[7]: phonopy.structure.atoms.PhonopyAtoms
```

By this, 1254 supercells with displacements were generated.

The generated displacement dataset is used in the force constants calculation.
Therefore, it is recommended to save it into a file if the python process having
the `Phono3py` class instance is expected to be terminated. The displacement
dataset and crystal structure information are saved to a file by
`Phono3py.save()` method:

```python
In [8]: ph3.save("phono3py_disp.yaml")
```

## Supercell force calculation

Forces of the generated supercells with displacements are calculated by some
external force calculator such as first-principles calculation code.

Calculated supercell forces will be stored in a `Phono3py` class instance
through `Phono3py.forces` attribute by setting an array_like variable with the
shape of `(num_supercells, num_atoms_in_supercell, 3)`. In the above example,
the array shape is `(1254, 72, 3)`.

If the calculated force sets are stored in the
{ref}`input-output_files_FORCES_FC3` file, the numpy array of `forces` is
obtained by

```python
forces = np.loadtxt("FORCES_FC3").reshape(-1, num_atoms_in_supercell, 3)
assert len(forces) == num_supercells
```

## Force constants calculation

The pair of the displacement dataset and force sets is required to calculate
force constants. {ref}`api-phono3py-load` is the convenient function to load
these data from files and to set up them in the `Phono3py` class instance.
However, in the case when only displacement dataset is expected, the low-level
phono3py-yaml parser in `Phono3pyYaml` is useful.

```python
In [1]: from phono3py.interface.phono3py_yaml import Phono3pyYaml

In [2]: ph3yml = Phono3pyYaml()

In [3]: ph3yml.read("phono3py_disp.yaml")

In [4]: disp_dataset = ph3yml.dataset
```

With this `ph3yml`, how to compute force constants is explained. In the
following, it is assumed that we have `FORCES_FC3` in the current directory. The
displacement dataset and force sets are set to the `Phono3py` class instance as
follows:

```python
In [5]: unitcell = ph3yml.unitcell

In [6]: from phono3py import Phono3py

In [7]: ph3 = Phono3py(unitcell, supercell_matrix=ph3yml.supercell_matrix, primitive_matrix=ph3yml.primitive_matrix)

In [8]: import numpy as np

In [9]: forces = np.loadtxt("FORCES_FC3").reshape(-1, len(ph3.supercell), 3)

In [10]: ph3.dataset = disp_dataset

In [11]: ph3.forces = forces
```

Now it is ready to compute force constants.

```python
In [12]: ph3.produce_fc3()
```

## Non-analytical term correction parameters

Users collects Born effective charges and dielectric constant tensor from
experiments or calculations, and they are used as parameters for non-analytical
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

Be careful that, in the case of using phono3py API, Born effective charges of
all atoms in the primitive cell are necessary, whereas in the
[`BORN` file](https://phonopy.github.io/phonopy/input-files.html#born-optional),
only Born effective charges of symmetrically independent atoms are written. Some
more information about NAC parameters is found
[here](https://phonopy.github.io/phonopy/phonopy-module.html#getting-parameters-for-non-analytical-term-correction).

This NAC parameters are set to the `Phono3py` class instance via the
`Phono3py.nac_params` attribute. The NAC parameters may be read from `BORN` file

```python
In [13]: from phonopy.file_IO import parse_BORN

In [14]: nac_params = parse_BORN(ph3.primitive, filename="BORN")

In [15]: ph3.nac_params = nan_params
```

where the `parse_BORN` function requires the corresponding primitive cell of the
`PhonopyAtoms` class.

## Regular grid for **q**-point sampling

Phonons are normally sampled on a $\Gamma$ centre regular grid in reciprocal
space. There are three ways to specify the regular grid.

1. Three integer values
2. One value
3. 3x3 integer matrix

One of these is set to `mesh_numbers` attribute of the `Phono3py` class. For
example,

```python
ph3.mesh_numbers = [10, 10, 10]
ph3.mesh_numbers = 50
ph3.mesh_numbers = [[-10, 10, 10], [10, -10, 10], [10, 10, -10]]
```

### Three integer values

Three integer values ($n_1$, $n_2$, $n_3$) are specified.

The conventional regular grid is defined by the three integer values. The
$\mathbf{q}$-points are given by linear combination of reciprocal basis vectors
divided by the respective integers, which is given as

```{math}
:label: three-integer-grid
\mathbf{q} = \left( \frac{\mathbf{b}^*_1}{n_1} \; \frac{\mathbf{b}^*_2}{n_2} \;
\frac{\mathbf{b}^*_3}{n_3} \right)
\begin{pmatrix} m_1 \\ m_2 \\ m_3 \end{pmatrix},
```

where $m_i \in \{ 0, 1, \ldots, n_i - 1 \}$.

### One value

A distance like value $l$ is specified. Using this value, a regular grid is
generated. As default, the three integer values of the conventional regular grid
are defined by the following calculation.

```{math}
:label: one-value-grid
n_i = \max[1, \mathrm{nint}(l|\mathbf{b}^*_i|)]
```

Experimentally, use of a generalized regular grid is supported. By specifying
`use_grg = True` at the `Phono3py` class instantiation, the generalized regular
grid is generated using the value $l$. First, the conventional unit cell of the
primitive cell is searched. Second, Eq. {eq}`one-value-grid` is applied to the
reciprocal basis vectors of the conventional unit cell. Then,
$\mathbf{q}$-points are sampled following Eq. {eq}`three-integer-grid` with
respect to the reciprocal basis vectors of the conventional unit cell. The
parallelepiped defined by $\mathbf{b}^*_i$ of the conventinal unit cell can be
smaller than that of the primitive cell. In this case, the $\mathbf{q}$-points
are sampled to fill the latter parallelepiped.

### 3x3 integer matrix (experimental)

This is used to define the generalized regular grid explicitly. The generalized
regular grid is designed to be automatically generated by one given value
considering symmetry. However a 3x3 integer matrix (array) is accepted if this
matrix follows the symmetry properly.

## Phonon-phonon interaction calculation

Three phonon interaction strength is calculated after defining the regular grid.
When we have `phono3py_disp.yaml` and `FORCES_FC3` used above (maybe `BORN`,
too), it is convenient to get the `Phono3py` class instance with settings.

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

Three phonon interaction calculation becomes ready to run by the following
lines:

```python
In [3]: ph3.mesh_numbers = 30

In [4]: ph3.init_phph_interaction()
```

The three phonon interaction calculation is implemented in
`phono3py.phonon3.interaction.Interaction` class. By phono3py's design choice,
$\mathbf{q}_1$ of the three phonons $(\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3)$
in this class is given, and calculation iterates over different $\mathbf{q}_2$.
$\mathbf{q}_3$ is uniquely determined from $\mathbf{q}_1$ and $\mathbf{q}_2$.
Symmetry is employed to avoid calculating symmetrically redundant $\mathbf{q}_2$
at the fixed $\mathbf{q}_1$. One `Interaction` class instance stores the data
only for one fixed $\mathbf{q}_1$.

Three phonon interaction strength requires large memory space. Therefore, in
lattice thermal conductivity calculation, it is calculated on-demand, and then
abandoned after use of the data. Three phonon interaction strength calculation
is the most computationally demanding part for usual applications. Detailed
techniques are used to avoid elements of the Three phonon interaction strength
if it is specified to do so.

Three phonon interaction strength calculation is the engine of more practical
(or closer to macroscopic physical properties) calculation such as lattice
thermal conductivity calculation. To minimize computational resources required
by each purpose of use, the outer function that calls this function determines
proper computational configuration of this function.

## Lattice thermal conductivity calculation

Once three phonon interaction calculation is prepared by
`ph3.init_phph_interaction()`, it is ready to run lattice thermal conductivity
calculation. The many parameters are explained in the docstring, but even with
the default parameters, the lattice thermal conductivity calculation under the
mode relaxation time approximation is performed as follows:

```python
In [5]: ph3.run_thermal_conductivity()
```

## Use of different supercell dimensions for 2nd and 3rd order FCs

Phono3py supports different supercell dimensions for second and third order
force constants (fc2 and fc3). The default setting is using the same dimensions.
Although fc3 requires much more number of supercells than fc2, in our
experience, fc3 have shorter interaction range in direct space. Therefore we
tend to expect to use smaller supercell dimension for fc3 than that for fc2. To
achieve this, `phonon_supercell_matrix` parameter exists to specify fc2
supercell dimension independently.

Using `POSCAR-unitcell` in the AlN-LDA example,

```python
In [1]: from phonopy.interface.calculator import read_crystal_structure

In [2]: unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode='vasp')

In [3]: from phono3py import Phono3py

In [4]: ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix='auto', phonon_supercell_matrix=[5, 5, 3])

In [5]: ph3.save("phono3py_disp.yaml")

In [6]: ph3.generate_displacements()
```

`Phono3py.generate_fc2_displacements()` is the method to generate displacement
dataset for fc2. This is normally unnecessary to be called because this is
called when `Phono3py.generate_displacements()` is called.
`Phono3py.generate_fc2_displacements()` may be called explicitly if non-default
control of displacement pattern is expected. See their docstrings for the
details.

When `phonon_supercell_matrix` is specified, the following attributes are
usable:

```python
Phono3py.phonon_supercell_matrix
Phono3py.phonon_dataset
Phono3py.phonon_forces
Phono3py.phonon_supercells_with_displacements
```

The meanings of them are found in their docstrings though they may be guessed
easily.

(api-phono3py-load)=

## `phono3py.load`

The purpose of `phono3py.load` is to create a `Phono3py` class instance with
basic parameters loaded from the phono3py-yaml-type file and also to try setting
up force constants and non-analytical term correction automatically from
phono3py files in the current directory.

In AlN-LDA example, the unit cell structure, supercell matrix, and primitive
matrix were recorded in the `phono3py_disp_dimfc2.yaml` file. This is easily
read a helper function of `phono3py.load`. Using ipython (or jupyter-notebook):

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
