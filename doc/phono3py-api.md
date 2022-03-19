(phono3py_api)=

# Phono3py API

```{contents}
:depth: 2
:local:
```

How to use phono3py API is described below along with snippets that work with
[`AlN-LDA` example](https://github.com/phonopy/phono3py/tree/develop/example/AlN-LDA).

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

There are many parameters that can be given to `Phono3py` class. The details are
written in the docstring, which is shown by

```python
help(Phono3py)
```

The `unitcell` is an instance of the
[`PhonopyAtoms` class](https://phonopy.github.io/phonopy/phonopy-module.html#phonopyatoms-class).
`supercell_matrix` and `primitive_matrix` are the transformation matrices to
generate supercell and primitive cell from `unitcell` (see
[definitions](https://phonopy.github.io/phonopy/phonopy-module.html#definitions-of-variables)).
This step is similar to the
[instantiation of `Phonopy` class](https://phonopy.github.io/phonopy/phonopy-module.html#pre-process).

When we want `unitcell` from a text file of a force-calculator format, the
`read_crystal_structure` function in phonopy can be used, e.g., in the AlN-LDA
example,

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

In AlN-LDA example, the unit cell structure, supercell matrix, and primitive
matrix were recorded in the `phono3py_disp_dimfc2.yamll` file. This is easily
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
In [6]: ph3 = Phono3py(unitcell, supercell_matrix=[3, 3, 2], primitive_matrix='auto')

In [7]: ph3.generate_displacements()

In [8]: len(ph3.supercells_with_displacements)
Out[8]: 1254

In [9]: type(ph3.supercells_with_displacements[0])
Out[9]: phonopy.structure.atoms.PhonopyAtoms
```

By this, 1254 supercells with displacements were generated.

The generated displacement dataset is used in the force constants calculation.
Therefore, it is recommended to save it into a file if the python process having
the `Phono3py` class instance is expected to be terminated. The displacement
dataset and crystal structure information are saved to a file by
`Phono3py.save()` method:

```python
In [11]: ph3.save("phono3py_disp.yaml")
```

## Supercell force calculation

Forces of the generated supercells with displacements are calculated by some
external force calculator such as first-principles calculation code.

Calculated supercell forces will be stored in `Phono3py` class instance through
`Phono3py.forces` attribute by setting an array_like variable with the shape of
`(num_supercells, num_atoms_in_supercell, 3)`. In the above example, the array
shape is `(1254, 72, 3)`.

If calculated force sets are stored in the {ref}`input-output_files_FORCES_FC3`
format, the numpy array of `forces` is obtained by

```python
forces = np.loadtxt("FORCES_FC3").reshape(-1, num_atoms_in_supercell, 3)
assert len(forces) == num_supercells
```
