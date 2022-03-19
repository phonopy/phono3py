(phono3py_api)=

# Phono3py API

```{contents}
:depth: 2
:local:
```

## `Phono3py` class

To operate phono3py from python, there is phono3py API. The main class is
`Phono3py` that is imported by

```python
from phono3py import Phono3py
```

The minimum set of inputs are `unitcell` (1st argument), `supercell_matrix` (2nd
argument), and `primitive_matrix`, which are used as

```python
ph3 = Phono3py(unitcell,
               [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
               primitive_matrix='auto')
```

The `unitcell` is an instance of the
[`PhonopyAtoms` class](https://phonopy.github.io/phonopy/phonopy-module.html#phonopyatoms-class).
`supercell_matrix` and `primitive_matrix` are the transformation matrices to
generate supercell and primitive cell from `unitcell` (see
[definitions](https://phonopy.github.io/phonopy/phonopy-module.html#definitions-of-variables)).
This step is similar to the
[instantiation of `Phonopy` class](https://phonopy.github.io/phonopy/phonopy-module.html#pre-process).
