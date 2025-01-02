(pypolymlp-interface)=

# Force constants calculation using pypolymlp (machine learning potential)

**This is an experimental feature.**

With the `--pypolymlp` option, phono3py can interface with the polynomial
machine learning potential (MLP) code,
[pypolymlp](https://github.com/sekocha/pypolymlp), to perform training and
evaluation tasks of MLPs. This feature aims to reduce the computational cost of
anharmonic force constant calculations by using MLPs as an intermediary layer,
efficiently representing atomic interactions.

The training process involves using a dataset consisting of supercell
displacements, forces, and energies. The trained MLPs are then employed to
compute forces for supercells with specific displacements.

For further details on combining phono3py calculations with pypolymlp, refer to
<u>A. Togo and A. Seko, J. Chem. Phys. **160**, 211001 (2024)</u>
[[doi](https://doi.org/10.1063/5.0211296)]
[[arxiv](https://arxiv.org/abs/2401.17531)].

An example of its usage can be found in the `example/NaCl-pypolymlp` directory
in the distribution from GitHub or PyPI.

## Citation of pypolymlp

"Tutorial: Systematic development of polynomial machine learning potentials for elemental and alloy systems", A. Seko, J. Appl. Phys. **133**, 011101 (2023).

```
@article{pypolymlp,
    author = {Seko, Atsuto},
    title = "{"Tutorial: Systematic development of polynomial machine learning potentials for elemental and alloy systems"}",
    journal = {J. Appl. Phys.},
    volume = {133},
    number = {1},
    pages = {011101},
    year = {2023},
    month = {01},
}
```

## Requirements

- [pypolymlp](https://github.com/sekocha/pypolymlp)

  For linux (x86-64), a compiled package of pypolymlp can be installed via
  conda-forge (recommended). Otherwise, pypolymlp can be installed from
  source-code.

## How to calculate

### Workflow

1. Generate random displacements in supercells. Use {ref}`--rd
   <random_displacements_option>` option.
2. Calculate corresponding forces and energies in supercells. Use of VASP
   interface is recommended for {ref}`--sp <sp_option>` option is supported.
3. Prepare dataset composed of displacements, forces, and energies in
   supercells. The dataset must be stored in a phono3py-yaml-like file, e.g.,
   `phono3py_params.yaml`. Use {ref}`--cf3 <cf3_option>` and {ref}`--sp
   <sp_option>` option simultaneously.
4. Develop MLPs. By default, 90 and 10 percents of the dataset are used for the
   training and test, respectively. At this step `phono3py.pmlp` is saved.
5. Generate displacements in supercells either systematic or random displacements.
6. Evaluate MLPs for forces of the supercells generated in step 5.
7. Calculate force constants from displacement-force dataset from steps 5 and 6.

The steps 4-7 are executed in running phono3py with `--pypolymlp`
option.

### Steps 1-3: Dataset preparation

For the training, the following supercell data are required in the phono3py
setting to use pypolymlp:

- Displacements
- Forces
- Total energies

These data must be stored in phono3py.yaml-like file.

The supercells with displacements are generated by

```
% phono3py --pa auto --rd 100 -c POSCAR-unitcell --dim 2 2 2
        _                      _____
  _ __ | |__   ___  _ __   ___|___ / _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ |_ \| '_ \| | | |
 | |_) | | | | (_) | | | | (_) |__) | |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___/____/| .__/ \__, |
 |_|                                |_|    |___/
                                           3.5.0

-------------------------[time 2024-09-19 14:40:00]-------------------------
Compiled with OpenMP support (max 10 threads).
Python version 3.12.4
Spglib version 2.5.0

Unit cell was read from "POSCAR-unitcell".
-------------------------------- unit cell ---------------------------------
Lattice vectors:
  a    5.603287477054753    0.000000000000000    0.000000000000000
  b    0.000000000000000    5.603287477054753    0.000000000000000
  c    0.000000000000000    0.000000000000000    5.603287477054753
Atomic positions (fractional):
    1 Na  0.00000000000000  0.00000000000000  0.00000000000000  22.990
    2 Na  0.00000000000000  0.50000000000000  0.50000000000000  22.990
    3 Na  0.50000000000000  0.00000000000000  0.50000000000000  22.990
    4 Na  0.50000000000000  0.50000000000000  0.00000000000000  22.990
    5 Cl  0.50000000000000  0.50000000000000  0.50000000000000  35.453
    6 Cl  0.50000000000000  0.00000000000000  0.00000000000000  35.453
    7 Cl  0.00000000000000  0.50000000000000  0.00000000000000  35.453
    8 Cl  0.00000000000000  0.00000000000000  0.50000000000000  35.453
----------------------------------------------------------------------------
Supercell (dim): [2 2 2]
Primitive matrix:
  [0.  0.5 0.5]
  [0.5 0.  0.5]
  [0.5 0.5 0. ]
Displacement distance: 0.03
Number of displacements: 100
NAC parameters were read from "BORN".
Spacegroup: Fm-3m (225)
Displacement dataset was written in "phono3py_disp.yaml".
-------------------------[time 2024-09-19 14:40:00]-------------------------
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
```

For the generated supercells, forces and energies are calculated. Here it is
assumed to use the VASP code. Once the calculations are complete, the data
(forces and energies) can be extracted using the following command:

```bash
% phono3py --sp --cf3 vasprun_xmls/vasprun-{00001..00100}.xml
```

This command extracts the necessary data and stores it in the
`phono3py_params.yaml` file. For more details, refer to the description of the
{ref}`--sp <sp_option>` option. Currently, supercell energy extraction from
calculator outputs is only supported when using the VASP interface.

``````{note}
A set of the VASP calculation results is placed in `example/NaCl-rd`. It is
obtained by

```bash
% tar xvfa ../NaCl-rd/vasprun_xmls.tar.xz
```
``````


### Steps 4-7: Force constants calculation (systematic displacements in step 5)

After developing MLPs, displacements are generated systematically considering
crystal symmetry.

Having `phono3py_params.yaml`, phono3py is executed with `--pypolymlp` option,

```
% phono3py-load --pypolymlp phono3py_params.yaml
        _                      _____
  _ __ | |__   ___  _ __   ___|___ / _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ |_ \| '_ \| | | |
 | |_) | | | | (_) | | | | (_) |__) | |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___/____/| .__/ \__, |
 |_|                                |_|    |___/
                                           3.5.0

-------------------------[time 2024-09-19 15:20:27]-------------------------
Compiled with OpenMP support (max 10 threads).
Running in phono3py.load mode.
Python version 3.12.6
Spglib version 2.5.0
----------------------------- General settings -----------------------------
HDF5 data compression filter: gzip
Crystal structure was read from "phono3py_params.yaml".
Supercell (dim): [2 2 2]
Primitive matrix:
  [0.  0.5 0.5]
  [0.5 0.  0.5]
  [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Use -v option to watch primitive cell, unit cell, and supercell structures.
NAC parameters were read from "phono3py_params.yaml".
----------------------------- Force constants ------------------------------
Displacement dataset for fc3 was read from "phono3py_params.yaml".
----------------------------- pypolymlp start ------------------------------
Pypolymlp is a generator of polynomial machine learning potentials.
Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).
Pypolymlp is developed at https://github.com/sekocha/pypolymlp.
Developing MLPs by pypolymlp...
Regression: cholesky decomposition ...
- alpha: 0.001
- alpha: 0.01
- alpha: 0.1
- alpha: 1.0
- alpha: 10.0
Clear training X.T @ X
Calculate X.T @ X for test data
Clear test X.T @ X
Regression: model selection ...
- alpha = 1.000e-03 : rmse (train, test) = 9.39542e+14 9.39543e+14
- alpha = 1.000e-02 : rmse (train, test) = 9.39542e+14 9.39543e+14
- alpha = 1.000e-01 : rmse (train, test) = 0.03738 0.04961
- alpha = 1.000e+00 : rmse (train, test) = 0.03900 0.04742
- alpha = 1.000e+01 : rmse (train, test) = 0.04058 0.04584
MLPs were written into "phono3py.pmlp"
------------------------------ pypolymlp end -------------------------------
Generate displacements
  Displacement distance: 0.001
Evaluate forces in 292 supercells by pypolymlp
Computing fc3[ 1, x, x ] using numpy.linalg.pinv.
Displacements (in Angstrom):
    [ 0.0010  0.0000  0.0000]
    [-0.0010  0.0000  0.0000]
Computing fc3[ 33, x, x ] using numpy.linalg.pinv.
Displacements (in Angstrom):
    [ 0.0010  0.0000  0.0000]
    [-0.0010  0.0000  0.0000]
Expanding fc3.
fc3 was symmetrized.
fc2 was symmetrized.
Max drift of fc3: -0.000000 (zzz) -0.000000 (zzz) -0.000000 (zzz)
Max drift of fc2: -0.000000 (zz) -0.000000 (zz)
fc3 was written into "fc3.hdf5".
fc2 was written into "fc2.hdf5".
----------- None of ph-ph interaction calculation was performed. -----------
Dataset generated using MMLPs was written in "phono3py_mlp_eval_dataset.yaml".
Summary of calculation was written in "phono3py.yaml".
-------------------------[time 2024-09-19 15:21:41]-------------------------
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
```

Information about the development of MLPs using pypolymlp is provided between
the `pypolymlp start` and `pypolymlp end` sections. The polynomial MLPs are
saved in the `phono3py.pmlp` file, which can be reused in subsequent phono3py
executions with the `--pypolymlp` option when only displacements (and no forces)
are provided.

After the MLPs are developed, systematic displacements, such as those involving
the displacement of one or two atoms in supercells, are generated with a
displacement distance of 0.001 Angstrom. The forces for these supercells are
then evaluated using pypolymlp. Both the generated displacements and the
corresponding forces are stored in the `phono3py_mlp_eval_dataset` file.

### Steps 4-7: Force constants calculation (random displacements in step 5)

After developing MLPs, random displacements are generated by specifying
{ref}`--rd <random_displacements_option>` option. To compute force constants
with random displacements, an external force constants calculator is necessary.
For this, symfc is used which is invoked by `--symfc` option.

Having `phono3py_params.yaml`, phono3py is executed with `--pypolymlp` option,

```
% phono3py-load --pypolymlp --rd 200 --symfc phono3py_params.yaml
        _                      _____
  _ __ | |__   ___  _ __   ___|___ / _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ |_ \| '_ \| | | |
 | |_) | | | | (_) | | | | (_) |__) | |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___/____/| .__/ \__, |
 |_|                                |_|    |___/
                                           3.5.0

-------------------------[time 2024-09-19 15:33:23]-------------------------
Compiled with OpenMP support (max 10 threads).
Running in phono3py.load mode.
Python version 3.12.6
Spglib version 2.5.0
----------------------------- General settings -----------------------------
HDF5 data compression filter: gzip
Crystal structure was read from "phono3py_params.yaml".
Supercell (dim): [2 2 2]
Primitive matrix:
  [0.  0.5 0.5]
  [0.5 0.  0.5]
  [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Use -v option to watch primitive cell, unit cell, and supercell structures.
NAC parameters were read from "phono3py_params.yaml".
----------------------------- Force constants ------------------------------
Displacement dataset for fc3 was read from "phono3py_params.yaml".
----------------------------- pypolymlp start ------------------------------
Pypolymlp is a generator of polynomial machine learning potentials.
Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).
Pypolymlp is developed at https://github.com/sekocha/pypolymlp.
Developing MLPs by pypolymlp...
Regression: cholesky decomposition ...
- alpha: 0.001
- alpha: 0.01
- alpha: 0.1
- alpha: 1.0
- alpha: 10.0
Clear training X.T @ X
Calculate X.T @ X for test data
Clear test X.T @ X
Regression: model selection ...
- alpha = 1.000e-03 : rmse (train, test) = 9.39542e+14 9.39543e+14
- alpha = 1.000e-02 : rmse (train, test) = 9.39542e+14 9.39543e+14
- alpha = 1.000e-01 : rmse (train, test) = 0.03738 0.04961
- alpha = 1.000e+00 : rmse (train, test) = 0.03900 0.04742
- alpha = 1.000e+01 : rmse (train, test) = 0.04058 0.04584
MLPs were written into "phono3py.pmlp"
------------------------------ pypolymlp end -------------------------------
Generate random displacements
  Twice of number of snapshots will be generated for plus-minus displacements.
  Displacement distance: 0.001
Evaluate forces in 400 supercells by pypolymlp
-------------------------------- Symfc start -------------------------------
Symfc is a non-trivial force constants calculator. Please cite the paper:
A. Seko and A. Togo, arXiv:2403.03588.
Symfc is developed at https://github.com/symfc/symfc.
Computing [2, 3] order force constants.
Increase log-level to watch detailed symfc log.
--------------------------------- Symfc end --------------------------------
-------------------------------- Symfc start -------------------------------
Symfc is a non-trivial force constants calculator. Please cite the paper:
A. Seko and A. Togo, arXiv:2403.03588.
Symfc is developed at https://github.com/symfc/symfc.
Computing [2] order force constants.
Increase log-level to watch detailed symfc log.
--------------------------------- Symfc end --------------------------------
Max drift of fc3: -0.000000 (xyx) 0.000000 (zyy) -0.000000 (xyx)
Max drift of fc2: 0.000000 (xx) 0.000000 (xx)
fc3 was written into "fc3.hdf5".
fc2 was written into "fc2.hdf5".
----------- None of ph-ph interaction calculation was performed. -----------
Dataset generated using MMLPs was written in "phono3py_mlp_eval_dataset.yaml".
Summary of calculation was written in "phono3py.yaml".
-------------------------[time 2024-09-19 15:34:41]-------------------------
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
```

The development of MLPs follows the same procedure as described for the
systematic displacements (in step 5) above.

After the MLPs are developed, 200 supercells with random directional
displacements are generated. These displacements are then inverted, resulting in
an additional 200 supercells. In total, 400 supercells are created. The forces
for these supercells are then evaluated. Finally, the force constants are
calculated using symfc.

## Parameters for developing MLPs

A few parameters can be specified using the `--mlp-params` option for the
development of MLPs. The parameters are provided as a string, e.g.,

```bash
% phono3py-load phono3py_params.yaml --pypolymlp --mlp-params="ntrain=80, ntest=20"
```

Parameters are separated by commas for configuration. A brief explanation of the
available parameters can be found in the docstring of `PypolymlpParams` that is
found by

```python
In [1]: from phonopy.interface.pypolymlp import PypolymlpParams

In [2]: help(PypolymlpParams)
```

`ntrain` and `ntest` are implemented in phono3py, while the remaining parameters
are directly passed to pypolymlp. Optimizing pypolymlp parameters can be
difficult, both in terms of achieving accuracy and managing the computational
resources required. The current default parameters are likely suitable for
systems up to ternary compounds. For binary systems, the calculations can
generally be run on standard laptop computers, but for ternary systems, around
40 GB of memory or more may be necessary.

For parameter adjustments, it is recommended to consult the
[pypolymlp](https://github.com/sekocha/pypolymlp) documentation and review the
 relevant research papers.

### `ntrain` and `ntest`

This method provides a straightforward dataset split: the first `ntrain`
supercells from the list are used for training, while the last `ntest`
supercells are reserved for testing.

## Convergence with respect to dataset size

In general, increasing the amount of data improves the accuracy of representing
force constants. Therefore, it is recommended to check the convergence of the
target property with respect to the number of supercells in the training
dataset. Lattice thermal conductivity may be a convenient property to monitor
when assessing convergence.

For example, by preparing an initial set with 100 supercell data, calculations
can then be performed by varying the size of the training dataset while keeping
the test dataset unchanged as follows:

```bash
% phono3py-load --pypolymlp --mlp-params="ntrain=20, ntest=20" --br --mesh 40 phono3py_params.yaml | tee log-20
% phono3py-load --pypolymlp --mlp-params="ntrain=40, ntest=20" --br --mesh 40 phono3py_params.yaml | tee log-40
% phono3py-load --pypolymlp --mlp-params="ntrain=60, ntest=20" --br --mesh 40 phono3py_params.yaml | tee log-60
% phono3py-load --pypolymlp --mlp-params="ntrain=80, ntest=20" --br --mesh 40 phono3py_params.yaml | tee log-80
```

The computed lattice thermal conductivities (LTCs) are plotted against the size
of the training dataset to observe LTC convergence. If the LTC has not
converged, an additional set of supercell data (e.g., forces and energies in
the next 100 supercells) will be computed and included. With this procedure in
mind, it may be convenient to generate a sufficiently large number of supercells
with random displacements in advance, such as 1000 supercells, before starting
the LTC calculation with pypolymlp.