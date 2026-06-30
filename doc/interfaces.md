(calculator_interfaces)=
# Interfaces to calculators

Currently the built-in interfaces for VASP, QUANTUM ESPRESSO (QE),
CRYSTAL, Abinit, TURBOMOLE, and LAMMPS are prepared. VASP is the default
interface and no special option is necessary to invoke it, but for the other
interfaces, each special option has to be specified, e.g. `--qe`,
`--crystal`, `--abinit`, `--turbomole`, or `--lammps`

```{toctree}
:maxdepth: 1
vasp
qe
crystal
turbomole
lammps
```

## Calculator specific behaviors

### Physical unit

The interfaces for VASP, QE (pw), CRYSTAL, Abinit, TURBOMOLE, and LAMMPS
are built in to the phono3py command.

For each calculator, each physical unit system is used. The physical
unit systems used for the calculators are summarized below.

```{table}
| calculator | unit-cell | FORCES_FC3  | phono3py_disp.yaml |
|------------|-----------|-------------|--------------------|
| VASP       | Angstrom  | eV/Angstrom | Angstrom           |
| QE (pw)    | au (bohr) | Ry/au       | au                 |
| CRYSTAL    | Angstrom  | eV/Angstrom | Angstrom           |
| Abinit     | au (bohr) | eV/Angstrom | au                 |
| TURBOMOLE  | au (bohr) | hartree/au  | au                 |
| LAMMPS     | Angstrom  | eV/Angstrom | Angstrom           |
```

`FORCES_FC3`, `FORCES_FC2`, and `phono3py_disp.yaml` have the same physical units.

Always (irrespective of calculator interface) the physical units of
2nd and 3rd order force constants that are to be stored in
`fc2.hdf5` and `fc3.hdf5` are $\text{eV}/\text{Angstrom}^2$ and
$\text{eV}/\text{Angstrom}^3$, respectively.

(default_unit_cell_file_name_for_calculator)=
### Default unit cell file name

Default unit cell file names are also changed according to the calculators::

```{table}
| calculator| filename    |
|-----------|-------------|
| VASP      | POSCAR      |
| QE        | unitcell.in |
| CRYSTAL   | crystal.o   |
| Abinit    | unitcell.in |
| TURBOMOLE | control     |
| LAMMPS    | unitcell    |
```

(default_displacement_distance_for_calculator)=
### Default displacement distance created

Default displacement distances created by `-d` option without
`--amplitude` option are respectively as follows::

```{table}
| calculator | distance       |
|------------|----------------|
| VASP       | 0.03 Angstrom  |
| QE         | 0.06 au (bohr) |
| CRYSTAL    | 0.03 Angstrom  |
| Abinit     | 0.06 au (bohr) |
| TURBOMOLE  | 0.06 au (bohr) |
| LAMMPS     | 0.03 Angstrom  |
```
