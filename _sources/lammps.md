(lammps_interface)=

# LAMMPS & phono3py calculation

Phono3py drives a LAMMPS force calculation in the same way as an ab-initio
calculator: it builds fc3 (and optionally fc2) supercells with atomic
displacements, LAMMPS evaluates the forces on every atom, and phono3py collects
them into `FORCES_FC3` (and `FORCES_FC2`). This is convenient in combination
with machine-learning potentials (for example the ACE `pace` or the polynomial
`polymlp` `pair_style`).

## Workflow

1. (Optional) Relax the unit cell with the potential
   ({ref}`lammps_p3_structure_optimization`).
2. Generate supercells with displacements
   ({ref}`lammps_p3_supercell_generation`).
3. Compute forces with LAMMPS for every supercell
   ({ref}`lammps_p3_force_calculation`).
4. Create `FORCES_FC3` (and `FORCES_FC2`) with `phono3py-init --cf3`
   ({ref}`lammps_p3_force_calculation`).
5. Calculate the lattice thermal conductivity with `phono3py-load`
   ({ref}`lammps_p3_conductivity`).

Assumptions:

- The LAMMPS calculation uses `units metal` and `atom_style atomic`.
- LAMMPS version 15Sep2022 or later is assumed.
- Forces are read from a LAMMPS dump written in a specific format
  ({ref}`lammps_p3_force_calculation`).

A worked example is in the [example
directory](https://github.com/phonopy/phono3py/tree/master/example):
`Si-lammps-ace` (ACE `pace` potential).

(lammps_p3_structure_format)=

## LAMMPS structure format

Phono3py reads and writes crystal structures in the LAMMPS
[read_data](https://docs.lammps.org/read_data.html) format, identically to
phonopy. The supported keywords, the rotation to the LAMMPS triclinic box
convention, and the handling of `Atom Type Labels` and `Masses` are documented
in the [phonopy LAMMPS
documentation](https://phonopy.github.io/phonopy/lammps.html#lammps-structure-input-format).

Two points are worth repeating here:

- A structure written for LAMMPS is a rotated copy of the phono3py cell. The
  forces from LAMMPS are rotated back automatically when `FORCES_FC3` is created.
- Phono3py writes a `Masses` section into every supercell it generates, so the
  LAMMPS input script for the force calculation needs no `mass` command. `Masses`
  may be omitted from the input unit cell file, since the masses are taken from
  the element labels.

(lammps_p3_supercell_generation)=

## Generating supercells with displacements

There are two routes, depending on how the unit cell is provided. After either
route, `phono3py_disp.yaml` and the supercell files (`supercell`,
`supercell-00001`, `supercell-00002`, ...) are created. Many fc3 supercells are
generated because fc3 needs displacement pairs; the file index uses five digits.

### Route (a): from a LAMMPS structure file

When the unit cell is already in the LAMMPS format (e.g. `lammps_structure_Si`),
generate the supercells directly:

```
% phono3py-init --lammps -c lammps_structure_Si -d --dim 2 2 2
```

A larger supercell for fc2 than for fc3 can be requested with `--dim-fc2`, which
adds `supercell_fc2-xxxxx` files:

```
% phono3py-init --lammps -c lammps_structure_Si -d --dim 2 2 2 --dim-fc2 4 4 4
```

### Route (b): from a unit cell defined in yaml

A LAMMPS structure file is expressed in the rotated triclinic convention. To keep
the cell in its original (for example, the symmetric, unrotated) orientation,
define it in yaml and generate the supercells with a short script. A silicon
conventional cell saved as `phono3py_unitcell.yaml`:

```yaml
unit_cell:
  lattice:
    - [5.431000000000000, 0.000000000000000, 0.000000000000000] # a
    - [0.000000000000000, 5.431000000000000, 0.000000000000000] # b
    - [0.000000000000000, 0.000000000000000, 5.431000000000000] # c
  points:
    - {symbol: Si, coordinates: [0.00, 0.00, 0.00]}
    - {symbol: Si, coordinates: [0.00, 0.50, 0.50]}
    - {symbol: Si, coordinates: [0.50, 0.00, 0.50]}
    - {symbol: Si, coordinates: [0.50, 0.50, 0.00]}
    - {symbol: Si, coordinates: [0.25, 0.25, 0.25]}
    - {symbol: Si, coordinates: [0.25, 0.75, 0.75]}
    - {symbol: Si, coordinates: [0.75, 0.25, 0.75]}
    - {symbol: Si, coordinates: [0.75, 0.75, 0.25]}
```

```python
import phono3py
from phonopy.interface.calculator import write_supercells_with_displacements
from phonopy.interface.phonopy_yaml import read_cell_yaml

cell = read_cell_yaml("phono3py_unitcell.yaml")
ph3 = phono3py.load(
    unitcell=cell, supercell_matrix=[2, 2, 2], calculator="lammps", produce_fc=False
)
ph3.generate_displacements()
ph3.save("phono3py_disp.yaml")

write_supercells_with_displacements(
    "lammps", ph3.supercell, ph3.supercells_with_displacements, zfill_width=5
)
```

`primitive_matrix` defaults to `"auto"`, so it is not passed. With route (b),
`phono3py_disp.yaml` stores the cell in the original (unrotated) orientation,
whereas route (a) stores the rotated triclinic cell. In both cases the supercells
follow the LAMMPS structure file format.

(lammps_p3_force_calculation)=

## Force calculation and FORCES_FC3

Phono3py reads forces from a LAMMPS dump written in a fixed text format. For each
`supercell-xxxxx`, evaluate the forces once with `run 0` (no time integration):

```
units metal

read_data supercell-00001

pair_style  <potential>
pair_coeff  <...>

dump phonopy all custom 1 force.* id type x y z fx fy fz
dump_modify phonopy format line "%d %d %15.10f %15.10f %15.10f %15.10f %15.10f %15.10f"
run 0
```

Only the `pair_style`/`pair_coeff` lines change between potentials. Keep the
`dump` and `dump_modify` lines verbatim so that phono3py can parse the output;
`force.*` expands to `force.0` for `run 0`. Loop over all supercells, e.g.:

```bash
for f in supercell-*; do
    num=${f#supercell-}
    sed "s/supercell-00001/$f/" in.force > in.tmp
    lmp -in in.tmp
    mv force.0 force.$num
done
```

Collect the forces into `FORCES_FC3` (and `FORCES_FC2` with `--cf2` if a separate
fc2 supercell was used):

```
% phono3py-init --cf3 force.{00001..00111}
```

`phono3py_disp.yaml` is required. Phono3py subtracts the drift force (the net
force on each supercell) and rotates the forces back from the LAMMPS frame to the
phono3py cell, printing the rotation matrix `R`:

```
Forces parsed from LAMMPS output were rotated by F=R.F(lammps) with R:
  1.00000 0.00000 0.00000
  0.00000 1.00000 0.00000
  0.00000 0.00000 1.00000
```

(lammps_p3_conductivity)=

## Running the thermal conductivity calculation

Once `FORCES_FC3` exists, run `phono3py-load`, which reads `phono3py_disp.yaml`
and `FORCES_FC3` automatically. The first run also creates `fc3.hdf5` and
`fc2.hdf5`. For example, the BTE-RTA lattice thermal conductivity at 300 K with
an 11x11x11 sampling mesh:

```
% phono3py-load --mesh 11 11 11 --br --ts 300
```

`kappa-m111111.hdf5` is written with the result.

(lammps_p3_examples)=

## Examples

### Si-lammps-ace (ACE)

This uses the LAMMPS `pair_style pace` (atomic cluster expansion). Any LAMMPS
build with the `ML-PACE` package can run it, including the conda-forge `lammps`
package:

```
pair_style pace
pair_coeff * * Si_npj_CompMat2021.ace Si
```

The potential file `Si_npj_CompMat2021.ace` is from the dataset accompanying Y.
Lysogorskiy *et al.*, "Performant implementation of the atomic cluster expansion
(PACE) and application to copper and silicon", npj Comput. Mater. **7**, 97
(2021), distributed on Zenodo (<https://doi.org/10.5281/zenodo.4734036>) under
the **CC-BY-4.0** license. Download it into the working directory; it is not
redistributed with phono3py.

The example uses the 8-atom conventional cell and a 2x2x2 supercell, the same as
the VASP `Si-PBEsol` example, so the lattice thermal conductivities can be
compared directly. At 300 K the ACE result is 118.9 W/m-K (11x11x11 mesh) and
132.7 W/m-K (19x19x19 mesh), somewhat higher than the PBEsol values of 109.1 and
124.4 W/m-K.

The unit cell uses the experimental room-temperature lattice constant of
silicon, a = 5.431 A. Because the ACE equilibrium constant differs slightly, a
small residual stress may appear as near-zero or slightly imaginary acoustic
frequencies around Gamma; relax the cell with the potential to remove it (see the
appendix below).

(lammps_p3_structure_optimization)=

## Appendix: structure optimization using LAMMPS

Relax the crystal structure with the potential before the phonon calculation so
that the residual forces, and the residual stress on the lattice, vanish. The
following relaxes both the cell and the internal coordinates:

```
units metal

read_data unitcell

pair_style  pace
pair_coeff * * Si_npj_CompMat2021.ace Si

variable etol equal 0.0
variable ftol equal 1e-8
variable maxiter equal 1000
variable maxeval equal 100000

fix relax all box/relax iso 0.0 vmax 0.001
minimize ${etol} ${ftol} ${maxiter} ${maxeval}

write_data dump.unitcell
```

Drop the `fix box/relax` line to relax only the internal coordinates. More
instruction is found at
<https://gist.github.com/lan496/e9dff8449cd7489f6722b276282e66a0>.
