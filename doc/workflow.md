(workflow)=

# Workflow

Phono3py calculate phonon-phonon interaction related properties. Diagram shown
below illustrates workflow of lattice thermal conductivity (LTC) calculation.
The other properties such as spectral function can be calculated in similar
workflow.

The LTC calculation is performed by the following three steps:

1. Calculation of supercell-force sets (force-sets)
2. Calculation of force constants (FC)
3. Calculation of lattice thermal conductivity (LTC)

Users will call phono3py at each step with at least unit cell and supercell
matrix, which define the supercell model, as inputs.

In the first step, supercell and sets of atomic displacements are generated,
where we call the sets of the displacements as "displacement dataset".
Supercells with displacements are built from them. Then forces of the supercell
models are calculated using a force calculator such as first-principles
calculation code, which we call "force sets".

In the second step, second and third order force constants (fc2 and fc3) are
computed from the displacement datasets and force sets obtained in the first
step.

In the third step, the force constants obtained in the second step are used to
calculate lattice thermal conductivity. When the input unit cell is not a
primitive cell, primitive cell matrix is required to be given. Long-range
dipole-dipole interaction can be included when parameters for non-analytical
term correction (NAC) are provided.

```{figure} procedure.png
:align: center

Work flow of lattice thermal conductivity calculation
```
