(workflow)=

# Workflow

Phono3py calculate phonon-phonon interaction related properties. Diagram shown
below is an example of workflow to calculate lattice thermal conductivity using
phono3py. The other properties such as spectral function can be calculated in
similar ways.

The calculation is divided into roughly three parts:

1. Calculation of supercell-force sets
2. Calculation of force constants
3. Calculation of lattice thermal conductivity (or other properties)

```{figure} procedure.png
:scale: 80
:align: center

Work flow of lattice thermal conductivity calculation
```
