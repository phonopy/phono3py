# Welcome to phono3py

This software calculates phonon-phonon interaction and related properties using
the supercell approach. For example, the following physical values are obtained:

- Lattice thermal conductivity by relaxation time approximation and
  direct-solution of phonon Boltzmann equation ({ref}`LTC_options`)
- {ref}`Cummulative lattice thermal conductivity and related properties <auxiliary_tools_kaccum>`
- {ref}`self_energy_options` (Phonon lifetime/linewidth)
- {ref}`jdos_option`
- {ref}`spectral_function_option`
- Built-in interfaces for {ref}`VASP <vasp_interface>`,
  {ref}`QE (pw) <qe_interface>`, {ref}`CRYSTAL <crystal_interface>`,
  {ref}`TURBOMOLE <turbomole_interface>`, and Abinit (see
  {ref}`calculator_interfaces`).
- API is prepared to operate phono3py from Python
  ([example](https://github.com/phonopy/phono3py/blob/master/example/Si-PBEsol/Si.py)).

Papers that may introduce phono3py:

- Theoretical background is summarized in this paper:
  http://dx.doi.org/10.1103/PhysRevB.91.094306 (arxiv
  http://arxiv.org/abs/1501.00691).
- Introduction to phono3py application:
  https://doi.org/10.1103/PhysRevB.97.224306 (open access).

```{image} Si-kaccum.png
:width: 20%
```

```{image} Si-kaccum-MFP.png
:width: 20%
```

```{image} Si-kdeplot.png
:width: 22%
```

## Documentation

```{toctree}
:maxdepth: 1
install
workflow
examples
Interfaces to calculators (VASP, QE, CRYSTAL, Abinit, TURBOMOLE) <interfaces>
command-options
input-output-files
hdf5_howto
auxiliary-tools
direct-solution
workload-distribution
cutoff-pair
external-tools
phono3py-api
tips
citation
changelog
```

## Mailing list

For questions, bug reports, and comments, please visit following mailing list:

https://lists.sourceforge.net/lists/listinfo/phonopy-users

Message body including attached files has to be smaller than 300 KB.

## License

BSD-3-Clause (New BSD)

## Contact

- Author: [Atsushi Togo](http://atztogo.github.io/)
