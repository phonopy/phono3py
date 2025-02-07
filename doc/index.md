# Welcome to phono3py

This software calculates phonon-phonon interaction and related properties using
the supercell approach. For example, the following physical values are obtained:

- {ref}`Lattice thermal conductivity by relaxation time approximation
and direct-solution of phonon Boltzmann equation and
the Wigner transport equation <LTC_options>`
- {ref}`Cumulative lattice thermal conductivity and related properties <auxiliary_tools_kaccum>`
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
  [PRB.91.094306](http://dx.doi.org/10.1103/PhysRevB.91.094306) (arxiv
  [1501.00691](http://arxiv.org/abs/1501.00691>)).
- Introduction to phono3py application:
  [JPSJ.92.012001](https://journals.jps.jp/doi/10.7566/JPSJ.92.012001) (open access),
  and phono3py inputs for 103 compounds found in Fig.17
  <https://github.com/atztogo/phonondb/blob/main/mdr/phono3py_103compounds_fd/README.md>
- Implementation of phono3py:
  [JPCM.35.353001](https://iopscience.iop.org/article/10.1088/1361-648X/acd831)
  (open access)

```{toctree}
:hidden:
install
workflow
examples
Interfaces to calculators (VASP, QE, CRYSTAL, Abinit, TURBOMOLE) <interfaces>
command-options
input-output-files
hdf5_howto
auxiliary-tools
direct-solution
wigner-solution
workload-distribution
random-displacements
pypolymlp
cutoff-pair
external-tools
phono3py-api
phono3py-load
tips
citation
reference
changelog
```

## Mailing list

For questions, bug reports, and comments, please visit following mailing list:

<https://lists.sourceforge.net/lists/listinfo/phonopy-users>

Message body including attached files has to be smaller than 300 KB.

## License

BSD-3-Clause (New BSD)

## Contributors

- Atsushi Togo, National Institute for Materials Science

## Acknowledgements

Phono3py development is supported by:

- National Institute for Materials Science
