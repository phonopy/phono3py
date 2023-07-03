(qe_interface)=
# Quantum ESPRESSO (pw) & phono3py calculation

Quantum espresso package itself has a set of the force constants
calculation environment based on DFPT. But the document here explains how
to calculate phonon-phonon interaction and related properties using
phono3py, i.e., using the finite displacement and supercell approach.

An example for QE (pw) is found in the `example-phono3py/Si-QE` directory.

Unless a proper `phono3py_disp.yaml` containing calculator information,
to invoke the QE (pw) interface, `--qe` option has to be specified,

```bash
% phono3py --qe [options] [arguments]
```

When the file name of the unit cell is different from the default one
(see {ref}`default_unit_cell_file_name_for_calculator`), `-c` option
is used to specify the file name. QE (pw) unit cell file parser used in
phono3py is the same as that in phonopy. It can read
only limited number of keywords that are shown in the phonopy web site
(http://phonopy.github.io/phonopy/qe.html#qe-interface).

(qe_workflow)=
## Workflow

1. Create supercells with displacements

   ```bash
   % phono3py --qe -d --dim="2 2 2" --pa="F" -c Si.in
   ```

   In this example, probably 111 different supercells with
   displacements are created. Supercell files (`supercell-xxx.in`)
   are created but they contain only the crystal
   structures. Calculation setting has to be added before running the
   calculation. In this step, the option `--qe` is necessary.

2. Run QE (pw) for supercell force calculations

   Let's assume that the calculations have been made in `disp-xxx`
   directories with the file names of `Si-supercell.in`. Then after
   finishing those calculations, `Si-supercell.out` may be created
   in each directory.

3. Collect forces

   `FORCES_FC3` is obtained with `--cf3` options collecting the
   forces on atoms in QE (pw) calculation results:

   ```bash
   % phono3py --cf3 disp-00001/Si-supercell.out disp-00002/Si-supercell.out ...
   ```

   or in recent bash or zsh:

   ```bash
   % phono3py --cf3 disp-{00001..00111}/Si-supercell.out
   ```

   `phono3py_disp.yaml` is used to create `FORCES_FC3`, therefore it
   must exist in current directory.

4) Calculate 3rd and 2nd order force constants

   `fc3.hdf5` and `fc2.hdf5` files are created by:

   ```bash
   % phono3py --sym-fc
   ```

   where `--sym-fc` symmetrizes fc3 and fc2.

5) Calculate lattice thermal conductivity, e.g., by:

   ```bash
   % phono3py --mesh="11 11 11" --fc3 --fc2 --br
   ```
