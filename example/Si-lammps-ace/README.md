# Si lattice thermal conductivity using the LAMMPS interface (ACE potential)

This example computes the lattice thermal conductivity of diamond silicon using
the LAMMPS `pair_style pace` (atomic cluster expansion, ACE). It mirrors the
VASP `Si-PBEsol` example (the same 8-atom conventional cell and 2x2x2 supercell),
so the results can be compared directly. Any LAMMPS build that includes the
`ML-PACE` package can run it, for instance the conda-forge `lammps` package.

## Potential file and license

The silicon ACE potential file `Si_npj_CompMat2021.ace` is taken from the
dataset accompanying

> Y. Lysogorskiy, C. van der Oord, A. Bochkarev, S. Menon, M. Rinaldi, T.
> Hammerschmidt, M. Mrovec, A. Thompson, G. Csanyi, C. Ortner, and R. Drautz,
> "Performant implementation of the atomic cluster expansion (PACE) and
> application to copper and silicon", npj Comput. Mater. 7, 97 (2021).

It is distributed on Zenodo under the **CC-BY-4.0** license:

> DOI: 10.5281/zenodo.4734036 (https://doi.org/10.5281/zenodo.4734036)

Download `Si_npj_CompMat2021.ace` from that record into this directory before
running LAMMPS. The file is not redistributed with phono3py; please observe the
CC-BY-4.0 attribution terms (cite the reference above).

## Lattice constant

The unit cell uses the experimental room-temperature (~300 K) lattice constant
of silicon, a = 5.431 A (`lammps_structure_Si` and `phono3py_unitcell.yaml`). The
ACE equilibrium lattice constant differs slightly from this value, so a small
residual stress may show up as near-zero or slightly imaginary acoustic
frequencies around Gamma. To remove it, relax the cell with the potential first
(see the structure-optimization appendix in the
[phono3py LAMMPS documentation](https://phonopy.github.io/phono3py/lammps.html))
and use the relaxed constant instead.

## Steps

1. Generate fc3 supercells with displacements. Two equivalent routes are
   provided.

   (a) From the LAMMPS structure file `lammps_structure_Si`:

   ```
   % phono3py-init --lammps -c lammps_structure_Si -d --dim 2 2 2
   ```

   (b) From the unit cell defined in `phono3py_unitcell.yaml`:

   ```
   % python generate_displacements.py
   ```

   Both create `phono3py_disp.yaml` and 111 supercell files
   (`supercell-00001` ... `supercell-00111`). Route (b) keeps the cell in its
   original (unrotated) orientation in `phono3py_disp.yaml`, whereas route (a)
   reads the cell in the rotated LAMMPS triclinic convention. Either way, the
   supercells follow the LAMMPS structure input format, and the forces obtained
   from LAMMPS are rotated back to the original coordinate system automatically
   in step 3.

2. Run LAMMPS for every supercell. `in.force` reads `supercell-00001`; loop over
   all supercells, e.g. in bash:

   ```bash
   for f in supercell-*; do
       num=${f#supercell-}
       sed "s/supercell-00001/$f/" in.force > in.tmp
       lmp -in in.tmp
       mv force.0 force.$num
   done
   ```

   The masses are written into the supercell files, so `in.force` needs no
   `mass` command.

3. Collect the forces into `FORCES_FC3`:

   ```
   % phono3py-init --cf3 force.{00001..00111}
   ```

   `phono3py_disp.yaml` created at step 1 is required. The drift force (the net
   force on each supercell) is subtracted and the forces are rotated back to the
   original orientation; the rotation matrix is printed in the output.

4. Calculate the lattice thermal conductivity (this also creates `fc3.hdf5` and
   `fc2.hdf5` on the first run):

   ```
   % phono3py-load --mesh 11 11 11 --br --ts 300
   ```

   `kappa-m111111.hdf5` is written. The lattice thermal conductivity at 300 K is
   118.9 W/m-K with the 11x11x11 sampling mesh, and 132.7 W/m-K with the
   19x19x19 mesh.

   For comparison, the VASP `Si-PBEsol` example (same cell and supercell) gives
   109.1 W/m-K (11x11x11) and 124.4 W/m-K (19x19x19). The ACE potential yields a
   somewhat higher conductivity, closer to the experimental value of silicon.
