(vasp_interface)=
# VASP & phono3py calculation

(vasp_workflow)=
## Workflow

1. Create POSCARs with displacements

   This is the same way as usual phonopy:

   ```bash
   % phono3py -d --dim="2 2 2" --pa="F" -c POSCAR-unitcell
   ```

   `phono3py_disp.yaml` and `POSCAR-xxxxx` files are created.

   If you want to use larger supercell size for
   second-order force constants (fc2) calculation than that
   for third-order force constants (fc3) calculation:

   ```bash
   % phono3py -d --dim-fc2="4 4 4" --dim="2 2 2" --pa="F" -c POSCAR-unitcell
   ```

   In this case, `POSCAR_FC2-xxxxx` files are also created.

2. Run VASP for supercell force calculations

   To calculate forces on atoms in supercells, `POSCAR-xxxxx` (and
   `POSCAR_FC2-xxxxx` if they exist) are used as VASP (or any force
   calculator) calculations.

   It is supposed that each force calculation is executed under the
   directory named `disp-xxxxx` (and `disp_fc2-xxxxx`), where
   `xxxxx` is sequential number.

3. Collect `vasprun.xml`'s

   When VASP is used as the force calculator, force sets to calculate
   fc3 and fc2 are created as follows.

   ```bash
   % phono3py --cf3 disp-{00001..00755}/vasprun.xml
   ```

   where 0755 is an example of the index of the last displacement
   supercell. To perform this collection, `phono3py_disp.yaml` created at
   step 1 is required. Then `FORCES_FC3` is created.

   When you use larger supercell for fc2 calculation:

   ```bash
   % phono3py --cf2 disp_fc2-{00001..00002}/vasprun.xml
   ```
   `phono3py_displ.yaml` is necessary in this case and `FORCES_FC2` is
   created.

4. Create `fc2.hdf` and `fc3.hdf`

   ```bash
   % phono3py --sym-fc
   ```

   `--sym-fc` symmetrizes fc3 and fc2. `fc2.hdf5` and `fc3.hdf5`
   are created from `FORCES_FC3` (and
   optionally `FORCES_FC2`) and `phono3py_disp.yaml`. This step is
   not mandatory, but you can avoid calculating fc2 and fc3 at every
   run time when reading force constants from these files with
   `--fc3` and `--fc2` options.

5. Thermal conductivity calculation

   An example of thermal conductivity calculation is:

   ```
   % phono3py --mesh="11 11 11" --br
   ```

   This calculation may take very long time. `--thm` invokes a
   tetrahedron method for Brillouin zone integration for phonon
   lifetime calculation, which is the default option. Instead,
   `--sigma` option can be used with the smearing widths.

   In this command, phonon lifetimes at many grid points are
   calculated in series. The phonon lifetime calculation at each grid
   point can be separately calculated since they
   are independent and no communication is necessary at the
   computation. The procedure is as follows:

   First run the same command with the addition option of `--wgp`:

   ```
   % phono3py --fc3 --fc2 --mesh="11 11 11" --br --wgp
   ```

   `ir_grid_points.yaml` is obtained. In this file, irreducible
   q-points are shown. Then distribute calculations of phonon
   lifetimes on grid points with `--write-gamma` option by:

   ```
   % phono3py --mesh="11 11 11" --br --write-gamma --gp="[grid ponit(s)]"
   ```

   After finishing all distributed calculations, run with
   `--read-gamma` option:

   ```
   % phono3py --fc3 --fc2 --mesh="11 11 11" --br --read-gamma
   ```

   Once this calculation runs without problem, separately calculated
   hdf5 files on grid points are no more necessary and may be deleted.
