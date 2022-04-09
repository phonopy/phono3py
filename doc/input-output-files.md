(input-output_files)=

# Input / Output files

```{contents}
:depth: 3
:local:
```

The calculation results are written into files. Mostly the data are stored in
HDF5 format, therefore how to read the data from HDF5 files is also shown.

## Intermediate text files

The following files are not compatible with phonopy. But phonopy's `FORCE_SETS`
file can be created using phono3py command options from the following files. See
the detail at {ref}`file_format_compatibility`.

### `phono3py_disp.yaml`

This is created with `-d` option. See {ref}`create_displacements_option`.

This file contains displacement dataset and crystal structure information.

(input-output_files_FORCES_FC3)=

### `FORCES_FC3`

This is created with `--cf3` option. See {ref}`cf3_option`.

There are two formats of `FORCES_FC3`. The type-I format is like that shown
below

```
# File: 1
# 1       0.0300000000000000   0.0000000000000000   0.0000000000000000
  -0.6458483000    0.0223064300   -0.0143299700
   0.0793497000    0.0088413200   -0.0052766800
   0.0768176500   -0.0095501600    0.0057262300
  -0.0016552800   -0.0366684600   -0.0059480700
  -0.0023432300    0.0373490000    0.0059468600
   0.0143901800    0.0000959800   -0.0001100900
  -0.0019487200   -0.0553591300   -0.0113649500
   0.0143732700   -0.0000614400    0.0000502600
  -0.0020311400    0.0554678300    0.0115355100
...
# File: 1254
# 37      0.0000000000000000   0.0000000000000000  -0.0300000000000000
# 68      0.0000000000000000   0.0000000000000000  -0.0300000000000000
  -0.0008300600   -0.0004792400    0.0515596200
  -0.0133197900   -0.0078480800    0.0298334900
   0.0141518600   -0.0105405200    0.0106313000
   0.0153762500   -0.0072671600    0.0112864200
  -0.0134565300   -0.0076112400    0.0298334900
  -0.0019180000   -0.0011073600    0.0272454300
   0.0013945800    0.0169498000    0.0112864200
   0.0006578200    0.0003797900    0.0085617600
  -0.0020524300    0.0175261300    0.0106313000
   0.0019515200    0.0011267100   -0.2083651200
   0.0148675800   -0.0516285500   -0.0924200600
  -0.0168043800    0.0074232400   -0.0122506300
  -0.0128831200    0.0114004400   -0.0110906700
...
```

This file contains supercell forces. Lines starting with `#` is ignored when
parsing. Each line gives forces of at atom in Cartesian coordinates. All forces
of atoms in each supercell are written in the same order as the atoms in the
supercell. All forces of all supercells are concatenated. If force sets are
stored in a numpy array (`forces`) of the shape of
`(num_supercells, num_atoms_in_supercell, 3)`, this file is generated using
numpy as follows:

```python
np.savetxt("FORCES_FC3", forces.reshape(-1, 3))
```

The type-II format is the same as
[phonopy's type-II format](https://phonopy.github.io/phonopy/input-files.html#type-2)
of `FORCE_SETS`.

### `FORCES_FC2`

This is created with `--cf2` option. See {ref}`cf2_option` and
{ref}`dim_fc2_option`.

The file formats (type-I and type-II) are same as those of `FORCES_FC3`.

## HDF5 files

### `kappa-*.hdf5`

See the detail at {ref}`kappa_hdf5_file`.

(fc3_hdf5_file)=

### `fc3.hdf5`

Third order force constants (in real space) are stored in
$\mathrm{eV}/\text{Angstrom}^3$.

In phono3py, this is stored in the numpy array `dtype='double'` and `order='C'`
in the shape of:

```
(num_atom, num_atom, num_atom, 3, 3, 3)
```

against $\Phi_{\alpha\beta\gamma}(l\kappa, l'\kappa',
l''\kappa'')$. The first
three `num_atom` are the atom indices in supercell corresponding to $l\kappa$,
$l'\kappa'$, $l''\kappa''$, respectively. The last three elements are the
Cartesian coordinates corresponding to $\alpha$, $\beta$, $\gamma$,
respectively.

If you want to import a supercell structure and its fc3, you may suffer from
matching its atom index between the supercell and an expected unit cell. This
may be easily dealt with by letting phono3py see your supercell as the unit cell
(e.g., `POSCAR`, `unitcell.in`, etc) and find the unit (primitive) cell using
{ref}`--pa option <pa_option>`. For example, let us assume your supercell is the
2x2x2 multiples of your unit cell that has no centring, then your `--pa` setting
will be `1/2 0 0 0 1/2 0 0 1/2 0`. If your unit cell is a conventional unit cell
and has a centring, e.g., the face centring,

$$
(\mathbf{a}_\text{p}, \mathbf{b}_\text{p}, \mathbf{c}_\text{p}) =
(\mathbf{a}_\text{s}, \mathbf{b}_\text{s}, \mathbf{c}_\text{s})
\begin{pmatrix}
\frac{{1}}{2} & 0 & 0 \\
0 & \frac{{1}}{2} & 0 \\
0 & 0 & \frac{{1}}{2}
\end{pmatrix}
\begin{pmatrix}
0 & \frac{{1}}{2} & \frac{{1}}{2} \\
\frac{{1}}{2} & 0 & \frac{{1}}{2} \\
\frac{{1}}{2} & \frac{{1}}{2} & 0
\end{pmatrix} =
(\mathbf{a}_\text{s}, \mathbf{b}_\text{s}, \mathbf{c}_\text{s})
\begin{pmatrix}
0 & \frac{{1}}{4} & \frac{{1}}{4} \\
\frac{{1}}{4} & 0 & \frac{{1}}{4} \\
\frac{{1}}{4} & \frac{{1}}{4} & 0
\end{pmatrix}.
$$

So what you have to set is `--pa="0 1/4 1/4 1/4 0 1/4 1/4 1/4 0"`.

(fc2_hdf5_file)=

### `fc2.hdf5`

Second order force constants are stored in $\mathrm{eV}/\text{Angstrom}^2$.

In phono3py, this is stored in the numpy array `dtype='double'` and `order='C'`
in the shape of:

```
(num_atom, num_atom, 3, 3)
```

against $\Phi_{\alpha\beta}(l\kappa, l'\kappa')$. More detail is similar to the
case for {ref}`fc3_hdf5_file`.

### `gamma-*.hdf5`

Imaginary parts of self energies at harmonic phonon frequencies
($\Gamma_\lambda(\omega_\lambda)$ = half linewidths) are stored in THz. See
{ref}`write_gamma_option`.

### `gamma_detail-*.hdf5`

Q-point triplet contributions to imaginary parts of self energies at phonon
frequencies (half linewidths) are stored in THz. See
{ref}`write_detailed_gamma_option`.

## Simple text file

### `gammas-*.dat`

Imaginary parts of self energies with respect to frequency
$\Gamma_\lambda(\omega)$ are stored in THz. See {ref}`ise_option`.

### `jdos-*.dat`

Joint densities of states are stored in Thz. See {ref}`jdos_option`.

### `linewidth-*.dat`
