(input-output_files)=

# Input / Output files

The calculation results are written into files. Mostly the data are stored in
HDF5 format, therefore how to read the data from HDF5 files is also shown.

## `phono3py_disp.yaml`

This is created with {ref}`-d <create_displacements_option>` or
{ref}`--rd <random_displacements_option>` option.
This file contains displacement dataset and crystal structure information.
Parameters for non-analytical term correction can be also included.

## `phono3py_params.yaml`


This is created with the combination of {ref}`--cf3 <cf3_option>` and {ref}`--sp
<sp_option>` options. This file contains displacement-force dataset and crystal
structure information. In addition, energies of supercells may be included in
the dataset. Parameters for non-analytical term correction can be also included.

(iofile_FORCES_FC3)=

## `FORCES_FC3`

This is created with {ref}`--cf3 <cf3_option>` option . There are two formats of
`FORCES_FC3`. The type-I format is like that shown below

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

## `FORCES_FC2`

This is created with {ref}`--cf2 <dim_fc2_option>` option. The file formats
(type-I and type-II) are same as those of `FORCES_FC3`.

(iofile_fc3_hdf5)=
## `fc3.hdf5`

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

(iofile_fc2_hdf5)=
## `fc2.hdf5`

Second order force constants are stored in $\mathrm{eV}/\text{Angstrom}^2$.

In phono3py, this is stored in the numpy array `dtype='double'` and `order='C'`
in the shape of:

```
(num_atom, num_atom, 3, 3)
```

against $\Phi_{\alpha\beta}(l\kappa, l'\kappa')$. More detail is similar to the
case for {ref}`iofile_fc3_hdf5`.

(iofile_kappa_hdf5)=
## `kappa-*.hdf5`

Files name, e.g. `kappa-m323220.hdf5`, is determined by some
specific options. `mxxx`, show the numbers of sampling
mesh. `sxxx` and `gxxx` appear optionally. `sxxx` gives the
smearing width in the smearing method for Brillouin zone integration
for phonon lifetime, and `gxxx` denotes the grid number. Using the
command option of `-o`, the file name can be modified slightly. For
example `-o nac` gives `kappa-m323220.nac.hdf5` to
memorize the option `--nac` was used.

### `mesh`

(Versions 1.10.11 or later)

The numbers of mesh points for reciprocal space sampling along
reciprocal axes, $a^*, b^*, c^*$.

### `frequency`

Phonon frequencies. The physical unit is THz, where THz
is in the ordinal frequency not the angular frequency.

The array shape is (irreducible q-point, phonon band).

(iofile_kappa_hdf5_gamma)=
### `gamma`

Imaginary part of self energy of phonon bubble diagram (phonon-phonon
scattering). The physical unit is THz, where THz is in the ordinal frequency not
the angular frequency.

The array shape for all grid-points (irreducible q-points) is
(temperature, irreducible q-point, phonon band).

The array shape for a specific grid-point is
(temperature, phonon band).

Phonon lifetime ($\tau_\lambda=1/2\Gamma_\lambda(\omega_\lambda)$) may
be estimated from `gamma`. $2\pi$ has to be multiplied with
`gamma` values in the hdf5 file to convert the unit of ordinal
frequency to angular frequency. Zeros in `gamma` values mean that
those elements were not calculated such as for three acoustic modes at
$\Gamma$ point. The below is the copy-and-paste from the
previous section to show how to obtain phonon lifetime in pico
second:

```python
In [8]: g = f['gamma'][30]

In [9]: import numpy as np

In [10]: g = np.where(g > 0, g, -1)

In [11]: lifetime = np.where(g > 0, 1.0 / (2 * 2 * np.pi * g), 0)
```

### `gamma_isotope`

Isotope scattering of $1/2\tau^\mathrm{iso}_\lambda$.
The physical unit is same as that of gamma.

The array shape is same as that of frequency.

### `group_velocity`

Phonon group velocity, $\nabla_\mathbf{q}\omega_\lambda$. The
physical unit is $\text{THz}\cdot\text{Angstrom}$, where THz
is in the ordinal frequency not the angular frequency.

The array shape is (irreducible q-point, phonon band, 3 = Cartesian coordinates).

### `heat_capacity`

Mode-heat-capacity defined by

$$
C_\lambda = k_\mathrm{B}
\left(\frac{\hbar\omega_\lambda}{k_\mathrm{B} T} \right)^2
\frac{\exp(\hbar\omega_\lambda/k_\mathrm{B}
T)}{[\exp(\hbar\omega_\lambda/k_\mathrm{B} T)-1]^2}.
$$

The physical unit is eV/K.

The array shape is (temperature, irreducible q-point, phonon band).

(iofile_kappa_hdf5_kappa)=
### `kappa`

Thermal conductivity tensor. The physical unit is W/m-K.

The array shape is (temperature, 6 = (xx, yy, zz, yz, xz, xy)).

### `mode-kappa`

Thermal conductivity tensors at k-stars (${}^*\mathbf{k}$):

$$
\sum_{\mathbf{q} \in {}^*\mathbf{k}} \kappa_{\mathbf{q}j}.
$$

The sum of this over ${}^*\mathbf{k}$ corresponding to
irreducible q-points divided by number of grid points gives
$\kappa$ ({ref}`iofile_kappa_hdf5_kappa`), e.g.,:

```python
kappa_xx_at_index_30 = mode_kappa[30, :, :, 0].sum()/ weight.sum()
```

Be careful that until version 1.12.7, mode-kappa values were divided
by number of grid points.

The physical unit is W/m-K. Each tensor element is the sum of tensor
elements on the members of ${}^*\mathbf{k}$, i.e., symmetrically
equivalent q-points by crystallographic point group and time reversal
symmetry.

The array shape is (temperature, irreducible q-point, phonon band, 6 =
(xx, yy, zz, yz, xz, xy)).

### `gv_by_gv`

Outer products of group velocities for k-stars
(${}^*\mathbf{k}$) for each irreducible q-point and phonon band
($j$):

$$
\sum_{\mathbf{q} \in {}^*\mathbf{k}} \mathbf{v}_{\mathbf{q}j} \otimes
\mathbf{v}_{\mathbf{q}j}.
$$

The physical unit is
$\text{THz}^2\cdot\text{Angstrom}^2$, where THz is in the
ordinal frequency not the angular frequency.

The array shape is (irreducible q-point, phonon band, 6 = (xx, yy, zz,
yz, xz, xy)).

### `q-point`

Irreducible q-points in reduced coordinates.

The array shape is (irreducible q-point, 3 = reduced
coordinates in reciprocal space).

### `temperature`

Temperatures where thermal conductivities are calculated. The physical
unit is K.

### `weight`

Weights corresponding to irreducible q-points. Sum of weights equals to
the number of mesh grid points.

### `ave_pp`

Averaged phonon-phonon interaction $P_{\mathbf{q}j}$ in $\text{eV}^2$:

$$
P_{\mathbf{q}j} = \frac{1}{(3n_\mathrm{a})^2} \sum_{\lambda'\lambda''}
|\Phi_{\lambda\lambda'\lambda''}|^2.
$$

This is not going to be calculated in the RTA thermal coductivity
calculation mode by default. To calculate this, `--full-pp` option
has to be specified (see {ref}`full_pp_option`).

### `boundary_mfp`

A value specified by {ref}`boundary_mfp_option`. The physical unit is
micrometer.

When `--boundary-mfp` option is explicitly specified, its value is stored here.

### `kappa_unit_conversion`

This is used to convert the physical unit of lattice thermal
conductivity made of `heat_capacity`, `group_velocity`, and
`gamma`, to W/m-K. In the single mode relaxation time (SMRT) method,
a mode contribution to the lattice thermal conductivity is given by

$$
\kappa_\lambda = \frac{1}{V_0} C_\lambda \mathbf{v}_\lambda \otimes
\mathbf{v}_\lambda \tau_\lambda^{\mathrm{SMRT}}.
$$

For example, $\kappa_{\lambda,{xx}}$ is calculated by:

```python
In [1]: import h5py

In [2]: f = h5py.File("kappa-m111111.hdf5")

In [3]: kappa_unit_conversion = f['kappa_unit_conversion'][()]

In [4]: weight = f['weight'][:]

In [5]: heat_capacity = f['heat_capacity'][:]

In [6]: gv_by_gv = f['gv_by_gv'][:]

In [7]: gamma = f['gamma'][:]

In [8]: kappa_unit_conversion * heat_capacity[30, 2, 0] * gv_by_gv[2, 0] / (2 * gamma[30, 2, 0])

Out[8]:
array([  1.02050241e+03,   1.02050241e+03,   1.02050241e+03,
         4.40486382e-15,   0.00000000e+00,  -4.40486382e-15])

In [9]: f['mode_kappa'][30, 2, 0]
Out[9]:
array([  1.02050201e+03,   1.02050201e+03,   1.02050201e+03,
         4.40486209e-15,   0.00000000e+00,  -4.40486209e-15])
```

(iofile_kappa_hdf5_gamma_NU)=
### `gamma_N` and `gamma_U`

The data are stored in `kappa-mxxx(-gx-sx-sdx).hdf5` file and accessed by
`gamma_N` and `gamma_U` keys. The shape of the arrays is the same as that of
`gamma` (see {ref}`iofile_kappa_hdf5_gamma`). An example (Si-PBEsol) is shown
below:

```bash
% phono3py-load --mesh 11 11 11 --fc3 --fc2 --br --nu
...
% ipython
```

```python
In [1]: import h5py

In [2]: f = h5py.File("kappa-m111111.hdf5", 'r')

In [3]: list(f)
Out[3]:
['frequency',
 'gamma',
 'gamma_N',
 'gamma_U',
 'group_velocity',
 'gv_by_gv',
 'heat_capacity',
 'kappa',
 'kappa_unit_conversion',
 'mesh',
 'mode_kappa',
 'qpoint',
 'temperature',
 'weight']

In [4]: f['gamma'].shape
Out[4]: (101, 56, 6)

In [5]: f['gamma_N'].shape
Out[5]: (101, 56, 6)

In [6]: f['gamma_U'].shape
Out[6]: (101, 56, 6)
```

## `gamma-*.hdf5`

Imaginary parts of self energies at harmonic phonon frequencies
($\Gamma_\lambda(\omega_\lambda)$ = half linewidths) are stored in THz. See
{ref}`write_gamma_option`.

(iofile_gamma_detail_hdf5)=
## `gamma_detail-*.hdf5`

Q-point triplet contributions to imaginary parts of self energies at phonon
frequencies (half linewidths) are stored in THz. See
{ref}`--write-gamma-detail <write_detailed_gamma_option>` option.

In the output file in hdf5, following keys are used to extract the detailed
information.

```{table}
| dataset                     | Array shape                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| gamma_detail for `--ise`    | (temperature, sampling frequency point, symmetry reduced set of triplets at given grid point, band1, band2, band3) in THz (without $2\pi$) |
| gamma_detail for `--br`     | (temperature, symmetry reduced set of triplets at gvien grid point, band1, band2, band3) in THz (without $2\pi$)                           |
| mesh                        | Numbers of sampling mesh along reciprocal axes.                                                                                            |
| frequency_point for `--ise` | Sampling frequency points in THz (without $2\pi$), i.e., $\omega$ in $\Gamma_\lambda(\omega)$                                              |
| temperature                 | (temperature,), Temperatures in K                                                                                                          |
| triplet                     | (symmetry reduced set of triplets at given grid point, 3), Triplets are given by the grid point indices (see below).                       |
| weight                      | (symmetry reduced set of triplets at given grid point,), Weight of each triplet to imaginary part of self energy                           |
| triplet_all                 | (triplets at given grid point, 3), symmetry non-reduced version of the triplet information.                                                |
```

See {ref}`grid_triplets` to recover the q-points of each triplet.

Imaginary part of self energy (linewidth/2) is recovered by the following
script:

```python
import h5py
import numpy as np

gd = h5py.File("gamma_detail-mxxx-gx.hdf5")
temp_index = 30 # index of temperature
temperature = gd['temperature'][temp_index]
gamma_tp = gd['gamma_detail'][:].sum(axis=-1).sum(axis=-1)
weight = gd['weight'][:]
gamma = np.dot(weight, gamma_tp[temp_index])
```

For example, for `--br`, this `gamma` gives $\Gamma_\lambda(\omega_\lambda)$ of
the band indices at the grid point indicated by $\lambda$ at the temperature of
index 30. If any bands are degenerated, those `gamma` in
`kappa-mxxx-gx(-sx-sdx).hdf5` or `gamma-mxxx-gx(-sx-sdx).hdf5` type file are
averaged, but the `gamma` obtained here in this way are not symmetrized. Apart
from this symmetrization, the values must be equivalent between them.

To understand each contribution of triptle to imaginary part of self energy,
reading `phonon-mxxx.hdf5` is useful (see {ref}`write_phonon_option`). For
example, phonon triplets of three phonon scatterings are obtained by

```python
import h5py
import numpy as np

gd = h5py.File("gamma_detail-mxxx-gx.hdf5", 'r')
ph = h5py.File("phonon-mxxx.hdf5", 'r')
gp1 = gd['grid_point'][()]
triplets = gd['triplet'][:] # Sets of (gp1, gp2, gp3) where gp1 is fixed
mesh = gd['mesh'][:]
grid_address = ph['grid_address'][:]
q_triplets = grid_address[triplets] / mesh.astype('double') # For conventional regular grid
# Phonons of triplets[2]
phonon_tp = [(ph['frequency'][i], ph['eigenvector'][i]) for i in triplets[2]]
# Fractions of contributions of triplets at this grid point and temperature index 30
gamma_sum_over_bands = np.dot(weight, gd['gamma_detail'][30].sum(axis=-1).sum(axis=-1).sum(axis=-1))
contrib_tp = [gd['gamma_detail'][30, i].sum() / gamma_sum_over_bands for i in range(len(weight))]
np.dot(weight, contrib_tp) # is one
```

(iofile_phonon_hdf5)=
## `phonon-*.hdf5`

Contents of `phonon-mxxx.hdf5` are watched by:

```python
In [1]: import h5py

In [2]: f = h5py.File("phonon-m111111.hdf5")

In [3]: list(f)
Out[3]:
['eigenvector',
 'frequency',
 'grid_address',
 'ir_grid_points',
 'ir_grid_weights',
 'mesh',
 'version']

In [4]: f['mesh'][:]
Out[4]: array([11, 11, 11])

In [5]: f['grid_address'].shape
Out[5]: (1367, 3)

In [6]: f['frequency'].shape
Out[6]: (1367, 6)

In [7]: f['eigenvector'].shape
Out[7]: (1367, 6, 6)

In [8]: f['ir_grid_points'].shape
Out[8]: (56,)
```

The first axis of `ph['grid_address']`, `ph['frequency']`, and
`ph['eigenvector']` corresponds to the number of q-points where phonons are
calculated. Here the number of phonons may not be equal to product of mesh
numbers ($1367 \neq 11^3$). This is because all q-points on Brillouin zone
boundary are included, i.e., even if multiple q-points are translationally
equivalent, those phonons are stored separately though these phonons are
physically equivalent within the equations employed in phono3py. Here Brillouin
zone is defined by Wigner–Seitz cell of reciprocal primitive basis vectors. This
is convenient to categorize phonon triplets into Umklapp and Normal scatterings
based on the Brillouin zone.


## `pp-*.hdf5`

This file contains phonon-phonon interaction strength
$\bigl|\Phi_{\lambda\lambda'\lambda''}\bigl|^2$. To use the data in this
file, it is recommended to generate with `--full-pp` option because the data
structure to access becomes simpler.

```bash
% phono3py-load phono3py.yaml --gp 5 --br --mesh 11 11 11 --write-pp --full-pp
```

```python
In [1]: import h5py

In [2]: f = h5py.File("pp-m111111-g5.hdf5")

In [3]: list(f)
Out[3]: ['pp', 'triplet', 'triplet_all', 'version', 'weight']

In [4]: f['pp'].shape
Out[4]: (146, 6, 6, 6)
```

Indices of the `pp` array are (symmetry reduced set of triplets at given grid
point, band1, band2, band3), and the values are given in $\text{eV}^2$. See
{ref}`grid_triplets` to recover the q-points of each triplet.

Except for `pp`, all the other information are equivalent to those found in
{ref}`iofile_gamma_detail_hdf5`.


## `gammas-*.dat`

Imaginary parts of self energies with respect to frequency
$\Gamma_\lambda(\omega)$ are stored in THz. See {ref}`ise_option`.

## `jdos-*.dat`

Joint densities of states are stored in Thz. See {ref}`jdos_option`.

## `linewidth-*.dat`
