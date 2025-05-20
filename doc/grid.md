(grid)=
# Grids in reciprocal space

The regular grid can be a conventional regular grid or a generalized regular
grid. Here the conventional regular grid means that the grids are cut parallel
to the reciprocal basis vectors. In most cases, the conventional regular grid is
used. In special case, e.g., for crystals with body center tetragonal symmetry,
the generalized regular grid can be useful. In phono3py, the generalized regular
grid is defined to be cut parallel to the reciprocal basis vectors of the
conventional unit cell.

Two types of grid data structure are used in phono3py. Normal regular grid
contains translationally unique grid points (regular grid). The other grid
includes the points on Brillouin zone (BZ) boundary (BZ grid).

## `BZGrid` class instance

Grid point information in reciprocal space is stored in the `BZGrid` class. This
class instance can be easily accessed in the following way.

```python
In [1]: import phono3py

In [2]: ph3 = phono3py.load("phono3py.yaml", produce_fc=False)

In [3]: ph3.mesh_numbers = [11, 11, 11]

In [4]: ph3.grid
Out[4]: <phono3py.phonon.grid.BZGrid at 0x1070f3b60>
```

It is recommended to read docstring in `BZGrid` by

```python
In [5]: help(ph3.grid)
```

Some attributes of this class are presented below.

```python
In [6]: ph3.grid.addresses.shape
Out[6]: (1367, 3)

In [7]: ph3.grid.D_diag
Out[7]: array([11, 11, 11])

In [8]: ph3.grid.P
Out[8]:
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])

In [9]: ph3.grid.Q
Out[9]:
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])

In [10]: ph3.grid.QDinv
Out[10]:
array([[0.09090909, 0.        , 0.        ],
       [0.        , 0.09090909, 0.        ],
       [0.        , 0.        , 0.09090909]])

In [11]: ph3.grid.PS
Out[11]: array([0, 0, 0])
```

The integer array `addresses` contains grid point addresses. Every grid point
address is represented by the unique series of three integers. These addresses
 are converted to the q-points in fractional coordinates as explained in the
{ref}`section below<grid_address_to_q>`.

Unless generalized regular grid is employed, the other attributes are not
important. `D_diag` is equivalent to the three integer numbers of the specified
conventional regular grid. `P` and `Q` are are the left and right unimodular
matrix after Smith normal form: $\mathrm{D}=\mathrm{PAQ}$, respectively, where
$\mathrm{A}$ is the grid matrix. `D_diag` is the three diagonal elements of the
matrix $\mathrm{D}$. The grid matrix is usually a diagonal matrix, then
 $\mathrm{P}$ and $\mathrm{Q}$ are chosen as identity matrix. `QDinv` is given
by $\mathrm{Q}\mathrm{D}^{-1}$. `PS` represents half-grid-shifts (usually always
`[0, 0, 0]` in phono3py).

## Find grid point index corresponding to grid point address

Grid point index corresponding to a grid point address is obtained using the
instance method `BZGrid.get_indices_from_addresses` as follows:

```python
In [1]: import phono3py

In [2]: ph3 = phono3py.load("phono3py_disp.yaml")

In [3]: ph3.mesh_numbers = [20, 20, 20]

In [4]: ph3.grid.get_indices_from_addresses([0, 10, 10])
Out[4]: 4448
```

This index number is different between phono3py version 1.x and 2.x.
To get the number corresponding to the phono3py version 1.x,
`store_dense_gp_map=False` should be specified in `phono3py.load`,

```python
In [5]: ph3 = phono3py.load("phono3py_disp.yaml", store_dense_gp_map=False)

In [6]: ph3.mesh_numbers = [20, 20, 20]

In [7]: ph3.grid.get_indices_from_addresses([0, 10, 10])
Out[7]: 4200
```

(grid_address_to_q)=
## q-points in fractional coordinates corresponding to grid addresses

For Gamma centered regular grid, q-points in fractional coordinates
are obtained by

```python
qpoints = addresses @ QDinv.T
```

For shifted regular grid (usually unused in phono3py),

```python
qpoints = (addresses * 2 + PS) @ (QDinv.T / 2.0)
```

(grid_triplets)=
## Grid point triplets

Three grid point indices are used to represent a q-point triplet. For example
the following command generates `gamma_detail-m111111-g5.hdf5`,

```bash
% phono3py-load phono3py.yaml --gp 5 --br --mesh 11 11 11 --write-gamma-detail
```

This file contains various information:

```python
In [1]: import h5py

In [2]: f = h5py.File("gamma_detail-m111111-g5.hdf5")

In [3]: list(f)
Out[3]:
['gamma_detail',
 'grid_point',
 'mesh',
 'temperature',
 'triplet',
 'triplet_all',
 'version',
 'weight']

In [4]: f['gamma_detail'].shape
Out[4]: (101, 146, 6, 6, 6)
```

For the detailed analysis of contributions of triplets to imaginary part of
self energy a phonon mode of the grid point, it is necessary to understand the
data structure of `triplet` and `weight`.

```python
In [5]: f['triplet'].shape
Out[5]: (146, 3)

In [6]: f['weight'].shape
Out[6]: (146,)

In [7]: f['triplet'][:10]
Out[7]:
array([[  5,   0,   6],
       [  5,   1,   5],
       [  5,   2,   4],
       [  5,   3,   3],
       [  5,   7,  10],
       [  5,   8,   9],
       [  5,  11, 118],
       [  5,  12, 117],
       [  5,  13, 116],
       [  5,  14, 115]])
```

The second index of `gamma_detail` corresponds to the first index of `triplet`.
Three integers of each triplet are the grid point indices, which means, the grid
addresses and their q-points are recovered by

```python
In [8]: import phono3py

In [9]: ph3 = phono3py.load("phono3py.yaml", produce_fc=False)

In [10]: ph3.mesh_numbers = [11, 11, 11]

In [11]: ph3.grid.addresses[f['triplet'][:10]]
Out[11]:
array([[[ 5,  0,  0],
        [ 0,  0,  0],
        [-5,  0,  0]],

       [[ 5,  0,  0],
        [ 1,  0,  0],
        [ 5,  0,  0]],

       [[ 5,  0,  0],
        [ 2,  0,  0],
        [ 4,  0,  0]],

       [[ 5,  0,  0],
        [ 3,  0,  0],
        [ 3,  0,  0]],

       [[ 5,  0,  0],
        [-4,  0,  0],
        [-1,  0,  0]],

       [[ 5,  0,  0],
        [-3,  0,  0],
        [-2,  0,  0]],

       [[ 5,  0,  0],
        [ 0,  1,  0],
        [-5, -1,  0]],

       [[ 5,  0,  0],
        [ 1,  1,  0],
        [ 5, -1,  0]],

       [[ 5,  0,  0],
        [ 2,  1,  0],
        [ 4, -1,  0]],

       [[ 5,  0,  0],
        [ 3,  1,  0],
        [ 3, -1,  0]]])

n [14]: ph3.grid.addresses[f['triplet'][:10]] @ ph3.grid.QDinv
Out[14]:
array([[[ 0.45454545,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [-0.45454545,  0.        ,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.09090909,  0.        ,  0.        ],
        [ 0.45454545,  0.        ,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.18181818,  0.        ,  0.        ],
        [ 0.36363636,  0.        ,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.27272727,  0.        ,  0.        ],
        [ 0.27272727,  0.        ,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [-0.36363636,  0.        ,  0.        ],
        [-0.09090909,  0.        ,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [-0.27272727,  0.        ,  0.        ],
        [-0.18181818,  0.        ,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.        ,  0.09090909,  0.        ],
        [-0.45454545, -0.09090909,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.09090909,  0.09090909,  0.        ],
        [ 0.45454545, -0.09090909,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.18181818,  0.09090909,  0.        ],
        [ 0.36363636, -0.09090909,  0.        ]],

       [[ 0.45454545,  0.        ,  0.        ],
        [ 0.27272727,  0.09090909,  0.        ],
        [ 0.27272727, -0.09090909,  0.        ]]])
```
