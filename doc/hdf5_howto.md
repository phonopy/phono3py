(hdf5_howto)=
# Using phono3py hdf5 files

## Using `h5py` in ipython

It is assumed that `python-h5py` is installed on the computer you interactively
use. In the following, how to see the contents of `.hdf5` files in the
interactive mode of Python. The basic usage of reading `.hdf5` files using
`h5py` is found at
[here](http://docs.h5py.org/en/latest/high/dataset.html#reading-writing-data>).
In the following example, an MgO result of thermal conductivity calculation
stored in `kappa-m111111.hdf5` (see {ref}`iofile_kappa_hdf5`) is loaded and
thermal conductivity tensor at 300 K is watched.

```python
In [1]: import h5py

In [2]: f = h5py.File("kappa-m111111.hdf5")

In [3]: list(f)
Out[3]:
['frequency',
 'gamma',
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

In [4]: f['kappa'].shape
Out[4]: (101, 6)

In [5]: f['kappa'][:]
Out[5]:
array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.11702476e+05,   2.11702476e+05,   2.11702476e+05,
          6.64531043e-13,   6.92618921e-13,  -1.34727352e-12],
       [  3.85304024e+04,   3.85304024e+04,   3.85304024e+04,
          3.52531412e-13,   3.72706406e-13,  -7.07290889e-13],
       ...,
       [  2.95769356e+01,   2.95769356e+01,   2.95769356e+01,
          3.01803322e-16,   3.21661793e-16,  -6.05271364e-16],
       [  2.92709650e+01,   2.92709650e+01,   2.92709650e+01,
          2.98674274e-16,   3.18330655e-16,  -5.98999091e-16],
       [  2.89713297e+01,   2.89713297e+01,   2.89713297e+01,
          2.95610215e-16,   3.15068595e-16,  -5.92857003e-16]])

In [6]: f['temperature'][:]
Out[6]:
array([    0.,    10.,    20.,    30.,    40.,    50.,    60.,    70.,
          80.,    90.,   100.,   110.,   120.,   130.,   140.,   150.,
         160.,   170.,   180.,   190.,   200.,   210.,   220.,   230.,
         240.,   250.,   260.,   270.,   280.,   290.,   300.,   310.,
         320.,   330.,   340.,   350.,   360.,   370.,   380.,   390.,
         400.,   410.,   420.,   430.,   440.,   450.,   460.,   470.,
         480.,   490.,   500.,   510.,   520.,   530.,   540.,   550.,
         560.,   570.,   580.,   590.,   600.,   610.,   620.,   630.,
         640.,   650.,   660.,   670.,   680.,   690.,   700.,   710.,
         720.,   730.,   740.,   750.,   760.,   770.,   780.,   790.,
         800.,   810.,   820.,   830.,   840.,   850.,   860.,   870.,
         880.,   890.,   900.,   910.,   920.,   930.,   940.,   950.,
         960.,   970.,   980.,   990.,  1000.])

In [7]: f['kappa'][30]
Out[7]:
array([  1.09089896e+02,   1.09089896e+02,   1.09089896e+02,
         1.12480528e-15,   1.19318349e-15,  -2.25126057e-15])

In [8]: f['mode_kappa'][30, :, :, :].sum(axis=0).sum(axis=0) / weight.sum()
Out[8]:
array([  1.09089896e+02,   1.09089896e+02,   1.09089896e+02,
         1.12480528e-15,   1.19318349e-15,  -2.25126057e-15])

In [9]: g = f['gamma'][30]

In [10]: import numpy as np

In [11]: g = np.where(g > 0, g, -1)

In [12]: lifetime = np.where(g > 0, 1.0 / (2 * 2 * np.pi * g), 0)
```
