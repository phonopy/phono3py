(command_cutoff_pair)=
# Force constants calculation with cutoff pair-distance

Here the detail of the command option {ref}`--cutoff_pair <cutoff_pair_option>` is explained.

```{contents}
:depth: 2
:local:
```

## What is cutff pair-distance in phono3py?

Using `--cutoff-pair` option, number of supercells with
displacements to be calculated is reduced. But of course this
sacrifices the accuracy of third-order force constants (fc3).

In phono3py, to obtain supercell-fc3,
$\Phi_{\alpha\beta\gamma}(jl, j'l', j''l'')$, forces in many
supercells having different pairs of displaced atoms are computed
using some force-calculator such as ab-initio code. In the phono3py
default behaviour, full elements of supercell-fc3 are computed. In
this case, though depending on the number of atoms in the supercell
and the crystal symmetry, the number of atomic-pair configuration can
be huge and beyond our computational resource.

Sometimes we may expect that interaction range of fc3 among triplets
of atoms is shorter than chosen supercell size. If it is the case, we
may be allowed to omit computing some elements of supercell-fc3. This
is what achieved by `--cutoff-pair` option.

A supercell-fc3 element is specified by three atomic
displacements. Two of three are finitely displaced
($\Delta\mathbf{r}_1$ and $\Delta\mathbf{r}_2$) but one of
them is included in a force given by the force calculator such as
$\mathbf{F}_3=-\frac{\partial V}{\partial\mathbf{r}_3}$. The
cutoff distance $d_\text{cut}$ is defined as the upper bound of
the distance between the atoms 1 and 2. By this, the set of atomic
pairs $\{(\mathbf{r}_1,\mathbf{r}_2)| |\mathbf{r}_1 -
\mathbf{r}_2| < d_\text{cut}\}$ is selected for the supercell-fc3
calculation. By this, when three distances among the three atoms of
triplets are all larger than $d_\text{cut}$, those fc3 elements
can not be obtained and so they are simply set zero.

## Usage

### Creating supercells with displacements

`--cutoff-pair` option is employed when creating supercells with
displacements, therefore this option must be used with `-d` option
when running phono3py, for the Si-PBEsol example:

```bash
% phono3py --cutoff-pair=5 -d --dim="2 2 2" -c POSCAR-unitcell
...

Unit cell was read from "POSCAR-unitcell".
Displacement distance: 0.03
Number of displacements: 111
Cutoff distance for displacements: 5.0
Number of displacement supercell files created: 51

Displacement dataset was written in "phono3py_disp.yaml".
...

% ls POSCAR-0*
POSCAR-00001  POSCAR-00032  POSCAR-00043  POSCAR-00080  POSCAR-00097
POSCAR-00002  POSCAR-00033  POSCAR-00070  POSCAR-00081  POSCAR-00098
POSCAR-00003  POSCAR-00034  POSCAR-00071  POSCAR-00082  POSCAR-00099
POSCAR-00016  POSCAR-00035  POSCAR-00072  POSCAR-00083  POSCAR-00100
POSCAR-00017  POSCAR-00036  POSCAR-00073  POSCAR-00084  POSCAR-00101
POSCAR-00018  POSCAR-00037  POSCAR-00074  POSCAR-00085  POSCAR-00102
POSCAR-00019  POSCAR-00038  POSCAR-00075  POSCAR-00086  POSCAR-00103
POSCAR-00024  POSCAR-00039  POSCAR-00076  POSCAR-00087
POSCAR-00025  POSCAR-00040  POSCAR-00077  POSCAR-00088
POSCAR-00026  POSCAR-00041  POSCAR-00078  POSCAR-00089
POSCAR-00027  POSCAR-00042  POSCAR-00079  POSCAR-00096
% ls POSCAR-0*|wc -l
      51
```

`Number of displacements: 111` shows the number of supercells with
displacements when this is run without `--cutoff-pair`
option. `Number of displacement supercell files created: 51` gives
the contracted number of supercells with displacements by
`--cutoff-pair` option. There number of `POSCAR-0xxxx` files is found 51.
At this step, a special `phono3py_disp.yaml` is created. This
contains information on this contraction and used in the other
calculation step, therefore this file must be kept carefully.

### Supercell files

`POSCAR-xxxxx` (in the other calculator interface, the prefix of the
filename is different) are not generated if distance between a pair of
atoms to be displaced is larger than the specified cutoff pair
distance. The indexing number (`xxxxx`) corresponds to that of the
case without setting this option, i.e., the same `POSCAR-xxxxx`
files are created for the same configurations of pairs of
displacements but `POSCAR-xxxxx` files not being included are not
generated. The reason of this indexing is that it can be useful when
changing the cutoff-pair-distance.

### Special `phono3py_disp.yaml`

Using `--cutoff-pair` option together with `-d` option, a special
`phono3py_disp.yaml` is created. This contains information on distances
between displaced atomic-pairs and whether those pairs are to be
computed or not. This special `phono3py_disp.yaml` is necessary to create
fc3, therefore be careful not to overwrite it by running the option
`-d` without `--cutoff-pair` or with different `--cutoff-pair`
with different value.

### Making `FORCES_FC3`

To create `FORCES_FC3`, only output files of the supercells created
using `--cutoff-pair` option are passed to `phono3py` as the
arguments. The special `phono3py_disp.yaml` file is necessary to be
located at current directory.

An example is shown below for the Si example. Here, it is supposed
that forces are calculated using VASP in `disp-xxxxx`
directories. After running force calculations, there should be the
output file containing forces in each directory (for VASP
`vasprun.xml`).

```bash
% phono3py --cf3 disp-{00001,00002,00003,00016,00017,00018,00019,00024,00025,00026,00027,00032,00033,00034,00035,00036,00037,00038,00039,00040,00041,00042,00043,00070,00071,00072,00073,00074,00075,00076,00077,00078,00079,00080,00081,00082,00083,00084,00085,00086,00087,00088,00089,00096,00097,00098,00099,00100,00101,00102,00103}/vasprun.xml
...

Displacement dataset is read from phono3py_disp.yaml.
counter (file index): 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51
FORCES_FC3 has been created.
```

Using {ref}`--cf3-file option <cf3_file_option>` may be recommended
when the number of force files is large.

```bash
% for i in `ls POSCAR-0*|sed s/POSCAR-//`;do echo disp-$i/vasprun.xml;done > file_list.dat
% phono3py --cf3-file file_list.dat
```

Using a python script, `phono3py_disp.yaml` is easily parsed. So
it is also easy to create the file list by a python
script:

```python
#!/usr/bin/env python

from phono3py.interface.phono3py_yaml import Phono3pyYaml

p3yaml = Phono3pyYaml()
p3yaml.read("phono3py_disp.yaml")
dds = p3yaml.dataset
file_name_tmpl = "disp-%05d/vasprun.xml"
count = 1
for d1 in dds['first_atoms']:
    print(file_name_tmpl % count)
    count += 1
for d1 in dds['first_atoms']:
    for d2 in d1['second_atoms']:
        if d2['included']:
            print(file_name_tmpl % count)
        count += 1
```

### Running phonon-phonon interaction calculation

To create fc3, `--cutoff-pair` option is not necessary but the
special `phono3py_disp.yaml` is required.

```bash
% phono3py
...

----------------------------- General settings -----------------------------
Run mode: None
HDF5 data compression filter: gzip
Crystal structure was read from "phono3py_disp.yaml".
Supercell (dim): [2 2 2]
Primitive matrix:
  [0.  0.5 0.5]
  [0.5 0.  0.5]
  [0.5 0.5 0. ]
Spacegroup: Fd-3m (227)
Use -v option to watch primitive cell, unit cell, and supercell structures.
----------------------------- Force constants ------------------------------
Imposing translational and index exchange symmetry to fc2: False
Imposing translational and index exchange symmetry to fc3: False
Imposing symmetry of index exchange to fc3 in reciprocal space: False
Displacement dataset for fc3 was read from "phono3py_disp.yaml".
Sets of supercell forces were read from "FORCES_FC3".
Computing fc3[ 1, x, x ] using numpy.linalg.pinv with a displacement:
    [ 0.0300  0.0000  0.0000]
Expanding fc3.
Cutting-off fc3 (cut-off distance: 5.000000)
Building atom mapping table...
Creating contracted fc3...
Writing fc3 to "fc3.hdf5".
Max drift of fc3: 0.047748 (yxz) 0.047748 (xyz) 0.047748 (xzy)
Displacement dataset for fc2 was read from "phono3py_disp.yaml".
Sets of supercell forces were read from "FORCES_FC3".
Writing fc2 to "fc2.hdf5".
Max drift of fc2: -0.000001 (zz) -0.000001 (zz)
----------- None of ph-ph interaction calculation was performed. -----------

Summary of calculation was written in "phono3py.yaml".
...
```

Once `fc3.hdf5` and `fc2.hdf5` are created, `--cutoff-pair`
option and the special `phono3py_disp.yaml` are not needed anymore.

```bash
% phono3py --mesh="11 11 11" --fc3 --fc2 --br
...

  300.0     108.051    108.051    108.051      0.000      0.000     -0.000
...
```

## A script extract supercell IDs from `phono3py_disp.yaml`

The following script is an example to collect supercell IDs
with the cutoff-pair distance, for which `included: true` is used
to hook them. `duplicates` in `phono3py_disp.yaml` gives the pairs of
exactly same supercells having different IDs. Therefore one of each
pair is necessary to calculate. As a ratio, the number is not many,
but if we want to save very much the computational demand, it is good
to consider.

```python
#!/usr/bin/env python

from phono3py.interface.phono3py_yaml import Phono3pyYaml

p3yaml = Phono3pyYaml()
p3yaml.read("phono3py_disp.yaml")
data = p3yaml.dataset

disp_ids = []
for data1 in data['first_atoms']:
    disp_ids.append(data1['id'])

for data1 in data['first_atoms']:
    for data2 in data1['second_atoms']:
        if data2['included']:
            disp_ids.append(data2['id'])

# To remove duplicates
# duplicates = dict(data['duplicates'])
# disp_ids_nodup = [i for i in disp_ids if i not in duplicates]

print(" ".join(["%05d" % i for i in disp_ids]))
```

Even for the case that `phono3py_disp.yaml` was created without
`--cutoff-pair` option, if we replace the line in the above script:

```python
if data2['included']:
```

by

```python
if data2['distance'] < 5.0:  # 5.0 is cutoff-pair distance
```

we can find the supercell IDs almost equivalent to those obtained
above for `--cutoff-pair="5.0"`.


## Tests

### Si-PBE

For testing, thermal conductivities with respect to `--cutoff-pair`
values are calculated as follows. Note that if `FORCES_FC3` for full
fc3 elements exists, the same `FORCES_FC3` file can be used for
generating contracted fc3 for each special `phono3py_disp.yaml`.

```bash
% egrep '^\s+pair_distance' phono3py_disp.yaml|awk '{print $2}'|sort|uniq
0.000000
2.366961
3.865232
4.532386
5.466263
5.956722
6.694777
7.100884
7.730463
9.467845
% cp phono3py_disp.yaml phono3py_disp.orig.yaml
% for i in {2..10};do d=`grep pair_distance phono3py_disp.orig.yaml|awk '{print $2}'|sort|uniq|sed "${i}q;d"`; d=$((d+0.1)); phono3py --cutoff-pair=$d -o $i -d --pa="F" --dim="2 2 2" -c POSCAR-unitcell; mv phono3py_disp.yaml phono3py_disp.$i.yaml; done
% ls phono3py_disp.*.yaml
% ls phono3py_disp.*.yaml
phono3py_disp.10.yaml  phono3py_disp.5.yaml  phono3py_disp.9.yaml
phono3py_disp.2.yaml   phono3py_disp.6.yaml  phono3py_disp.orig.yaml
phono3py_disp.3.yaml   phono3py_disp.7.yaml
phono3py_disp.4.yaml   phono3py_disp.8.yaml
% for i in {2..10};do grep number_of_pairs_in_cutoff phono3py_disp.$i.yaml;done
number_of_pairs_in_cutoff: 10
number_of_pairs_in_cutoff: 30
number_of_pairs_in_cutoff: 50
number_of_pairs_in_cutoff: 55
number_of_pairs_in_cutoff: 75
number_of_pairs_in_cutoff: 95
number_of_pairs_in_cutoff: 103
number_of_pairs_in_cutoff: 108
number_of_pairs_in_cutoff: 110
% for i in {2..10};do cp phono3py_disp.$i.yaml phono3py_disp.yaml; phono3py --mesh="11 11 11" --br|tee std.$i.out;done
% for i in {2..10};do egrep '^\s+300' std.$i.out;done
  300.0     123.606    123.606    123.606     -0.000     -0.000      0.000
  300.0     118.617    118.617    118.617     -0.000     -0.000      0.000
  300.0     118.818    118.818    118.818     -0.000     -0.000      0.000
  300.0     118.879    118.879    118.879     -0.000     -0.000      0.000
  300.0     119.468    119.468    119.468     -0.000     -0.000      0.000
  300.0     119.489    119.489    119.489     -0.000     -0.000      0.000
  300.0     119.501    119.501    119.501     -0.000     -0.000      0.000
  300.0     119.483    119.483    119.483     -0.000     -0.000      0.000
  300.0     119.481    119.481    119.481     -0.000     -0.000      0.000
% for i in {2..10};do cp phono3py_disp.$i.yaml phono3py_disp.yaml; phono3py --sym-fc --mesh="11 11 11" --br|tee std.sym-$i.out;done
% for i in {2..10};do egrep '^\s+300' std.sym-$i.out;done
  300.0     124.650    124.650    124.650     -0.000     -0.000      0.000
  300.0     119.765    119.765    119.765     -0.000     -0.000      0.000
  300.0     118.847    118.847    118.847     -0.000     -0.000      0.000
  300.0     118.782    118.782    118.782     -0.000     -0.000      0.000
  300.0     119.471    119.471    119.471     -0.000     -0.000      0.000
  300.0     119.366    119.366    119.366     -0.000     -0.000      0.000
  300.0     119.350    119.350    119.350     -0.000     -0.000      0.000
  300.0     119.339    119.339    119.339     -0.000     -0.000      0.000
  300.0     119.337    119.337    119.337     -0.000     -0.000      0.000
```

### AlN-LDA

```bash
% egrep '^\s+pair_distance' phono3py_disp.yaml|awk '{print $2}'|sort|uniq
0.000000
1.889907
1.901086
3.069402
3.076914
3.111000
3.640065
3.645881
4.370303
4.375582
4.743307
4.743308
4.788360
4.978000
5.364501
5.388410
5.672503
5.713938
5.870162
6.205027
6.469591
7.335901
% cp phono3py_disp.yaml phono3py_disp.orig.yaml
% for i in {2..21};do d=`grep pair_distance phono3py_disp.orig.yaml|awk '{print $2}'|sort|uniq|sed "${i}q;d"`; d=$((d+0.0001)); phono3py --cutoff-pair=$d -o $i -d --dim="3 3 2" -c POSCAR-unitcell; mv phono3py_disp.yaml phono3py_disp.$i.yaml; done
% for i in {2..21};do grep number_of_pairs_in_cutoff phono3py_disp.$i.yaml;done
number_of_pairs_in_cutoff: 72
number_of_pairs_in_cutoff: 92
number_of_pairs_in_cutoff: 196
number_of_pairs_in_cutoff: 216
number_of_pairs_in_cutoff: 312
number_of_pairs_in_cutoff: 364
number_of_pairs_in_cutoff: 460
number_of_pairs_in_cutoff: 564
number_of_pairs_in_cutoff: 660
number_of_pairs_in_cutoff: 712
number_of_pairs_in_cutoff: 764
number_of_pairs_in_cutoff: 784
number_of_pairs_in_cutoff: 888
number_of_pairs_in_cutoff: 928
number_of_pairs_in_cutoff: 980
number_of_pairs_in_cutoff: 1020
number_of_pairs_in_cutoff: 1116
number_of_pairs_in_cutoff: 1156
number_of_pairs_in_cutoff: 1208
number_of_pairs_in_cutoff: 1248
% for i in {2..21};do cp phono3py_disp.$i.yaml phono3py_disp.yaml; phono3py --mesh="13 13 9" --br --nac --io $i|tee std.$i.out; done
% for i in {2..21};do egrep '^\s+300\.0' std.$i.out;done
  300.0     205.550    205.550    193.665     -0.000     -0.000     -0.000
  300.0     218.963    218.963    204.942     -0.000     -0.000     -0.000
  300.0     213.624    213.624    193.863     -0.000     -0.000     -0.000
  300.0     219.932    219.932    199.819     -0.000     -0.000     -0.000
  300.0     235.516    235.516    218.843     -0.000     -0.000     -0.000
  300.0     234.750    234.750    217.384     -0.000     -0.000     -0.000
  300.0     234.355    234.355    218.030     -0.000     -0.000     -0.000
  300.0     235.381    235.381    218.609     -0.000     -0.000     -0.000
  300.0     235.996    235.996    219.785     -0.000     -0.000     -0.000
  300.0     236.220    236.220    219.867     -0.000     -0.000     -0.000
  300.0     236.161    236.161    219.298     -0.000     -0.000     -0.000
  300.0     236.096    236.096    219.313     -0.000     -0.000     -0.000
  300.0     234.602    234.602    217.064     -0.000     -0.000     -0.000
  300.0     235.914    235.914    218.689     -0.000     -0.000     -0.000
  300.0     235.049    235.049    217.935     -0.000     -0.000     -0.000
  300.0     235.877    235.877    219.065     -0.000     -0.000     -0.000
  300.0     236.133    236.133    219.364     -0.000     -0.000     -0.000
  300.0     236.207    236.207    219.595     -0.000     -0.000     -0.000
  300.0     236.035    236.035    219.463     -0.000     -0.000     -0.000
  300.0     236.104    236.104    219.348     -0.000     -0.000     -0.000
% for i in {2..21};do cp phono3py_disp.$i.yaml phono3py_disp.yaml; phono3py --mesh="13 13 9" --br --nac --io $i|tee std.$i.out; done|tee std.sym-$i.out; done
% for i in {2..21};do egrep '^\s+300\.0' std.sym-$i.out;done
  300.0     232.964    232.964    216.333      0.000     -0.000     -0.000
  300.0     235.442    235.442    219.602      0.000     -0.000     -0.000
  300.0     235.521    235.521    217.767      0.000     -0.000     -0.000
  300.0     235.581    235.581    217.687      0.000     -0.000     -0.000
  300.0     236.837    236.837    219.933      0.000     -0.000     -0.000
  300.0     236.020    236.020    219.324      0.000     -0.000     -0.000
  300.0     235.482    235.482    218.633      0.000     -0.000     -0.000
  300.0     236.313    236.313    219.677      0.000     -0.000     -0.000
  300.0     236.308    236.308    219.955      0.000     -0.000     -0.000
  300.0     236.074    236.074    219.882      0.000     -0.000     -0.000
  300.0     235.520    235.520    219.450      0.000     -0.000     -0.000
  300.0     235.769    235.769    219.562      0.000     -0.000     -0.000
  300.0     235.441    235.441    219.168      0.000     -0.000     -0.000
  300.0     235.892    235.892    219.590      0.000     -0.000     -0.000
  300.0     235.509    235.509    219.167      0.000     -0.000     -0.000
  300.0     235.646    235.646    219.521      0.000     -0.000     -0.000
  300.0     235.783    235.783    219.311      0.000     -0.000     -0.000
  300.0     235.887    235.887    219.301      0.000     -0.000     -0.000
  300.0     235.642    235.642    219.348      0.000     -0.000     -0.000
  300.0     235.728    235.728    219.102      0.000     -0.000     -0.000
```