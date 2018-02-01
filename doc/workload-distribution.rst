.. _workload_distribution:

Workload distribution
======================

Workload of thermal conductivity calculation can be distributed into
computer nodes. The distribution over q-point grid points (:ref:`--gp
option <gp_option>`), over phonon bands (:ref:`--bi option
<bi_option>`), and over both of them are supported. Unless necessary,
the distribution over bands is not recommended since it has some
amount of overhead in the part of Fourier transformation of force
constants. Therefore the distribution over grid-points is explained
below. However since the distribution over bands works quite similarly as
that over q-points, the usage can be easily guessed.

On each computer node, pieces of lattice thermal conductivity
calculation are executed. The resulting data for each grid point are
stored in its ``kappa-mxxx-gx.hdf5`` file on each node by setting
:ref:`--write_gamma option <write_gamma_option>`. Once all data are
obtained, those data are collected by :ref:`--read_gamma option
<read_gamma_option>` and the lattice thermal conductivity is obtained.

.. contents::
   :depth: 2
   :local:

How to do it
------------

The following example is executed in the ``Si-PBE`` example.

To avoid re-calculating fc3 and fc2, ``fc3.hdf5`` and ``fc2.hdf5`` are
created on a single node::

   % phono3py --dim="2 2 2" --sym-fc -c POSCAR-unitcell

The indices of the irreducible grid-points neccesarry to specify
``--ga`` option are found by :ref:`--wgp option <wgp_option>`

::

   % phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --fc3 --fc2 --br --wgp

and they are stored in ``ir_grid_points.yaml``.

::

   % egrep '^- grid_point:' ir_grid_points.yaml|awk '{printf("%d,",$3)}'
   0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,60,61,62,63,64,65,66,67,68,69,70,71,72,73,80,81,82,83,84,85,86,87,88,89,90,91,100,101,102,103,104,105,106,107,108,109,120,121,122,123,124,125,126,127,140,141,142,143,144,145,160,161,162,163,180,181,402,403,404,405,406,407,408,409,422,423,424,425,426,427,428,429,430,431,432,433,434,435,442,443,444,445,446,447,448,449,450,451,452,453,462,463,464,465,466,467,468,469,470,471,482,483,484,485,486,487,488,489,502,503,504,505,506,507,522,523,524,525,542,543,804,805,806,807,808,809,824,825,826,827,828,829,830,831,832,833,844,845,846,847,848,849,850,851,864,865,866,867,868,869,884,885,886,887,904,905,1206,1207,1208,1209,1226,1227,1228,1229,1230,1231,1246,1247,1248,1249,1266,1267,1608,1609,1628,1629,

The calculated data on all the grid points shown above as indices are
necessary to obtain lattice thermal conductivity. To distribute
computational demands into computer nodes, a set of the grid-point
indices are chosen and executed as follows::

   % phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --fc3 --fc2 --br --gp="0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25" --write-gamma

Then many ``kappa-m191919-gx.hdf5`` files are generated. These file
names should not be altered because in reading the data by phono3py,
those file names are supposed to be so, though there is a little
freedom to arrange those file names, for which see :ref:`-o
<output_filename_option>` and :ref:`-i <input_filename_option>`
options. After completing calculations for all irreducible grid-point
indices, the RTA thermal conductivity is computed by another run in a
short time from the stored data:

::

   % phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --fc3 --fc2 --br --read-gamma

A convenient script
--------------------

The following short script may be useful to splitting all irreducible
grid-point indices into of reasonable number of sets of grid-point
indices for workload distribution.

.. code-block:: python

   #!/usr/bin/env python

   import sys
   import yaml

   if len(sys.argv) > 1:
       num = int(sys.argv[1])
   else:
       num = 1

   with open("ir_grid_points.yaml") as f:
       data = yaml.load(f)
       gps = [gp['grid_point'] for gp in data['ir_grid_points']]
       gp_lists = [[] for i in range(num)]
       for i, gp in enumerate(gps):
           gp_lists[i % num].append(gp)
       for gp_set in gp_lists:
           print(",".join(["%d" % gp for gp in gp_set]))

Supposed that this script is saved as ``divide_gps.py``,

::

   % phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --wgp
   ...
   % python divide_gps.py 20
   0,30,52,82,120,402,434,468,524,844,1206
   1,31,53,83,121,403,435,469,525,845,1207
   2,32,54,84,122,404,442,470,542,846,1208
   3,33,55,85,123,405,443,471,543,847,1209
   4,34,60,86,124,406,444,482,804,848,1226
   5,35,61,87,125,407,445,483,805,849,1227
   6,36,62,88,126,408,446,484,806,850,1228
   7,37,63,89,127,409,447,485,807,851,1229
   8,40,64,90,140,422,448,486,808,864,1230
   9,41,65,91,141,423,449,487,809,865,1231
   20,42,66,100,142,424,450,488,824,866,1246
   21,43,67,101,143,425,451,489,825,867,1247
   22,44,68,102,144,426,452,502,826,868,1248
   23,45,69,103,145,427,453,503,827,869,1249
   24,46,70,104,160,428,462,504,828,884,1266
   25,47,71,105,161,429,463,505,829,885,1267
   26,48,72,106,162,430,464,506,830,886,1608
   27,49,73,107,163,431,465,507,831,887,1609
   28,50,80,108,180,432,466,522,832,904,1628
   29,51,81,109,181,433,467,523,833,905,1629

For example distributing into 20 computer nodes using a queueing
system,

.. code-block:: shell

   % j=1; for i in `python divide_gps.py 20`;do echo $i; sed -e s/gps/$i/g -e s/num/$j/g job.sh|qsub; j=$((j+1)); done

with ``job.sh`` (here for grid-engine):

.. code-block:: shell

   #$ -S /bin/zsh
   #$ -cwd
   #$ -N phono3py-num
   #$ -pe mpi* 16
   #$ -e err-phono3py-num.log
   #$ -o std-phono3py-num.log

   phono3py --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" -c POSCAR-unitcell --mesh="19 19 19" --fc3 --fc2 --br --gp="gps" --write-gamma
