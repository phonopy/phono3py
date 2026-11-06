/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __NIGGLI_H__
#define __NIGGLI_H__

#define NIGGLI_MAJOR_VERSION 0
#define NIGGLI_MINOR_VERSION 1
#define NIGGLI_MICRO_VERSION 2

int niggli_get_major_version(void);
int niggli_get_minor_version(void);
int niggli_get_micro_version(void);
int niggli_reduce(double* lattice_, const double eps_);
int niggli_reduce_periodic(double* lattice_, const double eps_,
                           const int aperiodic_axis);

#endif
