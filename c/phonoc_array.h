/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __phonoc_array_H__
#define __phonoc_array_H__

#include <stdint.h>

#define MAX_NUM_DIM 20

/* It is assumed that number of dimensions is known for each array. */
typedef struct {
    int64_t dims[MAX_NUM_DIM];
    int64_t *data;
} Larray;

typedef struct {
    int dims[MAX_NUM_DIM];
    double *data;
} Darray;

#endif
