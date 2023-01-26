#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

void lagmat_multiply_matrix_vector_l3(long v[3], const long a[3][3],
                                      const long b[3]) {
    long i;
    long c[3];
    for (i = 0; i < 3; i++) {
        c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
    }
    for (i = 0; i < 3; i++) {
        v[i] = c[i];
    }
}

void lagmat_multiply_matrix_l3(long m[3][3], const long a[3][3],
                               const long b[3][3]) {
    long i, j; /* a_ij */
    long c[3][3];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    lagmat_copy_matrix_l3(m, c);
}

void lagmat_copy_matrix_l3(long a[3][3], const long b[3][3]) {
    a[0][0] = b[0][0];
    a[0][1] = b[0][1];
    a[0][2] = b[0][2];
    a[1][0] = b[1][0];
    a[1][1] = b[1][1];
    a[1][2] = b[1][2];
    a[2][0] = b[2][0];
    a[2][1] = b[2][1];
    a[2][2] = b[2][2];
}

long lagmat_check_identity_matrix_l3(const long a[3][3], const long b[3][3]) {
    if (a[0][0] - b[0][0] || a[0][1] - b[0][1] || a[0][2] - b[0][2] ||
        a[1][0] - b[1][0] || a[1][1] - b[1][1] || a[1][2] - b[1][2] ||
        a[2][0] - b[2][0] || a[2][1] - b[2][1] || a[2][2] - b[2][2]) {
        return 0;
    } else {
        return 1;
    }
}

long get_num_unique_elems(const long array[], const long array_size) {
    long i, num_unique_elems;
    long *unique_elems;

    num_unique_elems = 0;
    unique_elems = (long *)malloc(sizeof(long) * array_size);
    for (i = 0; i < array_size; i++) {
        unique_elems[i] = 0;
    }
    for (i = 0; i < array_size; i++) {
        unique_elems[array[i]]++;
    }
    for (i = 0; i < array_size; i++) {
        if (unique_elems[i]) {
            num_unique_elems++;
        }
    }
    free(unique_elems);
    unique_elems = NULL;
    return num_unique_elems;
}
