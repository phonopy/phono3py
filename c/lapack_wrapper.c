/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include "lapack_wrapper.h"

#include <math.h>

#define min(a, b) ((a) > (b) ? (b) : (a))

#if defined(MKL_LAPACKE) || defined(SCIPY_MKL_H)
MKL_Complex16 lapack_make_complex_double(double re, double im) {
    MKL_Complex16 z;
    z.real = re;
    z.imag = im;
    return z;
}
#ifndef LAPACKE_malloc
#define LAPACKE_malloc(size) malloc(size)
#endif
#ifndef LAPACKE_free
#define LAPACKE_free(p) free(p)
#endif
#endif

int phonopy_zheev(double *w, lapack_complex_double *a, const int n,
                  const char uplo) {
    lapack_int info;
    info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', uplo, (lapack_int)n, a,
                         (lapack_int)n, w);
    return (int)info;
}

int phonopy_pinv(double *data_out, const double *data_in, const int m,
                 const int n, const double cutoff) {
    int i, j, k;
    lapack_int info;
    double *s, *a, *u, *vt, *superb;

    a = (double *)malloc(sizeof(double) * m * n);
    s = (double *)malloc(sizeof(double) * min(m, n));
    u = (double *)malloc(sizeof(double) * m * m);
    vt = (double *)malloc(sizeof(double) * n * n);
    superb = (double *)malloc(sizeof(double) * (min(m, n) - 1));

    for (i = 0; i < m * n; i++) {
        a[i] = data_in[i];
    }

    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', (lapack_int)m,
                          (lapack_int)n, a, (lapack_int)n, s, u, (lapack_int)m,
                          vt, (lapack_int)n, superb);

    for (i = 0; i < n * m; i++) {
        data_out[i] = 0;
    }

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < min(m, n); k++) {
                if (s[k] > cutoff) {
                    data_out[j * m + i] += u[i * m + k] / s[k] * vt[k * n + j];
                }
            }
        }
    }

    free(a);
    free(s);
    free(u);
    free(vt);
    free(superb);

    return (int)info;
}

void phonopy_pinv_mt(double *data_out, int *info_out, const double *data_in,
                     const int num_thread, const int *row_nums,
                     const int max_row_num, const int column_num,
                     const double cutoff) {
    int i;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_thread; i++) {
        info_out[i] = phonopy_pinv(data_out + i * max_row_num * column_num,
                                   data_in + i * max_row_num * column_num,
                                   row_nums[i], column_num, cutoff);
    }
}

int phonopy_dsyev(double *data, double *eigvals, const int size,
                  const int algorithm) {
    lapack_int info;

    lapack_int liwork;
    long lwork;
    lapack_int *iwork;
    double *work;
    lapack_int iwork_query;
    double work_query;

    info = 0;
    liwork = -1;
    lwork = -1;
    iwork = NULL;
    work = NULL;

    switch (algorithm) {
        case 0: /* dsyev */
            info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)size,
                                 data, (lapack_int)size, eigvals);
            break;
        case 1: /* dsyevd */
            info = LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'U',
                                       (lapack_int)size, data, (lapack_int)size,
                                       eigvals, &work_query, lwork,
                                       &iwork_query, liwork);
            liwork = iwork_query;
            lwork = (long)work_query;
            /* printf("liwork %d, lwork %ld\n", liwork, lwork); */
            if ((iwork = (lapack_int *)LAPACKE_malloc(sizeof(lapack_int) *
                                                      liwork)) == NULL) {
                goto end;
            };
            if ((work = (double *)LAPACKE_malloc(sizeof(double) * lwork)) ==
                NULL) {
                goto end;
            }

            info = LAPACKE_dsyevd_work(LAPACK_COL_MAJOR, 'V', 'U',
                                       (lapack_int)size, data, (lapack_int)size,
                                       eigvals, work, lwork, iwork, liwork);

        end:
            if (iwork) {
                LAPACKE_free(iwork);
                iwork = NULL;
            }
            if (work) {
                LAPACKE_free(work);
                work = NULL;
            }

            /* info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, */
            /*                       'V', */
            /*                       'U', */
            /*                       (lapack_int)size, */
            /*                       data, */
            /*                       (lapack_int)size, */
            /*                       eigvals); */
            break;
    }

    return (int)info;
}

lapack_complex_double phonoc_complex_prod(const lapack_complex_double a,
                                          const lapack_complex_double b) {
    lapack_complex_double c;
    c = lapack_make_complex_double(
        lapack_complex_double_real(a) * lapack_complex_double_real(b) -
            lapack_complex_double_imag(a) * lapack_complex_double_imag(b),
        lapack_complex_double_imag(a) * lapack_complex_double_real(b) +
            lapack_complex_double_real(a) * lapack_complex_double_imag(b));
    return c;
}

void pinv_from_eigensolution(double *data, const double *eigvals,
                             const long size, const double cutoff,
                             const long pinv_method) {
    long i, ib, j, k, max_l, i_s, j_s;
    double *tmp_data;
    double e, sum;
    long *l;

    l = NULL;
    tmp_data = NULL;

    tmp_data = (double *)malloc(sizeof(double) * size * size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < size * size; i++) {
        tmp_data[i] = data[i];
    }

    l = (long *)malloc(sizeof(long) * size);
    max_l = 0;
    for (i = 0; i < size; i++) {
        if (pinv_method == 0) {
            e = fabs(eigvals[i]);
        } else {
            e = eigvals[i];
        }
        if (e > cutoff) {
            l[max_l] = i;
            max_l++;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for private(ib, j, k, i_s, j_s, sum)
#endif
    for (i = 0; i < size / 2; i++) {
        /* from front */
        i_s = i * size;
        for (j = i; j < size; j++) {
            j_s = j * size;
            sum = 0;
            for (k = 0; k < max_l; k++) {
                sum +=
                    tmp_data[i_s + l[k]] * tmp_data[j_s + l[k]] / eigvals[l[k]];
            }
            data[i_s + j] = sum;
            data[j_s + i] = sum;
        }
        /* from back */
        ib = size - i - 1;
        i_s = ib * size;
        for (j = ib; j < size; j++) {
            j_s = j * size;
            sum = 0;
            for (k = 0; k < max_l; k++) {
                sum +=
                    tmp_data[i_s + l[k]] * tmp_data[j_s + l[k]] / eigvals[l[k]];
            }
            data[i_s + j] = sum;
            data[j_s + ib] = sum;
        }
    }

    /* when size is odd */
    if ((size % 2) == 1) {
        i = (size - 1) / 2;
        i_s = i * size;
        for (j = i; j < size; j++) {
            j_s = j * size;
            sum = 0;
            for (k = 0; k < max_l; k++) {
                sum +=
                    tmp_data[i_s + l[k]] * tmp_data[j_s + l[k]] / eigvals[l[k]];
            }
            data[i_s + j] = sum;
            data[j_s + i] = sum;
        }
    }

    free(l);
    l = NULL;

    free(tmp_data);
    tmp_data = NULL;
}
