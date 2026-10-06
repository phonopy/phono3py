/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __lapack_wrapper_H__
#define __lapack_wrapper_H__

#include <stdint.h>

#if defined(_MSC_VER) || defined(MKL_BLAS) || defined(SCIPY_MKL_H)
#if defined(_MSC_VER)
typedef struct {
    double real;
    double imag;
} lapack_complex_double;
#else
#include <mkl.h>
#define lapack_complex_double MKL_Complex16
#endif
lapack_complex_double lapack_make_complex_double(double re, double im);
#define lapack_complex_double_real(z) ((z).real)
#define lapack_complex_double_imag(z) ((z).imag)
#else
#if defined(NO_INCLUDE_LAPACKE)
#include <complex.h>
#define lapack_complex_double double _Complex
#ifdef CMPLX
#define lapack_make_complex_double(re, im) CMPLX(re, im)
#else
#define lapack_make_complex_double(re, im) ((double _Complex)((re) + (im) * I))
#endif
#define lapack_complex_double_real(z) (creal(z))
#define lapack_complex_double_imag(z) (cimag(z))
#else
#if !defined(MKL_BLAS) && !defined(SCIPY_MKL_H)
#include <lapacke.h>
#endif
#endif
#endif

lapack_complex_double phonoc_complex_prod(const lapack_complex_double a,
                                          const lapack_complex_double b);

#ifndef NO_INCLUDE_LAPACKE
int phonopy_zheev(double *w, lapack_complex_double *a, const int n,
                  const char uplo);
int phonopy_pinv(double *data_out, const double *data_in, const int m,
                 const int n, const double cutoff);
void phonopy_pinv_mt(double *data_out, int *info_out, const double *data_in,
                     const int num_thread, const int *row_nums,
                     const int max_row_num, const int column_num,
                     const double cutoff);
int phonopy_dsyev(double *data, double *eigvals, const int size,
                  const int algorithm);

void pinv_from_eigensolution(double *data, const double *eigvals,
                             const int64_t size, const double cutoff,
                             const int64_t pinv_method);
#endif

#endif
