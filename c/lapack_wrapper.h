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

#ifndef __lapack_wrapper_H__
#define __lapack_wrapper_H__

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
                             const long size, const double cutoff,
                             const long pinv_method);
#endif

#endif
