/* SPDX-License-Identifier: BSD-3-Clause */

#include "funcs.h"

#include <math.h>

#include "phonoc_const.h"

#define INVSQRT2PI 0.3989422804014327

double funcs_bose_einstein(const double x, const double temperature_THz) {
    return 1.0 / (exp(x / temperature_THz) - 1);
}

double funcs_gaussian(const double x, const double sigma) {
    return INVSQRT2PI / sigma * exp(-x * x / 2 / sigma / sigma);
}

double funcs_inv_sinh_occupation(const double x, const double temperature_THz) {
    return 1.0 / sinh(x / 2 / temperature_THz);
}
