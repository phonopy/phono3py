#ifndef __test_utils_H__
#define __test_utils_H__

void lagmat_multiply_matrix_vector_l3(long v[3], const long a[3][3],
                                      const long b[3]);
void lagmat_multiply_matrix_l3(long m[3][3], const long a[3][3],
                               const long b[3][3]);
void lagmat_copy_matrix_l3(long a[3][3], const long b[3][3]);
long lagmat_check_identity_matrix_l3(const long a[3][3], const long b[3][3]);
#endif  // __test_utils_H__
