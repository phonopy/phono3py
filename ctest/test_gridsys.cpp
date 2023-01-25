#include <gtest/gtest.h>

extern "C" {
#include <math.h>

#include "gridsys.h"
#include "utils.h"
}

// Point group operations of rutile {R^T}
const long rutile_rec_rotations[16][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}},   {{0, -1, 0}, {1, 0, 0}, {0, 0, -1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},  {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
    {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   {{0, 1, 0}, {-1, 0, 0}, {0, 0, -1}},
    {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}},  {{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{0, -1, 0}, {-1, 0, 0}, {0, 0, -1}}, {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}},  {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},   {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}}};

// Symmetry operations of rutile 1x1x2 {R}
const long rutile112_symmetry_operations[32][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
    {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   {{0, 1, 0}, {-1, 0, 0}, {0, 0, -1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},  {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}},   {{0, -1, 0}, {1, 0, 0}, {0, 0, -1}},
    {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}},  {{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{0, -1, 0}, {-1, 0, 0}, {0, 0, -1}}, {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}},  {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},   {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
    {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   {{0, 1, 0}, {-1, 0, 0}, {0, 0, -1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},  {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}},   {{0, -1, 0}, {1, 0, 0}, {0, 0, -1}},
    {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}},  {{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{0, -1, 0}, {-1, 0, 0}, {0, 0, -1}}, {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}},  {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},   {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}}};

//  Point group operations of zincblende {R^T} (with time reversal)
const long zincblende_rec_rotations_with_time_reversal[24][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},    {{1, 1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, -1, 0}, {0, 0, 1}},  {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{-1, -1, 0}, {1, 0, 0}, {0, 0, 1}},  {{0, -1, 0}, {1, 1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},    {{1, 1, 0}, {0, -1, 0}, {0, 0, 1}},
    {{1, 0, 0}, {-1, -1, 0}, {0, 0, 1}},  {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{-1, -1, 0}, {0, 1, 0}, {0, 0, 1}},  {{-1, 0, 0}, {1, 1, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, {{-1, -1, 0}, {1, 0, 0}, {0, 0, -1}},
    {{0, -1, 0}, {1, 1, 0}, {0, 0, -1}},  {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}},
    {{1, 1, 0}, {-1, 0, 0}, {0, 0, -1}},  {{0, 1, 0}, {-1, -1, 0}, {0, 0, -1}},
    {{0, -1, 0}, {-1, 0, 0}, {0, 0, -1}}, {{-1, -1, 0}, {0, 1, 0}, {0, 0, -1}},
    {{-1, 0, 0}, {1, 1, 0}, {0, 0, -1}},  {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},
    {{1, 1, 0}, {0, -1, 0}, {0, 0, -1}},  {{1, 0, 0}, {-1, -1, 0}, {0, 0, -1}}};

//  Point group operations of zincblende {R^T} (without time reversal)
const long zincblende_rec_rotations_without_time_reversal[12][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},   {{1, 1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, -1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{-1, -1, 0}, {1, 0, 0}, {0, 0, 1}}, {{0, -1, 0}, {1, 1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},   {{1, 1, 0}, {0, -1, 0}, {0, 0, 1}},
    {{1, 0, 0}, {-1, -1, 0}, {0, 0, 1}}, {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{-1, -1, 0}, {0, 1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {1, 1, 0}, {0, 0, 1}}};

// Symmetry operations of zincblende 1x1x2 {R}
const long zincblende112_symmetry_operations[24][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},   {{1, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, -1, 0}, {1, -1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{-1, 1, 0}, {-1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {-1, 1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},   {{1, 0, 0}, {1, -1, 0}, {0, 0, 1}},
    {{1, -1, 0}, {0, -1, 0}, {0, 0, 1}}, {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {-1, 1, 0}, {0, 0, 1}}, {{-1, 1, 0}, {0, 1, 0}, {0, 0, 1}},
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},   {{1, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, -1, 0}, {1, -1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{-1, 1, 0}, {-1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {-1, 1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}},   {{1, 0, 0}, {1, -1, 0}, {0, 0, 1}},
    {{1, -1, 0}, {0, -1, 0}, {0, 0, 1}}, {{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {-1, 1, 0}, {0, 0, 1}}, {{-1, 1, 0}, {0, 1, 0}, {0, 0, 1}}};

const long AgNO2_rec_rotations[8][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},    {{0, 1, 0}, {1, 0, 0}, {-1, -1, -1}},
    {{1, 1, 1}, {0, 0, -1}, {0, -1, 0}},  {{0, 0, -1}, {1, 1, 1}, {-1, 0, 0}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, {{0, -1, 0}, {-1, 0, 0}, {1, 1, 1}},
    {{-1, -1, -1}, {0, 0, 1}, {0, 1, 0}}, {{0, 0, 1}, {-1, -1, -1}, {1, 0, 0}}};

const long AgNO2_tilde_rec_rotations[8][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{5, -12, -2}, {12, -25, -4}, {-60, 120, 19}},
    {{1, 0, 0}, {0, 11, 2}, {0, -60, -11}},
    {{5, -12, -2}, {12, -35, -6}, {-60, 180, 31}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
    {{-5, 12, 2}, {-12, 25, 4}, {60, -120, -19}},
    {{-1, 0, 0}, {0, -11, -2}, {0, 60, 11}},
    {{-5, 12, 2}, {-12, 35, 6}, {60, -180, -31}}};

/**
 * @brief gridsys_get_all_grid_addresses test
 * Return all GR-grid addresses of {(x, y, z)} where x runs fastest.
 */
TEST(test_gridsys, test_gridsys_get_all_grid_addresses) {
    long(*gr_grid_addresses)[3];
    long D_diag[3] = {3, 4, 5};
    long n, i, j, k;
    long grid_index = 0;

    n = D_diag[0] * D_diag[1] * D_diag[2];
    gr_grid_addresses = (long(*)[3])malloc(sizeof(long[3]) * n);
    gridsys_get_all_grid_addresses(gr_grid_addresses, D_diag);

    for (k = 0; k < D_diag[2]; k++) {
        for (j = 0; j < D_diag[1]; j++) {
            for (i = 0; i < D_diag[0]; i++) {
                ASSERT_EQ(gr_grid_addresses[grid_index][0], i);
                ASSERT_EQ(gr_grid_addresses[grid_index][1], j);
                ASSERT_EQ(gr_grid_addresses[grid_index][2], k);
                grid_index++;
            }
        }
    }
    free(gr_grid_addresses);
    gr_grid_addresses = NULL;
}

/**
 * @brief gridsys_get_double_grid_address
 * Return double grid address of single grid address with shift in GR-grid.
 * PS can be other than 0 and 1 for non-diagonal grid matrix.
 */
TEST(test_gridsys, test_gridsys_get_double_grid_address) {
    long address_double[3];
    long address[3] = {1, 2, 3};
    long PS[8][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                     {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    long i, j;

    for (i = 0; i < 8; i++) {
        gridsys_get_double_grid_address(address_double, address, PS[i]);
        for (j = 0; j < 3; j++) {
            ASSERT_EQ(address_double[j], address[j] * 2 + PS[i][j]);
        }
    }
}

/**
 * @brief gridsys_get_grid_address_from_index
 * Return single grid address of grid point index in GR-grid. See the definition
 * of grid point index at gridsys_get_all_grid_addresses.
 */
TEST(test_gridsys, test_gridsys_get_grid_address_from_index) {
    long address[3];
    long D_diag[3] = {3, 4, 5};
    long i, j, k, l;
    long grid_index = 0;

    for (k = 0; k < D_diag[2]; k++) {
        for (j = 0; j < D_diag[1]; j++) {
            for (i = 0; i < D_diag[0]; i++) {
                gridsys_get_grid_address_from_index(address, grid_index,
                                                    D_diag);
                for (l = 0; l < 3; l++) {
                    ASSERT_EQ(address[0], i);
                    ASSERT_EQ(address[1], j);
                    ASSERT_EQ(address[2], k);
                }
                grid_index++;
            }
        }
    }
}

/**
 * @brief gridsys_get_double_grid_index
 * Return grid point index corresponding to double grid address in GR-grid.
 */
TEST(test_gridsys, test_gridsys_get_double_grid_index) {
    long address_double[3], address[3];
    long D_diag[3] = {3, 4, 5};
    long PS[8][3] = {{0, 0, 0}, {0, 0, 3}, {0, 3, 0}, {0, 3, 3},
                     {3, 0, 0}, {3, 0, 3}, {3, 3, 0}, {3, 3, 3}};
    long i, j, k, l, grid_index;

    for (l = 0; l < 8; l++) {
        grid_index = 0;
        for (k = 0; k < D_diag[2]; k++) {
            address[2] = k;
            for (j = 0; j < D_diag[1]; j++) {
                address[1] = j;
                for (i = 0; i < D_diag[0]; i++) {
                    address[0] = i;
                    gridsys_get_double_grid_address(address_double, address,
                                                    PS[l]);
                    ASSERT_EQ(grid_index, gridsys_get_double_grid_index(
                                              address_double, D_diag, PS[l]));
                    grid_index++;
                }
            }
        }
    }
}

/**
 * @brief gridsys_get_grid_index_from_address
 * Return grid point index corresponding to grid address in GR-grid.
 */
TEST(test_gridsys, test_gridsys_get_grid_index_from_address) {
    long address[3];
    long D_diag[3] = {3, 4, 5};
    long i, j, k;
    long grid_index = 0;

    for (k = 0; k < D_diag[2]; k++) {
        address[2] = k;
        for (j = 0; j < D_diag[1]; j++) {
            address[1] = j;
            for (i = 0; i < D_diag[0]; i++) {
                address[0] = i;
                ASSERT_EQ(grid_index,
                          gridsys_get_grid_index_from_address(address, D_diag));
                grid_index++;
            }
        }
    }
}

/**
 * @brief gridsys_rotate_grid_index
 * Return grid point index of rotated address of given grid point index.
 */
TEST(test_gridsys, test_gridsys_rotate_grid_index) {
    long address[3], rot_address[3];
    long D_diag[3] = {3, 4, 5};
    long PS[8][3] = {{0, 0, 0}, {0, 0, 3}, {0, 3, 0}, {0, 3, 3},
                     {3, 0, 0}, {3, 0, 3}, {3, 3, 0}, {3, 3, 3}};
    long i, j, k, l;
    long grid_index = 0;
    long rot_grid_index;
    long rot_grid_indices[8][60] = {
        {0,  9,  6,  1,  10, 7,  2,  11, 8,  0,  9,  6,  48, 57, 54,
         49, 58, 55, 50, 59, 56, 48, 57, 54, 36, 45, 42, 37, 46, 43,
         38, 47, 44, 36, 45, 42, 24, 33, 30, 25, 34, 31, 26, 35, 32,
         24, 33, 30, 12, 21, 18, 13, 22, 19, 14, 23, 20, 12, 21, 18},
        {24, 33, 30, 25, 34, 31, 26, 35, 32, 24, 33, 30, 12, 21, 18,
         13, 22, 19, 14, 23, 20, 12, 21, 18, 0,  9,  6,  1,  10, 7,
         2,  11, 8,  0,  9,  6,  48, 57, 54, 49, 58, 55, 50, 59, 56,
         48, 57, 54, 36, 45, 42, 37, 46, 43, 38, 47, 44, 36, 45, 42},
        {10, 7,  4,  11, 8,  5,  9,  6,  3,  10, 7,  4,  58, 55, 52,
         59, 56, 53, 57, 54, 51, 58, 55, 52, 46, 43, 40, 47, 44, 41,
         45, 42, 39, 46, 43, 40, 34, 31, 28, 35, 32, 29, 33, 30, 27,
         34, 31, 28, 22, 19, 16, 23, 20, 17, 21, 18, 15, 22, 19, 16},
        {34, 31, 28, 35, 32, 29, 33, 30, 27, 34, 31, 28, 22, 19, 16,
         23, 20, 17, 21, 18, 15, 22, 19, 16, 10, 7,  4,  11, 8,  5,
         9,  6,  3,  10, 7,  4,  58, 55, 52, 59, 56, 53, 57, 54, 51,
         58, 55, 52, 46, 43, 40, 47, 44, 41, 45, 42, 39, 46, 43, 40},
        {11, 8,  5,  9,  6,  3,  9,  6,  3,  10, 7,  4,  59, 56, 53,
         57, 54, 51, 57, 54, 51, 58, 55, 52, 47, 44, 41, 45, 42, 39,
         45, 42, 39, 46, 43, 40, 35, 32, 29, 33, 30, 27, 33, 30, 27,
         34, 31, 28, 23, 20, 17, 21, 18, 15, 21, 18, 15, 22, 19, 16},
        {35, 32, 29, 33, 30, 27, 33, 30, 27, 34, 31, 28, 23, 20, 17,
         21, 18, 15, 21, 18, 15, 22, 19, 16, 11, 8,  5,  9,  6,  3,
         9,  6,  3,  10, 7,  4,  59, 56, 53, 57, 54, 51, 57, 54, 51,
         58, 55, 52, 47, 44, 41, 45, 42, 39, 45, 42, 39, 46, 43, 40},
        {3,  0,  9,  4,  1,  10, 5,  2,  11, 3,  0,  9,  51, 48, 57,
         52, 49, 58, 53, 50, 59, 51, 48, 57, 39, 36, 45, 40, 37, 46,
         41, 38, 47, 39, 36, 45, 27, 24, 33, 28, 25, 34, 29, 26, 35,
         27, 24, 33, 15, 12, 21, 16, 13, 22, 17, 14, 23, 15, 12, 21},
        {27, 24, 33, 28, 25, 34, 29, 26, 35, 27, 24, 33, 15, 12, 21,
         16, 13, 22, 17, 14, 23, 15, 12, 21, 3,  0,  9,  4,  1,  10,
         5,  2,  11, 3,  0,  9,  51, 48, 57, 52, 49, 58, 53, 50, 59,
         51, 48, 57, 39, 36, 45, 40, 37, 46, 41, 38, 47, 39, 36, 45}};

    // Rutile R^T of a screw operation.
    long rotation[3][3] = {{0, 1, 0}, {-1, 0, 0}, {0, 0, -1}};

    grid_index = 0;
    for (k = 0; k < D_diag[2]; k++) {
        address[2] = k;
        for (j = 0; j < D_diag[1]; j++) {
            address[1] = j;
            for (i = 0; i < D_diag[0]; i++) {
                address[0] = i;
                lagmat_multiply_matrix_vector_l3(rot_address, rotation,
                                                 address);
                ASSERT_EQ(
                    (gridsys_get_grid_index_from_address(rot_address, D_diag)),
                    (gridsys_rotate_grid_index(grid_index, rotation, D_diag,
                                               PS[0])));
                grid_index++;
            }
        }
    }
    for (l = 0; l < 8; l++) {
        for (grid_index = 0; grid_index < 60; grid_index++) {
            ASSERT_EQ(
                rot_grid_indices[l][grid_index],
                gridsys_rotate_grid_index(grid_index, rotation, D_diag, PS[l]));
        }
    }
}

/**
 * @brief gridsys_get_reciprocal_point_group with rutile symmetry
 * Return {R^T} of crystallographic point group {R} with and without time
 * reversal symmetry.
 */
TEST(test_gridsys, test_gridsys_get_reciprocal_point_group_rutile) {
    long i, j, k, num_R;
    long rec_rotations[48][3][3];
    long is_time_reversal, is_found;

    for (is_time_reversal = 0; is_time_reversal < 2; is_time_reversal++) {
        num_R = gridsys_get_reciprocal_point_group(
            rec_rotations, rutile112_symmetry_operations, 32, is_time_reversal);
        ASSERT_EQ(16, num_R);
        for (i = 0; i < 16; i++) {
            is_found = 0;
            for (j = 0; j < 16; j++) {
                if (lagmat_check_identity_matrix_l3(rec_rotations[i],
                                                    rutile_rec_rotations[j])) {
                    is_found = 1;
                    break;
                }
            }
            ASSERT_TRUE(is_found);
        }
    }
}

/**
 * @brief gridsys_get_reciprocal_point_group with zincblende symmetry
 * Return {R^T} of crystallographic point group {R} with and without time
 * reversal symmetry.
 */
TEST(test_gridsys, test_gridsys_get_reciprocal_point_group_zincblende) {
    long i, j, k, num_R;
    long rec_rotations[48][3][3];
    long is_found;

    // Without time reversal symmetry.
    num_R = gridsys_get_reciprocal_point_group(
        rec_rotations, zincblende112_symmetry_operations, 24, 0);
    ASSERT_EQ(12, num_R);
    for (i = 0; i < 12; i++) {
        is_found = 0;
        for (j = 0; j < 12; j++) {
            if (lagmat_check_identity_matrix_l3(
                    rec_rotations[i],
                    zincblende_rec_rotations_without_time_reversal[j])) {
                is_found = 1;
                break;
            }
        }
        ASSERT_TRUE(is_found);
    }

    // With time reversal symmetry.
    num_R = gridsys_get_reciprocal_point_group(
        rec_rotations, zincblende112_symmetry_operations, 24, 1);
    ASSERT_EQ(24, num_R);
    for (i = 0; i < 24; i++) {
        is_found = 0;
        for (j = 0; j < 24; j++) {
            if (lagmat_check_identity_matrix_l3(
                    rec_rotations[i],
                    zincblende_rec_rotations_with_time_reversal[j])) {
                is_found = 1;
                break;
            }
        }
        ASSERT_TRUE(is_found);
    }
}

/**
 * @brief gridsys_get_snf3x3 (1)
 * Return D, P, Q of SNF.
 */
TEST(test_gridsys, test_gridsys_get_snf3x3) {
    long D_diag[3], P[3][3], Q[3][3];
    long A[3][3] = {{0, 16, 16}, {16, 0, 16}, {6, 6, 0}};
    const long ref_P[3][3] = {{0, -1, 3}, {1, 0, 0}, {-3, 3, -8}};
    const long ref_Q[3][3] = {{1, 8, 17}, {0, 0, -1}, {0, 1, 1}};
    long succeeded;

    succeeded = gridsys_get_snf3x3(D_diag, P, Q, A);
    ASSERT_TRUE(succeeded);
    ASSERT_EQ(2, D_diag[0]);
    ASSERT_EQ(16, D_diag[1]);
    ASSERT_EQ(96, D_diag[2]);
    ASSERT_TRUE(lagmat_check_identity_matrix_l3(ref_P, P));
    ASSERT_TRUE(lagmat_check_identity_matrix_l3(ref_Q, Q));
}

/**
 * @brief gridsys_get_snf3x3 (AgNO2)
 * Return D, P, Q of SNF.
 */
TEST(test_gridsys, test_gridsys_get_snf3x3_AgNO2) {
    long D_diag[3], P[3][3], Q[3][3];
    long A[3][3] = {{0, 5, 5}, {2, 0, 2}, {3, 3, 0}};
    const long ref_P[3][3] = {{0, -1, 1}, {-1, -3, 2}, {6, 15, -10}};
    const long ref_Q[3][3] = {{1, -3, -31}, {0, 1, 11}, {0, 0, 1}};
    long succeeded;

    succeeded = gridsys_get_snf3x3(D_diag, P, Q, A);
    ASSERT_TRUE(succeeded);
    ASSERT_EQ(1, D_diag[0]);
    ASSERT_EQ(1, D_diag[1]);
    ASSERT_EQ(60, D_diag[2]);
    ASSERT_TRUE(lagmat_check_identity_matrix_l3(ref_P, P));
    ASSERT_TRUE(lagmat_check_identity_matrix_l3(ref_Q, Q));
}

/**
 * @brief gridsys_transform_rotations
 * Transform {R^T} to {R^T} with respect to transformed microzone basis vectors
 * in GR-grid
 */
TEST(test_gridsys, test_gridsys_transform_rotations) {
    long transformed_rotations[8][3][3];
    const long D_diag[3] = {1, 1, 60};
    const long Q[3][3] = {{1, -3, -31}, {0, 1, 11}, {0, 0, 1}};
    long i, j, is_found, succeeded;

    succeeded = gridsys_transform_rotations(transformed_rotations,
                                            AgNO2_rec_rotations, 8, D_diag, Q);
    ASSERT_TRUE(succeeded);
    for (i = 0; i < 8; i++) {
        is_found = 0;
        for (j = 0; j < 8; j++) {
            if (lagmat_check_identity_matrix_l3(AgNO2_tilde_rec_rotations[j],
                                                transformed_rotations[i])) {
                is_found = 1;
                break;
            }
        }
        ASSERT_TRUE(is_found);
    }
}

/**
 * @brief gridsys_get_thm_integration_weight
 * Return integration weight of linear tetrahedron method
 */
TEST(test_gridsys, test_gridsys_get_thm_integration_weight) {
    const double freqs_at[2] = {7.75038996, 8.45225776};
    const double tetra_freqs[24][4] = {
        {8.31845176, 8.69248151, 8.78939432, 8.66179133},
        {8.31845176, 8.69248151, 8.57211855, 8.66179133},
        {8.31845176, 8.3073908, 8.78939432, 8.66179133},
        {8.31845176, 8.3073908, 8.16360975, 8.66179133},
        {8.31845176, 8.15781566, 8.57211855, 8.66179133},
        {8.31845176, 8.15781566, 8.16360975, 8.66179133},
        {8.31845176, 8.3073908, 8.16360975, 7.23665561},
        {8.31845176, 8.15781566, 8.16360975, 7.23665561},
        {8.31845176, 8.69248151, 8.57211855, 8.25247917},
        {8.31845176, 8.15781566, 8.57211855, 8.25247917},
        {8.31845176, 8.15781566, 7.40609306, 8.25247917},
        {8.31845176, 8.15781566, 7.40609306, 7.23665561},
        {8.31845176, 8.69248151, 8.78939432, 8.55165578},
        {8.31845176, 8.3073908, 8.78939432, 8.55165578},
        {8.31845176, 8.3073908, 7.56474684, 8.55165578},
        {8.31845176, 8.3073908, 7.56474684, 7.23665561},
        {8.31845176, 8.69248151, 8.60076148, 8.55165578},
        {8.31845176, 8.69248151, 8.60076148, 8.25247917},
        {8.31845176, 7.72920193, 8.60076148, 8.55165578},
        {8.31845176, 7.72920193, 8.60076148, 8.25247917},
        {8.31845176, 7.72920193, 7.56474684, 8.55165578},
        {8.31845176, 7.72920193, 7.56474684, 7.23665561},
        {8.31845176, 7.72920193, 7.40609306, 8.25247917},
        {8.31845176, 7.72920193, 7.40609306, 7.23665561},
    };
    const double iw_I_ref[2] = {0.37259443, 1.79993056};
    const double iw_J_ref[2] = {0.05740597, 0.76331859};
    double iw_I, iw_J;
    long i;

    for (i = 0; i < 2; i++) {
        ASSERT_LT((fabs(gridsys_get_thm_integration_weight(freqs_at[i],
                                                           tetra_freqs, 'I') -
                        iw_I_ref[i])),
                  1e-5);
        ASSERT_LT((fabs(gridsys_get_thm_integration_weight(freqs_at[i],
                                                           tetra_freqs, 'J') -
                        iw_J_ref[i])),
                  1e-5);
    }
}

/**
 * @brief gridsys_get_thm_relative_grid_address
 * Return relative grid addresses for linear tetrahedron method.
 */
TEST(test_gridsys, test_gridsys_get_thm_relative_grid_address) {
    long all_rel_grid_address[4][24][4][3];
    long rel_grid_addresses[24][4][3];
    double rec_vectors[4][3][3] = {{{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}},
                                   {{-1, -1, -1}, {1, 1, -1}, {1, -1, 1}},
                                   {{1, 1, -1}, {-1, -1, -1}, {-1, 1, 1}},
                                   {{1, -1, 1}, {-1, 1, 1}, {-1, -1, -1}}};
    long i, j, k, l, main_diagonal;

    gridsys_get_thm_all_relative_grid_address(all_rel_grid_address);
    for (i = 0; i < 4; i++) {
        main_diagonal = gridsys_get_thm_relative_grid_address(
            rel_grid_addresses, rec_vectors[i]);
        ASSERT_EQ(i, main_diagonal);
        for (j = 0; j < 24; j++) {
            for (k = 0; k < 4; k++) {
                for (l = 0; l < 3; l++) {
                    ASSERT_EQ(all_rel_grid_address[i][j][k][l],
                              rel_grid_addresses[j][k][l]);
                }
            }
        }
    }
}

/**
 * @brief gridsys_get_ir_grid_map tested by rutile rotations
 * Return grid point mapping table to ir-grid points
 */
TEST(test_gridsys, test_gridsys_get_ir_grid_map_rutile) {
    long *ir_grid_map;
    long n, i, j;
    long D_diag[3] = {4, 4, 6};
    long PS[4][3] = {{0, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}};

    long ref_ir_grid_maps[4][96] = {
        {0,  1,  2,  1,  1,  5,  6,  5,  2,  6,  10, 6,  1,  5,  6,  5,
         16, 17, 18, 17, 17, 21, 22, 21, 18, 22, 26, 22, 17, 21, 22, 21,
         32, 33, 34, 33, 33, 37, 38, 37, 34, 38, 42, 38, 33, 37, 38, 37,
         48, 49, 50, 49, 49, 53, 54, 53, 50, 54, 58, 54, 49, 53, 54, 53,
         32, 33, 34, 33, 33, 37, 38, 37, 34, 38, 42, 38, 33, 37, 38, 37,
         16, 17, 18, 17, 17, 21, 22, 21, 18, 22, 26, 22, 17, 21, 22, 21},
        {0,  1,  1,  0,  1,  5,  5,  1,  1,  5,  5,  1,  0,  1,  1,  0,
         16, 17, 17, 16, 17, 21, 21, 17, 17, 21, 21, 17, 16, 17, 17, 16,
         32, 33, 33, 32, 33, 37, 37, 33, 33, 37, 37, 33, 32, 33, 33, 32,
         48, 49, 49, 48, 49, 53, 53, 49, 49, 53, 53, 49, 48, 49, 49, 48,
         32, 33, 33, 32, 33, 37, 37, 33, 33, 37, 37, 33, 32, 33, 33, 32,
         16, 17, 17, 16, 17, 21, 21, 17, 17, 21, 21, 17, 16, 17, 17, 16},
        {0,  1,  2,  1,  1,  5,  6,  5,  2,  6,  10, 6,  1,  5,  6,  5,
         16, 17, 18, 17, 17, 21, 22, 21, 18, 22, 26, 22, 17, 21, 22, 21,
         32, 33, 34, 33, 33, 37, 38, 37, 34, 38, 42, 38, 33, 37, 38, 37,
         32, 33, 34, 33, 33, 37, 38, 37, 34, 38, 42, 38, 33, 37, 38, 37,
         16, 17, 18, 17, 17, 21, 22, 21, 18, 22, 26, 22, 17, 21, 22, 21,
         0,  1,  2,  1,  1,  5,  6,  5,  2,  6,  10, 6,  1,  5,  6,  5},
        {0,  1,  1,  0,  1,  5,  5,  1,  1,  5,  5,  1,  0,  1,  1,  0,
         16, 17, 17, 16, 17, 21, 21, 17, 17, 21, 21, 17, 16, 17, 17, 16,
         32, 33, 33, 32, 33, 37, 37, 33, 33, 37, 37, 33, 32, 33, 33, 32,
         32, 33, 33, 32, 33, 37, 37, 33, 33, 37, 37, 33, 32, 33, 33, 32,
         16, 17, 17, 16, 17, 21, 21, 17, 17, 21, 21, 17, 16, 17, 17, 16,
         0,  1,  1,  0,  1,  5,  5,  1,  1,  5,  5,  1,  0,  1,  1,  0}};

    n = D_diag[0] * D_diag[1] * D_diag[2];
    ir_grid_map = (long *)malloc(sizeof(long) * n);

    for (i = 0; i < 4; i++) {
        gridsys_get_ir_grid_map(ir_grid_map, rutile_rec_rotations, 16, D_diag,
                                PS[i]);
        for (j = 0; j < n; j++) {
            ASSERT_EQ(ref_ir_grid_maps[i][j], ir_grid_map[j]);
        }
    }

    free(ir_grid_map);
    ir_grid_map = NULL;
}

/**
 * @brief gridsys_get_ir_grid_map tested by wurtzite rotations
 * Return grid point mapping table to ir-grid points
 */
TEST(test_gridsys, test_gridsys_get_ir_grid_map_wurtzite) {
    long *ir_grid_map;
    long n, i, j;
    long D_diag[3] = {5, 5, 4};
    long PS[2][3] = {{0, 0, 0}, {0, 0, 1}};

    long ref_ir_grid_maps[2][100] = {
        {0,  1,  2,  2,  1,  1,  6,  7,  6,  1,  2,  7,  7,  2,  6,  2,  6,
         2,  7,  7,  1,  1,  6,  7,  6,  25, 26, 27, 27, 26, 26, 31, 32, 31,
         26, 27, 32, 32, 27, 31, 27, 31, 27, 32, 32, 26, 26, 31, 32, 31, 50,
         51, 52, 52, 51, 51, 56, 57, 56, 51, 52, 57, 57, 52, 56, 52, 56, 52,
         57, 57, 51, 51, 56, 57, 56, 25, 26, 27, 27, 26, 26, 31, 32, 31, 26,
         27, 32, 32, 27, 31, 27, 31, 27, 32, 32, 26, 26, 31, 32, 31},
        {0,  1,  2,  2,  1,  1,  6,  7,  6,  1,  2,  7,  7,  2,  6,  2,  6,
         2,  7,  7,  1,  1,  6,  7,  6,  25, 26, 27, 27, 26, 26, 31, 32, 31,
         26, 27, 32, 32, 27, 31, 27, 31, 27, 32, 32, 26, 26, 31, 32, 31, 25,
         26, 27, 27, 26, 26, 31, 32, 31, 26, 27, 32, 32, 27, 31, 27, 31, 27,
         32, 32, 26, 26, 31, 32, 31, 0,  1,  2,  2,  1,  1,  6,  7,  6,  1,
         2,  7,  7,  2,  6,  2,  6,  2,  7,  7,  1,  1,  6,  7,  6}};

    n = D_diag[0] * D_diag[1] * D_diag[2];
    ir_grid_map = (long *)malloc(sizeof(long) * n);

    for (i = 0; i < 2; i++) {
        gridsys_get_ir_grid_map(ir_grid_map,
                                zincblende_rec_rotations_with_time_reversal, 24,
                                D_diag, PS[i]);
        for (j = 0; j < n; j++) {
            ASSERT_EQ(ref_ir_grid_maps[i][j], ir_grid_map[j]);
        }
    }

    free(ir_grid_map);
    ir_grid_map = NULL;
}

/**
 * @brief gridsys_get_bz_grid_addresses by FCC
 * Return BZ grid addresses
 */
TEST(test_gridsys, test_gridsys_get_bz_grid_addresses_FCC) {
    long D_diag[3] = {4, 4, 4};
    long ref_bz_addresses[89][3] = {
        {0, 0, 0},    {1, 0, 0},    {-2, 0, 0},   {2, 0, 0},    {-1, 0, 0},
        {0, 1, 0},    {1, 1, 0},    {2, 1, 0},    {-1, 1, 0},   {0, -2, 0},
        {0, 2, 0},    {1, 2, 0},    {-2, -2, 0},  {2, 2, 0},    {-1, -2, 0},
        {0, -1, 0},   {1, -1, 0},   {-2, -1, 0},  {-1, -1, 0},  {0, 0, 1},
        {1, 0, 1},    {2, 0, 1},    {-1, 0, 1},   {0, 1, 1},    {1, 1, 1},
        {2, 1, 1},    {-1, 1, 1},   {0, 2, 1},    {1, 2, 1},    {2, 2, 1},
        {-1, -2, 1},  {-1, -2, -3}, {-1, 2, 1},   {3, 2, 1},    {0, -1, 1},
        {1, -1, 1},   {-2, -1, 1},  {-2, -1, -3}, {2, -1, 1},   {2, 3, 1},
        {-1, -1, 1},  {0, 0, -2},   {0, 0, 2},    {1, 0, 2},    {-2, 0, -2},
        {2, 0, 2},    {-1, 0, -2},  {0, 1, 2},    {1, 1, 2},    {2, 1, 2},
        {-1, 1, -2},  {-1, 1, 2},   {-1, -3, -2}, {3, 1, 2},    {0, -2, -2},
        {0, 2, 2},    {1, 2, 2},    {-2, -2, -2}, {2, 2, 2},    {-1, -2, -2},
        {0, -1, -2},  {1, -1, -2},  {1, -1, 2},   {1, 3, 2},    {-3, -1, -2},
        {-2, -1, -2}, {-1, -1, -2}, {0, 0, -1},   {1, 0, -1},   {-2, 0, -1},
        {-1, 0, -1},  {0, 1, -1},   {1, 1, -1},   {-2, 1, -1},  {-2, -3, -1},
        {2, 1, -1},   {2, 1, 3},    {-1, 1, -1},  {0, -2, -1},  {1, -2, -1},
        {1, 2, -1},   {1, 2, 3},    {-3, -2, -1}, {-2, -2, -1}, {-1, -2, -1},
        {0, -1, -1},  {1, -1, -1},  {-2, -1, -1}, {-1, -1, -1},
    };
    long ref_bz_map[65] = {0,  1,  2,  4,  5,  6,  7,  8,  9,  11, 12, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                           29, 30, 34, 35, 36, 40, 41, 43, 44, 46, 47, 48, 49,
                           50, 54, 56, 57, 59, 60, 61, 65, 66, 67, 68, 69, 70,
                           71, 72, 73, 77, 78, 79, 83, 84, 85, 86, 87, 88, 89};
    long ref_bzg2grg[89] = {
        0,  1,  2,  2,  3,  4,  5,  6,  7,  8,  8,  9,  10, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 27, 27, 27, 28, 29,
        30, 30, 30, 30, 31, 32, 32, 33, 34, 34, 35, 36, 37, 38, 39, 39, 39, 39,
        40, 40, 41, 42, 42, 43, 44, 45, 45, 45, 45, 46, 47, 48, 49, 50, 51, 52,
        53, 54, 54, 54, 54, 55, 56, 57, 57, 57, 57, 58, 59, 60, 61, 62, 63};
    double rec_lattice[3][3] = {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}};
    long PS[3] = {0, 0, 0};
    long Q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    long bz_grid_addresses[125][3];
    long bz_map[65];
    long bzg2grg[125];
    long bz_size, i, j;

    bz_size = gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg,
                                            D_diag, Q, PS, rec_lattice, 2);
    ASSERT_EQ(89, bz_size);
    for (i = 0; i < 89; i++) {
        for (j = 0; j < 3; j++) {
            ASSERT_EQ(ref_bz_addresses[i][j], bz_grid_addresses[i][j]);
        }
    }
    for (i = 0; i < 65; i++) {
        ASSERT_EQ(ref_bz_map[i], bz_map[i]);
    }
    for (i = 0; i < 89; i++) {
        ASSERT_EQ(ref_bzg2grg[i], bzg2grg[i]);
    }
}

/**
 * @brief gridsys_get_bz_grid_addresses by wurtzite (angle 120)
 * @details Niggli reduction changes (swaps) basis vectors.
 * Return BZ grid addresses
 */
TEST(test_gridsys, test_gridsys_get_bz_grid_addresses_wurtzite1) {
    long ref_bz_addresses[155][3] = {
        {0, 0, 0},     {0, 1, 0},     {0, 2, 0},     {0, -2, 0},
        {0, -1, 0},    {0, 0, 1},     {0, 1, 1},     {0, 2, 1},
        {0, -3, 1},    {0, -2, 1},    {0, -1, 1},    {-4, 0, -18},
        {0, 0, 2},     {-4, 1, -18},  {0, 1, 2},     {1, 1, 2},
        {5, 1, 22},    {-4, -3, -18}, {0, -3, 2},    {1, 2, 2},
        {5, 2, 22},    {-4, -2, -18}, {0, -2, 2},    {-4, -1, -18},
        {0, -1, 2},    {1, 0, 3},     {1, 1, 3},     {1, 2, 3},
        {1, 3, 3},     {-4, -2, -17}, {1, -1, 3},    {-4, -1, -17},
        {1, 0, 4},     {1, 1, 4},     {1, 2, 4},     {1, -2, 4},
        {1, 3, 4},     {1, -1, 4},    {1, 0, 5},     {1, 1, 5},
        {1, 2, 5},     {1, -2, 5},    {1, -1, 5},    {-3, 0, -14},
        {1, 0, 6},     {-3, 1, -14},  {1, 1, 6},     {-3, 2, -14},
        {1, 2, 6},     {-3, -3, -14}, {1, -3, 6},    {-3, -2, -14},
        {1, -2, 6},    {-3, -1, -14}, {1, -1, 6},    {-3, 0, -13},
        {-3, 1, -13},  {2, 1, 7},     {-3, -3, -13}, {2, 2, 7},
        {-3, -2, -13}, {-3, -1, -13}, {2, 0, 8},     {2, 1, 8},
        {2, 2, 8},     {2, 3, 8},     {-3, -2, -12}, {2, -1, 8},
        {-3, -1, -12}, {2, 0, 9},     {2, 1, 9},     {2, 2, 9},
        {2, -2, 9},    {2, 3, 9},     {2, -1, 9},    {-2, 0, -10},
        {2, 0, 10},    {-2, 1, -10},  {2, 1, 10},    {-2, 2, -10},
        {2, 2, 10},    {-2, -2, -10}, {2, -2, 10},   {-2, -1, -10},
        {2, -1, 10},   {-2, 0, -9},   {-2, 1, -9},   {-2, 2, -9},
        {-2, -3, -9},  {-2, -2, -9},  {-2, -1, -9},  {-2, 0, -8},
        {-2, 1, -8},   {3, 1, 12},    {-2, -3, -8},  {3, 2, 12},
        {-2, -2, -8},  {-2, -1, -8},  {3, 0, 13},    {3, 1, 13},
        {3, 2, 13},    {3, 3, 13},    {-2, -2, -7},  {3, -1, 13},
        {-2, -1, -7},  {-1, 0, -6},   {3, 0, 14},    {-1, 1, -6},
        {3, 1, 14},    {-1, 2, -6},   {3, 2, 14},    {-1, -2, -6},
        {3, -2, 14},   {-1, 3, -6},   {3, 3, 14},    {-1, -1, -6},
        {3, -1, 14},   {-1, 0, -5},   {-1, 1, -5},   {-1, 2, -5},
        {-1, -2, -5},  {-1, -1, -5},  {-1, 0, -4},   {-1, 1, -4},
        {-1, 2, -4},   {-1, -3, -4},  {-1, -2, -4},  {-1, -1, -4},
        {-1, 0, -3},   {-1, 1, -3},   {4, 1, 17},    {-1, -3, -3},
        {4, 2, 17},    {-1, -2, -3},  {-1, -1, -3},  {0, 0, -2},
        {4, 0, 18},    {0, 1, -2},    {4, 1, 18},    {0, 2, -2},
        {4, 2, 18},    {0, 3, -2},    {4, 3, 18},    {-5, -2, -22},
        {-1, -2, -2},  {0, -1, -2},   {4, -1, 18},   {-5, -1, -22},
        {-1, -1, -2},  {0, 0, -1},    {0, 1, -1},    {0, 2, -1},
        {0, -2, -1},   {0, 3, -1},    {0, -1, -1}};
    long ref_bz_map[101] = {
        0,   1,   2,   3,   4,   5,   6,   7,   9,   10,  11,  13,  17,
        21,  23,  25,  26,  27,  28,  30,  32,  33,  34,  35,  37,  38,
        39,  40,  41,  42,  43,  45,  47,  51,  53,  55,  56,  58,  60,
        61,  62,  63,  64,  65,  67,  69,  70,  71,  72,  74,  75,  77,
        79,  81,  83,  85,  86,  87,  89,  90,  91,  92,  94,  96,  97,
        98,  99,  100, 101, 103, 105, 107, 109, 111, 115, 117, 118, 119,
        120, 121, 122, 123, 124, 126, 127, 128, 129, 131, 133, 134, 135,
        137, 139, 141, 145, 149, 150, 151, 152, 154, 155};
    long ref_bzg2grg[155] = {
        0,  1,  2,  3,  4,  5,  6,  7,  7,  8,  9,  10, 10, 11, 11, 11, 11, 12,
        12, 12, 12, 13, 13, 14, 14, 15, 16, 17, 18, 18, 19, 19, 20, 21, 22, 23,
        23, 24, 25, 26, 27, 28, 29, 30, 30, 31, 31, 32, 32, 32, 32, 33, 33, 34,
        34, 35, 36, 36, 37, 37, 38, 39, 40, 41, 42, 43, 43, 44, 44, 45, 46, 47,
        48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 56, 57, 57, 58,
        59, 60, 61, 61, 62, 62, 63, 64, 65, 66, 67, 68, 68, 69, 69, 70, 70, 71,
        71, 72, 72, 73, 73, 73, 73, 74, 74, 75, 76, 77, 78, 79, 80, 81, 82, 82,
        83, 84, 85, 86, 86, 87, 87, 88, 89, 90, 90, 91, 91, 92, 92, 93, 93, 93,
        93, 94, 94, 94, 94, 95, 96, 97, 98, 98, 99};
    double rec_lattice[3][3] = {
        {3.111, -1.5555, 0}, {0, 2.694205031173388, 0}, {0, 0, 4.978}};
    long PS[3] = {0, 0, 0};
    long D_diag[3] = {1, 5, 20};  // [5, 5, 4]
    long Q[3][3] = {{-1, 0, 4}, {0, -1, 0}, {-1, 0, 5}};
    long bz_grid_addresses[180][3];
    long bz_map[101];
    long bzg2grg[180];
    long bz_size, i, j;

    bz_size = gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg,
                                            D_diag, Q, PS, rec_lattice, 2);
    ASSERT_EQ(155, bz_size);
    for (i = 0; i < 155; i++) {
        for (j = 0; j < 3; j++) {
            ASSERT_EQ(ref_bz_addresses[i][j], bz_grid_addresses[i][j]);
        }
    }
    for (i = 0; i < 101; i++) {
        ASSERT_EQ(ref_bz_map[i], bz_map[i]);
    }
    for (i = 0; i < 155; i++) {
        ASSERT_EQ(ref_bzg2grg[i], bzg2grg[i]);
    }
}
/**
 * @brief gridsys_get_bz_grid_addresses by wurtzite (angle 60)
 * @details Niggli reduction doesn't change (swap) basis vectors.
 * Return BZ grid addresses
 */
TEST(test_gridsys, test_gridsys_get_bz_grid_addresses_zincblende2) {
    long ref_bz_addresses[155][3] = {
        {0, 0, 0},     {0, 1, 0},     {0, 2, 0},    {0, -2, 0},
        {0, -1, 0},    {0, 0, 1},     {0, 1, 1},    {0, 2, 1},
        {0, -2, 1},    {0, 3, 1},     {0, -1, 1},   {-4, 0, -18},
        {0, 0, 2},     {-4, 1, -18},  {0, 1, 2},    {-4, 2, -18},
        {0, 2, 2},     {-4, 3, -18},  {1, -2, 2},   {0, 3, 2},
        {5, -2, 22},   {-4, -1, -18}, {1, -1, 2},   {0, -1, 2},
        {5, -1, 22},   {1, 0, 3},     {1, 1, 3},    {-4, 1, -17},
        {1, -3, 3},    {-4, 2, -17},  {1, -2, 3},   {1, -1, 3},
        {1, 0, 4},     {1, 1, 4},     {1, 2, 4},    {1, -3, 4},
        {1, -2, 4},    {1, -1, 4},    {1, 0, 5},    {1, 1, 5},
        {1, 2, 5},     {1, -2, 5},    {1, -1, 5},   {-3, 0, -14},
        {1, 0, 6},     {-3, 1, -14},  {1, 1, 6},    {-3, 2, -14},
        {1, 2, 6},     {-3, -2, -14}, {-3, 3, -14}, {1, -2, 6},
        {1, 3, 6},     {-3, -1, -14}, {1, -1, 6},   {-3, 0, -13},
        {-3, 1, -13},  {-3, 2, -13},  {-3, 3, -13}, {2, -2, 7},
        {-3, -1, -13}, {2, -1, 7},    {2, 0, 8},    {2, 1, 8},
        {-3, 1, -12},  {2, -3, 8},    {-3, 2, -12}, {2, -2, 8},
        {2, -1, 8},    {2, 0, 9},     {2, 1, 9},    {2, 2, 9},
        {2, -3, 9},    {2, -2, 9},    {2, -1, 9},   {-2, 0, -10},
        {2, 0, 10},    {-2, 1, -10},  {2, 1, 10},   {-2, 2, -10},
        {2, 2, 10},    {-2, -2, -10}, {2, -2, 10},  {-2, -1, -10},
        {2, -1, 10},   {-2, 0, -9},   {-2, 1, -9},  {-2, 2, -9},
        {-2, -2, -9},  {-2, 3, -9},   {-2, -1, -9}, {-2, 0, -8},
        {-2, 1, -8},   {-2, 2, -8},   {-2, 3, -8},  {3, -2, 12},
        {-2, -1, -8},  {3, -1, 12},   {3, 0, 13},   {3, 1, 13},
        {-2, 1, -7},   {3, -3, 13},   {-2, 2, -7},  {3, -2, 13},
        {3, -1, 13},   {-1, 0, -6},   {3, 0, 14},   {-1, 1, -6},
        {3, 1, 14},    {-1, 2, -6},   {-1, -3, -6}, {3, 2, 14},
        {3, -3, 14},   {-1, -2, -6},  {3, -2, 14},  {-1, -1, -6},
        {3, -1, 14},   {-1, 0, -5},   {-1, 1, -5},  {-1, 2, -5},
        {-1, -2, -5},  {-1, -1, -5},  {-1, 0, -4},  {-1, 1, -4},
        {-1, 2, -4},   {-1, -2, -4},  {-1, 3, -4},  {-1, -1, -4},
        {-1, 0, -3},   {-1, 1, -3},   {-1, 2, -3},  {-1, 3, -3},
        {4, -2, 17},   {-1, -1, -3},  {4, -1, 17},  {0, 0, -2},
        {4, 0, 18},    {0, 1, -2},    {-5, 1, -22}, {4, 1, 18},
        {-1, 1, -2},   {0, -3, -2},   {-5, 2, -22}, {4, -3, 18},
        {-1, 2, -2},   {0, -2, -2},   {4, -2, 18},  {0, -1, -2},
        {4, -1, 18},   {0, 0, -1},    {0, 1, -1},   {0, 2, -1},
        {0, -3, -1},   {0, -2, -1},   {0, -1, -1}};
    long ref_bz_map[101] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   10,  11,  13,  15,
        17,  21,  25,  26,  28,  30,  31,  32,  33,  34,  36,  37,  38,
        39,  40,  41,  42,  43,  45,  47,  49,  53,  55,  56,  57,  58,
        60,  62,  63,  65,  67,  68,  69,  70,  71,  73,  74,  75,  77,
        79,  81,  83,  85,  86,  87,  88,  90,  91,  92,  93,  94,  96,
        98,  99,  101, 103, 104, 105, 107, 109, 113, 115, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 133, 135,
        137, 141, 145, 147, 149, 150, 151, 153, 154, 155};
    long ref_bzg2grg[155] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  8,  9,  10, 10, 11, 11, 12, 12, 13,
        13, 13, 13, 14, 14, 14, 14, 15, 16, 16, 17, 17, 18, 19, 20, 21, 22, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 30, 31, 31, 32, 32, 33, 33, 33, 33, 34,
        34, 35, 36, 37, 38, 38, 39, 39, 40, 41, 41, 42, 42, 43, 44, 45, 46, 47,
        47, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 56, 57, 58, 58,
        59, 60, 61, 62, 63, 63, 64, 64, 65, 66, 66, 67, 67, 68, 69, 70, 70, 71,
        71, 72, 72, 72, 72, 73, 73, 74, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
        83, 84, 85, 86, 87, 88, 88, 89, 89, 90, 90, 91, 91, 91, 91, 92, 92, 92,
        92, 93, 93, 94, 94, 95, 96, 97, 97, 98, 99};
    double rec_lattice[3][3] = {{0.3214400514304082, 0.0, 0.0},
                                {0.1855835002216734, 0.3711670004433468, 0.0},
                                {0.0, 0.0, 0.20088388911209323}};
    long PS[3] = {0, 0, 0};
    long D_diag[3] = {1, 5, 20};  // [5, 5, 4]
    long Q[3][3] = {{-1, 0, 4}, {0, -1, 0}, {-1, 0, 5}};
    long bz_grid_addresses[180][3];
    long bz_map[101];
    long bzg2grg[180];
    long bz_size, i, j;

    bz_size = gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg,
                                            D_diag, Q, PS, rec_lattice, 2);
    ASSERT_EQ(155, bz_size);
    for (i = 0; i < 155; i++) {
        for (j = 0; j < 3; j++) {
            ASSERT_EQ(ref_bz_addresses[i][j], bz_grid_addresses[i][j]);
        }
    }
    for (i = 0; i < 101; i++) {
        ASSERT_EQ(ref_bz_map[i], bz_map[i]);
    }
    for (i = 0; i < 155; i++) {
        ASSERT_EQ(ref_bzg2grg[i], bzg2grg[i]);
    }
}
