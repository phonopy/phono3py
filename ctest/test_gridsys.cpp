#include <gtest/gtest.h>

extern "C" {
#include "gridsys.h"
#include "utils.h"
}

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

/* TEST(test_gridsys, test_gridsys_get_all_grid_addresses) {
    double lattice[3][3] = {{4, 0, 0}, {0, 4, 0}, {0, 0, 3}};
    double position[][3] = {
        {0, 0, 0},     {0.5, 0.5, 0.5}, {0.3, 0.3, 0},
        {0.7, 0.7, 0}, {0.2, 0.8, 0.5}, {0.8, 0.2, 0.5},
    };
    int num_ir, retval;
    int types[] = {1, 1, 2, 2, 2, 2};
    int num_atom = 6;
    int m = 40;
    int mesh[3];
    int is_shift[] = {1, 1, 1};
    int(*grid_address)[3];
    int *grid_mapping_table;

    mesh[0] = m;
    mesh[1] = m;
    mesh[2] = m;
    grid_address = (int(*)[3])malloc(sizeof(int[3]) * m * m * m);
    grid_mapping_table = (int *)malloc(sizeof(int) * m * m * m);

    printf("*** spg_get_ir_reciprocal_mesh of Rutile structure ***:\n");

    num_ir = spg_get_ir_reciprocal_mesh(grid_address, grid_mapping_table,
mesh, is_shift, 1, lattice, position, types, num_atom, 1e-5);
    ASSERT_EQ(num_ir, 4200);

    free(grid_address);
    grid_address = NULL;
    free(grid_mapping_table);
    grid_mapping_table = NULL;
}
 */
