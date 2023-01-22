#include <gtest/gtest.h>

extern "C" {
#include "gridsys.h"
}

/**
 * @brief gridsys_get_all_grid_addresses test
 * Return all GR-grid addresses of {(x, y, z)} where x runs fastest.
 */
TEST(test_gridsys, test_gridsys_get_all_grid_addresses) {
    long(*gr_grid_addresses)[3];
    long D_diag[3] = {3, 4, 5};
    long n, i, j, k, count;

    n = D_diag[0] * D_diag[1] * D_diag[2];
    gr_grid_addresses = (long(*)[3])malloc(sizeof(long[3]) * n);
    gridsys_get_all_grid_addresses(gr_grid_addresses, D_diag);

    count = 0;
    for (k = 0; k < D_diag[2]; k++) {
        for (j = 0; j < D_diag[1]; j++) {
            for (i = 0; i < D_diag[0]; i++) {
                ASSERT_EQ(gr_grid_addresses[count][0], i);
                ASSERT_EQ(gr_grid_addresses[count][1], j);
                ASSERT_EQ(gr_grid_addresses[count][2], k);
                count++;
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
