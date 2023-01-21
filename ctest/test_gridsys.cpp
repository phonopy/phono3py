#include <gtest/gtest.h>

extern "C" {
#include "gridsys.h"
}

TEST(test_gridsys, test_gridsys_get_all_grid_addresses) {
    long(*gr_grid_addresses)[3];
    long D_diag[3];
    long n;

    D_diag[0] = 3;
    D_diag[1] = 4;
    D_diag[2] = 5;
    n = D_diag[0] * D_diag[1] * D_diag[2];
    gr_grid_addresses = (long(*)[3])malloc(sizeof(long[3]) * n);
    gridsys_get_all_grid_addresses(gr_grid_addresses, D_diag);
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

    num_ir = spg_get_ir_reciprocal_mesh(grid_address, grid_mapping_table, mesh,
                                        is_shift, 1, lattice, position, types,
                                        num_atom, 1e-5);
    ASSERT_EQ(num_ir, 4200);

    free(grid_address);
    grid_address = NULL;
    free(grid_mapping_table);
    grid_mapping_table = NULL;
}
 */
