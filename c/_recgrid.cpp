#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "recgrid.h"

namespace nb = nanobind;

long py_get_grid_index_from_address(nb::ndarray<> py_address,
                                    nb::ndarray<> py_D_diag) {
    long *address;
    long *D_diag;
    long gp;

    address = (long *)py_address.data();
    D_diag = (long *)py_D_diag.data();

    gp = recgrid_get_grid_index_from_address(address, D_diag);

    return gp;
}

long py_get_ir_grid_map(nb::ndarray<> py_grid_mapping_table,
                        nb::ndarray<> py_D_diag, nb::ndarray<> py_is_shift,
                        nb::ndarray<> py_rotations) {
    long *D_diag;
    long *is_shift;
    long(*rot)[3][3];
    long num_rot;

    long *grid_mapping_table;
    long num_ir;

    D_diag = (long *)py_D_diag.data();
    is_shift = (long *)py_is_shift.data();
    rot = (long(*)[3][3])py_rotations.data();
    num_rot = (long)py_rotations.shape(0);
    grid_mapping_table = (long *)py_grid_mapping_table.data();

    num_ir = recgrid_get_ir_grid_map(grid_mapping_table, rot, num_rot, D_diag,
                                     is_shift);
    return num_ir;
}

void py_get_gr_grid_addresses(nb::ndarray<> py_gr_grid_addresses,
                              nb::ndarray<> py_D_diag) {
    long(*gr_grid_addresses)[3];
    long *D_diag;

    gr_grid_addresses = (long(*)[3])py_gr_grid_addresses.data();
    D_diag = (long *)py_D_diag.data();

    recgrid_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

long py_get_reciprocal_rotations(nb::ndarray<> py_rec_rotations,
                                 nb::ndarray<> py_rotations,
                                 long is_time_reversal) {
    long(*rec_rotations)[3][3];
    long(*rotations)[3][3];
    long num_rot, num_rec_rot;

    rec_rotations = (long(*)[3][3])py_rec_rotations.data();
    rotations = (long(*)[3][3])py_rotations.data();
    num_rot = (long)py_rotations.shape(0);

    num_rec_rot = recgrid_get_reciprocal_point_group(
        rec_rotations, rotations, num_rot, is_time_reversal, 1);

    return num_rec_rot;
}

bool py_transform_rotations(nb::ndarray<> py_transformed_rotations,
                            nb::ndarray<> py_rotations, nb::ndarray<> py_D_diag,
                            nb::ndarray<> py_Q) {
    long(*transformed_rotations)[3][3];
    long(*rotations)[3][3];
    long *D_diag;
    long(*Q)[3];
    long num_rot, succeeded;

    transformed_rotations = (long(*)[3][3])py_transformed_rotations.data();
    rotations = (long(*)[3][3])py_rotations.data();
    D_diag = (long *)py_D_diag.data();
    Q = (long(*)[3])py_Q.data();
    num_rot = (long)py_transformed_rotations.shape(0);

    succeeded = recgrid_transform_rotations(transformed_rotations, rotations,
                                            num_rot, D_diag, Q);
    if (succeeded) {
        return true;
    } else {
        return false;
    }
}

bool py_get_snf3x3(nb::ndarray<> py_D_diag, nb::ndarray<> py_P,
                   nb::ndarray<> py_Q, nb::ndarray<> py_A) {
    long *D_diag;
    long(*P)[3];
    long(*Q)[3];
    long(*A)[3];
    long succeeded;

    D_diag = (long *)py_D_diag.data();
    P = (long(*)[3])py_P.data();
    Q = (long(*)[3])py_Q.data();
    A = (long(*)[3])py_A.data();

    succeeded = recgrid_get_snf3x3(D_diag, P, Q, A);
    if (succeeded) {
        return true;
    } else {
        return false;
    }
}

long py_get_bz_grid_addresses(nb::ndarray<> py_bz_grid_addresses,
                              nb::ndarray<> py_bz_map, nb::ndarray<> py_bzg2grg,
                              nb::ndarray<> py_D_diag, nb::ndarray<> py_Q,
                              nb::ndarray<> py_PS,
                              nb::ndarray<> py_reciprocal_lattice, long type) {
    long(*bz_grid_addresses)[3];
    long *bz_map;
    long *bzg2grg;
    long *D_diag;
    long(*Q)[3];
    long *PS;
    double(*reciprocal_lattice)[3];
    long num_total_gp;

    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    bz_map = (long *)py_bz_map.data();
    bzg2grg = (long *)py_bzg2grg.data();
    D_diag = (long *)py_D_diag.data();
    Q = (long(*)[3])py_Q.data();
    PS = (long *)py_PS.data();
    reciprocal_lattice = (double(*)[3])py_reciprocal_lattice.data();

    num_total_gp =
        recgrid_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg,
                                      D_diag, Q, PS, reciprocal_lattice, type);

    return num_total_gp;
}

long py_rotate_bz_grid_addresses(long bz_grid_index, nb::ndarray<> py_rotation,
                                 nb::ndarray<> py_bz_grid_addresses,
                                 nb::ndarray<> py_bz_map,
                                 nb::ndarray<> py_D_diag, nb::ndarray<> py_PS,
                                 long type) {
    long(*bz_grid_addresses)[3];
    long(*rotation)[3];
    long *bz_map;
    long *D_diag;
    long *PS;
    long ret_bz_gp;

    bz_grid_addresses = (long(*)[3])py_bz_grid_addresses.data();
    rotation = (long(*)[3])py_rotation.data();
    bz_map = (long *)py_bz_map.data();
    D_diag = (long *)py_D_diag.data();
    PS = (long *)py_PS.data();

    ret_bz_gp = recgrid_rotate_bz_grid_index(
        bz_grid_index, rotation, bz_grid_addresses, bz_map, D_diag, PS, type);

    return ret_bz_gp;
}

NB_MODULE(_recgrid, m) {
    m.def("grid_index_from_address", &py_get_grid_index_from_address);
    m.def("ir_grid_map", &py_get_ir_grid_map);
    m.def("gr_grid_addresses", &py_get_gr_grid_addresses);
    m.def("reciprocal_rotations", &py_get_reciprocal_rotations);
    m.def("transform_rotations", &py_transform_rotations);
    m.def("snf3x3", &py_get_snf3x3);
    m.def("bz_grid_addresses", &py_get_bz_grid_addresses);
    m.def("rotate_bz_grid_index", &py_rotate_bz_grid_addresses);
}
