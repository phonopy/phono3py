program test_gridsysf
    use, intrinsic :: iso_c_binding
    use gridsysf, only : gridsys_get_bz_grid_addresses
    implicit none

    call test_gridsys_get_bz_grid_addresses()

contains
    subroutine test_gridsys_get_bz_grid_addresses() bind(C)
        integer(c_long) :: bz_size
        integer(c_long) :: PS(3), D_diag(3), Q(3, 3), bz_grid_addresses(144, 3)
        integer(c_long) :: bz_map(76), bzg2grg(144)
        real(c_double) :: rec_lattice(3, 3)

        PS(:) = [0, 0, 0]
        D_diag(:) = [5, 5, 3]
        Q(:, :) = reshape([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3])
        rec_lattice(:, :) = reshape([0.3214400514304082, 0.0, 0.0, &
                                    0.1855835002216734, 0.3711670004433468, 0.0, &
                                    0.0, 0.0, 0.20088388911209323], [3, 3])

        bz_size = gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg, &
        D_diag, Q, PS, rec_lattice, int(2, c_long))

        if (.false.) error stop
    end subroutine test_gridsys_get_bz_grid_addresses
end program test_gridsysf
