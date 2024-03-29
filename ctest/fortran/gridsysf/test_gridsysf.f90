program test_gridsysf
    use, intrinsic :: iso_c_binding
    use gridsysf, only: &
        gridsys_get_grid_index_from_address, &
        gridsys_rotate_grid_index, &
        gridsys_get_double_grid_address, &
        gridsys_get_grid_address_from_index, &
        gridsys_get_double_grid_index, &
        gridsys_get_bz_grid_addresses, &
        gridsys_rotate_bz_grid_index, &
        gridsys_get_triplets_at_q, &
        gridsys_get_bz_triplets_at_q
    implicit none

    integer(c_long) :: wurtzite_rec_rotations_without_time_reversal(3, 3, 12)
    integer(c_long) :: wurtzite_tilde_rec_rotations_without_time_reversal(3, 3, 12)

    wurtzite_rec_rotations_without_time_reversal(:, :, :) = &
        reshape([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, -1, 0, 0, 0, 0, 1, &
                 0, 1, 0, -1, -1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, 1, &
                 -1, -1, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 1, 0, 0, 0, 1, &
                 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, 1, &
                 1, 0, 0, -1, -1, 0, 0, 0, 1, 0, -1, 0, -1, 0, 0, 0, 0, 1, &
                 -1, -1, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 1, 1, 0, 0, 0, 1], [3, 3, 12])

    wurtzite_tilde_rec_rotations_without_time_reversal(:, :, :) = &
        reshape([1, 0, 0, 0, 1, 0, 0, 0, 1, &
                 1, -1, 0, -5, 0, -2, 0, 3, 1, &
                 6, -1, 2, -5, -1, -2, -15, 3, -5, &
                 11, 0, 4, 0, -1, 0, -30, 0, -11, &
                 11, 1, 4, 5, 0, 2, -30, -3, -11, &
                 6, 1, 2, 5, 1, 2, -15, -3, -5, &
                 6, -1, 2, 5, 0, 2, -15, 3, -5, &
                 1, -1, 0, 0, -1, 0, 0, 3, 1, &
                 1, 0, 0, -5, -1, -2, 0, 0, 1, &
                 6, 1, 2, -5, 0, -2, -15, -3, -5, &
                 11, 1, 4, 0, 1, 0, -30, -3, -11, &
                 11, 0, 4, 5, 1, 2, -30, 0, -11], [3, 3, 12])

    write (*, '("[test_gridsys_get_grid_index_from_address]")')
    call test_gridsys_get_grid_index_from_address()
    write (*, '("[test_gridsys_rotate_grid_index]")')
    call test_gridsys_rotate_grid_index()
    write (*, '("[test_gridsys_rotate_bz_grid_index]")')
    call test_gridsys_rotate_bz_grid_index()
    write (*, '("[test_gridsys_get_bz_grid_addresses_wurtzite]")')
    call test_gridsys_get_bz_grid_addresses_wurtzite()
    write (*, '("[test_gridsys_get_triplets_at_q_wurtzite]")')
    call test_gridsys_get_triplets_at_q_wurtzite()
    write (*, '("[test_gridsys_get_bz_triplets_at_q_wurtzite_force_SNF]")')
    call test_gridsys_get_bz_triplets_at_q_wurtzite_force_SNF()

contains
    subroutine test_gridsys_get_grid_index_from_address() bind(C)
        integer(c_long) :: address(3)
        integer(c_long) :: D_diag(3)
        integer :: i, j, k, grid_index, ret_grid_index

        D_diag(:) = [3, 4, 5]
        grid_index = 0
        do k = 0, D_diag(3) - 1
            address(3) = k
            do j = 0, D_diag(2) - 1
                address(2) = j
                do i = 0, D_diag(1) - 1
                    address(1) = i
                    ret_grid_index = gridsys_get_grid_index_from_address(address, D_diag)
                    call assert_int(ret_grid_index, grid_index)
                    grid_index = grid_index + 1
                end do
            end do
        end do
        write (*, '("  OK")')
    end subroutine test_gridsys_get_grid_index_from_address

    subroutine test_gridsys_rotate_grid_index() bind(C)
        integer(c_long) :: address(3), d_address(3), rot_address(3)
        integer(c_long) :: D_diag(3, 2)
        integer(c_long) :: PS(3, 2, 2)
        integer(c_long) :: grid_index
        integer(c_long) :: rotation(3, 3)
        integer :: rot_grid_index, ref_rot_grid_index, i_tilde, i_ps, i_rot

        rotation(:, :) = reshape([0, 1, 0, -1, 0, 0, 0, 0, -1], [3, 3])
        D_diag(:, :) = reshape([1, 5, 15, 5, 5, 3], [3, 2])
        PS(:, :, :) = reshape([ &
                              0, 0, 0, -2, 0, 5, 0, 0, 0, 0, 0, 1], &
                              [3, 2, 2])

        do i_tilde = 1, 2
        do i_ps = 1, 2
        do i_rot = 1, 12
            if (i_tilde == 1) then
                rotation(:, :) = &
                    wurtzite_tilde_rec_rotations_without_time_reversal(:, :, i_rot)
            else
                rotation(:, :) = &
                    wurtzite_rec_rotations_without_time_reversal(:, :, i_rot)
            end if
            do grid_index = 0, 74
                call gridsys_get_grid_address_from_index(address, grid_index, &
                                                         D_diag(:, i_tilde))
                call gridsys_get_double_grid_address(d_address, address, &
                                                     PS(:, i_ps, i_tilde))
                rot_address = matmul(transpose(rotation), d_address)
                ref_rot_grid_index = gridsys_get_double_grid_index( &
                                     rot_address, D_diag(:, i_tilde), &
                                     PS(:, i_ps, i_tilde))
                rot_grid_index = gridsys_rotate_grid_index( &
                                 grid_index, rotation, D_diag(:, i_tilde), &
                                 PS(:, i_ps, i_tilde))
                call assert_int(rot_grid_index, ref_rot_grid_index)
            end do
        end do
        end do
        end do
        write (*, '("  OK")')
    end subroutine test_gridsys_rotate_grid_index

    subroutine test_gridsys_rotate_bz_grid_index() bind(C)
        integer(c_long) :: d_address(3), rot_address(3), ref_d_address(3)
        integer(c_long) :: D_diag(3, 2)
        integer(c_long) :: PS(3, 2, 2)
        integer(c_long) :: grid_index
        integer(c_long) :: rotation(3, 3)
        integer :: i_tilde, i_ps, i_rot, rot_bz_gp, bz_size
        integer(c_long) :: Q(3, 3, 2)
        real(c_double) :: rec_lattice(3, 3)
        integer(c_long) :: bz_grid_addresses(3, 144)
        integer(c_long) :: bz_map(76)
        integer(c_long) :: bzg2grg(144)

        Q(:, :, :) = reshape([-1, 0, -6, 0, -1, 0, -1, 0, -5, &
                              1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3, 2])
        rec_lattice(:, :) = reshape([0.3214400514304082, 0.0, 0.0, &
                                     0.1855835002216734, 0.3711670004433468, 0.0, &
                                     0.0, 0.0, 0.20088388911209323], [3, 3])
        D_diag(:, :) = reshape([1, 5, 15, 5, 5, 3], [3, 2])
        PS(:, :, :) = reshape([ &
                              0, 0, 0, -2, 0, 5, 0, 0, 0, 0, 0, 1], &
                              [3, 2, 2])

        do i_tilde = 1, 2
        do i_ps = 1, 2
            bz_size = gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg, &
                                                    D_diag(:, i_tilde), Q(:, :, i_tilde), PS(:, i_ps, i_tilde), &
                                                    rec_lattice, int(2, c_long))
            do i_rot = 1, 12
                if (i_tilde == 1) then
                    rotation(:, :) = &
                        wurtzite_tilde_rec_rotations_without_time_reversal(:, :, i_rot)
                else
                    rotation(:, :) = &
                        wurtzite_rec_rotations_without_time_reversal(:, :, i_rot)
                end if
                do grid_index = 0, 74
                    call gridsys_get_double_grid_address( &
                        d_address, bz_grid_addresses(:, grid_index + 1), &
                        PS(:, i_ps, i_tilde))
                    rot_address = matmul(transpose(rotation), d_address)
                    rot_bz_gp = gridsys_rotate_bz_grid_index( &
                                grid_index, rotation, &
                                bz_grid_addresses, bz_map, D_diag(:, i_tilde), &
                                PS(:, i_ps, i_tilde), int(2, c_long));
                    call gridsys_get_double_grid_address( &
                        ref_d_address, &
                        bz_grid_addresses(:, rot_bz_gp + 1), PS(:, i_ps, i_tilde))
                    call assert_1D_array_c_long(ref_d_address, rot_address, 3)
                end do
            end do
        end do
        end do
        write (*, '("  OK")')
    end subroutine test_gridsys_rotate_bz_grid_index

    subroutine test_gridsys_get_bz_grid_addresses_wurtzite() bind(C)
        integer(c_long) :: bz_size
        integer(c_long) :: PS(3), D_diag(3), Q(3, 3), bz_grid_addresses(3, 144)
        integer(c_long) :: bz_map(76), bzg2grg(144)
        real(c_double) :: rec_lattice(3, 3)

        integer(c_long) :: ref_bz_grid_addresses(3, 93)
        integer(c_long) :: ref_bz_map(76)
        integer(c_long) :: ref_bzg2grg(93)

        ref_bz_grid_addresses(:, :) = &
            reshape([0, 0, 0, 1, 0, 0, 2, 0, 0, -2, 0, 0, -1, 0, 0, &
                     0, 1, 0, 1, 1, 0, 2, 1, 0, -3, 1, 0, -2, 1, 0, &
                     -1, 1, 0, 0, 2, 0, 1, 2, 0, 1, -3, 0, 2, -3, 0, &
                     -3, 2, 0, -2, 2, 0, -1, 2, 0, 0, -2, 0, 1, -2, 0, &
                     2, -2, 0, -2, 3, 0, 3, -2, 0, -1, -2, 0, -1, 3, 0, &
                     0, -1, 0, 1, -1, 0, 2, -1, 0, -2, -1, 0, 3, -1, 0, &
                     -1, -1, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, -2, 0, 1, &
                     -1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, -3, 1, 1, &
                     -2, 1, 1, -1, 1, 1, 0, 2, 1, 1, 2, 1, 1, -3, 1, &
                     2, -3, 1, -3, 2, 1, -2, 2, 1, -1, 2, 1, 0, -2, 1, &
                     1, -2, 1, 2, -2, 1, -2, 3, 1, 3, -2, 1, -1, -2, 1, &
                     -1, 3, 1, 0, -1, 1, 1, -1, 1, 2, -1, 1, -2, -1, 1, &
                     3, -1, 1, -1, -1, 1, 0, 0, -1, 1, 0, -1, 2, 0, -1, &
                     -2, 0, -1, -1, 0, -1, 0, 1, -1, 1, 1, -1, 2, 1, -1, &
                     -3, 1, -1, -2, 1, -1, -1, 1, -1, 0, 2, -1, 1, 2, -1, &
                     1, -3, -1, 2, -3, -1, -3, 2, -1, -2, 2, -1, -1, 2, -1, &
                     0, -2, -1, 1, -2, -1, 2, -2, -1, -2, 3, -1, 3, -2, -1, &
                     -1, -2, -1, -1, 3, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, &
                     -2, -1, -1, 3, -1, -1, -1, -1, -1], [3, 93])
        ref_bz_map(:) = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, &
                         17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 30, 31, 32, 33, &
                         34, 35, 36, 37, 38, 40, 41, 42, 43, 45, 47, 48, 49, 50, &
                         51, 52, 54, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, &
                         68, 69, 71, 72, 73, 74, 76, 78, 79, 80, 81, 82, 83, 85, &
                         87, 88, 89, 90, 92, 93]
        ref_bzg2grg(:) = [ &
                         0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 12, 12, &
                         13, 14, 15, 16, 17, 18, 18, 19, 19, 20, 21, 22, 23, 23, 24, 25, &
                         26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 36, 37, 37, 38, &
                         39, 40, 41, 42, 43, 43, 44, 44, 45, 46, 47, 48, 48, 49, 50, 51, &
                         52, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 61, 62, 62, 63, 64, &
                         65, 66, 67, 68, 68, 69, 69, 70, 71, 72, 73, 73, 74]

        PS(:) = [0, 0, 0]
        D_diag(:) = [5, 5, 3]
        Q(:, :) = reshape([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3])
        rec_lattice(:, :) = reshape([0.3214400514304082, 0.0, 0.0, &
                                     0.1855835002216734, 0.3711670004433468, 0.0, &
                                     0.0, 0.0, 0.20088388911209323], [3, 3])

        bz_size = gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, bzg2grg, &
                                                D_diag, Q, PS, rec_lattice, int(2, c_long))

        write (*, '("check bz_grid_addresses")', advance='no')
        call assert_2D_array_c_long(bz_grid_addresses, ref_bz_grid_addresses, &
                                    shape(ref_bz_grid_addresses))
        write (*, '("  OK")')

        write (*, '("check bz_map")', advance='no')
        call assert_1D_array_c_long(bz_map, ref_bz_map, 76)
        write (*, '("  OK")')

        write (*, '("check bzg2grg")', advance='no')
        call assert_1D_array_c_long(bzg2grg, ref_bzg2grg, 93)
        write (*, '("  OK")')
    end subroutine test_gridsys_get_bz_grid_addresses_wurtzite

    subroutine test_gridsys_get_triplets_at_q_wurtzite() bind(C)
        integer(c_long) :: D_diag(3)
        integer(c_long) :: map_triplets(36), map_q(36)
        integer(c_long) :: grid_point, is_time_reversal, num_rot, swappable
        integer :: i, j, k, n_ir_triplets

        integer :: ref_num_triplets(4)
        integer(c_long) :: ref_map_triplets(36, 4), ref_map_q(36, 4)

        grid_point = 1
        D_diag(:) = [3, 3, 4]
        num_rot = 12
        ref_num_triplets(:) = [12, 18, 14, 24]
        ref_map_triplets(:, :) = &
            reshape([ &
                    0, 1, 0, 3, 3, 5, 5, 3, 3, 9, 10, 9, 12, 12, 14, 14, 12, 12, &
                    18, 19, 18, 21, 21, 23, 23, 21, 21, 9, 10, 9, 12, 12, 14, 14, 12, 12, &
                    0, 1, 2, 3, 4, 5, 5, 3, 4, 9, 10, 11, 12, 13, 14, 14, 12, 13, &
                    18, 19, 20, 21, 22, 23, 23, 21, 22, 9, 10, 11, 12, 13, 14, 14, 12, 13, &
                    0, 1, 0, 3, 3, 5, 5, 3, 3, 9, 10, 11, &
                    12, 13, 14, 14, 12, 13, 18, 19, 18, 21, 21, 23, &
                    23, 21, 21, 11, 10, 9, 13, 12, 14, 14, 13, 12, &
                    0, 1, 2, 3, 4, 5, 5, 3, 4, 9, 10, 11, &
                    12, 13, 14, 14, 12, 13, 18, 19, 20, 21, 22, 23, &
                    23, 21, 22, 27, 28, 29, 30, 31, 32, 32, 30, 31 &
                    ], [36, 4])
        ref_map_q(:, :) = &
            reshape([ &
                    0, 1, 2, 3, 4, 5, 5, 3, 4, 9, 10, 11, 12, 13, 14, 14, 12, 13, &
                    18, 19, 20, 21, 22, 23, 23, 21, 22, 9, 10, 11, 12, 13, 14, 14, 12, 13, &
                    0, 1, 2, 3, 4, 5, 5, 3, 4, 9, 10, 11, 12, 13, 14, 14, 12, 13, &
                    18, 19, 20, 21, 22, 23, 23, 21, 22, 9, 10, 11, 12, 13, 14, 14, 12, 13, &
                    0, 1, 2, 3, 4, 5, 5, 3, 4, 9, 10, 11, &
                    12, 13, 14, 14, 12, 13, 18, 19, 20, 21, 22, 23, &
                    23, 21, 22, 27, 28, 29, 30, 31, 32, 32, 30, 31, &
                    0, 1, 2, 3, 4, 5, 5, 3, 4, 9, 10, 11, &
                    12, 13, 14, 14, 12, 13, 18, 19, 20, 21, 22, 23, &
                    23, 21, 22, 27, 28, 29, 30, 31, 32, 32, 30, 31], &
                    [36, 4])

        k = 1
        do i = 0, 1
            do j = 0, 1
                is_time_reversal = 1 - i
                swappable = 1 - j
                n_ir_triplets = gridsys_get_triplets_at_q( &
                                map_triplets, map_q, grid_point, &
                                D_diag, is_time_reversal, num_rot, &
                                wurtzite_rec_rotations_without_time_reversal, &
                                swappable)

                call assert_int(n_ir_triplets, ref_num_triplets(k))
                write (*, '("check map_triplets")', advance='no')
                call assert_1D_array_c_long(map_triplets, ref_map_triplets(:, k), 36)
                write (*, '("  OK")')
                write (*, '("check map_q")', advance='no')
                call assert_1D_array_c_long(map_q, ref_map_q(:, k), 36)
                write (*, '("  OK")')
                k = k + 1
            end do
        end do
    end subroutine test_gridsys_get_triplets_at_q_wurtzite

    subroutine test_gridsys_get_bz_triplets_at_q_wurtzite_force_SNF() bind(C)
        integer(c_long) :: D_diag(3)
        integer(c_long) :: PS(3)
        integer(c_long) :: Q(3, 3)
        integer(c_long) :: map_triplets(75), map_q(75)
        real(c_double) :: rec_lattice(3, 3)
        integer(c_long) :: grid_point, is_time_reversal, num_rot, num_gp, swappable
        integer :: i, j, k, num_triplets_1, num_triplets_2, bz_size, i_grgp
        integer(c_long) :: triplets(3, 75)
        integer(c_long) :: bz_grid_addresses(3, 108)
        integer(c_long) :: bz_map(75)
        integer(c_long) :: bzg2grg(108)

        integer :: ref_num_triplets(4, 2)
        integer(c_long) :: ref_triplets(3, 45, 4, 2)
        integer(c_long) :: ref_ir_weights(45, 4, 2)
        integer :: shape_of_array(2)
        integer :: grgp(2)

        grgp(:) = [1, 7]
        ref_num_triplets(:, :) = reshape([18, 24, 30, 45, 24, 24, 45, 45], [4, 2])
        ref_triplets(:, :, :, :) = &
            reshape([ &
                    1, 0, 4, 1, 1, 3, 1, 2, 2, 1, 5, 91, 1, 7, 90, &
                    1, 10, 87, 1, 12, 85, 1, 13, 84, 1, 14, 83, 1, 18, 79, &
                    1, 19, 77, 1, 23, 74, 1, 31, 66, 1, 32, 65, 1, 33, 64, &
                    1, 36, 60, 1, 38, 59, 1, 41, 56, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    1, 0, 4, 1, 1, 3, 1, 2, 2, 1, 5, 91, 1, 7, 90, &
                    1, 8, 88, 1, 10, 87, 1, 11, 86, 1, 12, 85, 1, 13, 84, &
                    1, 14, 83, 1, 15, 81, 1, 17, 80, 1, 18, 79, 1, 19, 77, &
                    1, 23, 74, 1, 31, 66, 1, 32, 65, 1, 33, 64, 1, 34, 63, &
                    1, 35, 62, 1, 36, 60, 1, 38, 59, 1, 41, 56, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    1, 0, 4, 1, 1, 3, 1, 2, 2, 1, 3, 1, 1, 4, 0, &
                    1, 5, 91, 1, 7, 90, 1, 8, 88, 1, 10, 87, 1, 11, 86, &
                    1, 12, 85, 1, 13, 84, 1, 14, 83, 1, 15, 81, 1, 17, 80, &
                    1, 18, 79, 1, 19, 77, 1, 21, 76, 1, 22, 75, 1, 23, 74, &
                    1, 31, 66, 1, 32, 65, 1, 33, 64, 1, 34, 63, 1, 35, 62, &
                    1, 36, 60, 1, 38, 59, 1, 39, 57, 1, 41, 56, 1, 42, 55, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    1, 0, 4, 1, 1, 3, 1, 2, 2, 1, 3, 1, 1, 4, 0, &
                    1, 5, 91, 1, 7, 90, 1, 8, 88, 1, 10, 87, 1, 11, 86, &
                    1, 12, 85, 1, 13, 84, 1, 14, 83, 1, 15, 81, 1, 17, 80, &
                    1, 18, 79, 1, 19, 77, 1, 21, 76, 1, 22, 75, 1, 23, 74, &
                    1, 31, 66, 1, 32, 65, 1, 33, 64, 1, 34, 63, 1, 35, 62, &
                    1, 36, 60, 1, 38, 59, 1, 39, 57, 1, 41, 56, 1, 42, 55, &
                    1, 43, 54, 1, 44, 53, 1, 45, 52, 1, 46, 50, 1, 48, 49, &
                    1, 62, 35, 1, 63, 34, 1, 64, 33, 1, 65, 32, 1, 66, 31, &
                    1, 67, 29, 1, 69, 28, 1, 70, 26, 1, 72, 25, 1, 73, 24, &
                    8, 0, 89, 8, 1, 88, 8, 2, 87, 8, 3, 86, 9, 4, 92, &
                    8, 5, 84, 8, 6, 82, 8, 8, 81, 8, 10, 80, 8, 11, 85, &
                    8, 12, 78, 8, 13, 76, 8, 14, 75, 8, 17, 79, 8, 19, 71, &
                    8, 20, 69, 9, 22, 67, 8, 24, 65, 8, 27, 62, 8, 29, 66, &
                    8, 31, 58, 8, 32, 57, 8, 40, 50, 8, 48, 48, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    8, 0, 89, 8, 1, 88, 8, 2, 87, 8, 3, 86, 9, 4, 92, &
                    8, 5, 84, 8, 6, 82, 8, 8, 81, 8, 10, 80, 8, 11, 85, &
                    8, 12, 78, 8, 13, 76, 8, 14, 75, 8, 17, 79, 8, 19, 71, &
                    8, 20, 69, 9, 22, 67, 8, 24, 65, 8, 27, 62, 8, 29, 66, &
                    8, 31, 58, 8, 32, 57, 8, 40, 50, 8, 48, 48, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    8, 0, 89, 8, 1, 88, 8, 2, 87, 8, 3, 86, 9, 4, 92, &
                    8, 5, 84, 8, 6, 82, 8, 8, 81, 8, 10, 80, 8, 11, 85, &
                    8, 12, 78, 8, 13, 76, 8, 14, 75, 8, 16, 74, 8, 17, 79, &
                    8, 19, 71, 8, 20, 69, 9, 22, 67, 9, 23, 73, 8, 24, 65, &
                    8, 25, 64, 8, 27, 62, 8, 29, 66, 8, 31, 58, 8, 32, 57, &
                    8, 33, 56, 8, 34, 55, 8, 40, 50, 9, 41, 49, 9, 42, 54, &
                    8, 43, 46, 8, 44, 45, 8, 48, 48, 8, 50, 40, 8, 51, 38, &
                    9, 53, 36, 8, 58, 31, 9, 61, 35, 8, 62, 27, 8, 63, 26, &
                    8, 71, 19, 9, 72, 18, 8, 79, 17, 8, 81, 8, 8, 89, 0, &
                    8, 0, 89, 8, 1, 88, 8, 2, 87, 8, 3, 86, 9, 4, 92, &
                    8, 5, 84, 8, 6, 82, 8, 8, 81, 8, 10, 80, 8, 11, 85, &
                    8, 12, 78, 8, 13, 76, 8, 14, 75, 8, 16, 74, 8, 17, 79, &
                    8, 19, 71, 8, 20, 69, 9, 22, 67, 9, 23, 73, 8, 24, 65, &
                    8, 25, 64, 8, 27, 62, 8, 29, 66, 8, 31, 58, 8, 32, 57, &
                    8, 33, 56, 8, 34, 55, 8, 40, 50, 9, 41, 49, 9, 42, 54, &
                    8, 43, 46, 8, 44, 45, 8, 48, 48, 8, 50, 40, 8, 51, 38, &
                    9, 53, 36, 8, 58, 31, 9, 61, 35, 8, 62, 27, 8, 63, 26, &
                    8, 71, 19, 9, 72, 18, 8, 79, 17, 8, 81, 8, 8, 89, 0 &
                    ], &
                    [3, 45, 4, 2])

        ref_ir_weights(:, :, :) = &
            reshape([ &
                    2, 2, 1, 8, 4, 8, 4, 8, 8, 4, 4, 2, 4, 4, 2, 4, 2, 4, 0, 0, 0, 0, 0, &
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    2, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 2, &
                    4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, &
                    2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, &
                    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, &
                    2, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 2, 2, 4, 4, 2, 2, 4, 2, 4, &
                    2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    2, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 2, 2, 4, 4, 2, 2, 4, 2, 4, &
                    2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                    1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, &
                    1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, &
                    1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, &
                    2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1 &
                    ], [45, 4, 2])

        num_rot = 12
        num_gp = 75
        D_diag(:) = [1, 5, 15]
        PS(:) = [0, 0, 0]
        Q(:, :) = reshape([-1, 0, -6, 0, -1, 0, -1, 0, -5], [3, 3])
        rec_lattice(:, :) = reshape([0.3214400514304082, 0.0, 0.0, &
                                     0.1855835002216734, 0.3711670004433468, 0.0, &
                                     0.0, 0.0, 0.20088388911209323], [3, 3])

        bz_size = gridsys_get_bz_grid_addresses( &
                  bz_grid_addresses, bz_map, bzg2grg, &
                  D_diag, Q, PS, rec_lattice, int(2, c_long))
        call assert_int(bz_size, 93)
        do i_grgp = 1, 2
            grid_point = grgp(i_grgp)
            i = 1
            do j = 0, 1
                swappable = 1 - j
                do k = 0, 1
                    is_time_reversal = 1 - k
                    num_triplets_1 = gridsys_get_triplets_at_q( &
                                     map_triplets, map_q, grid_point, D_diag, &
                                     is_time_reversal, num_rot, &
                                     wurtzite_tilde_rec_rotations_without_time_reversal, swappable)
                    num_triplets_2 = gridsys_get_bz_triplets_at_q( &
                                     triplets, bz_map(grid_point + 1), bz_grid_addresses, bz_map, &
                                     map_triplets, num_gp, D_diag, Q, int(2, c_long))
                    write (*, '("swappable:", i0, ", is_time_reversal:", i0)', advance='no') swappable, is_time_reversal
                    call assert_int(num_triplets_1, num_triplets_2)
                    call assert_int(num_triplets_1, ref_num_triplets(i, i_grgp))
                    shape_of_array(:) = [3, num_triplets_2]
                    call assert_2D_array_c_long( &
                        triplets, ref_triplets(:, :, i, i_grgp), shape_of_array)
                    write (*, '("  OK")')
                    i = i + 1
                end do
            end do
        end do

    end subroutine test_gridsys_get_bz_triplets_at_q_wurtzite_force_SNF

    subroutine assert_int(val, ref_val)
        integer, intent(in) :: val, ref_val
        if (val /= ref_val) then
            print '()'
            print '(i0, "/=", i0)', val, ref_val
            error stop
        end if
    end subroutine assert_int

    subroutine assert_1D_array_c_long(vals, ref_vals, size_of_array)
        integer(c_long), intent(in) :: vals(:)
        integer(c_long), intent(in) :: ref_vals(:)
        integer, intent(in) :: size_of_array
        integer :: i

        do i = 1, size_of_array
            if (vals(i) /= ref_vals(i)) then
                print '()'
                print '(i0, ":", i0, " ", i0)', i, vals(i), ref_vals(i)
                error stop
            end if
        end do
    end subroutine assert_1D_array_c_long

    subroutine assert_2D_array_c_long(vals, ref_vals, shape_of_array)
        integer(c_long), intent(in) :: vals(:, :)
        integer(c_long), intent(in) :: ref_vals(:, :)
        integer, intent(in) :: shape_of_array(:)
        integer :: i, j

        do i = 1, shape_of_array(1)
            do j = 1, shape_of_array(2)
                if (vals(j, i) /= ref_vals(j, i)) then
                    print '()'
                    print '("(", i0, ",", i0, "):", i0, " ", i0)', i, j, vals(j, i), ref_vals(j, i)
                    error stop
                end if
            end do
        end do
    end subroutine assert_2D_array_c_long

end program test_gridsysf
