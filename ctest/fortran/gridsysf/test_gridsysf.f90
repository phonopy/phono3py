program test_gridsysf
   use, intrinsic :: iso_c_binding
   use gridsysf, only: gridsys_get_bz_grid_addresses
   implicit none

   write (*, '("[test_gridsys_get_bz_grid_addresses]")')
   call test_gridsys_get_bz_grid_addresses()

contains
   subroutine test_gridsys_get_bz_grid_addresses() bind(C)
      integer(c_long) :: bz_size
      integer(c_long) :: PS(3), D_diag(3), Q(3, 3), bz_grid_addresses(3, 144)
      integer(c_long) :: bz_map(76), bzg2grg(144)
      real(c_double) :: rec_lattice(3, 3)
      integer :: i, j

      integer :: ref_bz_addresses(3, 93)
      integer :: ref_bz_map(76)
      integer :: ref_bzg2grg(93)

      ref_bz_addresses(:, :) = reshape([ &
                                       0, 0, 0, 1, 0, 0, 2, 0, 0, -2, 0, 0, -1, 0, 0, &
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
                       68, 69, 71, 72, 73, 74, 76, 78, 79, 80, 81, 82, 83, 85, 87, 88, 89, 90, 92, 93]
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

      ! check bz_grid_addresses
      write (*, '("check bz_grid_addresses")', advance='no')
      do i = 1, 93
         do j = 1, 3
            if (bz_grid_addresses(j, i) /= ref_bz_addresses(j, i)) then
               print '("(", i0, ",", i0, "):", i0, " ", i0)', i, j, bz_grid_addresses(j, i), ref_bz_addresses(j, i)
               error stop
            end if
         end do
      end do
      write (*, '("  OK")')

      ! check bz_map
      write (*, '("check bz_map")', advance='no')
      do i = 1, 76
         if (bz_map(i) /= ref_bz_map(i)) then
            print '(i0, ":", i0, " ", i0)', i, bz_map(i), ref_bz_map(i)
            error stop
         end if
      end do
      write (*, '("  OK")')

      ! check bz_map
      write (*, '("check bzg2grg")', advance='no')
      do i = 1, 93
         if (bzg2grg(i) /= ref_bzg2grg(i)) then
            print '(i0, ":", i0, " ", i0)', i, bzg2grg(i), ref_bzg2grg(i)
            error stop
         end if
      end do
      write (*, '("  OK")')
    end subroutine test_gridsys_get_bz_grid_addresses
end program test_gridsysf
