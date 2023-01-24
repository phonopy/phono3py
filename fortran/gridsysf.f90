! Copyright (C) 2021 Atsushi Togo
! All rights reserved.

! This file is part of kspclib.

! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:

! * Redistributions of source code must retain the above copyright
!   notice, this list of conditions and the following disclaimer.

! * Redistributions in binary form must reproduce the above copyright
!   notice, this list of conditions and the following disclaimer in
!   the documentation and/or other materials provided with the
!   distribution.

! * Neither the name of the kspclib project nor the names of its
!   contributors may be used to endorse or promote products derived
!   from this software without specific prior written permission.

! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.

module gridsysf

  use iso_c_binding, only: c_char, c_long, c_double

  implicit none

  private

  interface

     subroutine gridsys_get_all_grid_addresses(gr_grid_addresses, D_diag) bind(c)
       import c_long
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(inout) :: gr_grid_addresses(3, *)
     end subroutine gridsys_get_all_grid_addresses


     subroutine gridsys_get_double_grid_address(address_double, address, &
          PS) bind(c)
       import c_long
       integer(c_long), intent(inout) :: address_double(3)
       integer(c_long), intent(in) :: address(3)
       integer(c_long), intent(in) :: PS(3)
     end subroutine gridsys_get_double_grid_address


     subroutine gridsys_get_grid_address_from_index(address, grid_index, D_diag) &
          bind(c)
       import c_long
       integer(c_long), intent(inout) :: address(3)
       integer(c_long), intent(in), value :: grid_index
       integer(c_long), intent(in) :: D_diag(3)
     end subroutine gridsys_get_grid_address_from_index


     function gridsys_rotate_grid_index(grid_index, rotation, D_diag, PS) &
          bind(c)
       import c_long
       integer(c_long), intent(in), value :: grid_index
       integer(c_long), intent(in) :: rotation(3, 3)
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(in) :: PS(3)
       integer(c_long) :: gridsys_rotate_grid_index
     end function gridsys_rotate_grid_index


     function gridsys_get_double_grid_index(address_double, D_diag, PS) bind(c)
       import c_long
       integer(c_long), intent(in) :: address_double(3)
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(in) :: PS(3)
       integer(c_long) :: gridsys_get_double_grid_index
     end function gridsys_get_double_grid_index


     function gridsys_get_reciprocal_point_group(rec_rotations, rotations, &
          num_rot, is_time_reversal) bind(c)
       import c_long
       integer(c_long), intent(inout) :: rec_rotations(3, 3, 48)
       integer(c_long), intent(in) :: rotations(3, 3, *)
       integer(c_long), intent(in), value :: num_rot
       integer(c_long), intent(in), value :: is_time_reversal
       integer(c_long) :: gridsys_get_reciprocal_point_group
     end function gridsys_get_reciprocal_point_group


     function gridsys_get_snf3x3(D_diag, P, Q, A) bind(c)
       import c_long
       integer(c_long), intent(inout) :: D_diag(3)
       integer(c_long), intent(inout) :: P(3, 3)
       integer(c_long), intent(inout) :: Q(3, 3)
       integer(c_long), intent(in) :: A(3, 3)
       integer(c_long) :: gridsys_get_snf3x3
     end function gridsys_get_snf3x3


     function gridsys_transform_rotations(transformed_rots, &
          rotations, num_rot, D_diag, Q) bind(c)
       import c_long
       integer(c_long), intent(inout) :: transformed_rots(3, 3, *)
       integer(c_long), intent(in) :: rotations(3, 3, *)
       integer(c_long), intent(in), value :: num_rot
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(in) :: Q(3, 3)
       integer(c_long) :: gridsys_transform_rotations
     end function gridsys_transform_rotations


     function gridsys_get_thm_integration_weight(omega, &
          tetrahedra_omegas, function_char) bind(c)
       import c_char, c_double
       real(c_double), intent(in), value :: omega
       real(c_double), intent(in) :: tetrahedra_omegas(4, 24)
       character(kind=c_char), intent(in), value :: function_char
       real(c_double) :: gridsys_get_thm_integration_weight
     end function gridsys_get_thm_integration_weight


     subroutine gridsys_get_thm_relative_grid_address(relative_grid_addresses, &
          rec_lattice) bind(c)
       import c_long, c_double
       integer(c_long), intent(inout) :: relative_grid_addresses(3, 4, 24)
       real(c_double), intent(in) :: rec_lattice(3, 3)
     end subroutine gridsys_get_thm_relative_grid_address


     subroutine gridsys_get_ir_grid_map(ir_grid_map, rotations, num_rot, &
          D_diag, PS) bind(c)
       import c_long
       integer(c_long), intent(inout) :: ir_grid_map(*)
       integer(c_long), intent(in) :: rotations(3, 3, *)
       integer(c_long), intent(in), value :: num_rot
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(in) :: PS(3)
     end subroutine gridsys_get_ir_grid_map


     function gridsys_get_bz_grid_addresses(bz_grid_addresses, bz_map, &
          bzg2grg, D_diag, Q, PS, rec_lattice, type) bind(c)
       import c_long, c_double
       integer(c_long), intent(inout) :: bz_grid_addresses(3, *)
       integer(c_long), intent(inout) :: bz_map(*)
       integer(c_long), intent(inout) :: bzg2grg(*)
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(in) :: Q(3, 3)
       integer(c_long), intent(in) :: PS(3)
       real(c_double), intent(in) :: rec_lattice(3, 3)
       integer(c_long), intent(in), value :: type
       integer(c_long) :: gridsys_get_bz_grid_addresses
     end function gridsys_get_bz_grid_addresses


     function gridsys_get_triplets_at_q(map_triplets, map_q, grid_point, &
          D_diag, is_time_reversal, num_rot, rec_rotations, swappable) bind(c)
       import c_long
       integer(c_long), intent(inout) :: map_triplets(*)
       integer(c_long), intent(inout) :: map_q(*)
       integer(c_long), intent(in), value ::  grid_point
       integer(c_long), intent(in) :: D_diag(3)
       integer(c_long), intent(in), value :: is_time_reversal
       integer(c_long), intent(in), value :: num_rot
       integer(c_long), intent(in) :: rec_rotations(3, 3, *)
       integer(c_long), intent(in), value :: swappable
       integer(c_long) :: gridsys_get_triplets_at_q
     end function gridsys_get_triplets_at_q

  end interface


  public :: gridsys_get_all_grid_addresses, &
       gridsys_get_double_grid_address, &
       gridsys_get_grid_address_from_index, &
       gridsys_rotate_grid_index, &
       gridsys_get_thm_relative_grid_address, &
       gridsys_get_double_grid_index, &
       gridsys_get_reciprocal_point_group, &
       gridsys_get_snf3x3, &
       gridsys_transform_rotations, &
       gridsys_get_thm_integration_weight, &
       gridsys_get_ir_grid_map, &
       gridsys_get_bz_grid_addresses, &
       gridsys_get_triplets_at_q

end module gridsysf