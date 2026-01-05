subroutine test_selected_kind(i, a)
  implicit none
  integer(kind=selected_int_kind(9)), intent(out) :: i
  real(kind=selected_real_kind(13,300)), intent(out) :: a
end subroutine test_selected_kind

