module base
  implicit none
  real(kind=8) :: a, b  ! Module-level variables
contains
  function a_times_b_plus_c(a, b, c)
    real(kind=8) :: a_times_b_plus_c
    real(kind=8), intent(in) :: a, b, c
    a_times_b_plus_c = a * b + c
  end function a_times_b_plus_c
end module base
