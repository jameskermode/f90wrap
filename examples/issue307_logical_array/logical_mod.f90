module logical_mod
  implicit none
  private

  public :: get_flags, set_flags

contains

  subroutine get_flags(n, flags)
    integer, intent(in) :: n
    logical, dimension(n), intent(out) :: flags
    integer :: i

    do i = 1, n
      flags(i) = mod(i, 2) == 0
    end do
  end subroutine get_flags

  subroutine set_flags(n, flags, result)
    integer, intent(in) :: n
    logical, dimension(n), intent(in) :: flags
    integer, intent(out) :: result
    integer :: i

    result = 0
    do i = 1, n
      if (flags(i)) result = result + 1
    end do
  end subroutine set_flags

end module logical_mod
