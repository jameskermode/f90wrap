module global_mod
  implicit none
  integer :: number
contains
  subroutine print_number(number)
    integer, intent(in) :: number

    print *, number
  end subroutine print_number
end module global_mod
