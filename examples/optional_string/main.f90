
module m_string_test
  implicit none
  private
  public :: string_in
  public :: string_in_array
  public :: string_in_array_hardcoded_size
  public :: string_to_string
  public :: string_to_string_array
  public :: string_out
  public :: string_out_optional
  public :: string_out_optional_array

contains

  subroutine string_in(input)
    character(len=*), intent(in) :: input
  end subroutine

  subroutine string_in_array(input)
    character(len=6), intent(in) :: input(:)
    integer :: i

    if (input(1).ne."one   ") then
      call f90wrap_abort("First char input is incorrect, should be 'one', but is '" // input(1) // "'" )
    endif

    if (input(2).ne."two   ") then
      call f90wrap_abort("Second char input is incorrect, should be 'two', but is '" // input(2) // "'" )
    endif
  end subroutine

  subroutine string_in_array_hardcoded_size(input)
    character(len=6), intent(in) :: input(2)
    integer :: i

    if (input(1).ne."one   ") then
      call f90wrap_abort("First char input is incorrect, should be 'one', but is '" // input(1) // "'" )
    endif

    if (input(2).ne."two   ") then
      call f90wrap_abort("Second char input is incorrect, should be 'two', but is '" // input(2) // "'" )
    endif
  end subroutine

  subroutine string_to_string(input,output)
    character(len=*), intent(in)  :: input
    character(len=*), intent(out) :: output
    output=input
  end subroutine string_to_string

  subroutine string_to_string_array(input,output)
    character(len=*), intent(in) :: input(:)
    character(len=*), intent(out) :: output(:)
    output=input
  end subroutine string_to_string_array

  subroutine string_out(output)
    character(len=13), intent(out) :: output
    output = "output string"
  end subroutine

  subroutine string_out_optional(output)
    character(len=13), optional, intent(out) :: output
    if (present(output)) then
      output = "output string"
    endif
  end subroutine

  subroutine string_out_optional_array(output)
    character(len=13), optional, intent(out) :: output(:)
    if (present(output)) then
      output = "output string"
    endif
  end subroutine

end module m_string_test
