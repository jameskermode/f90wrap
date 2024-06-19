
module m_string_test
  implicit none
  private
  public :: string_in
  public :: string_to_string
  public :: string_to_string_array
  public :: string_out
  public :: string_out_optional
  public :: string_out_optional_array

contains

  subroutine string_in(input)
    character(len=*), intent(in) :: input
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

