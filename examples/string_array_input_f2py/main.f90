subroutine string_in_array(n0, input, output)
  implicit none

  integer :: n0
  !f2py intent(hide), depend(input) :: n0 = shape(input,0)
  character*(*), intent(in), dimension(n0) :: input
  integer, intent(out) :: output

  if(input(1) .eq. "one" .and. input(2) .eq. "two") then
    output=0
  else
    output=1
  endif
end subroutine string_in_array

subroutine string_in_array_optional(n0, input, output)
  implicit none

  integer :: n0
  !f2py intent(hide), depend(input) :: n0 = shape(input,0)
  character*(*), intent(in), optional, dimension(n0) :: input
  integer, intent(out) :: output

  output=2
  if(present(input)) then
    if(input(1) .eq. "one" .and. input(2) .eq. "two") then
      output=0
    else
      output=1
    endif
  endif
end subroutine string_in_array_optional
