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
    print *, input(1)
    print *, input(2)
end subroutine string_in_array

