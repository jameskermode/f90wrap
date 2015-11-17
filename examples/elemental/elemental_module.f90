module elemental_module
implicit none
contains
elemental real(kind=8) function sinc(x)
real(kind=8), intent(in) :: x
if(abs(x).gt.1d-5) then
sinc = sin(x)/x
else
sinc = 1.0d0
endif
return
end function sinc
end module elemental_module