module kind_param_mod
    implicit none
    integer, parameter :: jprb = 8

contains

    function multiply(a, b) result(c)
        real(kind=jprb), intent(in) :: a, b
        real(kind=jprb) :: c
        c = a * b
    end function multiply

end module kind_param_mod
