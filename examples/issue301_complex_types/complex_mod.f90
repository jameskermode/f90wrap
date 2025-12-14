! Minimal reproducer for issue #301: Complex types fail without kind_map
! Complex is a standard Fortran type and should work without extra configuration

module complex_mod
    implicit none

    integer, parameter :: dp = kind(1.0d0)

contains

    subroutine set_complex(z)
        complex(dp), intent(out) :: z
        z = (1.0_dp, 2.0_dp)
    end subroutine set_complex

    subroutine get_parts(z, re, im)
        complex(dp), intent(in) :: z
        real(dp), intent(out) :: re, im
        re = real(z)
        im = aimag(z)
    end subroutine get_parts

    function add_complex(a, b) result(c)
        complex(dp), intent(in) :: a, b
        complex(dp) :: c
        c = a + b
    end function add_complex

end module complex_mod
