! Regression example for optional complex scalar arguments.
!
! f90wrap previously dropped every optional complex scalar argument in
! transform.py, so present(z) could never be exercised from Python. Required
! complex scalars were always supported. This passes a complex value through
! present(z) on both the regular f2py path and the --direct-c path.
module m_complex
    implicit none
contains

    subroutine opt_complex(flag, z)
        integer, intent(out) :: flag
        complex, intent(in), optional :: z
        if (present(z)) then
            flag = nint(real(z)) + nint(aimag(z))
        else
            flag = -1
        end if
    end subroutine opt_complex

end module m_complex
