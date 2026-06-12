! Regression example for issue #369.
!
! With --direct-c, passing None for an OPTIONAL argument must make present(arg)
! return .false. inside Fortran. The bug passed a non-NULL pointer (scalar) or an
! allocated blank buffer (character), so present() wrongly returned .true.
!
! Each subroutine reports its decision through an intent(out) result so the Python
! test can assert which branch ran. Both the regular f90wrap+f2py path and the
! --direct-c path are exercised by the shared tests.py.
module m_optional
    implicit none
contains

    ! Optional scalar: the exact case from issue #369.
    subroutine scalar_default(x, n)
        real, intent(out) :: x
        integer, intent(in), optional :: n
        if (.not. present(n)) then
            x = 42.0
        else
            x = real(n)
        end if
    end subroutine scalar_default

    ! Optional character input.
    subroutine char_default(label, name)
        character(len=16), intent(out) :: label
        character(len=*), intent(in), optional :: name
        if (present(name)) then
            label = name
        else
            label = "ABSENT"
        end if
    end subroutine char_default

    ! Optional array input (already correct before the fix: a regression lock).
    subroutine array_default(s, v)
        integer, intent(out) :: s
        real, intent(in), optional :: v(:)
        if (present(v)) then
            s = size(v)
        else
            s = -1
        end if
    end subroutine array_default

end module m_optional
