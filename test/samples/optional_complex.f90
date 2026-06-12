module optional_complex
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
end module optional_complex
