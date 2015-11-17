module string_io

    implicit none

    CHARACTER(512) global_string

contains

    function func_generate_string(n) result(stringout)
        INTEGER(4) i, j, n, k
        CHARACTER(n) stringout
        DO i=1,n
            j = i + 1
            k = i + 33
            stringout(i:j) = achar(k)
        ENDDO
    end function func_generate_string

    function func_return_string() result(stringout)
        CHARACTER(51) stringout
        stringout = '-_-::this is a string with ASCII, / and 123...::-_-'
    end function func_return_string

    subroutine generate_string(n, stringout)
        INTEGER(4) i, j, k
        INTEGER(4), INTENT(in) :: n
        CHARACTER(n), INTENT(out) :: stringout
        DO i=1,n
            j = i + 1
            k = i + 33
            stringout(i:j) = achar(k)
        ENDDO
    end subroutine generate_string

    subroutine return_string(stringout)
        CHARACTER(51), INTENT(out) :: stringout
        stringout = '-_-::this is a string with ASCII, / and 123...::-_-'
    end subroutine return_string

    subroutine set_global_string(n, newstring)
        INTEGER(4), INTENT(in) :: n
        CHARACTER(n), INTENT(in) :: newstring
        global_string = newstring
    end subroutine set_global_string

    subroutine inout_string(n, stringinout)
        INTEGER(4) i, j
        INTEGER(4), INTENT(in) :: n
        CHARACTER(n), INTENT(inout) :: stringinout
        DO i=1,n
            j = i + 1
            stringinout(i:j) = 'Z'
        ENDDO
    end subroutine inout_string

end module string_io

