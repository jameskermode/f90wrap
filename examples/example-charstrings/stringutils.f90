module stringutils

    implicit none
    private
    public :: testing_chararray, testing_intarray, roundtrip, chararray_roundtrip
    public :: fill_global_string

    CHARACTER(512) long_global_string

contains

    function chararray2string(n, chararray) result(string)
        INTEGER(4) i, j, n
        CHARACTER(n) string
        CHARACTER(1), DIMENSION(n) :: chararray
        DO i=1,n
            j = i + 1
            string(i:j) = chararray(i)
        ENDDO
    end function chararray2string

    function string2chararray(n, string) result(chararray)
        INTEGER(4) i, j, n
        CHARACTER(n) string
        CHARACTER(1), DIMENSION(n) :: chararray
        DO i=1,n
            j = i + 1
            chararray(i) = string(i:j)
        ENDDO
    end function string2chararray

    function string2intarray(n, string) result(intarray)
        INTEGER(4) i, j, n
        CHARACTER(n) string
        INTEGER(4), DIMENSION(n) :: intarray
        DO i=1,n
            j = i + 1
            intarray(i) = ichar(string(i:j))
        ENDDO
    end function string2intarray

    function intarray2string(n, intarray) result(string)
        INTEGER(4) n, i, j
        INTEGER(4), DIMENSION(n) :: intarray
        CHARACTER(n) string
        DO i=1,n
            j = i + 1
            string(i:j) = achar(intarray(i))
        ENDDO
    end function intarray2string

    function chararray2intarray(n, chararray) result(intarray)
        INTEGER(4) i, n
        INTEGER(4), DIMENSION(n) :: intarray
        CHARACTER(1), DIMENSION(n) :: chararray
        DO i=1,n
            intarray(i) = ichar(chararray(i))
        ENDDO
    end function chararray2intarray

    function intarray2chararray(n, intarray) result(chararray)
        INTEGER(4) i, n
        INTEGER(4), DIMENSION(n) :: intarray
        CHARACTER(1), DIMENSION(n) :: chararray
        DO i=1,n
            chararray(i) = achar(intarray(i))
        ENDDO
    end function intarray2chararray

    subroutine chararray_roundtrip(n, chararray, chararray_out)
        INTEGER(4), INTENT(in) :: n
        CHARACTER, DIMENSION(n), INTENT(in) :: chararray
        CHARACTER, DIMENSION(n), INTENT(inout) :: chararray_out
        CHARACTER(n) string
        string = chararray2string(n, chararray)
        print*,"PRINT string: ", string
        write(*,*) "WRITE string: ", string
        chararray_out = string2chararray(n, string)
    end subroutine chararray_roundtrip

    subroutine testing_chararray(n, chararray)
        INTEGER(4), INTENT(in) :: n
        CHARACTER(1), DIMENSION(n), INTENT(in):: chararray
        CHARACTER(n) :: string
        string = chararray2string(n, chararray)
        write(*,*) "constructed string (between commas): ,", string, ","
    end subroutine testing_chararray

    subroutine testing_intarray(n, intarray)
        INTEGER(4), INTENT(in) :: n
        INTEGER(4), DIMENSION(n), INTENT(in):: intarray
        CHARACTER(n) :: string
        string = intarray2string(n, intarray)
        write(*,*) "constructed string (between commas): ,", string, ","
    end subroutine testing_intarray

    subroutine roundtrip(n, intarray, intarray_out)
        INTEGER(4), INTENT(in) :: n
        INTEGER(4), DIMENSION(n), INTENT(in):: intarray
        INTEGER(4), DIMENSION(n), INTENT(out):: intarray_out
        CHARACTER(1), DIMENSION(n) :: chararray
        CHARACTER(n) :: string
        string = intarray2string(n, intarray)
        chararray = string2chararray(n, string)
        intarray_out = chararray2intarray(n, chararray)
    end subroutine roundtrip

    subroutine fill_global_string(n, intarray)
        INTEGER(4), INTENT(in) :: n
        INTEGER(4), DIMENSION(n), INTENT(in):: intarray
        CHARACTER(n) :: string

        string = intarray2string(n, intarray)
        write(*,*) "constructed string (between commas): ,", string, ","

        long_global_string = string
        write(*,*) "long_global_string (between commas): ,", long_global_string, ","

        long_global_string = long_global_string(1:n)
        write(*,*) "long_global_string(1:n) (between commas): ,", long_global_string, ","
    end subroutine fill_global_string

end module stringutils

