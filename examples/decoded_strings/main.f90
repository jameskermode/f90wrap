module m_test
  implicit none

contains

    subroutine return_string(stringout)
        character(51), intent(out) :: stringout
        stringout = '-_-::this is a string with ASCII, / and 123...::-_-  '
    end subroutine return_string

    function func_return_string() result(stringout)
        character(51) stringout
        stringout = '-_-::this is a string with ASCII, / and 123...::-_-  '
    end function func_return_string

end module m_test
