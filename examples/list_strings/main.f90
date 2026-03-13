module list_strings

    implicit none

    type mytype
       character(len=512) :: strings(3) = (/'a  ', 'ab ', 'abc'/)
    end type


contains

    function mytype_get_strings(self) result(stringsout)
        class(mytype), intent(in) :: self
        character(len=512)        :: stringsout(3)
        stringsout = self%strings
    end function mytype_get_strings

    subroutine mytype_set_strings(self, stringsin)
        class(mytype),  intent(inout) :: self
        character(len=*), intent(in)    :: stringsin(3)
        self%strings = stringsin
    end subroutine mytype_set_strings

    subroutine mytype_set_strings_implicit(self, stringsin)
        class(mytype),  intent(inout) :: self
        character(len=*), intent(in)    :: stringsin(:)
        self%strings = stringsin(1:3)
    end subroutine mytype_set_strings_implicit

    subroutine mytype_set_string(self, stringin, i)
        class(mytype),  intent(inout) :: self
        character(len=*), intent(in)    :: stringin
        integer :: i
        self%strings(i) = stringin
    end subroutine mytype_set_string

end module list_strings

