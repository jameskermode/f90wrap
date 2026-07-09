module main
    implicit none
    public

    type :: testtype
        real :: value
    end type testtype

    type :: testclass
        real :: value
    contains
        procedure :: method => testclass_method
    end type testclass

    type(testtype), target :: global_testtype
    type(testclass), target :: global_testclass

contains

    function testclass_method(self) result(value)
        class(testclass), intent(in) :: self
        real :: value
        value = 2 * self%value
    end function testclass_method

end module main
