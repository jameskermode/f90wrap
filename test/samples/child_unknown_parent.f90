module child_mod
    implicit none

    ! This type extends a parent type that is not included in the wrapping
    ! The wrapper should handle this gracefully by ignoring the inheritance
    type, extends(unknown_parent_type) :: child_type
        integer :: child_value
    contains
        procedure :: get_child_value
    end type child_type

contains

    function get_child_value(self) result(val)
        class(child_type), intent(in) :: self
        integer :: val
        val = self%child_value
    end function get_child_value

end module child_mod
