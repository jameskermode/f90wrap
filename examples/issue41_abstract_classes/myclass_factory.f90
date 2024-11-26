module myclass_factory

use myclass_impl, only: myclass_impl_t
implicit none

contains

function create_myclass(impl_type) result(myobject)
    type(myclass_impl_t), allocatable :: myobject

    character(*), intent(in) :: impl_type

    select case(impl_type)
        case("impl")
            allocate(myclass_impl_t :: myobject)
        case default
            print *, "create_field_can: Unknown implementation: ", impl_type
            error stop
    end select
end function create_myclass

end module myclass_factory
