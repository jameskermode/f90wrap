module main

implicit none

type, abstract :: myclass_t
contains
    procedure(i_get_value), deferred :: get_value
    procedure(i_get_value2), deferred :: get_value2
end type myclass_t

type, extends(myclass_t) :: myclass_impl_t
contains
    procedure :: get_value => get_value_impl
    procedure :: get_value2 => get_value2_impl
    final :: myclass_impl_destroy
end type myclass_impl_t

abstract interface
    subroutine i_get_value(self, value)
        import :: myclass_t
        class(myclass_t), intent(in) :: self
        real, intent(out) :: value
    end subroutine i_get_value
end interface

abstract interface
    subroutine i_get_value2(self, value)
        import :: myclass_t
        class(myclass_t), intent(in) :: self
        integer, intent(out) :: value
    end subroutine i_get_value2
end interface

contains

function use_myclass() result(res)
    real :: res
    type(myclass_impl_t), allocatable :: obj

    obj = myclass_impl_t()
    call obj%get_value(res)
end function use_myclass

subroutine get_value_impl(self, value)
    class(myclass_impl_t), intent(in) :: self
    real, intent(out) :: value

    value = 1.0
end subroutine get_value_impl

subroutine get_value2_impl(self, value)
    class(myclass_impl_t), intent(in) :: self
    integer, intent(out) :: value

    value = 42
end subroutine get_value2_impl

subroutine myclass_impl_destroy(self)
    type(myclass_impl_t), intent(inout) :: self

    print *, "Finalising myclass_impl_t"
end subroutine myclass_impl_destroy

end module main
