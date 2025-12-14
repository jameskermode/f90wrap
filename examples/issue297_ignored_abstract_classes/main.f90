module main

use myclass_base, only: myclass_t
implicit none

type, extends(myclass_t) :: myclass_impl_t
contains
    procedure :: get_value => get_value_impl
    final :: myclass_impl_destroy
end type myclass_impl_t

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

subroutine myclass_impl_destroy(self)
    type(myclass_impl_t), intent(inout) :: self

    print *, "Finalising myclass_impl_t"
end subroutine myclass_impl_destroy

end module main
