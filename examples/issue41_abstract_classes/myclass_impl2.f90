module myclass_impl2

use myclass_base, only: myclass_t
implicit none

type, extends(myclass_t) :: myclass_impl2_t
contains
    procedure :: get_value => get_value_impl2
    final :: myclass_impl2_destroy
end type myclass_impl2_t

contains

subroutine get_value_impl2(self, value)
    class(myclass_impl2_t), intent(in) :: self
    real, intent(out) :: value

    value = 2.0
end subroutine get_value_impl2

subroutine myclass_impl2_destroy(self)
    type(myclass_impl2_t), intent(inout) :: self

    print *, "Finalising myclass_impl2_t"
end subroutine myclass_impl2_destroy

end module myclass_impl2
