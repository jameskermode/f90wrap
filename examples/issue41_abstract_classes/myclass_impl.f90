module myclass_impl

implicit none

type :: myclass_impl_t
contains
    procedure :: get_value => get_value_impl
    final :: myclass_impl_finalise
end type myclass_impl_t

contains

subroutine get_value_impl(self, value)
    class(myclass_impl_t), intent(in) :: self
    real, intent(out) :: value

    value = 1.0
end subroutine get_value_impl

subroutine myclass_impl_finalise(self)
    type(myclass_impl_t), intent(inout) :: self

    print *, "Finalising myclass_impl_t"
end subroutine myclass_impl_finalise

end module myclass_impl
