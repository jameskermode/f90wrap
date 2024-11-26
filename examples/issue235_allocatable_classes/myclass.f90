module myclass

implicit none

integer :: create_count = 0
integer :: destroy_count = 0

type :: myclass_t
    real :: val
contains
    procedure :: get_val => myclass_get_val
    procedure :: set_val => myclass_set_val
    final :: myclass_destroy
end type myclass_t

contains

subroutine myclass_get_val(self, val)
    class(myclass_t), intent(in) :: self
    real, intent(out) :: val

    val = self%val
end subroutine myclass_get_val

subroutine myclass_set_val(self, val)
    class(myclass_t), intent(inout) :: self
    real, intent(in) :: val

    self%val = val
end subroutine myclass_set_val

subroutine myclass_destroy(self)
    type(myclass_t), intent(inout) :: self

    destroy_count = destroy_count + 1
    print *, 'Destroying class_t with val = ', self%val
end subroutine myclass_destroy

end module myclass
