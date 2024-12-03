! Module myclass_base defined in file myclass_base.f90

subroutine f90wrap_myclass_base__myclass_t_finalise(this)
    use myclass_base, only: myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_myclass_base__myclass_t_finalise

subroutine f90wrap_myclass_base__get_value__binding__myclass_t(self, value)
    use myclass_impl, only: myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(out) :: value
    self_ptr = transfer(self, self_ptr)
    call self_ptr%p%obj%get_value(value=value)
end subroutine f90wrap_myclass_base__get_value__binding__myclass_t

! End of module myclass_base defined in file myclass_base.f90
