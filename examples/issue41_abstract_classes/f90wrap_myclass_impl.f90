! Module myclass_impl defined in file myclass_impl.f90

subroutine f90wrap_myclass_impl__myclass_impl_finalise__binding__mycla4a60(self)
    use myclass_impl, only: myclass_impl_t, myclass_impl_finalise
    implicit none

    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t
    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    self_ptr = transfer(self, self_ptr)
    !call myclass_impl_finalise(self=self_ptr%p)
    !deallocate(self_ptr%p)
end subroutine f90wrap_myclass_impl__myclass_impl_finalise__binding__mycla4a60

subroutine f90wrap_myclass_impl__myclass_impl_finalise(self)
    use myclass_impl, only: myclass_impl_t, myclass_impl_finalise
    implicit none

    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t
    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    self_ptr = transfer(self, self_ptr)
    print *, self
    !deallocate(self_ptr%p%obj)
end subroutine f90wrap_myclass_impl__myclass_impl_finalise

subroutine f90wrap_myclass_impl__myclass_impl_t_initialise(this)
    use myclass_impl, only: myclass_impl_t
    implicit none

    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t
    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_t_ptr_type) :: this_ptr
    integer, intent(out), dimension(2) :: this
    allocate(this_ptr%p)
    this = transfer(this_ptr, this)
end subroutine f90wrap_myclass_impl__myclass_impl_t_initialise

subroutine f90wrap_myclass_impl__get_value__binding__myclass_impl_t(self, value)
    use myclass_impl, only: myclass_impl_t
    implicit none

    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t
    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(out) :: value
    self_ptr = transfer(self, self_ptr)
    call self_ptr%p%obj%get_value(value=value)
end subroutine f90wrap_myclass_impl__get_value__binding__myclass_impl_t

subroutine f90wrap_myclass_impl__get_value_impl(self, value)
    use myclass_impl, only: myclass_impl_t, get_value_impl
    implicit none

    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t
    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(out) :: value
    self_ptr = transfer(self, self_ptr)
    call get_value_impl(self=self_ptr%p%obj, value=value)
end subroutine f90wrap_myclass_impl__get_value_impl

! End of module myclass_impl defined in file myclass_impl.f90
