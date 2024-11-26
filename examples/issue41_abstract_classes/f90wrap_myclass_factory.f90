! Module myclass_factory defined in file myclass_factory.f90

subroutine f90wrap_myclass_factory__create_myclass(ret_myobject, impl_type)
    use myclass_impl, only: myclass_impl_t
    use myclass_factory, only: create_myclass
    implicit none

    type :: myclass_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_wrapper_t
    type myclass_impl_t_ptr_type
        type(myclass_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_t_ptr_type) :: ret_myobject_ptr
    integer, intent(out), dimension(2) :: ret_myobject
    character*(*), intent(in) :: impl_type
    allocate(ret_myobject_ptr%p)
    ret_myobject_ptr%p%obj = create_myclass(impl_type=impl_type)
    ret_myobject = transfer(ret_myobject_ptr, ret_myobject)
end subroutine f90wrap_myclass_factory__create_myclass

! End of module myclass_factory defined in file myclass_factory.f90
