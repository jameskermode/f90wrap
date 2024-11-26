! Module myclass_factory defined in file myclass_factory.f90

subroutine f90wrap_myclass_factory__myclass_create(ret_myobject, val)
    use myclass_factory, only: myclass_create
    use myclass, only: myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: ret_myobject_ptr
    integer, intent(out), dimension(2) :: ret_myobject
    real, intent(in) :: val
    allocate(ret_myobject_ptr%p)
    ret_myobject_ptr%p%obj = myclass_create(val=val)
    ret_myobject = transfer(ret_myobject_ptr, ret_myobject)
end subroutine f90wrap_myclass_factory__myclass_create

! End of module myclass_factory defined in file myclass_factory.f90
