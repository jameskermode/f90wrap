! Module myclass_factory defined in file myclass_factory.f90

subroutine f90wrap_myclass_factory__create_myclass(ret_myobject, impl_type)
    use myclass_factory, only: create_myclass
    use myclass_impl, only: myclass_impl_t
    use myclass_impl_reference_storage, only: myclass_impl_wrapper_t, &
        myclass_impl_t_ptr_type, push_reference, reference_storage, &
        reference_storage_size
    implicit none

    type(myclass_impl_t_ptr_type) :: ret_myobject_ptr
    integer, intent(out), dimension(2) :: ret_myobject

    character*(*), intent(in) :: impl_type

    print *, "create_myclass: ", impl_type

    call push_reference()
    reference_storage(reference_storage_size)%obj = create_myclass(impl_type)

    ret_myobject_ptr%p => reference_storage(reference_storage_size)
    ret_myobject = transfer(ret_myobject_ptr, ret_myobject)
end subroutine f90wrap_myclass_factory__create_myclass

! End of module myclass_factory defined in file myclass_factory.f90
