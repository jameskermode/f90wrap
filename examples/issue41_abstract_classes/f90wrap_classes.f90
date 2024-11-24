module myclass_impl_reference_storage
    use myclass_impl, only: myclass_impl_t
    implicit none
    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t

    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type
    type(myclass_impl_wrapper_t), target :: reference_storage(1024)
    integer :: reference_storage_size = 0
contains

    subroutine push_reference()
        print *, "push_reference"
        reference_storage_size = reference_storage_size + 1
    end subroutine push_reference
end module myclass_impl_reference_storage
