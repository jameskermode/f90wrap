module myclass_impl_reference_storage
    use myclass_impl, only: myclass_impl_t
    implicit none
    type myclass_impl_wrapper_t
        class(myclass_impl_t), allocatable :: obj
    end type myclass_impl_wrapper_t

    type myclass_impl_t_ptr_type
        type(myclass_impl_wrapper_t), pointer :: p => NULL()
    end type myclass_impl_t_ptr_type

    type myclass_impl_wrapper_t_dict_t
        integer :: key(2)
        type(myclass_impl_wrapper_t) :: value
        type(myclass_impl_wrapper_t_dict_t), pointer :: next => NULL()
    end type myclass_impl_wrapper_t_dict_t

    type(myclass_impl_wrapper_t_dict_t), pointer :: reference_storage => NULL()
contains

    function add_reference(key) result(out)
        integer, intent(in) :: key(2)
        type(myclass_impl_wrapper_t_dict_t), pointer :: out

        if (.not. associated(reference_storage)) then
            allocate(reference_storage)
            reference_storage%key = key
            out => reference_storage
        else
            out => reference_storage
            do while (associated(out%next))
                out => out%next
            end do
            allocate(out%next)
            out%next%key = key
            out => out%next
        end if
    end function add_reference
end module myclass_impl_reference_storage
