module alloc_output
implicit none

type :: alloc_output_type
    real :: a
end type alloc_output_type

contains

! This should be used by the wrapper generator
function alloc_output_type_func(val) result(out)
    real, intent(in) :: val
    type(alloc_output_type), allocatable :: out
    allocate(out)
    out%a = val
end function alloc_output_type_func


! This should be discarded by the wrapper generator
function alloc_output_intrinsic_func(val) result(out)
    real, intent(in) :: val
    real, allocatable :: out
    allocate(out)
    out = val
end function alloc_output_intrinsic_func


! This should be discarded by the wrapper generator
function alloc_output_array_func(val) result(out)
    real, intent(in) :: val(:)
    real, allocatable :: out(:)
    allocate(out(size(val)))
    out(:) = val
end function alloc_output_array_func


subroutine noalloc_output_subroutine(val, out)
    real, intent(in) :: val
    type(alloc_output_type), intent(inout) :: out
    out%a = val
end subroutine noalloc_output_subroutine

end module alloc_output
