module alloc_output
implicit none

type :: alloc_output_type
    real :: a
end type alloc_output_type

contains

function alloc_output_func() result(out)
    type(alloc_output_type), allocatable :: out
    allocate(out)
    out%a = 1.0
end function alloc_output_func

end module alloc_output
