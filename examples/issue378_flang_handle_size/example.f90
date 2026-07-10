module example
    implicit none

    type :: item
        integer :: value
        real, allocatable :: values(:)
    end type item
end module example
