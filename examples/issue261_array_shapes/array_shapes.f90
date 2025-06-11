module array_shapes
    implicit none

    type :: container
        integer :: n_data
        real, dimension(:), allocatable :: data
    end type container

    contains

    function two_arrays_dynamic(y, x) result(res)
        real, intent(in), dimension(:) :: y
        real, intent(in), dimension(:) :: x
        real, dimension(size(x)) :: res

        res = x * y(1) + y(2)
    end function two_arrays_dynamic

    function two_arrays_fixed(y, x) result(res)
        real, intent(in), dimension(2) :: y
        real, intent(in), dimension(3) :: x
        real, dimension(3) :: res

        res = x * y(1) + y(2)
    end function two_arrays_fixed

    function two_arrays_mixed(y, x) result(res)
        real, intent(in), dimension(2) :: y
        real, intent(in), dimension(:) :: x
        real, dimension(size(x)) :: res

        res = x * y(1) + y(2)
    end function two_arrays_mixed

    function get_container(x) result(c)
        real, intent(in), dimension(:) :: x
        type(container) :: c

        c%n_data = size(x)
        c%data = x
    end function get_container

    function array_container_dynamic(c, y) result(res)
        type(container), intent(in) :: c
        real, intent(in), dimension(:) :: y
        real, dimension(c%n_data) :: res

        res = c%data * y(1) + y(2)
    end function array_container_dynamic

    function array_container_fixed(c, y) result(res)
        type(container), intent(in) :: c
        real, intent(in), dimension(2) :: y
        real, dimension(c%n_data) :: res

        res = c%data * y(1) + y(2)
    end function array_container_fixed

end module array_shapes
