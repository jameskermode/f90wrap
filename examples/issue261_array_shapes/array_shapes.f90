module array_shapes
    implicit none

    type :: container
        integer :: n_data
        real, dimension(:), allocatable :: data
    end type container

    contains

    function one_array_dynamic(x) result(res)
        real, intent(in), dimension(:) :: x
        real, dimension(size(x)) :: res

        res = x * 2.0
    end function one_array_dynamic

    function one_array_fixed(x) result(res)
        real, intent(in), dimension(3) :: x
        real, dimension(3) :: res

        res = x * 2.0
    end function one_array_fixed

    function one_array_fixed_range(x) result(res)
        real, intent(in), dimension(3) :: x
        real, dimension(1:3) :: res

        res = x * 2.0
    end function one_array_fixed_range

    function one_array_explicit(x, n) result(res)
        real, intent(in), dimension(n) :: x
        integer, intent(in) :: n
        real, dimension(n) :: res

        res = x * 2.0
    end function one_array_explicit

    function one_array_explicit_range(x, n) result(res)
        real, intent(in), dimension(n) :: x
        integer, intent(in) :: n
        real, dimension(1:n) :: res

        res = x * 2.0
    end function one_array_explicit_range

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

    ! ----- !

    function two_arrays_2d_dynamic(y, x) result(res)
        real, intent(in), dimension(:) :: y
        real, intent(in), dimension(:) :: x
        real, dimension(size(x), size(y)) :: res
        integer :: i

        do i = 1, size(y)
            res(:, i) = x * y(i)
        end do
    end function two_arrays_2d_dynamic

    function two_arrays_2d_fixed(y, x) result(res)
        real, intent(in), dimension(2) :: y
        real, intent(in), dimension(3) :: x
        real, dimension(3,2) :: res
        integer :: i

        do i = 1, size(y)
            res(:, i) = x * y(i)
        end do
    end function two_arrays_2d_fixed

    function two_arrays_2d_fixed_whitespace(y, x) result(res)
        real, intent(in), dimension(2) :: y
        real, intent(in), dimension(3) :: x
        real, dimension(3, 2) :: res
        integer :: i

        do i = 1, size(y)
            res(:, i) = x * y(i)
        end do
    end function two_arrays_2d_fixed_whitespace

    function two_arrays_2d_mixed(y, x) result(res)
        real, intent(in), dimension(2) :: y
        real, intent(in), dimension(:) :: x
        real, dimension(size(x), 2) :: res
        integer :: i

        do i = 1, size(y)
            res(:, i) = x * y(i)
        end do
    end function two_arrays_2d_mixed

    ! ----- !

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
    
    function array_container_dynamic_2d(n, c, y) result(res)
        type(container), intent(in) :: c
        integer, intent(in) :: n
        real, intent(in), dimension(:) :: y
        real, dimension(c%n_data, n) :: res
        integer :: i

        do i = 1, n
            res(:, i) = c%data * y(i)
        end do
    end function array_container_dynamic_2d

end module array_shapes
