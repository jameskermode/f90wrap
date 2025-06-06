module dta_tc
    implicit none
    
    type :: t_inner
        integer :: value
        contains
        procedure :: print => t_inner_print
    end type t_inner

    type :: t_outer
        integer :: value
        type(t_inner) :: inner
    end type t_outer

    interface t_inner
        module procedure new_inner
    end interface t_inner

    interface t_outer
        module procedure new_outer
    end interface t_outer

    contains

    function new_inner(value) result(inner)
        type(t_inner) :: inner
        integer, intent(in) :: value

        inner%value = value
    end function new_inner

    function new_outer(value, inner) result(node)
        type(t_outer) :: node
        integer, intent(in) :: value
        type(t_inner), intent(in) :: inner

        node%inner = inner
        node%value = value
    end function new_outer

    subroutine t_inner_print(inner)
        class(t_inner), intent(in) :: inner

        write(*,*) "value:", inner%value
    end subroutine t_inner_print

    function get_outer_inner(outer) result(inner)
        type(t_outer), intent(in) :: outer
        type(t_inner) :: inner

        inner = outer%inner
    end function get_outer_inner

end module dta_tc
