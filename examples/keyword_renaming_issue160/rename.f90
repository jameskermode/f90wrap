module global

    implicit none
    
    type class2
        integer :: x = 456
    end type class2

    integer :: abc=0
    integer, parameter :: lambda=1
    integer :: with(9)
    
    contains
    
    subroutine is(a)
        implicit none
        integer,intent(in) :: a

        abc = abc + a
        return 
    end subroutine is
end module global

integer function in(a)
    implicit none

    integer, intent(in) :: a

    in = a + 1

    return
end function in
