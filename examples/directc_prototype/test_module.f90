module test_interface_prototype
    implicit none

    interface generic_proc
        module procedure proc_int
        module procedure proc_real
    end interface generic_proc

contains

    subroutine proc_int(x, result)
        integer, intent(in) :: x
        integer, intent(out) :: result
        result = x * 2
    end subroutine proc_int

    subroutine proc_real(x, result)
        real, intent(in) :: x
        real, intent(out) :: result
        result = x * 2.0
    end subroutine proc_real

end module test_interface_prototype
