module test_module

integer, parameter :: m=5
integer, parameter :: n=3
type test_type2
    real, dimension(m) :: y
end type test_type2

contains

subroutine test_routine4(x1, x2, x3, x4, x5, x6)
    real, dimension(m) :: x1
    type(test_type2), dimension(n) :: x2
    ! another dimension(n) array, to see if the wrapper creates two super types on not (it should not)
    type(test_type2), dimension(n) :: x3
    ! another array of the same type, but different dimension : should lead to a different super-type
    type(test_type2), dimension(m) :: x4
    ! "normal" array of types
    type(test_type2), dimension(5) :: x5
    ! x6 is completely implicit : should be real, and will be treated as intent(in) unless the command-line flag --default-to-intent-inout is used

    x1(1) = 42
    x2(2)%y(2) = 42
    x3(3)%y(3) = 42
    x4(4)%y(4) = 42
    x5(5)%y(5) = 42
    x6 = x6 + 1

end subroutine test_routine4


end module test_module

