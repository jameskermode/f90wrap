subroutine foo(a,b)
    real(kind=8), intent(in) :: a
    integer :: b
    print *, a, b, a*b
end subroutine foo
