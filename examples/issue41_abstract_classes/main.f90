program main
    use myclass_factory, only: create_myclass
    use myclass_base, only: myclass_t
    implicit none

    print *, "Start"

    call test()

    print *, "Done"

contains

subroutine test
    real :: x
    class(myclass_t), allocatable :: myobject

    myobject = create_myclass("impl")
    call myobject%get_value(x)

    print *, "Value: ", x
end subroutine test

end program main
