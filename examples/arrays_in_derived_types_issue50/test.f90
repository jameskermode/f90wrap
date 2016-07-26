    module module_test

        type real_array
            real, dimension(6) :: item
        end type real_array

        contains

        subroutine testf(x)
            implicit none
            type(real_array) :: x

            print*, "This is received in fortran : ", x%item
            x%item(4) = 4
            print*, "This is sent back to python : ", x%item
        end subroutine testf

    end module module_test