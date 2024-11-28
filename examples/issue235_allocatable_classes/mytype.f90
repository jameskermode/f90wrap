module mytype

    implicit none

    integer :: create_count = 0
    integer :: destroy_count = 0

    type :: mytype_t
        real :: val
    contains
        final :: mytype_destroy
    end type mytype_t

    contains

    function mytype_create(val) result(self)
        type(mytype_t) :: self
        real, intent(in) :: val

        self%val = val
        create_count = create_count + 1
    end function mytype_create

    subroutine mytype_destroy(self)
        type(mytype_t), intent(inout) :: self

        destroy_count = destroy_count + 1
        print *, 'Destroying mytype_t with val = ', self%val
    end subroutine mytype_destroy

end module mytype
