! Minimal reproducer for issue #302: Pointer arrays silently skipped in direct-c
! f90wrap should warn when it cannot wrap a feature instead of silent skip

module pointer_mod
    implicit none

    integer, parameter :: dp = kind(1.0d0)

    type :: container_t
        real(dp), pointer :: data(:) => null()
        integer :: size = 0
    contains
        procedure :: init => container_init
        procedure :: free => container_free
    end type container_t

contains

    subroutine container_init(self, n)
        class(container_t), intent(inout) :: self
        integer, intent(in) :: n
        allocate(self%data(n))
        self%data = 0.0_dp
        self%size = n
    end subroutine container_init

    subroutine container_free(self)
        class(container_t), intent(inout) :: self
        if (associated(self%data)) deallocate(self%data)
        self%size = 0
    end subroutine container_free

end module pointer_mod
